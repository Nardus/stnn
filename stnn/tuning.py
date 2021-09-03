"""
Functions to tune an stnn model
"""

import os
import torch
import torch.nn.functional as functional
import torch.optim as optim

from ray import tune

from .utils import rmse
from .datasets import load_data
from .stnn import SpatioTemporalNN


"""
Default configuration
"""
default_config = {
    # stnn neural net params
    "activation": "identity",  # dynamic module activation function (either identity or tanh)
    "nhid": 0,                 # dynamic function hidden size
    "nlayers": 1,              # dynamic function num layers
    "dropout_f": 0.0,          # latent factors dropout
    "dropout_d": 0.0,          # dynamic function dropout
    "lambd": 0.1,              # lambda between reconstruction and dynamic losses

    # optim params
    "lr": 3e-3,                # Initial learning rate
    "beta1": 0.0,              # adam beta1
    "beta2": 0.999,            # adam beta2
    "eps": 1e-9,               # adam eps
    "wd": 1e-6,                # weight decay
    "wd_z": 1e-7,              # weight decay on latent factors
    "l2_z": 0.0,               # l2 between consecutives latent factors
    "l1_rel": 1e-8,            # l1 regularization on relation discovery mode

    # run-specific (user-specified) params: these currently have no defaults
    "obs_periodicity": None,   # periodicity of weights at initialization
    "mode": None,              # mode to run model in (None | "refine" | "discover")
    "nepoch": None,            # number of epochs to train for
    "patience": None,          # number of epoch to wait before trigerring learning rate decay
    "batch_size": None         # batch size for training
}


def train_stnn_model(config, obs_path, relation_paths, train_inds, random_seed, 
                     allow_gpu = False, validation_prop=0.2, checkpoint_dir=None):
    """Train an stnn model using a dictionary of hyper-parameters
    
    Parameters
    ----------
    config : dict
        A dictionary of model hyper-parameters
    obs_path : str
        Path to observation data.
    relation_paths: list
        List of paths to spatial relation matrices
    train_inds : list
        List of indices specifying the data rows to use for training and validation
    random_seed: int
        A random seed to pass to all utilized GPUs (not used when allow_gpu=False)
    allow_gpu : bool
        If true, all available GPUs will be used 
    validation_prop : float, optional
        Proportion of training data to hold back for model validation (i.e. evaluation of tuning)
    checkpoint_dir : str
        Directory for loading/saving model checkpoints
    
    Raises
    ----------
    ValueError : 
        Input invalid (e.g. missing hyperparameter values or values outside their valid range)
    """
    # Device setup
    if allow_gpu:
        device = torch.device("cuda:0")
        torch.cuda.manual_seed_all(random_seed)
    else:
        device = torch.device("cpu")

    # Load data
    all_data, relations = load_data(obs_path, relation_paths)

    # Data-related quantities / dimensions
    nd = 1                  # dimension of input observations (1 = univariate time series)
    nt = all_data.shape[0]  # number of observed timesteps
    nx = all_data.shape[1]  # number of locations (columns)
    nz = nd                 # dimension of observations in latent space

    if config["obs_periodicity"] > 0:
        # if periode < nt, latent factors will be initialized with a periodicity
        periode = config["obs_periodicity"]
    else:
        periode = nt

    # Split data
    if validation_prop <=0 or validation_prop >= 1:
        raise ValueError("validation_prop should be in (0, 1)")
    
    nt_train = len(train_inds)  # number of rows in training data
    nt_actual = int(nt_train * (1 - validation_prop))
    train_subset_inds = train_inds[:nt_actual]
    val_subset_inds = train_inds[nt_actual:nt_train]

    train_data = all_data[train_subset_inds]
    validation_data = all_data[val_subset_inds]

    train_data = train_data.to(device)
    validation_data = validation_data.to(device)
    relations = relations.to(device)

    # Set up model
    model = SpatioTemporalNN(relations, nx, nt_actual, nd, nz, 
                             mode = config["mode"], 
                             nhid = config["nhid"], 
                             nlayers = config["nlayers"],
                             dropout_f = config["dropout_f"], 
                             dropout_d = config["dropout_d"], 
                             activation = config["activation"],
                             periode = periode)
    model = model.to(device)

    # Train inputs
    # - indices
    t_idx = torch.arange(nt_actual, out = torch.LongTensor()).unsqueeze(1).expand(nt_actual, nx).contiguous()
    x_idx = torch.arange(nx, out = torch.LongTensor()).expand_as(t_idx).contiguous()

    # - dynamic
    idx_dyn = torch.stack((t_idx[1:], x_idx[1:])).view(2, -1).to(device)
    nex_dyn = idx_dyn.size(1)

    # - decoder
    idx_dec = torch.stack((t_idx, x_idx)).view(2, -1).to(device)
    nex_dec = idx_dec.size(1)

    # Optimizer
    params = [{"params": model.factors_parameters(), "weight_decay": config["wd_z"]},
              {"params": model.dynamic.parameters()},
              {"params": model.decoder.parameters()}]

    if config["mode"] in ("refine", "discover"):
        params.append({"params": model.rel_parameters(), "weight_decay": 0.})

    optimizer = optim.Adam(params, 
                           lr=config["lr"], 
                           betas=(config["beta1"], config["beta2"]), 
                           eps=config["eps"], weight_decay=config["wd"])

    if config["patience"] > 0:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = config["patience"])

    # Load from existing checkpoint:
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # Training
    lr = config["lr"]

    for epoch in range(config["nepoch"]):
        # ------------------------ Train ------------------------
        model.train()
        
        # --- decoder ---
        idx_perm = torch.randperm(nex_dec).to(device)
        batches = idx_perm.split(config["batch_size"])

        for i, batch in enumerate(batches):
            optimizer.zero_grad()

            # data
            input_t = idx_dec[0][batch]
            input_x = idx_dec[1][batch]
            x_target = train_data[input_t, input_x]

            # closure
            x_rec = model.dec_closure(input_t, input_x)
            mse_dec = functional.mse_loss(x_rec, x_target)

            # backward
            mse_dec.backward()

            # step
            optimizer.step()

        # --- dynamic ---
        idx_perm = torch.randperm(nex_dyn).to(device)
        batches = idx_perm.split(config["batch_size"])

        for i, batch in enumerate(batches):
            optimizer.zero_grad()

            # data
            input_t = idx_dyn[0][batch]
            input_x = idx_dyn[1][batch]

            # closure
            z_inf = model.factors[input_t, input_x]
            z_pred = model.dyn_closure(input_t - 1, input_x)

            # loss
            mse_dyn = z_pred.sub(z_inf).pow(2).mean()
            loss_dyn = mse_dyn * config["lambd"]

            if config["l2_z"] > 0:
                loss_dyn += config["l2_z"] * model.factors[input_t - 1, input_x].sub(model.factors[input_t, input_x]).pow(2).mean()

            if config["mode"] in("refine", "discover") and config["l1_rel"] > 0:
                loss_dyn += config["l1_rel"] * model.get_relations().abs().mean()

            # backward
            loss_dyn.backward()

            # step
            optimizer.step()

        # ------------------------ Validate ------------------------
        model.eval()

        with torch.no_grad():
            x_pred, _ = model.generate(validation_data.shape[0])
            score = rmse(x_pred, validation_data)
        
        # checkpoint / report
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(validation_rmse = score, learn_rate = lr)

        # schedule lr
        if config["patience"] > 0 and score < 1:
            lr_scheduler.step(score)

        lr = optimizer.param_groups[0]["lr"]

        if lr <= 1e-5:
            break


def restore_model(model_settings, config, allow_gpu=False):
        """
        Restore a checkpointed model after tuning

        Parameters
        ----------
        model_settings : dict
            Dictionary recording the conditions used during training, with the following keys:
                obs_path: Path to observation data
                relation_paths: List of paths to relation data
                nt_train: number of samples to use as training data
                validation_prop: Proportion of nt_train to use for validation
                random_seed: Random seed used
                checkpoint_path: Path to a checkpoint to restore

        config: dict
            Tuned hyper-parameters

        Returns
        -------
        (torch.Tensor, torch.Tensor, stnn.SpatioTemporalNN)
            A tuple representing observed data, relations, and the restored model, respectively
        """
        if allow_gpu:
            device = torch.device("cuda:0")
            torch.cuda.manual_seed_all(model_settings["random_seed"])
        else:
            device = torch.device("cpu")

        all_data, relations = load_data(model_settings["obs_path"], 
                                        model_settings["relation_paths"])
        all_data = all_data.to(device)
        relations = relations.to(device)

        nd = 1                  # dimension of input observations (1 = univariate time series)
        nt = all_data.shape[0]  # number of observed timesteps
        nx = all_data.shape[1]  # number of locations (columns)
        nz = nd                 # dimension of observations in latent space
        
        nt_actual = int(model_settings["nt_train"] * (1 - model_settings["validation_prop"]))

        if config["obs_periodicity"] > 0:
            # if periode < nt, latent factors will be initialized with a periodicity
            periode = config["obs_periodicity"]
        else:
            periode = nt

        model = SpatioTemporalNN(relations, nx, nt_actual, nd, nz, 
                                 mode = config["mode"], 
                                 nhid = config["nhid"], 
                                 nlayers = config["nlayers"],
                                 dropout_f = config["dropout_f"], 
                                 dropout_d = config["dropout_d"], 
                                 activation = config["activation"],
                                 periode = periode)
        model = model.to(device)

        checkpoint_path = os.path.join(model_settings["checkpoint_path"], "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint_path)
        model.load_state_dict(model_state)

        return all_data, relations, model
