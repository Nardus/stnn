"""
Functions to tune an stnn model
"""

import os
import torch
import torch.nn.functional as functional
import torch.optim as optim

from ray import tune
from copy import deepcopy

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
    "patience": None,          # number of epoch to wait before trigerring learning rate decay
    "batch_size": None,        # batch size for training
    "obs_path": None,          # path to observation data
    "relation_paths": None,    # list of paths to relation data
    "train_inds": None,        # list of observation data indices (rows) to use for training/validation
    "validation_prop": None,   # proportion of training samples to use for validation
    "random_seed": None,       # random seed
    "allow_gpu": None            # whether training should be performed on a gpu
}


class stnnTrainable(tune.Trainable):
    def setup(self, config):
        """
        Trainable wrapper for the Spatio-Temporal Neural net model, compatible with ray.tune
        
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
        # Data loading - performed only during initial setup 
        # (and not when setup is called by reset() below)
        if not hasattr(self, "initialized"):
            # Device setup
            if config["allow_gpu"]:
                self.device = torch.device("cuda:0")
                torch.cuda.manual_seed_all(config["random_seed"])
            else:
                self.device = torch.device("cpu")

            # Load data
            all_data, relations = load_data(config["obs_path"], config["relation_paths"])

            # Data-related quantities / dimensions
            self.nd = 1                  # dimension of input observations (1 = univariate time series)
            self.nt = all_data.shape[0]  # number of observed timesteps
            self.nx = all_data.shape[1]  # number of locations (columns)
            self.nz = self.nd            # dimension of observations in latent space

            # Split data
            if config["validation_prop"] <=0 or config["validation_prop"] >= 1:
                raise ValueError("validation_prop should be in (0, 1)")
            
            train_inds = config["train_inds"]
            nt_train = len(train_inds)  # number of rows in training data
            self.nt_actual = int(nt_train * (1 - config["validation_prop"]))
            train_subset_inds = train_inds[:self.nt_actual]
            val_subset_inds = train_inds[self.nt_actual:nt_train]

            train_data = all_data[train_subset_inds]
            validation_data = all_data[val_subset_inds]

            self.train_data = train_data.to(self.device)
            self.validation_data = validation_data.to(self.device)
            self.relations = relations.to(self.device)

            self.initialized = True

        # Always performed
        self.config = config

        # Set up model
        if config["obs_periodicity"] > 0:
            # if periode < nt, latent factors will be initialized with a periodicity
            periode = config["obs_periodicity"]
        else:
            periode = self.nt

        model = SpatioTemporalNN(self.relations, self.nx, self.nt_actual, self.nd, self.nz, 
                                 mode = config["mode"], 
                                 nhid = config["nhid"], 
                                 nlayers = config["nlayers"],
                                 dropout_f = config["dropout_f"], 
                                 dropout_d = config["dropout_d"], 
                                 activation = config["activation"],
                                 periode = periode)
        self.model = model.to(self.device)

        # Indices
        # - all
        t_idx = (torch.arange(self.nt_actual, out = torch.LongTensor())
                 .unsqueeze(1)
                 .expand(self.nt_actual, self.nx)
                 .contiguous())
        x_idx = (torch.arange(self.nx, out = torch.LongTensor())
                 .expand_as(t_idx)
                 .contiguous())

        # - dynamic
        self.idx_dyn = torch.stack((t_idx[1:], x_idx[1:])).view(2, -1).to(self.device)
        self.nex_dyn = self.idx_dyn.size(1)

        # - decoder
        self.idx_dec = torch.stack((t_idx, x_idx)).view(2, -1).to(self.device)
        self.nex_dec = self.idx_dec.size(1)

        # Optimizer
        params = [{"params": self.model.factors_parameters(), "weight_decay": config["wd_z"]},
                  {"params": self.model.dynamic.parameters()},
                  {"params": self.model.decoder.parameters()}]

        if config["mode"] in ("refine", "discover"):
            params.append({"params": self.model.rel_parameters(), "weight_decay": 0.})

        self.optimizer = optim.Adam(params, 
                                    lr=config["lr"], 
                                    betas=(config["beta1"], config["beta2"]), 
                                    eps=config["eps"], weight_decay=config["wd"])

        if config["patience"] > 0:
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience = config["patience"])
        

    def step(self):
        """Perform a single iteration (epoch) of training"""
        # ------------------------ Train ------------------------
        self.model.train()
        
        # --- decoder ---
        idx_perm = torch.randperm(self.nex_dec).to(self.device)
        batches = idx_perm.split(self.config["batch_size"])

        for batch in batches:
            self.optimizer.zero_grad()

            # data
            input_t = self.idx_dec[0][batch]
            input_x = self.idx_dec[1][batch]
            x_target = self.train_data[input_t, input_x]

            # closure
            x_rec = self.model.dec_closure(input_t, input_x)
            mse_dec = functional.mse_loss(x_rec, x_target)

            # backward
            mse_dec.backward()

            # step
            self.optimizer.step()

        # --- dynamic ---
        idx_perm = torch.randperm(self.nex_dyn).to(self.device)
        batches = idx_perm.split(self.config["batch_size"])

        for batch in batches:
            self.optimizer.zero_grad()

            # data
            input_t = self.idx_dyn[0][batch]
            input_x = self.idx_dyn[1][batch]

            # closure
            z_inf = self.model.factors[input_t, input_x]
            z_pred = self.model.dyn_closure(input_t - 1, input_x)

            # loss
            mse_dyn = z_pred.sub(z_inf).pow(2).mean()
            loss_dyn = mse_dyn * self.config["lambd"]

            if self.config["l2_z"] > 0:
                temp_factors = self.model.factors[input_t - 1, input_x]
                temp_factors = temp_factors.sub(self.model.factors[input_t, input_x])
                temp_factors = temp_factors.pow(2).mean()
                loss_dyn += self.config["l2_z"] * temp_factors

            if self.config["mode"] in("refine", "discover") and self.config["l1_rel"] > 0:
                loss_dyn += self.config["l1_rel"] * self.model.get_relations().abs().mean()

            # backward
            loss_dyn.backward()

            # step
            self.optimizer.step()

        # ------------------------ Validate ------------------------
        self.model.eval()

        with torch.no_grad():
            x_pred, _ = self.model.generate(self.validation_data.shape[0])
            score = rmse(x_pred, self.validation_data)

        # schedule lr reduction
        if self.config["patience"] > 0 and score < 1:
            self.lr_scheduler.step(score)

        # report 
        lr = self.optimizer.param_groups[0]["lr"]
        return {"validation_rmse": score, "learn_rate": lr}


    # Other methods
    def save_checkpoint(self, checkpoint_dir):
        """Save a checkpoint"""
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), path)
        return path


    def load_checkpoint(self, checkpoint):
        """Load a previously-saved checkpoint"""
        model_state, optimizer_state = torch.load(checkpoint)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)


    def reset_config(self, new_config):
        """Reset this object with a new config (allows re-use without loading data again)"""
        self.setup(deepcopy(new_config))
        return self.config == new_config



# Additional utility functions
def restore_model(checkpoint_path, config, allow_gpu=False):
        """
        Restore a checkpointed model after tuning

        Parameters
        ----------
        checkpoint_path: str
            Path to a checkpoint to restore

        config: dict
            Tuned hyper-parameters

        Returns
        -------
        (torch.Tensor, torch.Tensor, stnn.SpatioTemporalNN)
            A tuple representing observed data, relations, and the restored model, respectively
        """
        if allow_gpu:
            device = torch.device("cuda:0")
            torch.cuda.manual_seed_all(config["random_seed"])
        else:
            device = torch.device("cpu")

        all_data, relations = load_data(config["obs_path"], 
                                        config["relation_paths"])
        all_data = all_data.to(device)
        relations = relations.to(device)

        nd = 1                  # dimension of input observations (1 = univariate time series)
        nt = all_data.shape[0]  # number of observed timesteps
        nx = all_data.shape[1]  # number of locations (columns)
        nz = nd                 # dimension of observations in latent space
        
        nt_train = len(config["train_inds"])
        nt_actual = int(nt_train * (1 - config["validation_prop"]))

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
        
        model_state, _ = torch.load(checkpoint_path)
        model.load_state_dict(model_state)

        return all_data, relations, model
