import os

import numpy as np
import torch

from .utils import DotDict, normalize


def dataset_factory(data_dir, name, k=1):
    # get dataset
    if name[:4] == 'heat':
        opt, data, relations = heat(data_dir, '{}.csv'.format(name))
    else:
        raise ValueError('Non dataset named `{}`.'.format(name))
    # make k hop
    new_rels = [relations]
    for n in range(k - 1):
        new_rels.append(torch.stack([new_rels[-1][:, r].matmul(new_rels[0][:, r]) for r in range(relations.size(1))], 1))
    relations = torch.cat(new_rels, 1)
    # split train / test
    train_data = data[:opt.nt_train]
    test_data = data[opt.nt_train:]
    return opt, (train_data, test_data), relations


def heat(data_dir, file='heat.csv'):
    # dataset configuration
    opt = DotDict()
    opt.nt = 200
    opt.nt_train = 100
    opt.nx = 41
    opt.nd = 1
    opt.periode = opt.nt
    # loading data
    data = torch.Tensor(np.genfromtxt(os.path.join(data_dir, file))).view(opt.nt, opt.nx, opt.nd)
    # load relations
    relations = torch.Tensor(np.genfromtxt(os.path.join(data_dir, 'heat_relations.csv')))
    relations = normalize(relations).unsqueeze(1)
    return opt, data, relations


def load_data(obs_path, relation_paths):
    """Load observed data and spatial relations

    Parameters
    ----------
    obs_path : str
        Path to observation data.
    relation_paths: list
        List of paths to spatial relation matrices

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        A tuple representing observed data and relations, respectively
    """
    # Load data
    all_data = np.genfromtxt(obs_path)
    relations = (np.genfromtxt(r) for r in relation_paths)
    
    # Convert to tensors
    nrows = all_data.shape[0]
    ncols = all_data.shape[1]  
    all_data = torch.Tensor(all_data).view(nrows, ncols, 1) # assume univariate data

    relations = (torch.Tensor(x) for x in relations)
    relations = [normalize(x).unsqueeze(1) for x in relations]
    relations = torch.cat(relations, 1)

    return all_data, relations
