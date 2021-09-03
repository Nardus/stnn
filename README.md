# Spatio-Temporal Neural Networks for Space-Time Series Forecasting and Relation Discovery

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Packaged version of Delasalles *et al.*'s stnn model, with modifications to allow tuning with the [ray.tune](https://docs.ray.io/en/latest/tune/index.html) library.


## Original implementation:

The reference implementation is at [edouardelasalles/stnn/](https://github.com/edouardelasalles/stnn/), and is described in:

- Ziat A, Delasalles E, Denoyer L, Gallinari P. 2017. Spatio-Temporal Neural Networks for Space-Time Series Forecasting and Relations Discovery2017 IEEE International Conference on Data Mining (ICDM). Presented at the 2017 IEEE International Conference on Data Mining (ICDM). pp. 705–714. [doi:10.1109/ICDM.2017.80](https://doi.org/10.1109/ICDM.2017.80)
- Delasalles E, Ziat A, Denoyer L, Gallinari P. 2019. Spatio-temporal neural networks for space-time data modeling and relation discovery. Knowl Inf Syst 61:1241–1267. [doi:10.1007/s10115-018-1291-x](https://doi.org/10.1007/s10115-018-1291-x)

Commands for reproducing synthetic experiments:

### Heat Diffusion
#### STNN
`python stnn/train_stnn.py --dataset heat --outputdir output_heat --manualSeed 2021 --xp stnn`

#### STNN-R(efine)
`python stnn/train_stnn.py --dataset heat --outputdir output_heat --manualSeed 5718 --xp stnn_r --mode refine --patience 800 --l1_rel 1e-8`

#### STNN-D(iscovery)
`python stnn/train_stnn.py --dataset heat --outputdir output_heat --manualSeed 9690 --xp stnn_d --mode discover --patience 1000 --l1_rel 3e-6`


### Modulated Heat Diffusion
#### STNN
`python stnn/train_stnn.py --dataset heat_m --outputdir output_heat_m --manualSeed 679 --xp stnn`

#### STNN-R(efine)
`python stnn/train_stnn.py --dataset heat_m --outputdir output_heat_m --manualSeed 3488 --xp stnn_r --mode refine --l1_rel 1e-5`

#### STNN-D(iscovery)
`python stnn/train_stnn_.py --dataset heat_m --outputdir output_m --xp test --manualSeed 7664 --mode discover --patience 500 --l1_rel 3e-6`

## Data format
The file `heat.csv` contains the raw temperature data. The 200 rows correspond to the 200 timestep, and the 41 columns are the 41 space points.
The file `heat_relations.csv` contains the spatial relation between the 41 space points. It is a 41 by 41 adjacency matrix _A_, where _A(i, j)_ = 1 means that series _i_ is a direct neighbor of series _j_ in space, and is 0 otherwise.
