AGNNES: Emulating RIAF and jet emission with neural networks
=============================================


## Introduction

This repository allows the fast calculation of the electromagnetic emission from radiatively inefficient accretion flows and relativistic jets as appropriate for low-luminosity active galactic nuclei, using neural networks for emulation. AGNNES then allows spectral fitting using Markov-chain Monte Carlo methods and gives the parameter posterior distributions from a sparsely-sampled observed spectral energy distribution (SED). AGNNES's core is a neural network trained on the output of [RIAF and jet models](https://github.com/rsnemmen/riaf-sed). More details can be found in [Almeida, Duarte & Nemmen 2021](https://doi.org/10.1093/mnras/stab3353). 

Our code gives the posterior distribution of the following RIAF and jet parameters from observed SEDs: $\delta$, $\dot{M}_0$ (or Mdot0), $s$, $\dot{M}_j$, $p$, $\epsilon_e$, $\epsilon_B$. For an explanation of the meaning of these parameters, check out Almeida et al. (2021).


## Installation

We suggest creating a separate environment with conda for testing:

```bash
conda create --name agnnes
conda activate agnnes
```

Install dependencies:

```bash
# Basic dependencies
conda install jupyter jupyterlab numpy matplotlib scikit-learn pandas scipy astropy
# tensorflow-related
pip install tensorflow np_utils 
# Misc. packages
pip install nmmn tqdm emcee corner
# SED plotting package
git clone git@gist.github.com:94c046ee73d5da6211cac37e7da7e659.git
mv 94c046ee73d5da6211cac37e7da7e659/sedplot.py .
```

Install AGNNES:

```bash
git clone git@github.com:black-hole-group/AGNNES.git
```

## How to use

### Fitting a SED and getting posteriors

You should write the data as the example files `SED-NGC5128.txt` and `SED-NGC5128-limits.txt`, respectively data points and upper limits (in case of no upper limits put at least one very high value -- <i>to be corrected in future</i>). Please save your data/limits files as 

```
data_file='Library/SED-'+filename+'.txt'
limits_file='Library/SED-'+filename+'-limits.txt'
```

After setting the data, edit the file `INNANAS_params.py`. Instructions about the fitting parameters are in the next section, "Parameters".

In the AGNNES folder, run `python INNANAS.py` in the terminal. The code will start and print much information on the screen. In the end, the code will save in the folder `Results/` the file `sampler-<filename>.h5` with the MCMC results.

We made available a jupyter notebook `AGNNES.ipynb` in the AGNNES folder. This notebook extracts the posterior distribution for the fitting parameters and makes figures showing them. It is an example of getting information from the `sampler-<filename>.h5` file.

### Computing single RIAF and jet models

Please see the `inference.ipynb` notebook for an example.

## Parameters 

The file `INNANAS_params.py` has 

1. `filename`: The same name as the data/limits files
2. `real mass`: The supermassive black hole mass in solar masses
3. `ADAF/Jet`: Set `True` or `False` for the component to be modelled
4. `usePriorADAF / usePriorJet`: Set `True` or `False` to use priors (recommended =  `True` and use defined priors)
5. Priors for the seven parameters: Set priors min/max values.
6. `nwalkers`: number of walkers for the MCMC (default: 300)
7. `n1/n2`: number of steps of MCMC. `n1` is the burn-in and'n2'is the final chain

There are other optional parameters for fitting. These are other functions:

1. `nu_jet`: For modelling with both components, ADAF and jet. This parameter is the maximum value of frequency log<sub>10</sub>(&nu;) to fit the jet component. For frequencies higher than `nu_jet`, the jet component will read the data points as upper limits. In practice, nu_jet is the maximum frequency for the jet modelling; above `nu_jet`, the ADAF will dominate emission. If you do not want to limit the jet modelling, set this value as <b>20</b>. 
2. `overpredict`: It is a modification in the likelihood calculation. We set the error as

&chi; = [(y<sub>model</sub> - y<sub>data</sub>)<sup>2</sup> / (&sigma;<sub>model</sub><sup>2</sup> + &sigma;<sub>data</sub><sup>2</sup>)]<sup>0.5</sup> x &Theta;(y<sub>model</sub> - y<sub>data</sub>)

with

&Theta;(y<sub>model</sub> - y<sub>data</sub>) = 1,if  y<sub>model</sub> - y<sub>data</sub> < 0 ; X<sub>overpredict</sub>, otherwise

This is an option to penalize more overprediction as underprediction from the model. This feature can be helpful.

## Reference papers

AGNNES is an open code. You are morally obligated to cite the following paper in any scientific literature that results from the use of any part of this code: [*AGNNES paper*](https://doi.org/10.1093/mnras/stab3353).

We based our work on [`riaf-sed`](https://github.com/rsnemmen/riaf-sed) model developed by Feng Yuan and Rodrigo Nemmen. 

## License

Copyright (c) 2021, [Ivan Almeida](https://ivancalmeida.wordpress.com). All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

