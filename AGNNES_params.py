import numpy as np


#figures settings
save_figs=True
filename='NGC4261_UV'

#Physics
#BH mass in solar masses
real_mass=5.2e8
#modelling
ADAF=True
Jet=True
#priors on delta, mdot and s
#True or false if you wish to use previous values for prior
usePriorADAF=False
usePriorJet=True
#Insert here the values for ADAF priors
#The original prior for ADAF is from NN training and it is:
#prior=[0.30999981848284736, 0.010061087796428128, -1.0004012458104414, -6.997462186837931, 0.9999788652378581, 1.1144956607478385e-05]. The names are autoexplicative.
delta_max=0.3
delta_min=0.13
logmdot_max=-2.5
logmdot_min=-5.
s_max=0.325
s_min=0.285
#defining the prior array
#prior=np.array([delta_max, delta_min, logmdot_max, logmdot_min, s_max, s_min])

#Insert here the values for jet priors
#The original prior for jet is from NN training and it is:
#prior=[1.0000719278935584,-1.0000129507374993,-6.999812661753591,2.999997106701305,2.0007646636533907,-1.0000518107061462,-4.999965489503788,-1.0001224692507082,-4.999980812910607,]. The names are autoexplicative.
logmdot_jet_max=-4.7
logmdot_jet_min=-7.5


p_max=3.0
p_min=2.0
logepse_max=-1.
logepse_min=-5.
logepsb_max=-1
logepsb_min=-5
#defining the prior array
#prior_jet=np.array([logmdot_jet_max, logmdot_jet_min, p_max, p_min, logepse_max, logepse_min, logepsb_max, logepsb_min])

#MCMC parameters
nwalkers = 300
n1, n2 = 100,400 #12000,48000

#fitting steps
jet_max_nu=13.
overpredict=1.
radius=0.05
