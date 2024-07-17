import numpy as np
import glob
import pandas as pd
import keras
import matplotlib.pyplot as plt
import pylab
from random import randint

from keras.optimizers import Adam
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import np_utils

import emcee
import corner
import tqdm
import sedplot as sd
import pylab
import datetime
import time
import random

from scipy.interpolate import interp1d
from decimal import Decimal
import scipy.optimize as op

import AGNNES_functions as RF
import AGNNES_model as RM
import AGNNES_params as RP

#defining the prior array
prior=np.array([RP.delta_max, RP.delta_min, RP.logmdot_max, RP.logmdot_min, RP.s_max, RP.s_min])
#defining the prior array
prior_jet=np.array([RP.logmdot_jet_max, RP.logmdot_jet_min, RP.p_max, RP.p_min, RP.logepse_max, RP.logepse_min, RP.logepsb_max, RP.logepsb_min])

#Normalizations
X_delta_min,X_delta_max,X_ar_min,X_ar_max,X_mass_max,X_mass_min,X_s_max,X_s_min, Y_max, Y_min = RF.X_delta_min,RF.X_delta_max,RF.X_ar_min,RF.X_ar_max,RF.X_mass_max,RF.X_mass_min, RF.X_s_max,RF.X_s_min, RF.Y_max, RF.Y_min
X_ar_jet_min,X_ar_jet_max,X_p_min,X_p_max,X_mass_jet_max,X_mass_jet_min,X_epse_max,X_epse_min,X_epsb_max,X_epsb_min, Y_jet_max, Y_jet_min = RF.X_ar_jet_min,RF.X_ar_jet_max,RF.X_p_min,RF.X_p_max,RF.X_mass_jet_max,RF.X_mass_jet_min, RF.X_epse_max,RF.X_epse_min, RF.X_epsb_max,RF.X_epsb_min, RF.Y_jet_max, RF.Y_jet_min


#logprobability function
def lnprob_adaf(theta, x, y, yerr, mass, xlim=None, ylim=None, prior=RF.norms):
    '''
Probability for the model. Taking into account the likelihood and the priors.
For more details see RF.lnlike and RF.lnprior_adaf.
    '''
    lp = RF.lnprior_adaf(theta, prior)
    return (lp + RF.lnlike(theta, x, y, yerr, mass, xlim=xlim, ylim=ylim))


#MCMC start
def AGNNES_MCMC_adaf(real_mass, xdata, ydata, yerror, xlim=None, ylim=None, ndim=3, nwalkers=700, prior=RF.norms,  n1=100, n2=500):
   #Finding the initial parameters guess and verifying with the prior range for the initial sphere
   count=0
   while (count == 0):
       params_net=np.array([[real_mass,random.uniform(prior[1], prior[0]), random.uniform(prior[3], prior[2]), random.uniform(prior[5], prior[4])]])
       print('Initial parameters for MCMC: '+str(params_net))
       count5=0
       max_pr=[]
       min_pr=[]
       for count5 in range(len(params_net[0][1:])):
           print(count5)
           max_pr.append(params_net[0][1+count5]*(1+(0-1)**count5*RP.radius))
           min_pr.append(params_net[0][1+count5]*(1-(0-1)**count5*RP.radius))
       count2=0
       boolean=[]
       for count2 in range(len(max_pr)):
           boolean.append((max_pr[count2])>(prior[2*count2]))
           boolean.append((min_pr[count2])<(prior[2*count2+1]))
       count3=0
       print(max_pr, min_pr)
       print(prior)
       print(boolean)
       count4=0
       for count3 in range(len(boolean)):
           #print('hi')
           if(boolean[count3]==True):
               count4=1
       if(count4==0):
           count=1
   #defining the initial distribution with our grid searched initial values
   p0=[params_net[0][1:] *(1 + 0.25*(0-1)**(np.random.randint(10))*RP.radius*np.random.randn(int(ndim))) for i in range(int(nwalkers))]
   #Beginning time
   t0=datetime.datetime.now()
   print('Initializing MCMC at '+str(t0))
   t0=time.time()
   #Loading BH mass value
   mass_n=real_mass#(real_mass-X_mass_min)/(X_mass_max-X_mass_min)
   # Set up the backend
   # Don't forget to clear it in case the file already exists
   filename2 = 'sampler-'+RP.filename+'-pi.h5'
   backend = emcee.backends.HDFBackend(filename2)
   backend.reset(nwalkers, ndim)
   #Running the MCMC
   #First run only to find a better distribution
   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_adaf, args=(xdata, ydata, yerror, mass_n, xlim, ylim, prior), backend=backend)
   #Finding the distribution after a initial run
   pos, prob, state = tqdm.tqdm(sampler.run_mcmc(p0, n1)) #100 ~ 8min
   #Resetting sampler and start again with the new distribution
   sampler.reset()
   #Starting time for the real MCMC chain
   t1=datetime.datetime.now()
   print('Starting MCMC at '+str(t1))
   t1=time.time()
   #Running the real MCMC chain
   sampler.run_mcmc(pos, n2)
   #Ending time
   t2=datetime.datetime.now()
   print('Finished at '+str(t2))# Took %.2f seconds' % (t2-t0))
   t2=time.time()
   print('Took %2e seconds or %.2f hours' % ((t2-t0), (t2-t0)/60./60.))
   return(sampler)

#============================ JET ============================

def lnprob_jet(theta, x, y, yerr, mass, xlim=None, ylim=None, prior=RF.norms_jet, jet_max_nu=20., x_model=RF.nu_jet):
    '''
Probability for the model. Taking into account the likelihood and the priors.
For more details see RF.lnlike and RF.lnprior_jet.
    '''
    id_cut=RF.find_nearest(x, jet_max_nu)
    lp = RF.lnprior_jet(theta, prior)
    return (lp + RF.lnlike(theta, x[:id_cut], y[:id_cut], yerr[:id_cut], mass, xlim=xlim, ylim=ylim, jet=True, x_model=RF.nu_jet))



def AGNNES_MCMC_jet(real_mass, xdata, ydata, yerror, xlim=None, ylim=None, ndim=4, nwalkers=700, prior_jet=RF.norms_jet[2:10],  n1=100, n2=500, jet_max_nu=20.):
   #Finding the initial parameters guess
   count=0
   while (count == 0):
       params_net=np.array([[real_mass,random.uniform(prior_jet[1], prior_jet[0]), random.uniform(prior_jet[3], prior_jet[2]), random.uniform(prior_jet[5], prior_jet[4]), random.uniform(prior_jet[7], prior_jet[6])]])
       print('Initial parameters for MCMC: '+str(params_net))
       count5=0
       max_pr=[]
       min_pr=[]
       for count5 in range(len(params_net[0][1:])):
           print(count5)
           max_pr.append(params_net[0][1+count5]*(1+(0-1)**count5*RP.radius))
           min_pr.append(params_net[0][1+count5]*(1-(0-1)**count5*RP.radius))
       count2=0
       boolean=[]
       for count2 in range(len(max_pr)):
           boolean.append((max_pr[count2])>(prior_jet[2*count2]))
           boolean.append((min_pr[count2])<(prior_jet[2*count2+1]))
       count3=0
       print(max_pr, min_pr)
       print(prior_jet)
       print(boolean)
       count4=0
       for count3 in range(len(boolean)):
           #print('hi')
           if(boolean[count3]==True):
               count4=1
       if(count4==0):
           count=1
   #defining the initial distribution with our grid searched initial values
   p0=[params_net[0][1:] + (0-1)**(np.random.randint(10))*1e-2*np.random.randn(int(ndim)) for i in range(int(nwalkers))]
   #Beginning time
   t0=datetime.datetime.now()
   print('Initializing MCMC at '+str(t0))
   t0=time.time()
   #Loading BH mass value
   mass_n=real_mass#(real_mass-X_mass_min)/(X_mass_max-X_mass_min)
   #print('I am alive dude '+str(len(prior)))
   #Running the MCMC
   # Don't forget to clear it in case the file already exists
   filename2 = 'sampler-'+RP.filename+'-jet.h5'
   backend = emcee.backends.HDFBackend(filename2)
   backend.reset(nwalkers, ndim)
   #First run only to find a better distribution
   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_jet, args=(xdata, ydata, yerror, mass_n, xlim, ylim, prior_jet, jet_max_nu), backend=backend)
   #Finding the distribution after a initial run

   pos, prob, state = sampler.run_mcmc(p0, n1) #100 ~ 8min
   #Resetting sampler and start again with the new distribution
   sampler.reset()
   #Starting time for the real MCMC chain
   t1=datetime.datetime.now()
   print('Starting MCMC at '+str(t1))
   t1=time.time()
   #Running the real MCMC chain
   sampler.run_mcmc(pos, n2)
   #Ending time
   t2=datetime.datetime.now()
   print('Finished at '+str(t2))# Took %.2f seconds' % (t2-t0))
   t2=time.time()
   print('Took %2e seconds or %.2f hours' % ((t2-t0), (t2-t0)/60./60.))
   return(sampler)


#=========================== total ========================================
def lnprob(theta, x, y, yerr, mass, xlim=None, ylim=None, prior_adaf=RF.norms, prior_jet=RF.norms_jet, jet_max_nu=20., x_model_adaf=RF.nu_off[:-1], x_model_jet=RF.nu_jet):
    '''
Probability for the model. Taking into account the likelihood and the priors.
For more details see RF.lnlike, RF.lnprior_jet and RF.lnprior_adaf.

Here we separated the data in two sets. The original one is used for ADAF modelling. In sequence we defined a new set:
    newdata = original data - ADAF modelling
The newdata it will be fitted with jet emission.

There is two auxiliar parameters here: RP.jet_max_nu and RP.xray_ratio
    RP:jet_max_nu is a value between 8-20, which is the maximum log10(nu) where the jet will fit the data.
    For log10(nu) > RP.jet_max_nu, the available data will become upperlimits.
    '''
    id_cut=RF.find_nearest(x, jet_max_nu)
    if(y[id_cut]>jet_max_nu):
        id_cut=id_cut-1

    newdata2=np.copy(y[id_cut:])
    #newdata2=newdata2-RP.xray_ratio
    newdata=np.copy(y[:id_cut])
    yerr_new=np.copy(yerr[:id_cut])
    x_new=np.copy(x[:id_cut])
    xlim_new=np.array(list(xlim)+list(x[id_cut:]))
    ylim_new=np.array(list(ylim)+list(newdata2))
    lp = RF.lnprior_adaf(theta[:3], prior_adaf) + RF.lnprior_jet(theta[3:], prior_jet)
    return (lp + RF.lnlike(theta[:3], x, y, yerr, mass, xlim=xlim, ylim=ylim, jet=False, x_model=RF.nu_off[:-1]) + RF.lnlike(theta[3:], x_new, newdata, yerr_new, mass, xlim=xlim_new, ylim=ylim_new, jet=True, x_model=RF.nu_jet))


def AGNNES_MCMC(real_mass, xdata, ydata, yerror, xlim=None, ylim=None, ndim=7, nwalkers=700, prior_adaf=RF.norms, prior_jet=RF.norms_jet[2:10],  n1=100, n2=500, jet_max_nu=20.):
   #Finding the initial parameters guess and verifying with the prior range for the initial sphere
   count=0
   while (count == 0):
       params_net=np.array([[random.uniform(prior[1], prior[0]), random.uniform(prior[3],
        prior[2]), random.uniform(prior[5], prior[4]),random.uniform(prior_jet[1], prior_jet[0]),
        random.uniform(prior_jet[3], prior_jet[2]),random.uniform(prior_jet[5], prior_jet[4]),
        random.uniform(prior_jet[7], prior_jet[6])]])

       print('Initial parameters for MCMC: '+str(params_net))
       count5=0
       max_pr=[]
       min_pr=[]
       for count5 in range(len(params_net[0][:])):
           aux_mc=1
           if(count5==6):
               aux_mc=-1
           max_pr.append(params_net[0][count5]*(1+aux_mc*(0-1)**count5*RP.radius))
           min_pr.append(params_net[0][count5]*(1-aux_mc*(0-1)**count5*RP.radius))
       count2=0
       boolean=[]
       for count2 in range(len(max_pr)):
           if(count2 < 3):
                boolean.append((max_pr[count2])>(prior[2*count2]))
                boolean.append((min_pr[count2])<(prior[2*count2+1]))
                print(count2)
           if(count2 > 2):
                boolean.append((max_pr[count2])>(prior_jet[2*count2-6]))
                boolean.append((min_pr[count2])<(prior_jet[2*count2+1-6]))

       count3=0
       print(max_pr)
       print(min_pr)
       print(prior, prior_jet)
       print(boolean)
       count4=0
       for count3 in range(len(boolean)):
           #print('hi')
           if(boolean[count3]==True):
               count4=1
       if(count4==0):
           count=1
   #defining the initial distribution with our grid searched initial values
   p0=[params_net[0] + (0-1)**(np.random.randint(10))*2e-2*np.random.randn(int(ndim)) for i in range(int(nwalkers))]
   #Beginning time
   t0=time.time()
   print('Initializing MCMC at '+str(datetime.datetime.now()))
   #Loading BH mass value
   mass_n=real_mass

   #Running the MCMC
   # Set up the backend
   # Don't forget to clear it in case the file already exists
   filename2 = 'sampler-'+RP.filename+'.h5'
   backend = emcee.backends.HDFBackend(filename2)
   backend.reset(nwalkers, ndim)

   #First run only to find a better distribution
   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xdata, ydata, yerror, mass_n, xlim, ylim, prior_adaf, prior_jet, jet_max_nu), backend=backend)
   #Finding the distribution after a initial run
   pos, prob, state = tqdm.tqdm(sampler.run_mcmc(p0, n1, progress=True)) #100 ~ 8min
   #Resetting sampler and start again with the new distribution
   sampler.reset()

   #Starting time for the real MCMC chain
   t1=time.time()
   print('Starting MCMC at '+str(datetime.datetime.now()))

   #Running the real MCMC chain
   sampler.run_mcmc(pos, n2)
   #Ending time
   t2=time.time()
   print('Finished at '+str(datetime.datetime.now()))# Took %.2f seconds' % (t2-t0))
   print('Took %2e seconds or %.2f hours' % ((t2-t0), (t2-t0)/60./60.))
   return(sampler)

#===============================================================================================
