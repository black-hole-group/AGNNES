import numpy as np
import glob
import pandas as pd
import keras
import matplotlib.pyplot as plt
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

from scipy.interpolate import interp1d
import scipy.optimize as op
import AGNNES_model as RM
import AGNNES_params as RP


#===========================================================
#=================DO NOT CHANGE THESE LINES=================
#Global variables (Yes I will use this)
#Frequency array
nu_off=np.linspace(9.12,21, 100)
nu_jet=np.linspace(8.2, 21.1, 130)
#Normalizations for the NN
X_mass_max,X_mass_min,X_delta_max,X_delta_min,X_ar_max,X_ar_min,X_s_max,X_s_min,Y_max,Y_min = np.load('normalization_ADAF.npy', allow_pickle=True, encoding='latin1')
norms=[X_delta_max,X_delta_min,X_ar_max,X_ar_min,X_s_max,X_s_min]
X_mass_jet_max,X_mass_jet_min,X_ar_jet_max,X_ar_jet_min,X_p_max,X_p_min,X_epse_max,X_epse_min,X_epsb_max,X_epsb_min,Y_jet_max,Y_jet_min = np.load('normalization_jet.npy', allow_pickle=True, encoding='latin1')
norms_jet=[X_mass_jet_max,X_mass_jet_min,X_ar_jet_max,X_ar_jet_min,X_p_max,X_p_min,X_epse_max,X_epse_min,X_epsb_max,X_epsb_min,Y_jet_max,Y_jet_min]



#=================END OF THE FORBIDDEN ZONE=================
#===========================================================
#Useful functions

def interpolation(sed, x_sed=nu_off[:-1]):
    '''Routine that creates an interpolation function for give x (x_sed) and y (sed) data.

Parameters:
    sed (array): y data from sed in log_10(nu*L_nu / [erg/s])) unit
    x_sed (array): x data from sed in log_10(nu / [Hz])) unit

Returns:
    f2: a function that interpolates sed in x_sed domain

Example:
    func = interpolation(sed=y, x_sed=x)

    func(x1) = sed(x1) (approximate)
    '''
    x = x_sed
    y = sed
    #interpolating
    f2 = interp1d(x, y, kind='cubic')
    return(f2)

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.argmin(np.abs(a - a0))
    if(a[idx]<a0):
        idx=idx+1
    return idx


def error_func(y_data, y_model, x_data, y_error=1., x_model=nu_off[:-1]):
    ''' Calculates chi square error of data vs model

Parameters:
    y_data, x_data, y_error: (array) real data
    y_model, x_model: (array) model data

Returns:
    Value of the error for a given model compared to data (float)

Example:
    err = error_func(y_data=ydata, x_data=xdata, y_model=ytheory, x_model=xtheory)
    err = 13.7647
    '''
    #interpolating model
    sed = interpolation(y_model, x_sed=x_model)
    #initializing error
    sq_error=0.
    for tmp in range(len(x_data)):
        #calculating error
        overpredict=1.
        if((sed(x_data[tmp]) - y_data[tmp] - y_error[tmp])>0.):
            overpredict=RP.overpredict
        sq_error = (sq_error + ((sed(x_data[tmp]) - y_data[tmp])/y_error[tmp])**2)*overpredict

    return(sq_error)

def error_lims(y_data, y_model, x_data, x_model=nu_off[:-1]):
    ''' Verify if model obeys the upper limits

Parameters:
    y_data, x_data, y_error: (array) real data
    y_model, x_model: (array) model data

Returns:
    If the fit exceeded the upper limits returns True, if not, False

Example:
    err = error_func(y_data=ydata, x_data=xdata, y_model=ytheory, x_model=xtheory)
    err = False/True
    '''

    #interpolating model
    sed = interpolation(y_model, x_sed=x_model)
    verifier=-(sed(x_data) - y_data)/abs(sed(x_data) - y_data)
    return(sum(1 for number in verifier if number < 0) > 0)

def normalizer(x,xmin, xmax):
    return((x-xmin)/(xmax-xmin))
#===========================================================
#Bayesian functions

#ln(likelihood)
def lnlike(theta, x, y, yerr,mass, jet=False, x_model=nu_off[:-1], xlim=None, ylim=None):
    ''' Returns the log of the likelihood.

The likelihood is related to the quadratic error here.
We use ln(likelihood) = -0.5 * chi_square (of the data x model)

The likelihood here is tied to a restriction that the model can not be higher than the upperlimits. If this occurs the likelihood will be extremely small.

Parameters:
    jet: Boolean variable. Put 'False' to calculate the likelihood for ADAF and 'True' for the Jet
	Theta: an array with [delta, mdot, s] (ADAF) or [mdot, p, epse, epsb] (jet) that it will be used to generate the model;
	x, y, yerr: They come from the data. x is log10(nu/[Hz]), y is log10(nu*L_nu/[erg/s]) and yerr is the error associated to y
	mass: BH mass
    xlim, ylim: They come from the data. They are the upperlimits, xlim is log10(nu/[Hz]), ylim is log10(nu*L_nu/[erg/s])

Returns:
    Value of the likelihood for the given model

Example:
	lik = lnlike([delta, mdot, s], xdata, ydata, ydata_error, mass)
lik is the likelihood that should be maximized in order to find the posteriori
    '''
    if(x_model.any==None):
        if(jet==False):
            x_model=nu_off[:-1]
        if(jet==True):
            x_model=nu_jet
    if (jet==False):
        #modelling parameters
        delta, mdot, s = theta
        #normalizing
        mass=(mass-X_mass_min)/(X_mass_max-X_mass_min)
        delta=(delta-X_delta_min)/(X_delta_max-X_delta_min)
        mdot=(mdot-X_ar_min)/(X_ar_max-X_ar_min)
        s=(s-X_s_min)/(X_s_max-X_s_min)
        #calculating the model value
        ms=RM.adaf.predict(np.array([[mass,delta,mdot,s]]))[0]*Y_min + Y_max#*(Y_max - Y_min) + Y_min#
        #we cannot allow this
        if(delta > 1. or mdot > 1. or s > 1.):
            ms = ms*1e10
    if (jet==True):
        #modelling parameters
        mdot, p, epse, epsb = theta
        #normalizing
        mass=10.**mass
        mass=(mass-X_mass_jet_min)/(X_mass_jet_max-X_mass_jet_min)
        mdot=(mdot-X_ar_jet_min)/(X_ar_jet_max-X_ar_jet_min)
        p=(p-X_p_min)/(X_p_max-X_p_min)
        epse=(epse-X_epse_min)/(X_epse_max-X_epse_min)
        epsb=(epsb-X_epsb_min)/(X_epsb_max-X_epsb_min)
        #calculating the model value
        ms=RM.jet.predict(np.array([[mass,mdot,p, epse, epsb]]))[0]*(Y_jet_max - Y_jet_min) + Y_jet_min
        #we cannot allow this
        if(epsb > 1. or epse > 1. or mdot > 1. or p > 1.):
            ms = ms*1e10
    #accounting upperlimits
    upper=0.
    if ((type(xlim) != type(None)) and (error_lims(y_data=ylim, y_model=ms, x_data=xlim, x_model=x_model) == True)):
        upper=1e10
    #calculating error
    erro_tmp=error_func(y,ms,x, y_error=yerr, x_model=x_model)+upper
    return -0.5*erro_tmp

#ln(prior)
def lnprior_adaf(theta, prior=norms):
    ''' Calculates the prior for given parameters

Parameters:
	theta: an array with [delta, mdot, s] that it will be used to generate the model;
    prior: an array with [delta_max, delta_min, mdot_max, mdot_min, s_max, s_min], the values X_max/min are the constraints in X

Returns:
    log(Prior): (float) this function is used to allow or not a range of values for theta components

Example:
    log_prior = lnprior(theta=[0.1, 1e-5, 0.4], prior=[0.3, 0.01, 1e-2, 1e-7, 1.0, 0.3])
    log_prior = 0.

    log_prior = lnprior(theta=[0.4, 1e-5, 0.4], prior=[0.3, 0.01, 1e-2, 1e-7, 1.0, 0.3])
    log_prior = -np.inf
    (0.4 > 0.3)
    '''
    #the theta values
    delta, mdot, s= theta
    #comparing with previous prior
    delta_max, delta_min, mdot_max, mdot_min, s_max, s_min = prior

    #values
    prior=-np.inf
    if delta_min < delta <= delta_max and mdot_min < (mdot) <= mdot_max and s_min < s <=s_max:
        prior=0.
    return prior

def lnprior_jet(theta, prior=norms_jet):
    ''' Calculates the prior for given parameters

Parameters:
	theta: an array with [mdot, p, epse, epsb] that it will be used to generate the model;
    prior: an array with [mdot_max, mdot_min, p_max, p_min, epse_max, epse_min, epsb_max, epsb_min], the values X_max/min are the constraints in X

Returns:
    log(Prior): (float) this function is used to allow or not a range of values for theta components

Example:
    log_prior = lnprior(theta=[1e-5, 2.5, 0.04, 0.01], prior=[1e-7, 1e-2, 2.0, 3.0, 0., 0.1, 0., 0.1])
    log_prior = 0.

    log_prior = lnprior(theta=[1e-5, 3.5, 0.04, 0.01], prior=[1e-7, 1e-2, 2.0, 3.0, 0., 0.1, 0., 0.1])
    log_prior = -np.inf
    (3.5 > 3.)
    '''
    #the theta values
    mdot, p, epse, epsb = theta
    #comparing with previous prior
    mdot_max, mdot_min, p_max, p_min, epse_max, epse_min, epsb_max, epsb_min = prior

    #values
    prior=-np.inf
    if p_min < p <= p_max and mdot_min < (mdot) <= mdot_max and epse_min < epse <=epse_max and epsb_min < epsb <=epsb_max:
        prior=0.
    return prior


#===========================================================
#Single SED calculation functions
def sed_calculator_jet(real_mass, mdotj, p, epse, epsb, plot=False, title=' ', save=False, savefile='sed.png'):
    ''' Calculates the jet SED for the given parameters

Parameters:
	real_mass, mdotj, p epse, epsb. [Jet SED parameters]

Plot:
    plot: If True, the code shows a SED figure
    save: save=True saves a figure
    savefile: name of the saved figure
    title: title of the plot

Returns:
    log(nu), log(nu*L_nu): (array) Values of frequency and SED, respectively

Example:
    nu, jet = sed_calculator_jet(real_mass=1e8, mdotj=2e-7, p=2.4, epse=1e-2, epsb=1e-2)
    [If you want to plot the SED] plt.plot(nu,jet)
    '''
    #Normalizing
    real_mass=np.log10(RP.real_mass/1.e6)
    mass_jet=(10**real_mass-X_mass_jet_min)/(X_mass_jet_max-X_mass_jet_min)
    mdotj=normalizer(mdotj, X_ar_jet_min, X_ar_jet_max)
    p=normalizer(p, X_p_min, X_p_max)
    epse=normalizer(epse, X_epse_min, X_epse_max)
    epsb=normalizer(epsb, X_epsb_min, X_epsb_max)
    #calculating SED
    sed_calc=RM.jet.predict(np.array([[mass_jet,mdotj,p,epse,epsb]]))[0]*(Y_jet_max - Y_jet_min) + Y_jet_min

    #plotting
    if(plot==True):
        sed_plot(nu,sed_total, title=title, save=save, savefile=savefile)
    return(sed_calc)

def sed_calculator_adaf(real_mass, delta, mdot, s, plot=False, title=' ', save=False, savefile='sed.png'):
    ''' Calculates the ADAF SED for the given parameters

Parameters:
	real_mass, delta, mdot, s. [ADAF SED parameters]

Plot:
    plot: If True, the code shows a SED figure
    save: save=True saves a figure
    savefile: name of the saved figure
    title: title of the plot

Returns:
    log(nu), log(nu*L_nu): (array) Values of frequency and SED, respectively

Example:
    nu, adaf = sed_calculator_adaf(real_mass=1e8, delta=0.25, mdot=2e-3, 2=0.4)
    [If you want to plot the SED] plt.plot(nu,adaf)
    '''
    #Normalizing
    real_mass=np.log10(real_mass/1.e6)
    mass_n=(real_mass-X_mass_min)/(X_mass_max-X_mass_min)
    delta=normalizer(delta, X_delta_min, X_delta_max)
    mdot=normalizer(mdot, X_ar_min, X_ar_max)
    s=normalizer(s, X_s_min, X_s_max)
    #calculating SED
    sed_calc=RM.adaf.predict(np.array([[mass_n, delta, mdot, s]]))[0]*Y_min + Y_max

    #plotting
    if(plot==True):
        sed_plot(nu,sed_total, title=title, save=save, savefile=savefile)
    return(sed_calc)

def sed_calculator(component,real_mass=0, delta=0, mdot=0, s=0, mdotj=0, p=0, epse=0, epsb=0, plot=False, title=' ', save=False, savefile='sed.png'):
    ''' Calculates the ADAF + Jet SED for the given parameters

Components:
    Select the component to calculate: adaf, jet, or both

Parameters:
	real_mass, delta, mdot, s. [ADAF SED parameters], mdotj, p epse, epsb. [Jet SED parameters]

Returns:
    log(nu), log(nu*L_nu): (array) Values of frequency and SED, respectively

Plot:
    plot: If True, the code shows a SED figure
    save: save=True saves a figure
    savefile: name of the saved figure
    title: title of the plot

Example:
    nu, adaf = sed_calculator(component=both,real_mass=1e8, delta=0.25, mdot=2e-3, 2=0.4, mdotj=2e-7, p=2.4, epse=1e-2, epsb=1e-2)
    [If you want to plot the SED] plt.plot(nu,adaf)
    '''
    if(component=='adaf'):
        sed_total=sed_calculator_adaf(real_mass, delta, mdot, s)
        nu=nu_off[:-1]
    if(component=='jet'):
        sed_total=sed_calculator_jet(real_mass, mdotj, p, epse, epsb)
        nu=nu_jet
    if (component=='both'):
        #seds for the two components
        sed_adaf=sed_calculator_adaf(real_mass, delta, mdot, s)
        sed_jet=sed_calculator_jet(real_mass, mdotj, p, epse, epsb)
        #interpolating jet sed in order to sum jet+adaf seds
        jet_interp=interpolation(sed_jet, x_sed=nu_jet)
        sed_total= np.log10(10**(sed_adaf-30.) + 10**(jet_interp(nu_off[:-1])-30.))+30.
        nu=nu_off[:-1]
    if(component!='jet' and component!='adaf' and component!='both'):
        print('Wrong name of \'component\'. Please select the desired component: \'adaf\', \'jet\' or \'both\'')
        nu,sed_total=None,None
    if(plot==True):
        sed_plot(nu,sed_total, title=title, save=save, savefile=savefile)
    return([nu,sed_total])

def sed_plot(x, y, title=' ', save=False, savefile='sed.png'):
    '''
Auxiliary function to plot SEDs.

Parameters:
    x, y: x-axis and y-axis of the plot
    save: save=True saves a figure
    savefile: name of the saved figure
    title: title of the plot
Returns:
    The plot of x,y with an adequated axis
    '''
    #Figure 3: The fitting
    fig, ax = plt.subplots()
    fig.set_size_inches((15,10))
    ax.set_xlim(8,20)
    ax.set_ylim(np.min(y)-0.5, np.max(y)+0.5)

    plt.plot(x,y, '-k')

    #Plot details
    plt.title(title, fontsize=28)#str(title)+r' delta = '+str(round(delta_mcmc[0],2))+' + '+str(round(delta_mcmc[2],2))+' - '+str(round(delta_mcmc[1],2))+r' mdot = '+str('{:.2e}'.format(mdot_exp[0]))+' + '+str('{:.2e}'.format(mdot_exp[2]))+' - '+str('{:.2e}'.format(mdot_exp[1]))+r' s = '+str(round(s_mcmc[0],2))+' + '+str(round(s_mcmc[2],2))+' - '+str(round(s_mcmc[1],2)))
    plt.xlabel(r'$\log_{10}(\nu / [Hz])$', fontsize=24)
    plt.ylabel(r'$\log_{10}(\nu L_{\nu} / [erg/s])$', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    #upper x-axis
    ax2=pylab.twiny()
    ax2.set_xlim(8,20)    # set this to match the lower X axis
    ax2.set_xticks([8.477,9.477,10.477,11.477,12.477,13.477,14.477,15.4768,16.383,17.383,18.383,19.383])
    ax2.set_xticklabels(['1m','10cm','1cm','1mm','100$\mu$m','10$\mu$m','1$\mu$m','1000$\AA$','.1keV','1keV','10keV','100keV'],size=12)
    pylab.minorticks_on()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    #saving figure
    if (save==True):
        plt.savefig(savefig)
    plt.show()
