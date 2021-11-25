import AGNNES_params as RP
import AGNNES_functions as RF
import AGNNES_MCMC as RS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data file
fit_title='AGNNES fit for '+RP.filename
data_file='Library/SED-'+RP.filename+'.txt'
limits_file='Library/SED-'+RP.filename+'-limits.txt'


#Reading the parameters file
observations = pd.read_csv(data_file, sep=",")
limits = pd.read_csv(limits_file, sep=",")


#Data points
xdata=observations['nu']
ydata=observations['nulnu']
yerror=(observations['unulnu']-observations['lnulnu'])/2.
yerror[yerror<0.005]=0.005 #NEED TO IMPROVE THIS
#upper limits (no need of errorbars)
xlims=limits['nu']
#print(xlims)
ylims=limits['lnulnu']
#print(ylims)
#ylerr=limits['unulnu']-limits['lnulnu']

#mass
real_mass=np.log10(RP.real_mass/1.e6)
#real_mass=(RP.real_mass/1.e6)


#Priors

#defining the prior array
prior=np.array([RP.delta_max, RP.delta_min, RP.logmdot_max, RP.logmdot_min, RP.s_max, RP.s_min])
#defining the prior array
prior_jet=np.array([RP.logmdot_jet_max, RP.logmdot_jet_min, RP.p_max, RP.p_min, RP.logepse_max, RP.logepse_min, RP.logepsb_max, RP.logepsb_min])


#prior=RF.norms
#prior_jet=RF.norms_jet
#Loading prior from parameters files if requested
#if (RP.usePriorADAF == True):
#prior=prior
#if (RP.usePriorJet == True):
#prior_jet=prior_jet

#number of free parameters for adaf/jet
ndim_jet=4
ndim_adaf=3

#Running
if(RP.ADAF==True and RP.Jet==False):
    #Calculations
    print('Only ADAF')
    sampler=RS.AGNNES_MCMC_adaf(real_mass,xdata,ydata,yerror, xlim=xlims, ylim=ylims,nwalkers=RP.nwalkers, prior=prior, ndim=ndim_adaf, n1=RP.n1, n2=RP.n2)
    #plotting
    #fit_params = RI.plotting(sampler, real_mass, xdata,ydata, yerror, xlim=xlims, ylim=ylims, save=RP.save_figs, title=RP.fit_title, filename=RP.filename, ndim=ndim_adaf)
    #np.save('sampler', sampler)
if(RP.ADAF==False and RP.Jet==True):
    #Calculations
    print('Only Jet')
    sampler=RS.AGNNES_MCMC_jet(real_mass,xdata,ydata,yerror, xlim=xlims, ylim=ylims,nwalkers=RP.nwalkers, prior_jet=prior_jet, ndim=ndim_jet, n1=RP.n1, n2=RP.n2, jet_max_nu=RP.jet_max_nu)
    #plotting
    #fit_params = RI.plotting(sampler, real_mass, xdata,ydata, yerror, xlim=xlims, ylim=ylims, save=RP.save_figs, title=RP.fit_title, filename=RP.filename, ndim=ndim_jet)
    #np.save('sampler', sampler)
if(RP.ADAF==True and RP.Jet==True):
    #Calculations
    print('ADAF+Jet')
    sampler=RS.AGNNES_MCMC(real_mass,xdata,ydata,yerror, xlim=xlims, ylim=ylims,nwalkers=RP.nwalkers, prior_adaf=prior, prior_jet=prior_jet, ndim=ndim_jet+ndim_adaf, n1=RP.n1, n2=RP.n2, jet_max_nu=RP.jet_max_nu)

    #plotting
    #fit_params,spectrum,chi2_final= RI.plotting(sampler, real_mass, xdata,ydata, yerror, xlim=xlims, ylim=ylims, save=RP.save_figs, title=RP.fit_title, filename=RP.filename, ndim=ndim_jet+ndim_adaf)
    #Saving results
    #np.save('Results/'+RP.filename+'sampler', sampler)
    #spectrum[0].to_csv('Results/'+RP.filename+'-spectrum-jet.csv', index=False)
    #spectrum[1].to_csv('Results/'+RP.filename+'-spectrum-adaf.csv', index=False)
    #spectrum[2].to_csv('Results/'+RP.filename+'-spectrum-total.csv', index=False)
    #fit_params.to_csv('Results/'+RP.filename+'-parameters.csv', index=False)


    #with open('Results/'+RP.filename+'-chi2.txt', 'w') as f:
        #print(chi2_final, file=f)

print('Tutorial-'+RP.filename+'.h5 saved')
