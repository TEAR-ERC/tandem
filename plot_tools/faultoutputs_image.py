#!/usr/bin/env python3
'''
Functions related to plotting spatio-temporal evolution of variables as an image
By Jeena Yun
Last modification: 2023.05.18.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from cmcrameri import cm
import change_params

ch = change_params.variate()

mypink = (230/255,128/255,128/255)
myblue = (118/255,177/255,230/255)
myburgundy = (214/255,0,0)
mynavy = (17/255,34/255,133/255)

yr2sec = 365*24*60*60
wk2sec = 7*24*60*60

# fields: Time [s] | state [s?] | cumslip0 [m] | traction0 [MPa] | slip-rate0 [m/s] | normal-stress [MPa]
# Index:     0     |      1     |       2      |        3        |         4        |          5 

def fout_image(sliprate,shearT,normalT,state_var,outputs,dep,plot_in_timestep,save_dir,prefix,vmin,vmax,plot_in_sec=False,save_on=True):
    which = np.where([sliprate,shearT,normalT,state_var])[0]
    for w in which:
        tf = [False,False,False,False]
        tf[w] = True
        X,Y,var,lab = get_var(tf[0],tf[1],tf[2],tf[3],outputs,dep,plot_in_timestep,plot_in_sec)
        plot_image(X,Y,var,lab,save_dir,prefix,plot_in_timestep,vmin,vmax,plot_in_sec,save_on)

def get_var(sliprate,shearT,normalT,state_var,outputs,dep,plot_in_timestep,plot_in_sec):
    if sliprate:
        lab = 'sliprate'
        idx = 4
    elif shearT:
        lab = 'shearT'
        idx = 3
    elif normalT:
        lab = 'normalT'
        idx = 5
    elif state_var:
        lab = 'statevar'
        idx = 1
    var = np.array([outputs[i][:,idx] for i in np.argsort(abs(dep))])
    if shearT or normalT:
        var = abs(var)

    if plot_in_timestep:
        print('Plot in time steps')
        xax = np.arange(var.shape[1])
    elif plot_in_sec:
        print('Plot in time [s]')
        xax = np.array(outputs[0][:,0])
    else:
        print('Plot in time [yrs]')
        xax = np.array(outputs[0][:,0]/ch.yr2sec)
    X,Y = np.meshgrid(xax,np.sort(abs(dep)))

    return X,Y,var,lab

def plot_image(X,Y,var,lab,save_dir,prefix,plot_in_timestep,vmin,vmax,plot_in_sec,save_on):
    Hs = ch.load_parameter(prefix)[1]

    plt.rcParams['font.size'] = '27'
    plt.figure(figsize=(20.6,11))

    if vmin is None:
        vmin = np.min(var)
    if vmax is None:
        vmax = np.max(var)

    if lab == 'sliprate':
        cb_label = 'Slip Rate [m/s]'
        cmap_n = 'magma'
    elif lab == 'shearT':
        cb_label = 'Shear Stress [MPa]'
        cmap_n = cm.davos
    elif lab == 'normalT':
        cb_label = 'Normal Stress [MPa]'
        cmap_n = cm.davos
    elif lab == 'statevar':
        cb_label = 'State Variable [1/s]'
        cmap_n = 'magma'

    if lab == 'sliprate':
        cb = plt.pcolormesh(X,Y,var,cmap=cmap_n,norm=colors.LogNorm(vmin,vmax))
    else:
        cb = plt.pcolormesh(X,Y,var,cmap=cmap_n,vmin=vmin,vmax=vmax)
        
    plt.colorbar(cb).set_label(cb_label,fontsize=30,rotation=270,labelpad=30)

    fsigma,ff0,fab,fdc,newb,newL = ch.what_is_varied(prefix)
    ver_info = ''
    if fsigma is not None:
        ver_info += 'Stress ver.%d '%fsigma
    if ff0 is not None: 
        ver_info += '+ f0 ver.%d '%fsigma
    if fab is not None:
        ver_info += '+ a-b ver.%d '%fab
    if fdc is not None:
        ver_info += '+ Dc ver.%d '%fdc
    if newb is not None:
        ver_info += '+ b = %2.4f '%newb
    if newL is not None:
        ver_info += '+ L = %2.4f'%newL

    xl = plt.gca().get_xlim()
    if len(ver_info) > 0:
        if ver_info[:2] == '+ ':
            ver_info = ver_info[2:]
        plt.text(xl[1]*0.025,Hs[0]*0.975,ver_info,color='w',fontsize=45,fontweight='bold',ha='left',va='bottom')
    plt.xlim(0,xl[1])
    plt.ylim(0,Hs[0])
    plt.gca().invert_yaxis()
    plt.ylabel('Depth [km]',fontsize=30)
    if plot_in_timestep:
        plt.xlabel('Timesteps',fontsize=30)
    elif plot_in_sec:        
        plt.xlabel('Time [s]',fontsize=30)
    else:        
        plt.xlabel('Time [yrs]',fontsize=30)

    plt.tight_layout()
    if save_on:
        if plot_in_timestep:
            plt.savefig('%s/%s_image_timestep.png'%(save_dir,lab),dpi=300)
        else:
            plt.savefig('%s/%s_image.png'%(save_dir,lab),dpi=300)
