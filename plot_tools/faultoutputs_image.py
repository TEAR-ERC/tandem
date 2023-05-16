#!/usr/bin/env python3
'''
Functions related to plotting spatio-temporal evolution of variables as an image
By Jeena Yun
Last modification: 2023.05.16.
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


def fout_image(sliprate,shearT,normalT,state_var,outputs,dep,plot_in_timestep,save_dir,prefix,vmin,vmax):
    X,Y,var,lab = get_var(sliprate,shearT,normalT,state_var,outputs,dep,plot_in_timestep)
    plot_image(X,Y,var,lab,save_dir,prefix,plot_in_timestep,vmin,vmax)

def get_var(sliprate,shearT,normalT,state_var,outputs,dep,plot_in_timestep):
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
    else:
        print('Plot in time')
        xax = np.array(outputs[0][:,0])
    X,Y = np.meshgrid(xax,np.sort(abs(dep)))

    return X,Y,var,lab

def plot_image(X,Y,var,lab,save_dir,prefix,plot_in_timestep,vmin,vmax):
    Hs = ch.load_parameter(prefix)[1]

    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(15,8))

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
        
    plt.colorbar(cb).set_label(cb_label,fontsize=15,rotation=270,labelpad=25)

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
        plt.text(xl[1]*0.025,Hs[0]*0.975,ver_info,color='w',fontsize=25,fontweight='bold',ha='left',va='bottom')
    plt.xlim(0,xl[1])
    plt.ylim(0,Hs[0])
    plt.gca().invert_yaxis()
    plt.ylabel('Depth [km]',fontsize=17)
    if plot_in_timestep:
        plt.xlabel('Timesteps',fontsize=17)
    else:
        plt.xlabel('Time [s]',fontsize=17)

    plt.tight_layout()
    if plot_in_timestep:
        plt.savefig('%s/%s_image_timestep.png'%(save_dir,lab),dpi=300)
    else:
        plt.savefig('%s/%s_image.png'%(save_dir,lab),dpi=300)

# def stress_image(save_dir,prefix,outputs,dep,plot_in_timestep,vmin=None,vmax=None):
#     Hs = ch.load_parameter(prefix)[1]
#     # z = np.zeros(len(dep))
#     # shear_stress = []
#     # c = 0
#     # for i in np.argsort(abs(dep)):
#     #     z[c] = abs(dep[i])
#     #     shear_stress.append(outputs[i][:,3])
#     #     c += 1        
#     # shear_stress = np.array(shear_stress)

#     shear_stress = np.array([outputs[i][:,3] for i in np.argsort(abs(dep))])
#     z = abs(np.sort(dep))

#     if plot_in_timestep:
#         print('Plot in time steps')
#         xax = np.arange(shear_stress.shape[1])
#     else:
#         print('Plot in time')
#         xax = np.array(outputs[0][:,0])
#     X,Y = np.meshgrid(xax,z)

#     plt.rcParams['font.size'] = '15'
#     plt.figure(figsize=(15,8))
#     if vmin is None:
#         vmin = np.min(abs(shear_stress))
#     if vmax is None:
#         vmax = np.max(abs(shear_stress))
#     cb = plt.pcolormesh(X,Y,abs(shear_stress),cmap='magma',vmin=vmin,vmax=vmax)
#     plt.ylabel('Depth [km]',fontsize=17)
#     if plot_in_timestep:
#         plt.xlabel('Timesteps',fontsize=17)
#     else:
#         plt.xlabel('Time [s]',fontsize=17)
#     plt.colorbar(cb,extend='max').set_label('Shear Stress [MPa]',fontsize=15,rotation=270,labelpad=25)
#     xl = plt.gca().get_xlim()
#     plt.xlim(0,xl[1])
#     plt.ylim(0,Hs[0])
#     plt.gca().invert_yaxis()

#     plt.tight_layout()
#     plt.savefig('%s/stress_image.png'%(save_dir),dpi=300)
