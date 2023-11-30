#!/usr/bin/env python3
'''
Functions related to plotting variables as time series
By Jeena Yun
Last modification: 2023.10.13.
'''
import numpy as np
import matplotlib.pyplot as plt

mypink = (230/255,128/255,128/255)
myblue = (118/255,177/255,230/255)
myburgundy = (214/255,0,0)
mynavy = (17/255,34/255,133/255)

yr2sec = 365*24*60*60
wk2sec = 7*24*60*60

# fields: Time [s] | state [s?] | cumslip0 [m] | traction0 [MPa] | slip-rate0 [m/s] | normal-stress [MPa]
# Index:     0     |      1     |       2      |        3        |         4        |          5 

def fout_time_plt(save_dir,outputs,dep,target_depth,target_var,plot_in_sec,ls='-',col='k',lab='',save_on=True):
    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(8,6))
    indx = np.argmin(abs(abs(dep) - abs(target_depth)))
    print('Depth = %1.1f [km]'%abs(dep[indx]))
    time = np.array(outputs[indx])[:,0]
    if target_var == 'state':
        var = np.array(outputs[indx])[:,1]
        ylab = 'State Variable'
        fign = 'state'
    elif target_var == 'slip':
        var = np.array(outputs[indx])[:,2]
        ylab = 'Cumulative Slip [m]'
        fign = 'cumslip'
    elif target_var == 'shearT':
        var = abs(np.array(outputs[indx])[:,3])
        ylab = 'Absolute Shear Stress [MPa]'
        fign = 'shearT'
    elif target_var == 'sliprate':
        if np.all(np.array(outputs[indx])[:,4]>0):
            var = np.log10(np.array(outputs[indx])[:,4])
        else:
            print('Negative slip rate - taking absolute')
            var = np.log10(abs(np.array(outputs[indx])[:,4]))
        ylab = 'log$_{10}$(Slip Rate [m/s])'
        fign = 'sliprate'
    elif target_var == 'normalT':
        var = abs(np.array(outputs[indx])[:,5])
        ylab = 'Absolute Normal Stress [MPa]'
        fign = 'normalT'

    if plot_in_sec:        # --- Plot in seconds
        plt.plot(time,var,color='k',lw=2.5)
        plt.xlabel('Time [s]',fontsize=17)
    else:        # --- Plot in years
        plt.plot(time/yr2sec,var, color='k',lw=2.5)
        plt.xlabel('Time [s]',fontsize=17)
    plt.ylabel(ylab,fontsize=17)
    if target_depth < 1e-1:
        plt.title('Depth = surface',fontsize=20,fontweight = 'bold')
    else:
        plt.title('Depth = %1.1f [km]'%abs(dep[indx]),fontsize=20,fontweight = 'bold')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/%s.png'%(save_dir,fign))

def fout_time_ax(ax,outputs,dep,target_depth,target_var,plot_in_sec,ls='-',col='k',lab=''):
    indx = np.argmin(abs(abs(dep) - abs(target_depth)))
    print('Depth = %1.1f [km]'%abs(dep[indx]))
    time = np.array(outputs[indx])[:,0]
    if target_var == 'state':
        var = np.array(outputs[indx])[:,1]
        ylab = 'State Variable'
    elif target_var == 'slip':
        var = np.array(outputs[indx])[:,2]
        ylab = 'Cumulative Slip [m]'
    elif target_var == 'shearT':
        var = abs(np.array(outputs[indx])[:,3])
        ylab = 'Absolute Shear Stress [MPa]'
    elif target_var == 'sliprate':
        if np.all(np.array(outputs[indx])[:,4]>0):
            var = np.log10(np.array(outputs[indx])[:,4])
        else:
            print('Negative slip rate - taking absolute')
            var = np.log10(abs(np.array(outputs[indx])[:,4]))
        ylab = 'log$_{10}$(Slip Rate [m/s])'
    elif target_var == 'normalT':
        var = abs(np.array(outputs[indx])[:,5])
        ylab = 'Absolute Normal Stress [MPa]'

    if plot_in_sec:        # --- Plot in seconds
        ax.plot(time,var,color=col, lw=2.5,label=lab,linestyle=ls)
        ax.set_xlabel('Time [s]',fontsize=17)
    else:        # --- Plot in years
        ax.plot(time/yr2sec,var, color=col, lw=2.5,label=lab,linestyle=ls)
        ax.set_xlabel('Time [s]',fontsize=17)
    ax.ylabel(ylab,fontsize=17)
    if target_depth < 1e-1:
        ax.set_title('Depth = surface',fontsize=20,fontweight = 'bold')
    else:
        ax.set_title('Depth = %1.1f [km]'%abs(dep[indx]),fontsize=20,fontweight = 'bold')

def sliprate_time(save_dir,outputs,dep,sliprate,plot_in_sec,save_on=True):
    target_depth = sliprate # in km
    indx = np.argmin(abs(abs(dep) - abs(target_depth)))
    print('Depth = %1.1f [km]'%abs(dep[indx]))

    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(8,6))
    if plot_in_sec:        # --- Plot in seconds
        if np.all(np.array(outputs[indx])[:,4]>0):
            plt.plot(np.array(outputs[indx])[:,0], np.log10(np.array(outputs[indx])[:,4]), color='k', lw=2.5)
        else:
            print('Negative slip rate - taking absolute')
            plt.plot(np.array(outputs[indx])[:,0], np.log10(abs(np.array(outputs[indx])[:,4])), color='k', lw=2.5)
        plt.xlabel('Time [s]',fontsize=17)
    else:        # --- Plot in years
        if np.all(np.array(outputs[indx])[:,4]>0):
            plt.plot(np.array(outputs[indx])[:,0]/yr2sec, np.log10(np.array(outputs[indx])[:,4]), color='k', lw=2.5)
        else:
            print('Negative slip rate - taking absolute')
            plt.plot(np.array(outputs[indx])[:,0]/yr2sec, np.log10(abs(np.array(outputs[indx])[:,4])), color='k', lw=2.5)
        plt.xlabel('Time [yr]',fontsize=17)
    plt.ylabel('log$_{10}$(Slip Rate [m/s])',fontsize=17)
    if target_depth < 1e-1:
        plt.title('Depth = surface',fontsize=20,fontweight = 'bold')
    else:
        plt.title('Depth = %1.1f [km]'%abs(dep[indx]),fontsize=20,fontweight = 'bold')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/sliprate.png'%(save_dir),dpi=300)

def slip_time(save_dir,outputs,dep,slip,plot_in_sec,save_on=True):
    target_depth = slip # in km
    indx = np.argmin(abs(abs(dep) - abs(target_depth)))
    print('Depth = %1.1f [km]'%abs(dep[indx]))
    time = np.array(outputs[indx])[:,0]
    cumslip = np.array(outputs[indx])[:,2]

    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(8,6))
    if plot_in_sec:        # --- Plot in seconds
        plt.plot(time,cumslip, color='k', lw=2.5)
        plt.xlabel('Time [s]',fontsize=17)
    else:        # --- Plot in years
        plt.plot(time/yr2sec,cumslip, color='k', lw=2.5)
        plt.xlabel('Time [s]',fontsize=17)
    plt.ylabel('Cumulative Slip [m]',fontsize=17)
    if target_depth < 1e-1:
        plt.title('Depth = surface',fontsize=20,fontweight = 'bold')
    else:
        plt.title('Depth = %1.1f [km]'%abs(dep[indx]),fontsize=20,fontweight = 'bold')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/slip.png'%(save_dir))

def stress_time(save_dir,outputs,dep,stress,plot_in_sec,save_on=True):
    target_depth = stress # in km
    indx = np.argmin(abs(abs(dep) - abs(target_depth)))
    print('Depth = %1.1f [km]'%abs(dep[indx]))

    plt.rcParams['font.size'] = '15'
    fig,ax = plt.subplots(ncols=2,figsize=(14,6))
    if plot_in_sec:        # --- Plot in seconds
        ax[0].plot(np.array(outputs[indx])[:,0], np.array(outputs[indx])[:,3], color='k', lw=2.5)
        ax[0].set_xlabel('Time [s]',fontsize=17)
        ax[1].plot(np.array(outputs[indx])[:,0], np.array(outputs[indx])[:,5], color='k', lw=2.5)
        ax[1].set_xlabel('Time [s]',fontsize=17)
    else:        # --- Plot in years
        ax[0].plot(np.array(outputs[indx])[:,0]/yr2sec, np.array(outputs[indx])[:,3], color='k', lw=2.5)
        ax[0].set_xlabel('Time [yr]',fontsize=17)
        ax[1].plot(np.array(outputs[indx])[:,0]/yr2sec, np.array(outputs[indx])[:,5], color='k', lw=2.5)
        ax[1].set_xlabel('Time [yr]',fontsize=17)
    ax[0].set_ylabel('Shear Stress [MPa]',fontsize=17)
    ax[1].set_ylabel('Normal Stress [MPa]',fontsize=17)
    if target_depth < 1e-1:
        ax[0].set_title('Depth = surface',fontsize=20,fontweight = 'bold')
        ax[1].set_title('Depth = surface',fontsize=20,fontweight = 'bold')
    else:
        ax[0].set_title('Depth = %1.1f [km]'%abs(dep[indx]),fontsize=20,fontweight = 'bold')
        ax[1].set_title('Depth = %1.1f [km]'%abs(dep[indx]),fontsize=20,fontweight = 'bold')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/stresses.png'%(save_dir))

def state_time(save_dir,outputs,dep,state_var,plot_in_sec,save_on=True):
    target_depth = state_var # in km
    indx = np.argmin(abs(abs(dep) - abs(target_depth)))
    print('Depth = %1.1f [km]'%abs(dep[indx]))
    time = np.array(outputs[indx])[:,0]

    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(8,6))
    if plot_in_sec:        # --- Plot in seconds
        plt.plot(time,np.array(outputs[indx])[:,1], color='k', lw=2.5)
        plt.xlabel('Time [s]',fontsize=17)
    else:        # --- Plot in years
        plt.plot(time/yr2sec,np.array(outputs[indx])[:,1], color='k', lw=2.5)
        plt.xlabel('Time [s]',fontsize=17)
    plt.ylabel('State Variable',fontsize=17)
    if target_depth < 1e-1:
        plt.title('Depth = surface',fontsize=20,fontweight = 'bold')
    else:
        plt.title('Depth = %1.1f [km]'%abs(dep[indx]),fontsize=20,fontweight = 'bold')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/state.png'%(save_dir))