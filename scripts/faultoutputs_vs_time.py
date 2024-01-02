#!/usr/bin/env python3
'''
Functions related to plotting variables as time series
By Jeena Yun
Last modification: 2023.12.04.
'''
import numpy as np
import matplotlib.pyplot as plt
import myplots
mp = myplots.Figpref()

mypink = (230/255,128/255,128/255)
myblue = (118/255,177/255,230/255)
myburgundy = (214/255,0,0)
mynavy = (17/255,34/255,133/255)

yr2sec = 365*24*60*60
wk2sec = 7*24*60*60

# fields: Time [s] | state [s?] | cumslip0 [m] | traction0 [MPa] | slip-rate0 [m/s] | normal-stress [MPa]
# Index:     0     |      1     |       2      |        3        |         4        |          5 

def phi2theta(prefix,raw_phi,dep,dep_index):
    import change_params
    ch = change_params.variate()
    y,_,_,raw_b,a_b,_,_,raw_Dc,others = ch.load_parameter(prefix,print_on=False)
    if dep_index is None:
        phi = np.max(raw_phi,axis=0)
        dep_index = np.argmax(raw_phi,axis=0)
    else:
        phi = raw_phi[dep_index,:]
    if len(raw_Dc) == 2:
        L = ch.same_length(raw_Dc[1],raw_Dc[0],dep[dep_index])
    else:
        L = ch.same_length(y,raw_Dc,dep[dep_index])
    if isinstance(dep_index,np.int64): L = L*np.ones(len(phi))
    if abs(np.mean(np.diff(raw_b)))<1e-3:
        b = raw_b[0]*np.ones(len(phi))
    # else:
    #     if len(a_b) == 2:
    #         b = ch.same_length(a_b[1],raw_b,dep[dep_index])
    #     else:
    #         b = ch.same_length(y,raw_b,dep[dep_index])
    f0 = others[-1]*np.ones(len(phi))
    V0 = others[-2]*np.ones(len(phi))

    theta = np.exp(np.divide(phi-f0,b) - np.log(V0)+np.log(L))
    return theta

def get_var(outputs,dep,target_depth,target_var,plot_in_sec,abs_on,prefix):
    if target_depth == None:
        print('Mode: Maximum along fault')
        indx = None
    else:
        indx = np.argmin(abs(abs(dep) - abs(target_depth)))
        print('Depth = %1.1f [km]'%abs(dep[indx]))

    if target_var == 'state':
        var_idx,ylab,fign = 1,'log$_{10}$(State Variable [s])','state'
        theta = phi2theta(prefix,np.array(outputs[:,:,var_idx]),dep,indx)
    elif target_var == 'slip':
        var_idx,ylab,fign = 2,'Cumulative Slip [m]','cumslip'
    elif target_var == 'shearT':
        var_idx,ylab,fign = 3,'Shear Stress [MPa]','shearT'
    elif target_var == 'sliprate':
        var_idx,ylab,fign = 4,'log$_{10}$(Slip Rate [m/s])','sliprate'
    elif target_var == 'normalT':
        var_idx,ylab,fign = 5,'Normal Stress [MPa]','normalT'

    if target_depth == None:
        if var_idx == 1:
            var = np.log10(theta)
            ylab = 'log$_{10}$(Peak State Variable [s])'
        elif var_idx == 3 and np.all(outputs[:,:,var_idx]<0):
            ii = np.argmax(np.array(abs(outputs[:,:,var_idx])),axis=0)
            var = np.array([outputs[ii[i],i,var_idx] for i in range(len(ii))])
            ylab = 'Peak ' + ylab
        elif var_idx == 4:
            var = np.log10(np.max(np.array(outputs[:,:,var_idx]),axis=0))
            ylab = 'log$_{10}$(Peak Slip Rate [m/s])'
        else:
            var = np.max(np.array(outputs[:,:,var_idx]),axis=0)
            ylab = 'Peak ' + ylab
    else:
        if var_idx == 1:
            var = np.log10(theta)
        elif var_idx == 4:
            if np.all(np.array(outputs[indx])[:,var_idx]>0):
                var = np.log10(np.array(outputs[indx])[:,var_idx])
            else:
                print('Negative slip rate - taking absolute')
                var = np.log10(abs(np.array(outputs[indx])[:,var_idx]))
        else:
            var = np.array(outputs[indx])[:,var_idx]
    if abs_on:
        var = abs(var)
        ylab = 'Absolute ' + ylab
    if plot_in_sec:
        time = np.array(outputs[0])[:,0]
        xlab = 'Time [s]'
    else:
        time = np.array(outputs[0])[:,0]/yr2sec
        xlab = 'Time [yrs]'
    return time,var,xlab,ylab,fign,indx

def get_lag(base_t,var_ref,t_ref,var_pert,t_pert,print_on=False):
    from scipy import signal,interpolate
    dat_ref = interpolate.interp1d(t_ref,var_ref)(base_t) # reference model 
    if base_t.max()>t_pert.max() or base_t.min()<t_pert.max():
        dat_pert = interpolate.interp1d(t_pert,var_pert)(base_t[np.logical_and(base_t<=t_pert.max(),base_t>=t_pert.min())])
    else:
        dat_pert = interpolate.interp1d(t_pert,var_pert)(base_t)
    dat_ref -= np.mean(dat_ref)
    dat_pert -= np.mean(dat_pert)
    corr = signal.correlate(dat_ref,dat_pert)
    lags = signal.correlation_lags(len(dat_ref),len(dat_pert))
    dat_corr = corr/max(abs(corr))
    dt = np.diff(base_t)[0]
    lag = lags[np.argmax(dat_corr)]*dt
    if print_on:
        if lag > 0:
            print('Time advance of %1.4f s'%lag)
        elif lag == 0:
            print('No change in event time')
        else:
            print('Time delay of %1.4f s'%lag)
    return lag

def fout_time(save_dir,outputs,dep,target_depth,target_var,plot_in_sec,ls='-',col='k',lab='',abs_on=False,prefix=None,save_on=True):
    time,var,xlab,ylab,fign,indx = get_var(outputs,dep,target_depth,target_var,plot_in_sec,abs_on,prefix)
    plt.plot(time,var,color=col, lw=2.5,label=lab,linestyle=ls)
    plt.xlabel(xlab,fontsize=17)
    plt.ylabel(ylab,fontsize=17)
    if target_depth < 1e-1:
        plt.title('Depth = surface',fontsize=20,fontweight = 'bold')
    else:
        plt.title('Depth = %1.1f [km]'%abs(dep[indx]),fontsize=20,fontweight = 'bold')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/%s.png'%(save_dir,fign))

def fout_time_max(save_dir,outputs,target_var,plot_in_sec,toff=0,ls='-',col='k',lab='',abs_on=False,prefix=None,dep=None,save_on=True):
    time,var,xlab,ylab,fign,_ = get_var(outputs,dep,None,target_var,plot_in_sec,abs_on,prefix)
    if abs(toff) > 0: time += toff
    plt.plot(time,var,color=col, lw=2.5,label=lab,linestyle=ls)
    plt.xlabel(xlab,fontsize=17)
    plt.ylabel(ylab,fontsize=17)
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/%s.png'%(save_dir,fign))

def fout_time_max_diff(save_dir,outputs,outputs2,target_var,plot_in_sec,ls='-',col='k',lab='',abs_on=False,prefix=None,save_on=True):
    time1,var1,xlab,ylab,fign,_ = get_var(outputs,None,None,target_var,plot_in_sec,abs_on,prefix)
    time2,var2,_,_,_,_ = get_var(outputs2,None,None,target_var,plot_in_sec,abs_on)
    plt.plot(time1,var2-var1,color=col, lw=2,label=lab,linestyle=ls)
    plt.xlabel(xlab,fontsize=17)
    plt.ylabel(ylab,fontsize=17)
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/%s.png'%(save_dir,fign))

def stress_pert_at_depth(save_dir,ref_outputs,dep,delVar,depth_range,target_depth,target_var,plot_in_sec,dt=0.01,ls='-',col='k',lab='',abs_on=False,print_on=True,save_on=True):
    indx = np.argmin(abs(abs(dep) - abs(target_depth)))
    print('Depth = %1.1f [km]'%abs(dep[indx]))
    t0 = ref_outputs[0,0,0]
    time = np.linspace(t0,t0+delVar.shape[0]*dt,delVar.shape[0])
    di = np.argmin(abs(depth_range+target_depth))
    if target_var == 'shearT':
        if abs_on:
            var = abs(np.array(ref_outputs[indx])[:,3])
            ylab = 'Absolute Shear Stress [MPa]'
        else:
            var = np.array(ref_outputs[indx])[:,3]
            ylab = 'Shear Stress [MPa]'
        fign = 'shearT'
    elif target_var == 'normalT':
        if abs_on:
            var = abs(np.array(ref_outputs[indx])[:,5])
            ylab = 'Absolute Normal Stress [MPa]'
        else:
            var = np.array(ref_outputs[indx])[:,5]
            ylab = 'Normal Stress [MPa]'
        fign = 'normalT'
    if plot_in_sec:        # --- Plot in seconds
        plt.plot(time,delVar[:,di]+var[0],color=col,lw=2.5,label=lab,linestyle=ls)
        # plt.plot(time,delVar[:,di]+var[1],color=col,lw=2.5,label=lab,linestyle=ls)
        plt.xlabel('Time [s]',fontsize=17)
        otime = time
    else:        # --- Plot in years
        plt.plot(time/yr2sec,delVar[:,di]+var[0],color=col, lw=2.5,label=lab,linestyle=ls)
        # plt.plot(time/yr2sec,delVar[:,di]+var[1],color=col, lw=2.5,label=lab,linestyle=ls)
        plt.xlabel('Time [yrs]',fontsize=17)
        otime = time/yr2sec
    plt.ylabel(ylab,fontsize=17)
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/%s.png'%(save_dir,fign))
    return otime,delVar[:,di]+var[1]

def dCFS_at_depth(save_dir,outputs,dep,dCFSt_seissol,depth_range,target_depth,mu,dt=0.01,ls1='-',ls2='-',col1='k',col2=mp.myburgundy,lab1='',lab2='',save_on=True):
    from scipy import interpolate
    indx = np.argmin(abs(abs(dep) - abs(target_depth)))
    print('Depth = %1.2f [km]'%abs(dep[indx]))
    pn = np.array(outputs[indx])[:,5]-np.array(outputs[indx])[0,5]
    ts = np.array(outputs[indx])[:,3]-np.array(outputs[indx])[0,3]
    dCFSt_in = [interpolate.interp1d(depth_range,dCFSt_seissol[ti])(-target_depth) for ti in range(dCFSt_seissol.shape[0])]
    dCFSt_out = -ts - mu*pn
    # dCFSt_out = -ts + mu*pn
    time_in = np.linspace(0,dCFSt_seissol.shape[0]*dt,dCFSt_seissol.shape[0])
    time_out = np.array(outputs[indx])[:,0]-np.array(outputs[indx])[0,0]
    plt.plot(time_in,dCFSt_in,color=col1,lw=2,label='Input Perturbation'+lab1,linestyle=ls1)
    plt.plot(time_out,dCFSt_out,color=col2,lw=2,label='Output Perturbation'+lab2,linestyle=ls2)
    plt.xlabel('Time [s]',fontsize=17)
    plt.ylabel('dCFS [Mpa]',fontsize=17)
    plt.grid(True,alpha=0.5)
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/dCFS_model.png'%(save_dir))

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