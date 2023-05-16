#!/usr/bin/env python3
'''
Functions related to plotting cumulative slip vs. depth plot
By Jeena Yun
Last modification: 2023.05.03.
'''
import numpy as np
import matplotlib.pylab as plt
import change_params

ch = change_params.variate()

mypink = (230/255,128/255,128/255)
myblue = (118/255,177/255,230/255)
myburgundy = (214/255,0,0)
mynavy = (17/255,34/255,133/255)

yr2sec = 365*24*60*60
wk2sec = 7*24*60*60

def with_ab(ax,prefix):
    from ab_profile import ab_with_cumslip
    ab_with_cumslip(ax,prefix,fs_label=30,fs_legend=15,ytick_on=False)

def with_stress(ax,prefix,outputs,dep):
    from stress_profile import stress_with_cumslip
    stress_with_cumslip(ax,prefix,outputs,dep,fs_label=30,fs_legend=15,ytick_on=False)

def with_dc(ax,prefix):
    from Dc_profile import Dc_with_cumslip
    Dc_with_cumslip(ax,prefix,fs_label=30,fs_legend=15,ytick_on=False)

def with_depth_dist(ax,prefix,cumslip_outputs):
    Hs = ch.load_parameter(prefix)[1]
    ax.hist(cumslip_outputs[1][1],bins=np.arange(0,Hs[0]+0.2,0.2),color='k',edgecolor='k',orientation='horizontal')
    ax.set_xlabel('Counts',fontsize=30)
    ax.axes.yaxis.set_ticklabels([])
    xl = ax.get_xlim()

    ax.hlines(y=Hs[1],xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')
    ax.hlines(y=(Hs[1]+Hs[2]),xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')
    ax.set_ylim(Hs[0],0)
    if len(Hs) > 3:
        ax.hlines(y=Hs[3],xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')
        if len(Hs) > 4:
            ax.hlines(y=(Hs[3]-Hs[4]),xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')
            ax.set_ylim(Hs[0]/2,0)
    ax.set_xlim(xl)

def cumslip_basic(ax,prefix,cumslip_outputs):
    # cumslip_outputs = [timeout, evout, creepout, coseisout, intermout]
    # cumslip_outputs[0] = [tstart_coseis,tend_coseis]
    # cumslip_outputs[1] = [evslip,evdep]
    # cumslip_outputs[2] = [cscreep,depcreep]
    # cumslip_outputs[3] = [cscoseis,depcoseis]
    # cumslip_outputs[4] = [csinterm,depinterm]
    if len(cumslip_outputs) > 4:
        interm = True
    else:
        interm = False
    Hs = ch.load_parameter(prefix)[1]
    
    ax.plot(cumslip_outputs[2][0],cumslip_outputs[2][1],color='royalblue',lw=1)
    if interm:
        ax.plot(cumslip_outputs[-1][0],cumslip_outputs[-1][1],color='yellowgreen',lw=1)
    ax.plot(cumslip_outputs[3][0],cumslip_outputs[3][1],color='chocolate',lw=1)
    ax.set_ylabel('Depth [km]',fontsize=30)
    ax.set_xlabel('Cumulative Slip [m]',fontsize=30)
    ev = ax.scatter(cumslip_outputs[1][0],cumslip_outputs[1][1],marker='*',s=700,facecolor=myburgundy,edgecolors='k',lw=2,zorder=3)
    ax.legend([ev],['Hypocenters'],fontsize=25,framealpha=1,loc='lower right')
    xl = ax.get_xlim()
    ax.set_xlim(0,xl[1])
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()

def cumslip_spinup(ax,prefix,cumslip_outputs,spup_cumslip_outputs):
    # spup_cumslip_outputs = [new_inits, spup_evslip, spup_cscreep, spup_cscoseis, spup_csinterm]
    # new_inits = [new_init_Sl,new_init_dp]
    if len(cumslip_outputs) > 4:
        interm = True
    else:
        interm = False
    Hs = ch.load_parameter(prefix)[1]

    ax.plot(spup_cumslip_outputs[2],cumslip_outputs[2][1],color='royalblue',lw=1)
    if interm:
        ax.plot(spup_cumslip_outputs[4],cumslip_outputs[4][1],color='yellowgreen',lw=1)
    ax.plot(spup_cumslip_outputs[3],cumslip_outputs[3][1],color='chocolate',lw=1)
    ev = ax.scatter(spup_cumslip_outputs[1],cumslip_outputs[1][1],marker='*',s=700,facecolor=myburgundy,edgecolors='k',lw=2,zorder=3)
    xl = ax.get_xlim()
    ax.set_xlim(0,xl[1])

    ax.set_ylabel('Depth [km]',fontsize=30)
    ax.set_xlabel('Cumulative Slip [m]',fontsize=30)
    ax.legend([ev],['Hypocenters'],fontsize=25,framealpha=1,loc='lower right')
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()

def three_set(save_dir,prefix,outputs,dep,cumslip_outputs,Vths,dt_coseismic,plot_dd,plot_ab,plot_stress,plot_dc,spup_cumslip_outputs=None):
    plt.rcParams['font.size'] = '27'
    fig,ax = plt.subplots(ncols=3, figsize=(29,11), gridspec_kw={'width_ratios': [4, 1, 1]})
    if spup_cumslip_outputs is not None:
        spin_up = True
    else:
        spin_up = False
    if not spin_up:
        cumslip_basic(ax[0],prefix,cumslip_outputs)
    else:
        cumslip_spinup(ax[0],prefix,cumslip_outputs,spup_cumslip_outputs)
    c = 1; lab = ''
    if plot_ab:
        lab += '_ab'
        with_ab(ax[c],prefix)
        c += 1
    if plot_stress:
        lab += '_stress'
        with_stress(ax[c],prefix,outputs,dep)
        c += 1
    if plot_dc:
        lab += '_dc'
        with_dc(ax[c],prefix)
        c += 1
    if plot_dd:
        lab += '_withdepth'
        with_depth_dist(ax[c],prefix,cumslip_outputs)
        c += 1
    plt.tight_layout()
    if not spin_up:
        plt.savefig('%s/cumslip_%d_%d%s.png'%(save_dir,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)
    else:
        plt.savefig('%s/spinup_cumslip_%d_%d%s.png'%(save_dir,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

def two_set(save_dir,prefix,outputs,dep,cumslip_outputs,Vths,dt_coseismic,plot_dd,plot_ab,plot_stress,plot_dc,spup_cumslip_outputs=None):
    plt.rcParams['font.size'] = '27'
    fig,ax = plt.subplots(ncols=2, figsize=(24,11), gridspec_kw={'width_ratios': [4, 1]})
    plt.subplots_adjust(wspace=0.05)
    if spup_cumslip_outputs is not None:
        spin_up = True
    else:
        spin_up = False
    if not spin_up:
        cumslip_basic(ax[0],prefix,cumslip_outputs)
    else:
        cumslip_spinup(ax[0],prefix,cumslip_outputs,spup_cumslip_outputs)
    c = 1; lab = ''
    if plot_ab:
        lab += '_ab'
        with_ab(ax[c],prefix)
        c += 1
    if plot_stress:
        lab += '_stress'
        with_stress(ax[c],prefix,outputs,dep)
        c += 1
    if plot_dc:
        lab += '_dc'
        with_dc(ax[c],prefix)
        c += 1
    if plot_dd:
        lab += '_withdepth'
        with_depth_dist(ax[c],prefix,cumslip_outputs)
        c += 1
    plt.tight_layout()
    if not spin_up:
        plt.savefig('%s/cumslip_%d_%d%s.png'%(save_dir,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)
    else:
        plt.savefig('%s/spinup_cumslip_%d_%d%s.png'%(save_dir,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)
    plt.show(block=False)
plt.pause(0.1)
plt.close()

def only_cumslip(save_dir,prefix,cumslip_outputs,Vths,dt_coseismic,spup_cumslip_outputs=None):
    plt.rcParams['font.size'] = '27'
    fig,ax = plt.subplots(figsize=(18,11))
    if spup_cumslip_outputs is not None:
        spin_up = True
    else:
        spin_up = False
    if not spin_up:
        cumslip_basic(ax,prefix,cumslip_outputs)
    else:
        cumslip_spinup(ax,prefix,cumslip_outputs,spup_cumslip_outputs)
    plt.tight_layout()
    if not spin_up:
        plt.savefig('%s/cumslip_%d_%d.png'%(save_dir,int(Vths*100),int(dt_coseismic*10)),dpi=300)
    else:
        plt.savefig('%s/spinup_cumslip_%d_%d.png'%(save_dir,int(Vths*100),int(dt_coseismic*10)),dpi=300)
    plt.show(block=False)
plt.pause(0.1)
plt.close()

def spup_where(save_dir,prefix,cumslip_outputs,spup_cumslip_outputs,Vths,dt_coseismic):
    if len(cumslip_outputs) > 4:
        interm = True
    else:
        interm = False
    Hs = ch.load_parameter(prefix)[1]
    plt.rcParams['font.size'] = '27'
    plt.figure(figsize=(19,11))
    plt.plot(cumslip_outputs[2][0],cumslip_outputs[2][1],color='royalblue',lw=1)
    if interm:
        plt.plot(cumslip_outputs[4][0],cumslip_outputs[4][1],color='yellowgreen',lw=1)
    plt.plot(cumslip_outputs[3][0],cumslip_outputs[3][1],color='chocolate',lw=1)
    plt.plot(spup_cumslip_outputs[0][0],spup_cumslip_outputs[0][1],color='yellowgreen',lw=5)
    ev = plt.scatter(cumslip_outputs[1][0],cumslip_outputs[1][1],marker='*',s=700,facecolor=myburgundy,edgecolors='k',lw=2,zorder=3)
    plt.ylabel('Depth [km]',fontsize=30)
    plt.xlabel('Cumulative Slip [m]',fontsize=30)
    plt.legend([ev],['Hypocenters'],fontsize=25,framealpha=1,loc='lower right')
    xl = plt.gca().get_xlim()
    plt.xlim(0,xl[1])
    plt.ylim(0,Hs[0])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('%s/spinup_cumslip_%d_%d_where.png'%(save_dir,int(Vths*100),int(dt_coseismic*10)),dpi=300)
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()