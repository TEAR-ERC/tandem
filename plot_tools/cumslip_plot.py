#!/usr/bin/env python3
'''
Functions related to plotting cumulative slip vs. depth plot
By Jeena Yun
Last modification: 2023.09.05.
'''
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import change_params
from cumslip_compute import analyze_events

ch = change_params.variate()

mypink = (230/255,128/255,128/255)
mydarkpink = (200/255,110/255,110/255)
# mydarkpink = (210/255,115/255,115/255)
myblue = (118/255,177/255,230/255)
myburgundy = (214/255,0,0)
mynavy = (17/255,34/255,133/255)
mylightblue = (218/255,230/255,240/255)
myygreen = (120/255,180/255,30/255)
mylavender = (170/255,100/255,215/255)
# mydarkviolet = (120/255,55/255,145/255)
mydarkviolet = (145/255,80/255,180/255)
pptyellow = (255/255,217/255,102/255)

yr2sec = 365*24*60*60
wk2sec = 7*24*60*60

def with_ab(ax,prefix):
    from ab_profile import ab_with_cumslip
    asp = ab_with_cumslip(ax,prefix,fs_label=30,fs_legend=15,ytick_on=False)
    return asp

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

def cumslip_basic(ax,prefix,cumslip_outputs,rths,publish):
    # cumslip_outputs = [timeout, evout, creepout, coseisout, intermout]
    # cumslip_outputs[0] = [tstart_coseis,tend_coseis]
    # cumslip_outputs[1] = [evslip,evdep,fault_slip]
    # cumslip_outputs[2] = [cscreep,depcreep]
    # cumslip_outputs[3] = [cscoseis,depcoseis]
    # cumslip_outputs[4] = [csinterm,depinterm]
    system_wide,partial_rupture,event_cluster,lead_fs = analyze_events(cumslip_outputs,rths)[2:6]
    Hs = ch.load_parameter(prefix)[1]
    ver_info = ch.version_info(prefix)

    if len(cumslip_outputs) > 4:
        ax.plot(cumslip_outputs[-1][0],cumslip_outputs[-1][1],color='yellowgreen',lw=1)
    ax.plot(cumslip_outputs[3][0],cumslip_outputs[3][1],color=mydarkpink,lw=1)
    ax.plot(cumslip_outputs[2][0],cumslip_outputs[2][1],color='0.62',lw=1)
    ax.set_ylabel('Depth [km]',fontsize=30)
    ax.set_xlabel('Cumulative Slip [m]',fontsize=30)
    if len(system_wide) > 0:
        ax.scatter(cumslip_outputs[1][0][system_wide],cumslip_outputs[1][1][system_wide],marker='*',s=700,facecolor=mydarkviolet,edgecolors='k',lw=1,zorder=3,label='System-wide events')
    if len(lead_fs) > 0:
        ax.scatter(cumslip_outputs[1][0][lead_fs],cumslip_outputs[1][1][lead_fs],marker='d',s=250,facecolor=mydarkviolet,edgecolors='k',lw=1,zorder=3,label='Leading foreshocks')
    if len(partial_rupture) > 0:
        # ax.scatter(cumslip_outputs[1][0][partial_rupture],cumslip_outputs[1][1][partial_rupture],marker='*',s=700,facecolor=mylightblue,edgecolors='k',lw=1,zorder=3,label='Partial rupture events')
        ax.scatter(cumslip_outputs[1][0][partial_rupture],cumslip_outputs[1][1][partial_rupture],marker='d',s=250,facecolor=mylightblue,edgecolors='k',lw=1,zorder=2,label='Partial rupture events')
    # ax.legend(fontsize=25,framealpha=1,loc='lower right')
    ax.legend(fontsize=20,framealpha=1,loc='lower right')
    xl = ax.get_xlim()
    if len(ver_info) > 0 and not publish:
        ax.text(xl[1]*0.025,Hs[0]*0.975,ver_info,color='k',fontsize=35,fontweight='bold',ha='left',va='bottom')
    ax.set_xlim(0,xl[1])
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()

def cumslip_spinup(ax,prefix,cumslip_outputs,spup_cumslip_outputs,rths,publish):
    # spup_cumslip_outputs = [new_inits, spup_evslip, spup_cscreep, spup_cscoseis, spup_csinterm, spin_up_idx]
    # new_inits = [new_init_Sl,new_init_dp]
    system_wide,partial_rupture,event_cluster,lead_fs = analyze_events(cumslip_outputs,rths)[2:6]
    # partial_rupture = partial_rupture[np.arange(0,len(partial_rupture),2)]
    Hs = ch.load_parameter(prefix)[1]
    ver_info = ch.version_info(prefix)

    if len(cumslip_outputs) > 4:
        ax.plot(spup_cumslip_outputs[4],cumslip_outputs[4][1],color='yellowgreen',lw=1)
    ax.plot(spup_cumslip_outputs[3],cumslip_outputs[3][1],color=mydarkpink,lw=1)
    ax.plot(spup_cumslip_outputs[2],cumslip_outputs[2][1],color='0.62',lw=1)
    if len(system_wide) > 0:
        ax.scatter(spup_cumslip_outputs[1][system_wide],cumslip_outputs[1][1][system_wide],marker='*',s=700,facecolor=mydarkviolet,edgecolors='k',lw=1,zorder=3,label='System-wide events')
    if len(lead_fs) > 0:
        ax.scatter(spup_cumslip_outputs[1][lead_fs],cumslip_outputs[1][1][lead_fs],marker='d',s=250,facecolor=mydarkviolet,edgecolors='k',lw=1,zorder=3,label='Leading foreshocks')
    if len(partial_rupture) > 0:
        # ax.scatter(spup_cumslip_outputs[1][partial_rupture],cumslip_outputs[1][1][partial_rupture],marker='*',s=700,facecolor=mylightblue,edgecolors='k',lw=1,zorder=3,label='Partial rupture events')
        ax.scatter(spup_cumslip_outputs[1][partial_rupture],cumslip_outputs[1][1][partial_rupture],marker='d',s=250,facecolor=mylightblue,edgecolors='k',lw=1,zorder=2,label='Partial rupture events')
    # ax.legend(fontsize=25,framealpha=1,loc='lower right')
    ax.legend(fontsize=20,framealpha=1,loc='lower right')
    xl = ax.get_xlim()
    if len(ver_info) > 0 and not publish:
        ax.text(xl[1]*0.025,Hs[0]*0.975,ver_info,color='k',fontsize=35,fontweight='bold',ha='left',va='bottom')
    ax.set_xlim(0,xl[1])
    ax.set_ylabel('Depth [km]',fontsize=30)
    ax.set_xlabel('Cumulative Slip [m]',fontsize=30)
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()

def three_set(save_dir,prefix,outputs,dep,cumslip_outputs,Vths,dt_coseismic,plot_dd,plot_ab,plot_stress,plot_dc,rths,spup_cumslip_outputs=None,publish=False,save_on=True):
    plt.rcParams['font.size'] = '27'
    fig,ax = plt.subplots(ncols=3, figsize=(29,11), gridspec_kw={'width_ratios': [4, 1, 1]})
    if spup_cumslip_outputs is not None:
        spin_up = True
    else:
        spin_up = False
    if publish:
        decor = 'publish_'
    else:
        decor = ''
    if not spin_up:
        cumslip_basic(ax[0],prefix,cumslip_outputs,rths,publish)
    else:
        cumslip_spinup(ax[0],prefix,cumslip_outputs,spup_cumslip_outputs,rths,publish)
    c = 1; lab = ''
    if plot_ab:
        lab += '_ab'
        asp = with_ab(ax[c],prefix)
        ax[0].hlines(y=asp,xmin=ax[0].get_xlim()[0],xmax=ax[0].get_xlim()[1],color='0.62',lw=1,zorder=0)
        ax[c].hlines(y=asp,xmin=ax[c].get_xlim()[0],xmax=ax[c].get_xlim()[1],color='0.62',lw=1,zorder=0)
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
    if save_on:
        if not spin_up:
            plt.savefig('%s/%scumslip_%d_%d%s.png'%(save_dir,decor,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)
        else:
            plt.savefig('%s/%sspinup_cumslip_%d_%d%s.png'%(save_dir,decor,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)

def two_set(save_dir,prefix,outputs,dep,cumslip_outputs,Vths,dt_coseismic,plot_dd,plot_ab,plot_stress,plot_dc,rths,spup_cumslip_outputs=None,publish=False,save_on=True):
    plt.rcParams['font.size'] = '27'
    fig,ax = plt.subplots(ncols=2, figsize=(24,11), gridspec_kw={'width_ratios': [4, 1]})
    plt.subplots_adjust(wspace=0.05)
    if spup_cumslip_outputs is not None:
        spin_up = True
    else:
        spin_up = False
    if publish:
        decor = 'publish_'
    else:
        decor = ''
    if not spin_up:
        cumslip_basic(ax[0],prefix,cumslip_outputs,rths,publish)
    else:
        cumslip_spinup(ax[0],prefix,cumslip_outputs,spup_cumslip_outputs,rths,publish)
    c = 1; lab = ''
    if plot_ab:
        lab += '_ab'
        asp = with_ab(ax[c],prefix)
        ax[0].hlines(y=asp,xmin=ax[0].get_xlim()[0],xmax=ax[0].get_xlim()[1],color='0.62',lw=1,zorder=0)
        ax[c].hlines(y=asp,xmin=ax[c].get_xlim()[0],xmax=ax[c].get_xlim()[1],color='0.62',lw=1,zorder=0)
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
    if save_on:
        if not spin_up:
            plt.savefig('%s/%scumslip_%d_%d%s.png'%(save_dir,decor,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)
        else:
            plt.savefig('%s/%sspinup_cumslip_%d_%d%s.png'%(save_dir,decor,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)

def only_cumslip(save_dir,prefix,cumslip_outputs,Vths,dt_coseismic,rths,spup_cumslip_outputs=None,publish=False,save_on=True):
    plt.rcParams['font.size'] = '27'
    fig,ax = plt.subplots(figsize=(18,11))
    if spup_cumslip_outputs is not None:
        spin_up = True
    else:
        spin_up = False
    if publish:
        decor = 'publish_'
    else:
        decor = ''
    if not spin_up:
        cumslip_basic(ax,prefix,cumslip_outputs,rths,publish)
    else:
        cumslip_spinup(ax,prefix,cumslip_outputs,spup_cumslip_outputs,rths,publish)
    plt.tight_layout()
    if save_on:
        if not spin_up:
            plt.savefig('%s/%scumslip_%d_%d.png'%(save_dir,decor,int(Vths*100),int(dt_coseismic*10)),dpi=300)
        else:
            plt.savefig('%s/%sspinup_cumslip_%d_%d.png'%(save_dir,decor,int(Vths*100),int(dt_coseismic*10)),dpi=300)

def spup_where(save_dir,prefix,cumslip_outputs,spup_cumslip_outputs,Vths,dt_coseismic,rths,publish=False,save_on=True):
    plt.rcParams['font.size'] = '27'
    fig,ax=plt.subplots(figsize=(19,11))
    cumslip_basic(ax,prefix,cumslip_outputs,rths,publish)
    ax.plot(spup_cumslip_outputs[0][0],spup_cumslip_outputs[0][1],color='yellowgreen',lw=5)
    if publish:
        decor = 'publish_'
    else:
        decor = ''
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/%sspinup_cumslip_%d_%d_where.png'%(save_dir,decor,int(Vths*100),int(dt_coseismic*10)),dpi=300)