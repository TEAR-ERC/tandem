#!/usr/bin/env python3
'''
Functions related to plotting cumulative slip vs. depth plot
By Jeena Yun
Last modification: 2023.05.18.
'''
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as pltcm
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

def version_info(prefix):
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
    if ver_info[:2] == '+ ':
        ver_info = ver_info[2:]
    return ver_info

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

def cumslip_basic(ax,prefix,cumslip_outputs,rths):
    # cumslip_outputs = [timeout, evout, creepout, coseisout, intermout]
    # cumslip_outputs[0] = [tstart_coseis,tend_coseis]
    # cumslip_outputs[1] = [evslip,evdep,fault_slip]
    # cumslip_outputs[2] = [cscreep,depcreep]
    # cumslip_outputs[3] = [cscoseis,depcoseis]
    # cumslip_outputs[4] = [csinterm,depinterm]
    system_wide,partial_rupture = analyze_events(cumslip_outputs,rths)[2:]
    Hs = ch.load_parameter(prefix)[1]
    ver_info = version_info(prefix)

    if len(cumslip_outputs) > 4:
        ax.plot(cumslip_outputs[-1][0],cumslip_outputs[-1][1],color='yellowgreen',lw=1)
    ax.plot(cumslip_outputs[3][0],cumslip_outputs[3][1],color=mydarkpink,lw=1)
    ax.plot(cumslip_outputs[2][0],cumslip_outputs[2][1],color='0.62',lw=1)
    ax.set_ylabel('Depth [km]',fontsize=30)
    ax.set_xlabel('Cumulative Slip [m]',fontsize=30)
    ev_part = ax.scatter(cumslip_outputs[1][0][partial_rupture],cumslip_outputs[1][1][partial_rupture],marker='*',s=700,facecolor=mylightblue,edgecolors='k',lw=1,zorder=3)
    ev_sys = ax.scatter(cumslip_outputs[1][0][system_wide],cumslip_outputs[1][1][system_wide],marker='*',s=700,facecolor=mydarkviolet,edgecolors='k',lw=1,zorder=3)
    ax.legend([ev_sys,ev_part],['System-wide events','Partial rupture events'],fontsize=25,framealpha=1,loc='lower right')
    xl = ax.get_xlim()
    if len(ver_info) > 0:
        ax.text(xl[1]*0.025,Hs[0]*0.975,ver_info,color='k',fontsize=45,fontweight='bold',ha='left',va='bottom')
    ax.set_xlim(0,xl[1])
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()

def cumslip_spinup(ax,prefix,cumslip_outputs,spup_cumslip_outputs,rths):
    # spup_cumslip_outputs = [new_inits, spup_evslip, spup_cscreep, spup_cscoseis, spup_csinterm, spin_up_idx]
    # new_inits = [new_init_Sl,new_init_dp]
    system_wide,partial_rupture = analyze_events(cumslip_outputs,rths)[2:]
    Hs = ch.load_parameter(prefix)[1]
    ver_info = version_info(prefix)

    if len(cumslip_outputs) > 4:
        ax.plot(spup_cumslip_outputs[4],cumslip_outputs[4][1],color='yellowgreen',lw=1)
    ax.plot(spup_cumslip_outputs[3],cumslip_outputs[3][1],color=mydarkpink,lw=1)
    ax.plot(spup_cumslip_outputs[2],cumslip_outputs[2][1],color='0.62',lw=1)
    ev_part = ax.scatter(spup_cumslip_outputs[1][partial_rupture],cumslip_outputs[1][1][partial_rupture],marker='*',s=700,facecolor=mylightblue,edgecolors='k',lw=1,zorder=3)
    ev_sys = ax.scatter(spup_cumslip_outputs[1][system_wide],cumslip_outputs[1][1][system_wide],marker='*',s=700,facecolor=mydarkviolet,edgecolors='k',lw=1,zorder=3)
    ax.legend([ev_sys,ev_part],['System-wide events','Partial rupture events'],fontsize=25,framealpha=1,loc='lower right')
    xl = ax.get_xlim()
    if len(ver_info) > 0:
        ax.text(xl[1]*0.025,Hs[0]*0.975,ver_info,color='k',fontsize=45,fontweight='bold',ha='left',va='bottom')
    ax.set_xlim(0,xl[1])
    ax.set_ylabel('Depth [km]',fontsize=30)
    ax.set_xlabel('Cumulative Slip [m]',fontsize=30)
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()

def three_set(save_dir,prefix,outputs,dep,cumslip_outputs,Vths,dt_coseismic,plot_dd,plot_ab,plot_stress,plot_dc,rths,spup_cumslip_outputs=None,save_on=True):
    plt.rcParams['font.size'] = '27'
    fig,ax = plt.subplots(ncols=3, figsize=(29,11), gridspec_kw={'width_ratios': [4, 1, 1]})
    if spup_cumslip_outputs is not None:
        spin_up = True
    else:
        spin_up = False
    if not spin_up:
        cumslip_basic(ax[0],prefix,cumslip_outputs,rths)
    else:
        cumslip_spinup(ax[0],prefix,cumslip_outputs,spup_cumslip_outputs,rths)
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
            plt.savefig('%s/cumslip_%d_%d%s.png'%(save_dir,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)
        else:
            plt.savefig('%s/spinup_cumslip_%d_%d%s.png'%(save_dir,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)

def two_set(save_dir,prefix,outputs,dep,cumslip_outputs,Vths,dt_coseismic,plot_dd,plot_ab,plot_stress,plot_dc,rths,spup_cumslip_outputs=None,save_on=True):
    plt.rcParams['font.size'] = '27'
    fig,ax = plt.subplots(ncols=2, figsize=(24,11), gridspec_kw={'width_ratios': [4, 1]})
    plt.subplots_adjust(wspace=0.05)
    if spup_cumslip_outputs is not None:
        spin_up = True
    else:
        spin_up = False
    if not spin_up:
        cumslip_basic(ax[0],prefix,cumslip_outputs,rths)
    else:
        cumslip_spinup(ax[0],prefix,cumslip_outputs,spup_cumslip_outputs,rths)
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
            plt.savefig('%s/cumslip_%d_%d%s.png'%(save_dir,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)
        else:
            plt.savefig('%s/spinup_cumslip_%d_%d%s.png'%(save_dir,int(Vths*100),int(dt_coseismic*10),lab),dpi=300)

def only_cumslip(save_dir,prefix,cumslip_outputs,Vths,dt_coseismic,rths,spup_cumslip_outputs=None,save_on=True):
    plt.rcParams['font.size'] = '27'
    fig,ax = plt.subplots(figsize=(18,11))
    if spup_cumslip_outputs is not None:
        spin_up = True
    else:
        spin_up = False
    if not spin_up:
        cumslip_basic(ax,prefix,cumslip_outputs,rths)
    else:
        cumslip_spinup(ax,prefix,cumslip_outputs,spup_cumslip_outputs,rths)
    plt.tight_layout()
    if save_on:
        if not spin_up:
            plt.savefig('%s/cumslip_%d_%d.png'%(save_dir,int(Vths*100),int(dt_coseismic*10)),dpi=300)
        else:
            plt.savefig('%s/spinup_cumslip_%d_%d.png'%(save_dir,int(Vths*100),int(dt_coseismic*10)),dpi=300)

def spup_where(save_dir,prefix,cumslip_outputs,spup_cumslip_outputs,Vths,dt_coseismic,rths,save_on=True):
    system_wide,partial_rupture = analyze_events(cumslip_outputs,rths)[2:]
    Hs = ch.load_parameter(prefix)[1]
    plt.rcParams['font.size'] = '27'
    plt.figure(figsize=(19,11))

    if len(cumslip_outputs) > 4:
        plt.plot(cumslip_outputs[4][0],cumslip_outputs[4][1],color='yellowgreen',lw=1)
    plt.plot(cumslip_outputs[3][0],cumslip_outputs[3][1],color=mydarkpink,lw=1)
    plt.plot(cumslip_outputs[2][0],cumslip_outputs[2][1],color='0.62',lw=1)
    plt.plot(spup_cumslip_outputs[0][0],spup_cumslip_outputs[0][1],color='yellowgreen',lw=5)
    ev_part = plt.scatter(cumslip_outputs[1][0][partial_rupture],cumslip_outputs[1][1][partial_rupture],marker='*',s=700,facecolor=mylightblue,edgecolors='k',lw=1,zorder=3)
    ev_sys = plt.scatter(cumslip_outputs[1][0][system_wide],cumslip_outputs[1][1][system_wide],marker='*',s=700,facecolor=mydarkviolet,edgecolors='k',lw=1,zorder=3)
    plt.legend([ev_sys,ev_part],['System-wide events','Partial rupture events'],fontsize=25,framealpha=1,loc='lower right')
    plt.ylabel('Depth [km]',fontsize=30)
    plt.xlabel('Cumulative Slip [m]',fontsize=30)
    xl = plt.gca().get_xlim()
    plt.xlim(0,xl[1])
    plt.ylim(0,Hs[0])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/spinup_cumslip_%d_%d_where.png'%(save_dir,int(Vths*100),int(dt_coseismic*10)),dpi=300)

def plot_event_analyze(save_dir,prefix,cumslip_outputs,rths,save_on=True):
    print('Rupture length criterion:',rths,'m')
    rupture_length,av_slip,system_wide,partial_rupture = analyze_events(cumslip_outputs,rths)
    Hs = ch.load_parameter(prefix)[1]
    ver_info = version_info(prefix)

    # ------ Define figure properties
    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(14,11))
    ax1 = plt.subplot2grid(shape=(2,5),loc=(0,0),colspan=3)
    plt.subplots_adjust(hspace=0.1)
    ax2 = plt.subplot2grid(shape=(2,5),loc=(1,0),colspan=3)
    ax3 = plt.subplot2grid(shape=(2,5),loc=(0,3),colspan=2,rowspan=2)
    plt.subplots_adjust(wspace=0.8)

    # cmap = pltcm.get_cmap('ocean')
    cmap = pltcm.get_cmap('gnuplot')

    # ------ Rupture length
    markers, stemlines, baseline = ax1.stem(np.arange(1,len(rupture_length)+1),rupture_length)
    plt.setp(stemlines, color='k', linewidth=2.5)
    plt.setp(markers, color='k')
    plt.setp(baseline, color='0.62')
    for i in range(len(system_wide)):
        markers, stemlines, baseline = ax1.stem(system_wide[i]+1,rupture_length[system_wide[i]])
        plt.setp(stemlines, color=cmap(np.linspace(0.5,0.9,len(system_wide))[i]), linewidth=2.6)
        plt.setp(markers, color=cmap(np.linspace(0.5,0.9,len(system_wide))[i]))
    ax1.hlines(y=rths,xmax=len(rupture_length)+1,xmin=0,color=mynavy,lw=1.5,linestyles='--')
    ax1.set_xticks(np.arange(-5,len(rupture_length)+5,5), minor=True)
    ax1.set_xlim(1-len(rupture_length)*0.05,len(rupture_length)*1.05)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.set_ylabel('Rupture Length [km]',fontsize=17)
    ax1.grid(True,alpha=0.4,which='both')

    # ------ Average slip
    # markers, stemlines, baseline = ax2.stem(np.arange(1,len(av_slip)+1),av_slip)
    # plt.setp(stemlines, color='k', linewidth=2.5)
    # plt.setp(markers, color='k')
    # plt.setp(baseline, color='0.62')
    # for i in range(len(system_wide)):
    #     markers, stemlines, baseline = ax2.stem(system_wide[i]+1,av_slip[system_wide[i]])
    #     plt.setp(stemlines, color=cmap(np.linspace(0.5,0.9,len(system_wide))[i]), linewidth=2.6)
    #     plt.setp(markers, color=cmap(np.linspace(0.5,0.9,len(system_wide))[i]))
    # ax2.set_xticks(np.arange(-5,len(av_slip)+5,5), minor=True)
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xlabel('Event Index',fontsize=17)
    # ax2.set_ylabel('Average Slip [m]',fontsize=17)
    # ax2.grid(True,alpha=0.4,which='both')
    evdep = cumslip_outputs[1][1]
    markers, stemlines, baseline = ax2.stem(np.arange(1,len(evdep)+1),evdep)
    plt.setp(stemlines, color='k', linewidth=2.5)
    plt.setp(markers, color='k')
    plt.setp(baseline, color='0.62')
    for i in range(len(system_wide)):
        markers, stemlines, baseline = ax2.stem(system_wide[i]+1,evdep[system_wide[i]])
        plt.setp(stemlines, color=cmap(np.linspace(0.5,0.9,len(system_wide))[i]), linewidth=2.6)
        plt.setp(markers, color=cmap(np.linspace(0.5,0.9,len(system_wide))[i]))
    ax2.set_xticks(np.arange(-5,len(evdep)+5,5), minor=True)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel('Event Index',fontsize=17)
    ax2.set_ylabel('Hypocenter Depth [km]',fontsize=17)
    ax2.grid(True,alpha=0.4,which='both')

    # ------ Slip along fault for each event
    fault_z = np.array(cumslip_outputs[3][1]).T[0]
    ax3.set_prop_cycle('color',[cmap(i) for i in np.linspace(0.5,0.9,len(system_wide))])
    ax3.plot(np.array(cumslip_outputs[1][2]).T[partial_rupture[0]].T,fault_z,lw=2.5,color='0.62',label='Partial rupture events')
    ax3.plot(np.array(cumslip_outputs[1][2]).T[partial_rupture[1:]].T,np.array([fault_z for i in range(len(partial_rupture[1:]))]).T,lw=2.5,color='0.62')
    ax3.plot(np.array(cumslip_outputs[1][2]).T[system_wide].T,np.array([fault_z for i in range(len(system_wide))]).T,lw=3,label=[r'Event %d ($\bar{D}$ = %2.2f m)'%(i+1,av_slip[i]) for i in system_wide])
    hyp_dep = cumslip_outputs[1][1][system_wide]
    hyp_slip = [np.array(cumslip_outputs[1][2]).T[system_wide][i][np.where(fault_z==hyp_dep[i])[0][0]] for i in range(len(system_wide))]
    ax3.scatter(hyp_slip,hyp_dep,marker='*',s=300,facecolor=mydarkviolet,edgecolors='k',lw=1,zorder=3,label='Hypocenter')
    ax3.legend(fontsize=13)
    xl = ax3.get_xlim()
    ax3.hlines(y=Hs[1],xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')
    ax3.hlines(y=(Hs[1]+Hs[2]),xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')
    if len(Hs) > 3:
        ax3.hlines(y=Hs[3],xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')
        if len(Hs) > 4:
            ax3.hlines(y=(Hs[3]-Hs[4]),xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')            
    ax3.set_xlabel('Slip [m]',fontsize=17)
    ax3.set_ylabel('Depth [km]',fontsize=17)
    ax3.set_xlim(xl)
    ax3.set_ylim(0,Hs[0])
    ax3.invert_yaxis()
    ax3.grid(True,alpha=0.4)

    if len(ver_info) > 0:
        plt.suptitle(ver_info,fontsize=25,fontweight='bold')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/analyze_events.png'%(save_dir),dpi=300)