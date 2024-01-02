#!/usr/bin/env python3
import numpy as np
from cumslip_compute import *
import matplotlib.pylab as plt
import os
import glob
import change_params
import myplots

mp = myplots.Figpref()
ch = change_params.variate()

want_spinup = True
if want_spinup:
    print('spin up')
else:
    print('no spin up')

prefix = 'perturb_stress/reference'

# ----------
if 'j4yun/' in os.getcwd(): # local
    print('local')
    save_dir = 'models/'+prefix
elif 'di75weg/' in os.getcwd(): # supermuc
    print('supermuc')
    save_dir = '/hppfs/scratch/06/di75weg/'+prefix
elif 'jyun/' in os.getcwd(): # LMU server
    print('LMU server')
    save_dir = '/export/dump/jyun/'+prefix


print('Load saved data: %s/outputs_depthinfo'%(save_dir))
dep = np.load('%s/outputs_depthinfo.npy'%(save_dir))
print('Load saved data: %s/const_params.npy'%(save_dir))
params = np.load('%s/const_params.npy'%(save_dir),allow_pickle=True)

y,Hs,a,b,a_b,tau0,sigma0,Dc,others = ch.load_parameter(prefix)

# ----------
Vths = 2e-1
Vlb = 0
intv = 0.15
dt_interm = 0
cuttime = 0
rths = 10
dt_creep = 2*ch.yr2sec
dt_coseismic = 0.5

def save_csl(ii,save_dir,prefix,cumslip_outputs,spup_cumslip_outputs,cumslip_outputs_events,spup_cumslip_outputs_events,system_wide,partial_rupture,lead_fs,Vths,dt_coseismic,save_on=True):
    Hs = ch.load_parameter(prefix)[1]
    if spup_cumslip_outputs is None:
        contour_x,event_x = [],[]
        contour_x.append(cumslip_outputs[0][0])
        event_x.append(cumslip_outputs_events[0][0])
        contour_x.append(cumslip_outputs[1][0])
        event_x.append(cumslip_outputs_events[1][0])
        contour_x.append(cumslip_outputs[2][0])
        event_x.append(cumslip_outputs_events[2][0])
        contour_x.append(cumslip_outputs[3][0])
        event_x.append(cumslip_outputs_events[3][0])
        if len(cumslip_outputs) > 4:
            contour_x.append(cumslip_outputs[4][0])
            event_x.append(cumslip_outputs_events[4][0])
    else:
        contour_x = spup_cumslip_outputs    
        event_x = spup_cumslip_outputs_events

    plt.rcParams['font.size'] = '27'
    fig,ax = plt.subplots(figsize=(18,11))

    if len(cumslip_outputs) > 4:
        ax.plot(contour_x[-1],cumslip_outputs[4][1],color='yellowgreen',lw=1)
    ax.plot(contour_x[3],cumslip_outputs[3][1],color=mp.mydarkpink,lw=1)
    ax.plot(contour_x[2],cumslip_outputs[2][1],color='0.62',lw=1)
    if len(system_wide) > 0:
        ax.scatter(event_x[1][system_wide],cumslip_outputs_events[1][1][system_wide],marker='*',s=700,facecolor=mp.mydarkviolet,edgecolors='k',lw=1,zorder=3,label='System-size events')
    if len(lead_fs) > 0:
        ax.scatter(event_x[1][lead_fs],cumslip_outputs_events[1][1][lead_fs],marker='d',s=250,facecolor=mp.mydarkviolet,edgecolors='k',lw=1,zorder=3,label='Leading foreshocks')
    if len(partial_rupture) > 0:
        ax.scatter(event_x[1][partial_rupture],cumslip_outputs_events[1][1][partial_rupture],marker='d',s=250,facecolor=mp.mylightblue,edgecolors='k',lw=1,zorder=2,label='Partial rupture events')
    ax.legend(fontsize=20,framealpha=1,loc='lower right')
    xl = ax.get_xlim()
    ax.set_xlim(0,xl[1])
    ax.set_ylabel('Depth [km]',fontsize=30)
    ax.set_xlabel('Cumulative Slip [m]',fontsize=30)
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()
    plt.tight_layout()
    if save_on:
        if spup_cumslip_outputs is None:
            plt.savefig('%s/adjusted_cumslip_%d_%d_short_%d.png'%(save_dir,int(Vths*100),int(dt_coseismic*10),ii),dpi=300)
        else:
            plt.savefig('%s/adjusted_spinup_cumslip_%d_%d_short_%d.png'%(save_dir,int(Vths*100),int(dt_coseismic*10),ii),dpi=300)

# fnames = glob.glob('%s/short_outputs_*.npy'%(save_dir))
fnames = ['%s/short_outputs_0.npy'%(save_dir)]
for fn in fnames:
    print(fn)
    outputs = np.load(fn)
    ii = int(fn.split('.npy')[0].split('_')[-1])
    # if ii == 0:
    #     print('Skip ii = 0')
    #     continue
    cumslip_outputs_events = compute_cumslip(outputs,dep,cuttime,Vlb,Vths,dt_creep,dt_coseismic,dt_interm,intv)
    if want_spinup:
        spup_cumslip_outputs_events = compute_spinup(outputs,dep,cuttime,cumslip_outputs_events,['yrs',200],rths)
    else:
        spup_cumslip_outputs_events = None
    system_wide,partial_rupture,event_cluster,lead_fs,major_pr,minor_pr = analyze_events(cumslip_outputs_events,rths)[2:]

    cumslip_outputs = compute_cumslip(outputs,dep,cuttime,Vlb,1e-2,dt_creep,dt_coseismic,dt_interm,intv)
    if want_spinup:
        spup_cumslip_outputs = compute_spinup(outputs,dep,cuttime,cumslip_outputs,['yrs',200],rths)
    else:
        spup_cumslip_outputs = None
    save_csl(ii,save_dir,prefix,cumslip_outputs,spup_cumslip_outputs,cumslip_outputs_events,spup_cumslip_outputs_events,system_wide,partial_rupture,lead_fs,Vths,dt_coseismic,save_on=True)
