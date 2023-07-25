#!/usr/bin/env python3
'''
Functions related to plotting Dc profile
By Jeena Yun
Last modification: 2023.05.16.
'''
import numpy as np
import matplotlib.pylab as plt
import change_params

ch = change_params.variate()

mypink = (230/255,128/255,128/255)
myblue = (118/255,177/255,230/255)
myburgundy = (214/255,0,0)
mynavy = (17/255,34/255,133/255)

# ------------------ Retrieved
def read_output(fname):
    fid = open(fname,'r')
    lines = fid.readlines()
    mesh_y = []
    Dc = []
    c = 0
    for line in lines:
        try:
            _y, _dc = line.split('\t')
        except:
            print('skip line %d:'%(c),line.strip())
            continue
        if float(_dc) > 1:
            print('skip line %d:'%(c),line.strip())
            continue
        try:
            mesh_y.append(float(_y.strip())); Dc.append(float(_dc.strip()))
        except:
            print('skip line %d:'%(c),line.strip())
            continue
        c += 1
    print('Total %d points'%c)
    fid.close()
    idx = np.argsort(np.array(mesh_y))
    mesh_y = np.array(mesh_y)[idx]; Dc = np.array(Dc)[idx]
    return mesh_y,Dc

def plot_Dc_vs_depth(save_dir,prefix,save_on=True):
    y,Hs,a,b,a_b,tau0,sigma0,_Dc,others = ch.load_parameter(prefix)
    # ---------- Inputs
    if len(_Dc) == 2:
        Dc0 = _Dc[0]
        y_in = _Dc[1]
    else:
        Dc0 = _Dc
        y_in = y
    if len(prefix.split('/')) > 1:
        fname = '%s/%s/dc_profile_%s'%(ch.get_setup_dir(),prefix.split('/')[0],prefix.split('/')[-1])
    else:
        fname = '%s/%s/dc_profile'%(ch.get_setup_dir(),prefix)
    # ---------- Outputs
    y_ret,Dc = read_output(fname)
    
    plt.rcParams['font.size'] = '15'
    fig,ax = plt.subplots(figsize=(9,7))
    # ---------- Inputs
    ax.plot(Dc0,abs(y_in),color='k',lw=3,label='Input',zorder=3)
    # ---------- Outputs
    ax.scatter(Dc,abs(y_ret),25,color=myburgundy,label='Output',zorder=3)
    ax.set_xlabel('Dc [m]',fontsize=17)
    ax.set_ylabel('Depth [km]',fontsize=17)
    xl = ax.get_xlim()
    ax.set_xlim(0,xl[1])
    ax.set_ylim(0,Hs[0])
    ax.hlines(y=Hs[1],xmin=0,xmax=xl[1],lw=2.5,linestyles='--',color='0.62')
    ax.hlines(y=Hs[3],xmin=0,xmax=xl[1],lw=2.5,linestyles='--',color='0.62')
    ax.hlines(y=Hs[1]+Hs[2],xmin=0,xmax=xl[1],lw=2.5,linestyles='--',color='0.62')
    ax.invert_yaxis()
    plt.grid(True)
    ax.legend(fontsize=13,loc='lower left')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/Dc_profile.png'%(save_dir))

# ------------------ With cumslip plot
def Dc_with_cumslip(ax,prefix,fs_label=30,fs_legend=15,ytick_on=False):
    y,Hs,a,b,a_b,tau0,sigma0,_Dc,others = ch.load_parameter(prefix)
    # ---------- Inputs
    if len(_Dc) == 2:
        Dc0 = _Dc[0]
        y_in = _Dc[1]
    else:
        Dc0 = _Dc
        y_in = y
    if len(prefix.split('/')) > 1:
        fname = '%s/%s/dc_profile_%s'%(ch.get_setup_dir(),prefix.split('/')[0],prefix.split('/')[-1])
    else:
        fname = '%s/%s/dc_profile'%(ch.get_setup_dir(),prefix)
    # ---------- Outputs
    y_ret,Dc = read_output(fname)

    # ---------- Inputs
    ax.plot(Dc0,abs(y_in),color='k',lw=3,label='Input',zorder=3)
    # ---------- Outputs
    ax.scatter(Dc,abs(y_ret),lw=2.5,color=myburgundy,label='Output',zorder=3)
    if not ytick_on:
        ax.axes.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel('Depth [km]',fontsize=fs_label)
    xl = ax.get_xlim()
    ax.hlines(y=Hs[1],xmin=0,xmax=xl[1],lw=2.5,linestyles='--',color='0.62')
    ax.hlines(y=Hs[3],xmin=0,xmax=xl[1],lw=2.5,linestyles='--',color='0.62')
    ax.hlines(y=Hs[1]+Hs[2],xmin=0,xmax=xl[1],lw=2.5,linestyles='--',color='0.62') 
    ax.set_xlabel('Dc [m]',fontsize=fs_label)
    ax.set_xlim(0,xl[1])
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()
    ax.legend(fontsize=fs_legend,loc='lower left')
