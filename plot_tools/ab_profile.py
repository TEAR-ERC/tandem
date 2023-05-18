#!/usr/bin/env python3
'''
Functions related to plotting a-b profile
By Jeena Yun
Last modification: 2023.05.18.
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
    a = []
    b = []
    c = 0
    for line in lines:
        try:
            _y, _a, _b = line.split('\t')
        except:
            print('skip line %d:'%(c),line.strip())
            continue
        try:
            mesh_y.append(float(_y.strip())); a.append(float(_a.strip())); b.append(float(_b.strip()))
        except:
            print('skip line %d:'%(c),line.strip())
            continue
        c += 1
    print('Total %d points'%c)
    fid.close()
    idx = np.argsort(np.array(mesh_y))
    mesh_y = np.array(mesh_y)[idx]; a = np.array(a)[idx]; b = np.array(b)[idx]
    return mesh_y,a,b        

# ------------------ Initial profile check
def plot_ab_vs_depth(save_dir,prefix,save_on=True):
    y,Hs,a0,b0,_a_b = ch.load_parameter(prefix)[0:5]
    if len(_a_b) == 2:
        y_in = _a_b[1]
    else:
        y_in = y
    if len(prefix.split('/')) > 1:
        fname = '%s/%s/ab_profile_%s'%(ch.setup_dir,prefix.split('/')[0],prefix.split('/')[-1])
    else:
        fname = '%s/%s/ab_profile'%(ch.setup_dir,prefix)
    y_ret,a,b = read_output(fname)
    
    plt.rcParams['font.size'] = '15'
    fig,ax = plt.subplots(figsize=(9,7))
    # ---------- Inputs
    # ax.plot(a0,y_in,color='k',lw=3,label='a',zorder=3)
    ax.plot(b0,abs(y_in),color=myblue,lw=3,label='b (Input)',zorder=3)
    ax.plot(a0-b0,abs(y_in),color=mypink,lw=3,label='a-b (Input)',zorder=3)
    # ---------- Outputs
    ax.scatter(b,abs(y_ret),25,color=mynavy,label='b (Output)',zorder=3)
    ax.scatter(a-b,abs(y_ret),25,color=myburgundy,label='a-b (Output)',zorder=3)
    ax.set_xlabel('Friction Paramters',fontsize=17)
    ax.set_ylabel('Depth [km]',fontsize=17)
    xl = ax.get_xlim()
    ax.set_xlim(xl[0],0.025)
    ax.set_ylim(0,Hs[0])
    ax.hlines(y=Hs[3],xmin=xl[0],xmax=0.025,lw=1.5,linestyles='--',color='0.62')
    ax.hlines(y=Hs[1],xmin=xl[0],xmax=0.025,lw=1.5,linestyles='--',color='0.62')
    ax.hlines(y=(Hs[1]+Hs[2]),xmin=xl[0],xmax=0.025,lw=1.5,linestyles='--',color='0.62')
    ax.invert_yaxis()
    plt.grid(True)
    ax.legend(fontsize=13,loc='lower left')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/ab_profile.png'%(save_dir))

# ------------------ With cumslip plot
def ab_with_cumslip(ax,prefix,fs_label=30,fs_legend=15,ytick_on=False):
    y,Hs,a0,b0,_a_b = ch.load_parameter(prefix)[0:5]
    if len(_a_b) == 2:
        y_in = _a_b[1]
    else:
        y_in = y
    if len(prefix.split('/')) > 1:
        fname = '%s/%s/ab_profile_%s'%(ch.setup_dir,prefix.split('/')[0],prefix.split('/')[-1])
    else:
        fname = '%s/%s/ab_profile'%(ch.setup_dir,prefix)
    y_ret,a,b = read_output(fname)
    asp = -_a_b[1][np.logical_and(_a_b[0]>0,np.logical_and(abs(_a_b[1])>Hs[3],abs(_a_b[1])<Hs[1]))]

    # ---------- Inputs
    ax.plot(b0,abs(y_in),color=myblue,lw=3,label='b (Input)',zorder=3)
    ax.plot(a0-b0,abs(y_in),color=mypink,lw=3,label='a-b (Input)',zorder=3)
    # ---------- Outputs
    ax.scatter(b,abs(y_ret),lw=2.5,color=mynavy,label='b (Output)',zorder=3)
    ax.scatter(a-b,abs(y_ret),lw=2.5,color=myburgundy,label='a-b (Output)',zorder=3)
    if not ytick_on:
        ax.axes.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel('Depth [km]',fontsize=fs_label)
    xl = ax.get_xlim()
    ax.hlines(y=Hs[3],xmin=xl[0],xmax=0.025,lw=1.5,linestyles='--',color='0.62')
    ax.hlines(y=Hs[1],xmin=xl[0],xmax=0.025,lw=1.5,linestyles='--',color='0.62')
    ax.hlines(y=(Hs[1]+Hs[2]),xmin=xl[0],xmax=0.025,lw=1.5,linestyles='--',color='0.62')
    ax.vlines(x=0,ymin=0,ymax=Hs[0],lw=1.5,linestyles='--',color='0.62')
    ax.set_xlabel('Friction Paramters',fontsize=fs_label)
    ax.set_xlim(xl[0],0.025)
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()
    ax.legend(fontsize=fs_legend,loc='lower left')
    return asp
