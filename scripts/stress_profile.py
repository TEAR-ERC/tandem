#!/usr/bin/env python3
'''
Functions related to plotting initial stress conditions
By Jeena Yun
Last modification: 2023.05.22.
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
def read_output(outputs,dep):
    z = np.zeros(len(dep))
    tau = np.zeros(len(dep))
    sigma = np.zeros(len(dep))

    c = 0
    for i in np.argsort(abs(dep)):
        z[c] = abs(dep[i])
        tau[c] = np.array(outputs[i])[0,3]
        sigma[c] = np.array(outputs[i])[0,5]
        c += 1
    return z,tau,sigma

# ------------------ Initial stress check
def plot_stress_vs_depth(save_dir,prefix,outputs,dep,save_on=True):
    plt.rcParams['font.size'] = '15'
    fig,ax = plt.subplots(figsize=(9,7))
    y,Hs,a,b,_a_b,tau0,_sigma0,L,others = ch.load_parameter(prefix)
    if len(_sigma0) == 2:
        sigma0 = _sigma0[0]
        y_in = _sigma0[1]
        tau0 = ch.same_length(y,tau0,y_in)
    else:
        sigma0 = _sigma0
        y_in = y
    y_ret,tau,sigma = read_output(outputs,dep)

    ax.plot(-tau0,abs(y_in),lw=2.5,color=mypink,label='Shear Stress')
    ax.plot(sigma0,abs(y_in),lw=2.5,color=myblue,label='Normal Stress')
    ax.scatter(-tau,abs(y_ret),lw=2.5,color=myburgundy,label='Shear Stress (retrieved)',zorder=3)
    ax.scatter(sigma,abs(y_ret),lw=2.5,color=mynavy,label='Normal Stress (retrieved)',zorder=3)
    ax.set_xlabel('Stress [MPa]',fontsize=17)
    ax.set_ylabel('Depth [km]',fontsize=17)
    ax.set_xlim(-2,60)
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()
    plt.grid(True)
    ax.legend(fontsize=13,loc='lower left')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/stress_profile.png'%(save_dir))

# ------------------ With cumslip plot
def stress_with_cumslip(ax,prefix,outputs,dep,fs_label=30,fs_legend=15,ytick_on=False):
    y,Hs,a,b,_a_b,tau0,_sigma0,L,others = ch.load_parameter(prefix)
    if len(_sigma0) == 2:
        sigma0 = _sigma0[0]
        y_in = _sigma0[1]
        tau0 = ch.same_length(y,tau0,y_in)
    else:
        sigma0 = _sigma0
        y_in = y
    y_ret,tau,sigma = read_output(outputs,dep)

    ax.plot(-tau0,abs(y_in),lw=3,color=mypink,label='Shear Stress')
    ax.plot(sigma0,abs(y_in),lw=3,color=myblue,label='Normal Stress')
    ax.scatter(-tau,abs(y_ret),lw=2.5,color=myburgundy,label='Shear Stress (retrieved)',zorder=3)
    ax.scatter(sigma,abs(y_ret),lw=2.5,color=mynavy,label='Normal Stress (retrieved)',zorder=3)
    if not ytick_on:
        ax.axes.yaxis.set_ticklabels([])
    else:
        ax.set_ylabel('Depth [km]',fontsize=fs_label)
    xl = ax.get_xlim()
    ax.hlines(y=Hs[1],xmax=max(xl)*1.2,xmin=0,color='0.62',lw=1.5,linestyles='--')
    ax.hlines(y=(Hs[1]+Hs[2]),xmax=max(xl)*1.2,xmin=0,color='0.62',lw=1.5,linestyles='--')
    if len(Hs) > 3:
        ax.hlines(y=Hs[3],xmax=max(xl)*1.2,xmin=0,color='0.62',lw=1.5,linestyles='--')
    ax.set_xlabel('Stress [MPa]',fontsize=fs_label)
    ax.set_xlim([0,xl[1]])
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()
    ax.legend(fontsize=fs_legend,loc='lower left')

# ------------------ Histogram
def plot_hist(save_dir,outputs,dep,save_on=True):
    y_ret,tau,sigma = read_output(outputs,dep)
    plt.rcParams['font.size'] = '15'
    fig,ax=plt.subplots(ncols=2,figsize=(18,7))
    ax[0].hist(sigma,color=myblue,edgecolor='k',lw=1)
    ax[0].set_xlabel('Stress [MPa]',fontsize=17)
    ax[0].set_ylabel('Count',fontsize=17)
    ax[0].set_title('All depth',fontsize=20,fontweight='bold')

    ax[1].hist(sigma[abs(y_ret)>2],color=myblue,edgecolor='k',lw=1)
    ax[1].set_xlabel('Stress [MPa]',fontsize=17)
    ax[1].set_title('Depth > 2 km',fontsize=20,fontweight='bold')

    plt.tight_layout()
    if save_on:
        plt.savefig('%s/sigma_hist.png'%(save_dir))

# ------------------ Only input profile check
def plot_stress_init(save_dir,prefix,save_on=True):
    plt.rcParams['font.size'] = '15'
    fig,ax = plt.subplots(figsize=(9,7))
    y,Hs,a,b,_a_b,tau0,_sigma0,L,others = ch.load_parameter(prefix)
    if len(_sigma0) == 2:
        sigma0 = _sigma0[0]
        y_in = _sigma0[1]
        tau0 = ch.same_length(y,tau0,y_in)
    else:
        sigma0 = _sigma0
        y_in = y

    ax.plot(-tau0,abs(y_in),lw=2.5,color=mypink,label='Shear Stress')
    ax.plot(sigma0,abs(y_in),lw=2.5,color=myblue,label='Normal Stress')
    ax.set_xlabel('Stress [MPa]',fontsize=17)
    ax.set_ylabel('Depth [km]',fontsize=17)
    # ax.set_xlim(-2,60)
    ax.set_ylim(0,Hs[0])
    ax.invert_yaxis()
    plt.grid(True)
    ax.legend(fontsize=13,loc='lower left')
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/stress_profile.png'%(save_dir))