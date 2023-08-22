# -------------------------------------------- Plotting functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from cumslip_compute import *
from fcomputeG import *
import myplots
mp = myplots.Figpref()

def slip_vs_stress_test(ax,x,y,txt,col='k',lw=2.5,yl=None):
    ax.plot(x,y,color=col,lw=lw)
    ax.text(np.max(x),np.max(y),txt,fontsize=15,fontweight='bold',ha='right',va='top')
    ax.set_xlabel('Slip [m]',fontsize=17)
    ax.set_ylabel('Shear Stress [MPa]',fontsize=17)
    if yl is not None:
        ax.set_ylim(yl)
    ax.grid(True,alpha=0.5)

def slip_vs_stress(ax,x,y,txt):
    xb,yb,xr,yr,err_note = get_xy(x,y)
    ax.fill_between(xb,yb,np.min(yb),color=mp.mylavender,alpha=0.3)
    ax.fill_between(xr,yr,np.min(yr),color=mp.myblue,alpha=0.3)
    ax.plot(x,y,'k',lw=2.5)
    ax.text(np.max(x),np.max(y),txt,fontsize=15,fontweight='bold',ha='right',va='top')
    ax.set_xlabel('Slip [m]',fontsize=17)
    ax.set_ylabel('Shear Stress [MPa]',fontsize=17)
    ax.grid(True,alpha=0.5)

def show_zoom(ax,dep,time,tstart,tend,var,idep,iev,add,varname):
    its,ite = np.argmin(abs(time[idep]-tstart[iev])), np.argmin(abs(time[idep]-tend[iev]))
    rect=plt.Rectangle(xy=(time[idep][its],np.min(var[idep][its-add:ite+add])),width=time[idep][ite]-time[idep][its],height=np.max(var[idep][its-add:ite+add])-np.min(var[idep][its-add:ite+add]),facecolor='lemonchiffon',linewidth=2,edgecolor='goldenrod')
    ax.add_patch(rect)
    ax.plot(time[idep][its-add:ite+add],var[idep][its-add:ite+add],'k',lw=2)
    ax.text(np.min(time[idep][its-add:ite+add]),np.max(var[idep][its-add:ite+add]),'Depth: %2.1f km\nEvent #: %d'%(np.sort(abs(dep))[idep],iev),fontsize=15,fontweight='bold',ha='left',va='top')
    ax.set_xlabel('Time [s]',fontsize=17)
    if varname == 'sr':
        ax.hlines(y=-2,xmin=time[idep][its-add],xmax=time[idep][ite+add],color=mp.myburgundy,linestyles='--',lw=2)
        ax.set_ylabel(r'$\log_{10}$(Slip Rate [m/s])',fontsize=17)
    elif varname == 'cumslip':
        ax.set_ylabel('Cumulative Slip [m]',fontsize=17)
    elif varname == 'shearT':
        ax.set_ylabel('Shear Stress [MPa]',fontsize=17)
    ax.grid(True,alpha=0.5)
    # ax.scatter(time[idep][its-add:ite+add],var[idep][its-add:ite+add],s=9,color='k')

def G_prof_single(ax,G,dep,iev,Hs):
    ax.plot(G,np.sort(abs(dep)),'k',lw=3)
    ax.set_xscale('log')
    ax.set_ylim(0,Hs[0])
    ax.set_title('Event # %d'%(iev),fontsize=20,fontweight='bold')
    ax.set_xlabel('G [J/m$^2$]',fontsize=17)
    ax.set_ylabel('Depth [km]',fontsize=17)
    ax.grid(True,alpha=0.5)
    ax.invert_yaxis()

def G_prof_all(ax,G,cumslip_outputs,avD,ev_idx,ev_type,Hs,var_mode):
    if ev_type == 'sw':
        # system-wide
        cmap = mpl.colormaps['gnuplot']
    elif ev_type == 'pr':
        # partial rupture
        cmap = mpl.colormaps['ocean']
    else:
        NameError('Wrong value for event type - choose either ''sw'' or ''pr''')
    fault_z = np.array(cumslip_outputs[3][1]).T[0]
    if len(ev_idx) > 0:
        ax.set_prop_cycle('color',[cmap(i) for i in np.linspace(0.5,0.9,len(ev_idx))]) 
        ax.plot(G[ev_idx].T,np.array([fault_z for i in range(len(ev_idx))]).T,lw=3,label=[r'Event %d ($\bar{D}$ = %2.2f m)'%(i,avD[i]) for i in ev_idx])
        hyp_dep = cumslip_outputs[1][1][ev_idx]
        hyp_slip = [G[ev_idx][i][np.where(fault_z==hyp_dep[i])[0][0]] for i in range(len(ev_idx))]
        # ax.scatter(hyp_slip,hyp_dep,marker='*',s=300,facecolor=mp.mydarkviolet,edgecolors='k',lw=1,zorder=3,label='Hypocenter')
        ax.scatter(hyp_slip,hyp_dep,300,marker='*',c=[cmap(i) for i in np.linspace(0.5,0.9,len(ev_idx))],edgecolors='k',lw=1.5,zorder=3,label='Hypocenter')
        ax.legend(fontsize=13,loc='lower right')
    else:
        IndexError('No event for chosen type')
    ax.set_xscale('log')
    ax.set_ylim(0,Hs[0])
    if var_mode == 'Wb':
        ax.set_xlabel('Wb [J/m$^2$]',fontsize=17)
    elif var_mode == 'Wr':
        ax.set_xlabel('Wr [J/m$^2$]',fontsize=17)
    ax.set_ylabel('Depth [km]',fontsize=17)
    ax.grid(True,alpha=0.5)
    ax.invert_yaxis()

def scaling(ax,D,G,col,lab=None,s=4,marker='o',ec=None,lw=1.5,zord=1,alpha=None,xl=[1e-9,1e2],yl=[1e-6,1e9],av=True):
    if len(col) == len(D):
        cmap = mpl.colormaps['magma_r']
        col = cmap(col.flatten())
    if ec is not None:
        if len(ec) == len(D):
            cmap = mpl.colormaps['magma_r']
            ec = cmap(ec.flatten())
        psc = ax.scatter(D,G,s=s,marker=marker,fc=col,label=lab,zorder=zord,ec=ec,lw=lw,alpha=alpha)
    else:
        psc = ax.scatter(D,G,s=s,marker=marker,fc=col,label=lab,zorder=zord,alpha=alpha)
    if lab is not None:
        ax.legend(fontsize=13,loc='lower right')
    if av:
        ax.set_xlabel('Average Slip [m]',fontsize=17)
        ax.set_ylabel('Average Fracture Energy Density [J/m$^2$]',fontsize=17)
    else:
        ax.set_xlabel('Slip [m]',fontsize=17)
        ax.set_ylabel('Fracture Energy Density [J/m$^2$]',fontsize=17)
    if xl != 'auto':
        ax.set_xlim(xl)
    if yl != 'auto':
        ax.set_ylim(yl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True,alpha=0.5)
    return psc

def plot_Coccos(ax,follow_color=False,legend_on=True):
    cocco_outputs = np.load('/Users/j4yun/Dropbox/Coursework/2023-S_Advanced_Seismology_2/Final_project/outputs/Cocco/outputs.npy',allow_pickle=True)
    Gcc = cocco_outputs.item().get('G')
    Dcc = cocco_outputs.item().get('D')
    Lcc = cocco_outputs.item().get('L')
    labels = cocco_outputs.item().get('labels')

    if not follow_color:
        cmap = mpl.colormaps['YlGn']
        spacing = np.linspace(0.7,0.3,len(Lcc))
        colset = cmap(spacing)
    else:
        colset = ['k',(128/255,205/255,230/255),'tomato','goldenrod']

    c = 0
    pts = []
    for i in range(len(Lcc)):
        if i==0:
            if legend_on:
                sc = scaling(ax,Dcc[c:c+Lcc[i]],Gcc[c:c+Lcc[i]],colset[i],lab=labels[i],zord=3,alpha=0.25,yl='auto')
            else:
                sc = scaling(ax,Dcc[c:c+Lcc[i]],Gcc[c:c+Lcc[i]],colset[i],zord=3,alpha=0.25,yl='auto')
        else:
            if legend_on:
                sc = scaling(ax,Dcc[c:c+Lcc[i]],Gcc[c:c+Lcc[i]],colset[i],lab=labels[i],zord=2,alpha=0.25,yl='auto')
            else:
                sc = scaling(ax,Dcc[c:c+Lcc[i]],Gcc[c:c+Lcc[i]],colset[i],zord=2,alpha=0.25,yl='auto')
        c += Lcc[i]
        pts.append(sc)
    return pts,labels

def avG_event(ax,var,tstart,system_wide,partial_rupture,type,spin_up_idx,c1=mp.myburgundy,c2=mp.mynavy):
    yl = [4,11.2]
    # ax.add_patch(Rectangle((spin_up_idx+0.5,0),len(tstart)-spin_up_idx-1,yl[1],fc='0.8',alpha=0.5))
    ax.add_patch(Rectangle((-0.5,0),spin_up_idx+1,yl[1],fc='0.8',alpha=0.5))
    markers, stemlines, baseline = ax.stem(system_wide,np.log10(var[system_wide]), label='System-wide')
    plt.setp(stemlines, color=c1, linewidth=2.5)
    plt.setp(markers, color=c1, marker='s')

    if len(partial_rupture) > 0:
        markers, stemlines, baseline = ax.stem(partial_rupture,np.log10(var[partial_rupture]), label='Partial rupture')
        plt.setp(stemlines, color=c2, linewidth=2.5)
        plt.setp(markers, color=c2, marker='d', ms=9)

    ax.legend(fontsize=11.5,loc='upper left')
    ax.set_xlabel('Event index',fontsize=17)
    if type == 'G':
        ax.set_ylabel('$\log$(G [J/m$^2$])',fontsize=17)
    elif type == 'D':
        ax.set_ylabel('$\log$(slip [m])',fontsize=17)
    ax.set_xticks(np.arange(0,len(tstart),1))
    ax.set_xlim(-len(tstart)*0.05,(len(tstart)-1)*1.05)
    ax.set_ylim(yl)

def scatter_xy(ax,x,y,col,lab=None,s=4,marker='o',ec=None,lw=1.5,zord=1,alpha=None,xl='auto',yl='auto',labx='Wr [J/m$^2$]',laby='Wb [J/m$^2$]'):
    psc = ax.scatter(x,y,s=s,marker=marker,fc=col,label=lab,zorder=zord,ec=ec,lw=lw,alpha=alpha)
    ax.set_xlabel(labx,fontsize=17)
    ax.set_ylabel(laby,fontsize=17)
    if xl != 'auto':
        ax.set_xlim(xl)
    if yl != 'auto':
        ax.set_ylim(yl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True,alpha=0.5)
    return psc