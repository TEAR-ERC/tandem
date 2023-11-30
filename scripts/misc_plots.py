'''
Miscellaneous plotting scripts
By Jeena Yun
Last modification: 2023.10.10.
'''
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from event_analyze import *
import change_params
ch = change_params.variate()

mypink = (230/255,128/255,128/255)
myblue = (118/255,177/255,230/255)
myburgundy = (214/255,0,0)
mynavy = (17/255,34/255,133/255)
mylightblue = (218/255,230/255,240/255)
mydarkviolet = (145/255,80/255,180/255)
yr2sec = 365*24*60*60

def plot_STF(save_dir,outputs,dep,cumslip_outputs,spin_up_idx,rths=10,save_on=True):
    print('Rupture length criterion:',rths,'m')
    system_wide,partial_rupture = analyze_events(cumslip_outputs,rths)[2:4]
    t,Fdot = compute_STF(save_dir,outputs,dep,cumslip_outputs)
    plt.rcParams['font.size'] = '15'
    fig,ax = plt.subplots(nrows=2,figsize=(10,10))
    cmap0 = mpl.colormaps['gnuplot']
    ax[0].set_prop_cycle('color',[cmap0(i) for i in np.linspace(0.5,0.9,len(system_wide))]) 
    ax[0].plot(t[system_wide].T,Fdot[system_wide].T/1e12,lw=2,
            label=[r'Event %d'%(i) for i in system_wide])
    ax[0].legend(fontsize=11,loc='upper right',framealpha=0)
    ax[0].set_ylabel('Force rate [$10^{12}$ N/s]',fontsize=17)
    ax[0].grid(True,alpha=0.4)
    cmap1 = mpl.colormaps['ocean']
    ax[1].set_prop_cycle('color',[cmap1(i) for i in np.linspace(0.5,0.9,len(partial_rupture))]) 
    ax[1].plot(t[partial_rupture].T,Fdot[partial_rupture].T/1e12,lw=2,
            label=[r'Event %d'%(i) for i in partial_rupture])
    # if len(partial_rupture) > 0:
    #     ax[1].legend(fontsize=11,loc='upper right',framealpha=0)
    ax[1].set_xlabel('Time [s]',fontsize=17)
    ax[1].set_ylabel('Force rate [$10^{12}$ N/s]',fontsize=17)
    ax[1].grid(True,alpha=0.4)

    plt.tight_layout()
    if save_on:
        plt.savefig('%s/STF.png'%(save_dir),dpi=300)

def plot_event_analyze(save_dir,prefix,cumslip_outputs,rths=10,publish=False,save_on=True):
    print('Rupture length criterion:',rths,'m')
    rupture_length,av_slip,system_wide,partial_rupture = analyze_events(cumslip_outputs,rths)[0:4]
    
    Hs = ch.load_parameter(prefix)[1]
    ver_info = ch.version_info(prefix)

    # ------ Define figure properties
    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(14,11))
    ax0 = plt.subplot2grid(shape=(2,5),loc=(0,0),colspan=3)
    plt.subplots_adjust(hspace=0.1)
    ax1 = plt.subplot2grid(shape=(2,5),loc=(1,0),colspan=3)
    ax2 = plt.subplot2grid(shape=(2,5),loc=(0,3),colspan=2,rowspan=2)
    plt.subplots_adjust(wspace=0.8)

    # cmap = mpl.colormaps('ocean')
    cmap = mpl.colormaps['gnuplot']

    # ------ Ax0: Rupture length
    markers, stemlines, baseline = ax0.stem(np.arange(1,len(rupture_length)+1),rupture_length)
    plt.setp(stemlines, color='0.7', linewidth=2.5)
    plt.setp(markers, color='0.7')
    plt.setp(baseline, color='0.7')
    if len(system_wide) > 0:
        for i in range(len(system_wide)):
            markers, stemlines, baseline = ax0.stem(system_wide[i]+1,rupture_length[system_wide[i]])
            plt.setp(stemlines, color=cmap(np.linspace(0.5,0.9,len(system_wide))[i]), linewidth=2.6)
            plt.setp(markers, color=cmap(np.linspace(0.5,0.9,len(system_wide))[i]))
    ax0.hlines(y=rths,xmax=len(rupture_length)+1,xmin=0,color=mynavy,lw=1.5,linestyles='--')
    ax0.set_xticks(np.arange(-5,len(rupture_length)+5,5), minor=True)
    ax0.set_xlim(1-len(rupture_length)*0.05,len(rupture_length)*1.05)
    ax0.axes.xaxis.set_ticklabels([])
    ax0.set_ylabel('Rupture Length [km]',fontsize=17)
    ax0.grid(True,alpha=0.4,which='both')

    # ------ Ax1: Hypocenter depth
    evdep = cumslip_outputs[1][1]
    markers, stemlines, baseline = ax1.stem(np.arange(1,len(evdep)+1),evdep)
    plt.setp(stemlines, color='0.7', linewidth=2.5)
    plt.setp(markers, color='0.7')
    plt.setp(baseline, color='0.7')
    if len(system_wide) > 0:
        for i in range(len(system_wide)):
            markers, stemlines, baseline = ax1.stem(system_wide[i]+1,evdep[system_wide[i]])
            plt.setp(stemlines, color=cmap(np.linspace(0.5,0.9,len(system_wide))[i]), linewidth=2.6)
            plt.setp(markers, color=cmap(np.linspace(0.5,0.9,len(system_wide))[i]))
    ax1.set_xticks(np.arange(-5,len(evdep)+5,5), minor=True)
    ax1.set_xlim(ax0.get_xlim())
    ax1.invert_yaxis()
    ax1.set_xlabel('Event Index',fontsize=17)
    ax1.set_ylabel('Hypocenter Depth [km]',fontsize=17)
    ax1.grid(True,alpha=0.4,which='both')

    # ------ Ax2: Slip along fault for each event
    fault_z = np.array(cumslip_outputs[3][1]).T[0]
    if len(partial_rupture) > 0:
        ax2.plot(np.array(cumslip_outputs[1][2]).T[partial_rupture[0]].T,fault_z,lw=2.5,color='0.7',label='Partial rupture events')
        if len(partial_rupture) > 1:
            ax2.plot(np.array(cumslip_outputs[1][2]).T[partial_rupture[1:]].T,np.array([fault_z for i in range(len(partial_rupture[1:]))]).T,lw=2.5,color='0.7')
    if len(system_wide) > 0:
        ax2.set_prop_cycle('color',[cmap(i) for i in np.linspace(0.5,0.9,len(system_wide))]) 
        ax2.plot(np.array(cumslip_outputs[1][2]).T[system_wide].T,np.array([fault_z for i in range(len(system_wide))]).T,lw=3,label=[r'Event %d ($\bar{D}$ = %2.2f m)'%(i,av_slip[i]) for i in system_wide])
        hyp_dep = cumslip_outputs[1][1][system_wide]
        hyp_slip = [np.array(cumslip_outputs[1][2]).T[system_wide][i][np.where(fault_z==hyp_dep[i])[0][0]] for i in range(len(system_wide))]
        ax2.scatter(hyp_slip,hyp_dep,marker='*',s=300,facecolor=mydarkviolet,edgecolors='k',lw=1,zorder=3,label='Hypocenter')
    if len(partial_rupture) > 0 or len(system_wide) > 0:
        ax2.legend(fontsize=13)
    xl = ax2.get_xlim()
    ax2.hlines(y=Hs[1],xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')
    ax2.hlines(y=(Hs[1]+Hs[2]),xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')
    if len(Hs) > 3:
        ax2.hlines(y=Hs[3],xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')
        if len(Hs) > 4:
            ax2.hlines(y=(Hs[3]-Hs[4]),xmax=max(xl)*1.2,xmin=min(xl)*1.2,color='0.62',lw=1.5,linestyles='--')            
    ax2.set_xlabel('Slip [m]',fontsize=17)
    ax2.set_ylabel('Depth [km]',fontsize=17)
    ax2.set_xlim(xl)
    ax2.set_ylim(0,Hs[0])
    ax2.invert_yaxis()
    ax2.grid(True,alpha=0.4)

    if len(ver_info) > 0 and not publish:
        plt.suptitle(ver_info,fontsize=25,fontweight='bold')
    plt.tight_layout()
    if save_on:
        if publish:
            plt.savefig('%s/publish_analyze_events.png'%(save_dir),dpi=300)
        else:
            plt.savefig('%s/analyze_events.png'%(save_dir),dpi=300)

def plot_M0(save_dir,cumslip_outputs,spin_up_idx,rths,mode='1d',Mw=False,save_on=True):
    if Mw:
        plot_in_log = False
    else:
        plot_in_log = True
    rupture_length,av_slip,system_wide,partial_rupture,event_cluster,lead_fs = analyze_events(cumslip_outputs,rths)[:6]
    M0_1D = compute_M0(save_dir,rupture_length,av_slip,mode,Mw)
    if spin_up_idx > 0:
        print('Spin-up: shows only after %d'%(spin_up_idx))
        system_wide = system_wide[system_wide>=spin_up_idx]
        partial_rupture = partial_rupture[partial_rupture>=spin_up_idx]
        lead_fs = lead_fs[lead_fs>=spin_up_idx]

    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(10,6))
    if not plot_in_log:
        maxM0 = np.max(M0_1D)
        order = 0
        while maxM0 >= 10:
            maxM0 /= 10
            order += 1
        scale = 10**(order)
        plt.scatter(system_wide,M0_1D[system_wide]/scale,s=121,color=mydarkviolet,edgecolors='k',lw=0.5,marker='*',label='System-wide events',zorder=3)
        if len(lead_fs) > 0:
            plt.scatter(lead_fs,M0_1D[lead_fs]/scale,s=36,color=mydarkviolet,edgecolors='k',lw=0.5,marker='d',label='Leading foreshocks',zorder=3)
        if len(partial_rupture) > 0:
            plt.scatter(partial_rupture,M0_1D[partial_rupture]/scale,s=36,color=mylightblue,edgecolors='k',lw=0.5,marker='d',label='Partial rupture events')
        if mode == '1d':
            ylab = 'Moment per Length [$10^{%d}$N]'%(order)
        elif mode == 'approx2d':
            ylab = 'Moment [$10^{%d}$Nm]'%(order)
        yl = plt.gca().get_ylim()
        plt.ylim(yl[0]-(yl[1]-yl[0])*0.2,yl[1])
    else:
        plt.scatter(system_wide,M0_1D[system_wide],s=121,color=mydarkviolet,edgecolors='k',lw=0.5,marker='*',label='System-wide events',zorder=3)
        if len(lead_fs) > 0:
            plt.scatter(lead_fs,M0_1D[lead_fs],s=36,color=mydarkviolet,edgecolors='k',lw=0.5,marker='d',label='Leading foreshocks',zorder=3)
        if len(partial_rupture) > 0:
            plt.scatter(partial_rupture,M0_1D[partial_rupture],s=36,color=mylightblue,edgecolors='k',lw=0.5,marker='d',label='Partial rupture events')
        if mode == '1d':
            ylab = 'Moment per Length [N]'
        elif mode == 'approx2d':
            ylab = 'Moment [Nm]'
        plt.yscale('log')
        yl = plt.gca().get_ylim()
        plt.ylim(yl[0]*0.2,yl[1])
    if Mw:
        ylab = 'Moment Magnitude (Mw)'
    plt.legend(loc='lower right',fontsize=13)
    plt.xlabel('Event Index',fontsize=17)
    plt.ylabel(ylab,fontsize=17)
    plt.grid(True,alpha=0.5)
    plt.tight_layout()
    if save_on:
        if Mw:
            plt.savefig('%s/Moments_%s_inMw.png'%(save_dir,mode),dpi=300)
        else:
            plt.savefig('%s/Moments_%s.png'%(save_dir,mode),dpi=300)

def plot_GR(save_dir,prefix,cumslip_outputs,spin_up_idx,rths,cutoff_Mw,npts=50,publish=False,save_on=True):
    baseval,N,b,x,yN,a = compute_GR(save_dir,cumslip_outputs,spin_up_idx,rths,cutoff_Mw,int(npts))
    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(8,6))
    ver_info = ch.version_info(prefix)
    plt.plot(x,yN,color=myburgundy,lw=2)
    plt.scatter(baseval,N,s=50,color='k',edgecolors='k',lw=0.5)
    plt.text(min(baseval),min(N),'b = %2.3f'%(-b),ha='left',va='bottom',fontsize=17,fontweight='bold',color=myburgundy)
    if not publish:
        plt.title(ver_info,fontsize=20,fontweight='bold')
    plt.xlabel('Moment Magnitude (Mw)',fontsize=17)
    plt.ylabel('N (events > Mw)',fontsize=17)
    plt.yscale('log')
    if publish:
        decor = 'publish_'
    else:
        decor = ''
    plt.tight_layout()
    if save_on:
        if spin_up_idx == 0:
            plt.savefig('%s/%sGRrelation.png'%(save_dir,decor),dpi=300)
        else:
            plt.savefig('%s/%sGRrelation_spup.png'%(save_dir,decor),dpi=300)

def mainshock_analyze(save_dir,cumslip_outputs,rths,spin_up_idx,save_on=True):
    rupture_length,av_slip,system_wide = analyze_events(cumslip_outputs,rths)[0:3]
    Mw = compute_M0(save_dir,rupture_length,av_slip,mode='approx2d',Mw=True)
    evdep = cumslip_outputs[1][1]

    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(10,6))
    plt.scatter(Mw[system_wide[system_wide>=spin_up_idx]],evdep[system_wide[system_wide>=spin_up_idx]],49,c=np.arange(sum(system_wide>=spin_up_idx)),zorder=3)
    plt.colorbar().set_label('Event Index',fontsize=15,rotation=270,labelpad=20)
    plt.xlabel('Magnitude [Mw]',fontsize=17)
    plt.ylabel('Depth [km]',fontsize=17)
    plt.gca().invert_yaxis()
    plt.grid(True,alpha=0.5)
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/sw_dep_Mw.png'%(save_dir),dpi=300)

def plot_displacements(save_dir,outputs,xyloc,disp_along_opt,save_on=True):
    time = outputs[:,:,0]
    u = outputs[:,:,1]
    xloc = xyloc[:,0]
    plt.rcParams['font.size'] = '15'
    if disp_along_opt == 'x':
        plt.figure(figsize=(8,6))
        for ti in np.linspace(0,time.shape[1],5,dtype=int)[0:-1]:
            plt.plot(xloc,u[:,ti]*1e3,lw=2.5,label='t = %d yrs'%(time[0][ti]/yr2sec))
        plt.plot(xloc,u[:,-1]*1e3,'k',lw=2.5,label='t = %d yrs'%(time[0][-1]/yr2sec))
        plt.legend(fontsize=15)
        plt.xlabel('Distance from Fault [km]',fontsize=17)
    if disp_along_opt == 't':
        plt.figure(figsize=(10,6))
        for i in np.arange(0,len(xloc),4)[0:-1]:
            plt.plot(time[i][np.linspace(0,len(u[i])-1,int(len(u[i])/5),dtype=int)]/yr2sec,u[i][np.linspace(0,len(u[i])-1,int(len(u[i])/5),dtype=int)]*1e3,lw=2.5,label='x = %1.1f km'%(xloc[i]))
        i = len(xloc)-1
        plt.plot(time[i][np.linspace(0,len(u[i])-1,int(len(u[i])/5),dtype=int)]/yr2sec,u[i][np.linspace(0,len(u[i])-1,int(len(u[i])/5),dtype=int)]*1e3,'k',lw=2.5,label='x = %1.1f km'%(xloc[-1]))
        plt.legend(fontsize=15,bbox_to_anchor=(1.42,1.))
        plt.xlabel('Time [yr]',fontsize=17)
    plt.ylabel('Displacement [m]',fontsize=17)
    plt.tight_layout()
    if save_on:
        plt.savefig('%s/disp_along_%s.png'%(save_dir,disp_along_opt),dpi=300)
