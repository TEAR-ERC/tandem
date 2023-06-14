'''
Miscellaneous plotting scripts
By Jeena Yun
Last modification: 2023.06.13.
'''
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import change_params
ch = change_params.variate()

mypink = (230/255,128/255,128/255)
myblue = (118/255,177/255,230/255)
myburgundy = (214/255,0,0)
mynavy = (17/255,34/255,133/255)
mydarkviolet = (145/255,80/255,180/255)

def compute_STF(save_dir,outputs,dep):
    from scipy import integrate
    from scipy import interpolate
    from cumslip_compute import event_times
    tstart, tend, evdep = event_times(dep,outputs,print_on=False)
    params = np.load('%s/const_params.npy'%(save_dir),allow_pickle=True)
    time = np.array([outputs[i][:,0] for i in np.argsort(abs(dep))])
    sr = abs(np.array([outputs[i][:,4] for i in np.argsort(abs(dep))]))
    z = np.sort(abs(dep))*1e3
    mu = params.item().get('mu')*1e9

    npoints = 500
    f = np.array([mu * integrate.simpson(sr[:,t],z) for t in range(sr.shape[1])])
    stf = interpolate.interp1d(time[0],f)
    Fdot=np.array([stf(np.linspace(tstart[iev],tend[iev],npoints)) for iev in range(len(tstart))])
    t = np.array([np.linspace(tstart[iev],tend[iev],npoints)-tstart[iev] for iev in range(len(tstart))])
    return t,Fdot

def plot_STF(save_dir,outputs,dep,cumslip_outputs,spin_up_idx,rths=10,save_on=True):
    from cumslip_compute import analyze_events
    print('Rupture length criterion:',rths,'m')
    rupture_length,av_slip,system_wide,partial_rupture = analyze_events(cumslip_outputs,rths)
    t,Fdot = compute_STF(save_dir,outputs,dep)
    plt.rcParams['font.size'] = '15'
    fig,ax = plt.subplots(nrows=2,figsize=(10,10))
    cmap0 = mpl.colormaps['gnuplot']
    ax[0].set_prop_cycle('color',[cmap0(i) for i in np.linspace(0.5,0.9,len(system_wide[system_wide>=spin_up_idx]))]) 
    ax[0].plot(t[system_wide[system_wide>=spin_up_idx]].T,Fdot[system_wide[system_wide>=spin_up_idx]].T/1e12,lw=2,
            label=[r'Event %d'%(i) for i in system_wide[system_wide>=spin_up_idx]])
    ax[0].legend(fontsize=11,loc='upper right',framealpha=0)
    ax[0].set_ylabel('Force rate [$10^{12}$ N/s]',fontsize=17)
    ax[0].grid(True,alpha=0.4)
    cmap1 = mpl.colormaps['ocean']
    ax[1].set_prop_cycle('color',[cmap1(i) for i in np.linspace(0.5,0.9,len(partial_rupture[partial_rupture>=spin_up_idx]))]) 
    ax[1].plot(t[partial_rupture[partial_rupture>=spin_up_idx]].T,Fdot[partial_rupture[partial_rupture>=spin_up_idx]].T/1e12,lw=2,
            label=[r'Event %d'%(i) for i in partial_rupture[partial_rupture>=spin_up_idx]])
    if len(partial_rupture) > 0:
        ax[1].legend(fontsize=11,loc='upper right',framealpha=0)
    ax[1].set_xlabel('Time [s]',fontsize=17)
    ax[1].set_ylabel('Force rate [$10^{12}$ N/s]',fontsize=17)
    ax[1].grid(True,alpha=0.4)

    plt.tight_layout()
    if save_on:
        plt.savefig('%s/STF.png'%(save_dir),dpi=300)

def plot_event_analyze(save_dir,prefix,cumslip_outputs,rths=10,save_on=True):
    from cumslip_compute import analyze_events
    print('Rupture length criterion:',rths,'m')
    rupture_length,av_slip,system_wide,partial_rupture = analyze_events(cumslip_outputs,rths)
    Hs = ch.load_parameter(prefix)[1]
    ver_info = ch.version_info(prefix)

    # ------ Define figure properties
    plt.rcParams['font.size'] = '15'
    plt.figure(figsize=(14,11))
    ax1 = plt.subplot2grid(shape=(2,5),loc=(0,0),colspan=3)
    plt.subplots_adjust(hspace=0.1)
    ax2 = plt.subplot2grid(shape=(2,5),loc=(1,0),colspan=3)
    ax3 = plt.subplot2grid(shape=(2,5),loc=(0,3),colspan=2,rowspan=2)
    plt.subplots_adjust(wspace=0.8)

    # cmap = mpl.colormaps('ocean')
    cmap = mpl.colormaps['gnuplot']

    # ------ Rupture length
    markers, stemlines, baseline = ax1.stem(np.arange(1,len(rupture_length)+1),rupture_length)
    plt.setp(stemlines, color='k', linewidth=2.5)
    plt.setp(markers, color='k')
    plt.setp(baseline, color='0.62')
    if len(system_wide) > 0:
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

    # ------ Hypocenter depth
    evdep = cumslip_outputs[1][1]
    markers, stemlines, baseline = ax2.stem(np.arange(1,len(evdep)+1),evdep)
    plt.setp(stemlines, color='k', linewidth=2.5)
    plt.setp(markers, color='k')
    plt.setp(baseline, color='0.62')
    if len(system_wide) > 0:
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
    if len(partial_rupture) > 0:
        ax3.plot(np.array(cumslip_outputs[1][2]).T[partial_rupture[0]].T,fault_z,lw=2.5,color='0.62',label='Partial rupture events')
        if len(partial_rupture) > 1:
            ax3.plot(np.array(cumslip_outputs[1][2]).T[partial_rupture[1:]].T,np.array([fault_z for i in range(len(partial_rupture[1:]))]).T,lw=2.5,color='0.62')
    if len(system_wide) > 0:       
        ax3.set_prop_cycle('color',[cmap(i) for i in np.linspace(0.5,0.9,len(system_wide))]) 
        ax3.plot(np.array(cumslip_outputs[1][2]).T[system_wide].T,np.array([fault_z for i in range(len(system_wide))]).T,lw=3,label=[r'Event %d ($\bar{D}$ = %2.2f m)'%(i+1,av_slip[i]) for i in system_wide])
        hyp_dep = cumslip_outputs[1][1][system_wide]
        hyp_slip = [np.array(cumslip_outputs[1][2]).T[system_wide][i][np.where(fault_z==hyp_dep[i])[0][0]] for i in range(len(system_wide))]
        ax3.scatter(hyp_slip,hyp_dep,marker='*',s=300,facecolor=mydarkviolet,edgecolors='k',lw=1,zorder=3,label='Hypocenter')
    if len(partial_rupture) > 0 or len(system_wide) > 0:
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