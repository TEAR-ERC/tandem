import numpy as np
import matplotlib.pyplot as plt
import os
from cumslip_compute import *
from fcompute import *
from fplots import *
import myplots
import change_params
import setup_shortcut

sc = setup_shortcut.setups()
mp = myplots.Figpref()
ch = change_params.variate()

compute_and_save = 1
plot_G_prof_all = 1
plot_G_prof_spinup = 1
plot_avG_event = 1
plot_scaling_relation = 1
plot_scaling_relation_no_av = 1
plot_linear_prop_onlyspup = 1

# prefix_list = ['BP1',
#                'Thakur20_hetero_stress/n8',
#                'Thakur20_various_fractal_profiles/ab2',
#                'Thakur20_various_fractal_profiles/Dc1',
#                'Thakur20_various_fractal_profiles/v6_ab2_Dc1_long',
#                'Thakur20_various_fractal_profiles/v6_Dc1_long']
# labs = ['BP1','Fractal Stress','Fractal a-b','Fractal Dc','Fractal Stress & a-b & Dc','Fractal Stress & Dc']
# cols = [mp.myburgundy,mp.mylavender,mp.mynavy,mp.myblue,mp.myyellow,mp.mypalepink]
prefix_list = ['Thakur20_various_fractal_profiles/ab2_Dc1']
labs = ['Fractal a-b & Dc']
cols = ['turquoise']
dir = '/Users/j4yun/Library/CloudStorage/Dropbox/Codes/Ridgecrest_CSC/Tandem'

for uu,prefix in enumerate(prefix_list):
    print(prefix)
    save_dir = dir + '/models/'+prefix
    plot_dir = 'plots/' + prefix

    if not os.path.exists(plot_dir):
        print('Generating directory',plot_dir)
        strloc = ''
        for i in range(len(prefix.split('/'))):
            if i == 0:
                strloc += prefix.split('/')[i]
            else:
                strloc += '/' + prefix.split('/')[i]
            if not os.path.exists('plots/%s'%(strloc)):
                os.mkdir('plots/%s'%(strloc))
                os.mkdir('outputs/%s'%(strloc))

    # ---------- Load outputs
    print('Load saved data: %s/outputs'%(save_dir))
    outputs = np.load('%s/outputs.npy'%(save_dir))
    print('Load saved data: %s/outputs_depthinfo'%(save_dir))
    dep = np.load('%s/outputs_depthinfo.npy'%(save_dir))
    Hs = ch.load_parameter(prefix)[1]

    time = np.array([outputs[i][:,0] for i in np.argsort(abs(dep))])
    cumslip = np.array([outputs[i][:,2] for i in np.argsort(abs(dep))])
    shearT = abs(np.array([outputs[i][:,3] for i in np.argsort(abs(dep))]))

    Vths = 1e-2
    Vlb = 0
    dt_interm = 0
    cuttime = 0
    mingap = 60
    rths = 10
    dt_creep = 2*ch.yr2sec
    dt_coseismic = 0.5

    cumslip_outputs = compute_cumslip(outputs,dep,cuttime,Vlb,Vths,dt_creep,dt_coseismic,dt_interm,mingap)
    rupture_length,av_slip,system_wide,partial_rupture = analyze_events(cumslip_outputs,rths)
    tstart,tend = cumslip_outputs[0][0],cumslip_outputs[0][1]
    spin_up_idx = compute_spinup(outputs,dep,cuttime,cumslip_outputs,2.5)[-1]

    dep = np.sort(abs(dep))

    if compute_and_save:
        G,D,sd = compute_G(dep,time,tstart,tend,cumslip,shearT)
        av_G = np.nanmean(G,axis=1)
        av_slip = np.nanmean(D,axis=1)
        av_stressdrop = np.nanmean(sd,axis=1)
        # av_slip = np.array(av_slip)
        outputs = {
            "G": G,
            "D": D,
            "sd": sd,
            "avG": av_G,
            "avD": av_slip,
            "avsd": av_stressdrop,
            "sw": system_wide,
            "pr": partial_rupture,
            "spinup": spin_up_idx
        }
        np.save('outputs/%s/outputs'%(prefix),outputs)
    else:
        print('Load saved data: outputs/%s/outputs.npy'%(prefix))
        outputs = np.load('outputs/%s/outputs.npy'%(prefix),allow_pickle=True)
        G = outputs.item().get('G')
        D = outputs.item().get('D')
        sd = outputs.item().get('sd')
        av_G = outputs.item().get('avG')
        av_slip = outputs.item().get('avD')
        av_stressdrop = outputs.item().get('avsd')
        system_wide = outputs.item().get('sw')
        partial_rupture = outputs.item().get('pr')
        spin_up_idx = outputs.item().get('spinup')

    # -------------- Plot G profile for all events
    if plot_G_prof_all:
        plt.rcParams['font.size'] = '15'
        fig,ax = plt.subplots(ncols=2,figsize=(15,11))
        G_prof_all(ax[0],G,cumslip_outputs,av_slip,system_wide,'sw',Hs)
        ax[0].set_title('System-wide Events',fontsize=20,fontweight='bold')
        G_prof_all(ax[1],G,cumslip_outputs,av_slip,partial_rupture,'pr',Hs)
        ax[1].set_title('Partial Rupture Events',fontsize=20,fontweight='bold')
        plt.tight_layout()
        plt.savefig('%s/G_profile_all'%(plot_dir),dpi=300)

    # -------------- Plot G profile for spin-up events only
    if plot_G_prof_spinup:
        plt.rcParams['font.size'] = '15'
        fig,ax = plt.subplots(ncols=2,figsize=(15,11))
        G_prof_all(ax[0],G,cumslip_outputs,av_slip,system_wide[system_wide>spin_up_idx],'sw',Hs)
        ax[0].set_title('System-wide Events',fontsize=20,fontweight='bold')
        G_prof_all(ax[1],G,cumslip_outputs,av_slip,partial_rupture[partial_rupture>spin_up_idx],'pr',Hs)
        ax[1].set_title('Partial Rupture Events',fontsize=20,fontweight='bold')
        plt.tight_layout()
        plt.savefig('%s/G_profile_spinup'%(plot_dir),dpi=300)

    # -------------- AvG by event #
    if plot_avG_event:
        plt.rcParams['font.size'] = '15'
        fig,ax = plt.subplots(figsize=(8,5))
        avG_event(ax,av_G,tstart,system_wide,partial_rupture,'G',spin_up_idx)
        plt.tight_layout()
        plt.savefig('%s/avG_event'%(plot_dir),dpi=300)

    # -------------- Scaling plot
    if plot_scaling_relation:
        plt.rcParams['font.size'] = '15'
        fig,ax = plt.subplots(figsize=(12,6))    
        pts,labels = plot_Coccos(ax,follow_color=False,legend_on=False)
        first_legend = plt.legend(handles=pts,labels=labels,fontsize=12,framealpha=0,loc='lower right',markerscale=2.)
        plt.gca().add_artist(first_legend)

        pts = []
        sw = scaling(ax,av_slip[system_wide],av_G[system_wide],mp.myburgundy,lab='%s; System-wide'%(labs[uu]),s=36,marker='s',ec='k',lw=1,zord=3,yl='auto')
        if len(partial_rupture) > 0:
            pr = scaling(ax,av_slip[partial_rupture],av_G[partial_rupture],mp.mynavy,lab='%s; Partial rupture'%(labs[uu]),s=50,marker='d',ec='k',lw=1,zord=3,yl='auto')
            pts.append(sw)
            pts.append(pr)
        else:
            pts.append(sw)
        ax.legend(fontsize=12,framealpha=0,loc='upper left')
        plt.tight_layout()
        plt.savefig('%s/scaling_relation'%(plot_dir),dpi=300)

    # -------------- Scaling plot - no averaging
    if plot_scaling_relation_no_av:
        plt.rcParams['font.size'] = '15'
        fig,ax = plt.subplots(figsize=(12,6))    
        pts,labels = plot_Coccos(ax,follow_color=False,legend_on=False)
        first_legend = plt.legend(handles=pts,labels=labels,fontsize=12,framealpha=0,loc='lower right',markerscale=2.)
        plt.gca().add_artist(first_legend)

        pts = []
        sw = scaling(ax,D[system_wide[system_wide>spin_up_idx]],G[system_wide[system_wide>spin_up_idx]],cols[uu],lab='%s; System-wide'%(labs[uu]),s=36,marker='s',ec='k',lw=1,zord=3,yl='auto')
        if len(partial_rupture) > 0 and sum(partial_rupture<=spin_up_idx) > 0:
            pr = scaling(ax,D[partial_rupture[partial_rupture>spin_up_idx]],G[partial_rupture[partial_rupture>spin_up_idx]],cols[uu],lab='%s; Partial rupture'%(labs[uu]),s=50,marker='d',ec='k',lw=1,zord=3,yl='auto')
            pts.append(sw)
            pts.append(pr)
        else:
            pts.append(sw)
        ax.legend(handles=pts,fontsize=12,framealpha=0,loc='upper left')
        plt.tight_layout()
        plt.savefig('%s/scaling_relation_after_spinup_not_av'%(plot_dir),dpi=300)

    # -------------- Linear proportionality
    if plot_linear_prop_onlyspup:
        plt.rcParams['font.size'] = '15'
        fig,ax = plt.subplots(figsize=(10,8))
        plot_Coccos(ax,follow_color=False,legend_on=False)

        xr = [1e-3,1e2]
        pts = []
        sw=scaling(ax,av_slip[system_wide[system_wide>spin_up_idx]],av_G[system_wide[system_wide>spin_up_idx]],cols[uu],lab='%s; System-wide'%(labs[uu]),s=36,marker='s',ec='k',lw=1,zord=3,yl=[1e2,2.5e11],xl=xr)
        if len(partial_rupture) > 0 and sum(partial_rupture>spin_up_idx) > 0:
            pr=scaling(ax,av_slip[partial_rupture[partial_rupture>spin_up_idx]],av_G[partial_rupture[partial_rupture>spin_up_idx]],cols[uu],lab='%s; Partial rupture'%(labs[uu]),s=50,marker='d',ec='k',lw=1,zord=3,yl=[1e2,2.5e11],xl=xr)
            pts.append(sw)
            pts.append(pr)
        else:
            pts.append(sw)
        first_legend = plt.legend(handles=pts,fontsize=12,framealpha=0,loc='upper left')
        plt.gca().add_artist(first_legend)

        if len(partial_rupture) > 0 and sum(partial_rupture>spin_up_idx) > 0:
            x = np.hstack((av_slip[system_wide[system_wide>spin_up_idx]],av_slip[partial_rupture[partial_rupture>spin_up_idx]]))
            y = np.hstack((av_G[system_wide[system_wide>spin_up_idx]],av_G[partial_rupture[partial_rupture>spin_up_idx]]))
        else:
            x = av_slip[system_wide[system_wide>spin_up_idx]]
            y = av_G[system_wide[system_wide>spin_up_idx]]
        b = estimate_coef(np.log10(x), np.log10(y))
        y_pred = logxy(xr,b[1],b[0])

        l1,=ax.plot(xr,logxy(xr,1.39,7.3),'0.5',lw=2.5,label='1.39 Cocco et al. 2016')
        l2,=ax.plot(xr,logxy(xr,1.35,7.2),'0.6',lw=2.5,label='1.35 Causse et al. 2014')
        l3,=ax.plot(xr,logxy(xr,1.28,6.975),'0.7',lw=2.5,label='1.28 Abercrombie & Rice 2005')
        l4,=ax.plot(xr,y_pred,color=cols[uu],lw=3,label='%2.2f %s'%(b[1],labs[uu]))
        ax.legend(handles=[l4,l1,l2,l3],fontsize=10,framealpha=0,loc='lower right')

        plt.tight_layout()
        plt.savefig('%s/linear_prop_onlyspup'%(plot_dir),dpi=300)