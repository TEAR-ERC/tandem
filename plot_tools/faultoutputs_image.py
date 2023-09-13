#!/usr/bin/env python3
'''
Functions related to plotting spatio-temporal evolution of variables as an image
By Jeena Yun
Last modification: 2023.09.09.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmcrameri.cm as cram
from event_analyze import analyze_events
import change_params
import warnings

ch = change_params.variate()

mypink = (230/255,128/255,128/255)
myblue = (118/255,177/255,230/255)
myburgundy = (214/255,0,0)
mynavy = (17/255,34/255,133/255)
mylightblue = (218/255,230/255,240/255)
myygreen = (120/255,180/255,30/255)
mylavender = (170/255,100/255,215/255)
mydarkviolet = (145/255,80/255,180/255)


# fields: Time [s] | state [s?] | cumslip0 [m] | traction0 [MPa] | slip-rate0 [m/s] | normal-stress [MPa]
# Index:     0     |      1     |       2      |        3        |         4        |          5 

def fout_image(lab,outputs,dep,params,cumslip_outputs,save_dir,prefix,
               rths,vmin,vmax,Vths,zoom_frame,plot_in_timestep=True,plot_in_sec=False,cb_on=True,publish=False,save_on=True):
    processed_outputs = class_figtype(zoom_frame,outputs,cumslip_outputs,publish)[0]
    X,Y,var = get_var(lab,processed_outputs,dep,plot_in_timestep,plot_in_sec)
    if vmin is None:
        vmin = np.min(var)
    if vmax is None:
        vmax = np.max(var)
    ax = plot_image(X,Y,var,lab,outputs,cumslip_outputs,save_dir,prefix,params,zoom_frame,rths,vmin,vmax,Vths,plot_in_timestep,plot_in_sec,cb_on,publish,save_on)
    return ax

def get_var(lab,outputs,dep,plot_in_timestep,plot_in_sec):
    if lab == 'sliprate':
        idx = 4
    elif lab == 'shearT':
        idx = 3
    elif lab == 'normalT':
        idx = 5
    elif lab == 'state_var':
        idx = 1
    var = np.array([outputs[i][:,idx] for i in np.argsort(abs(dep))])
    if lab == 'shearT' or lab == 'normalT':
        var = abs(var)

    if plot_in_timestep:
        print('Plot in time steps')
        xax = np.arange(var.shape[1])
    elif plot_in_sec:
        print('Plot in time [s]')
        xax = np.array(outputs[0][:,0])
    else:
        print('Plot in time [yrs]')
        xax = np.array(outputs[0][:,0]/ch.yr2sec)
    X,Y = np.meshgrid(xax,np.sort(abs(dep)))

    return X,Y,var

def gen_cmap(lab,params,vmin,vmax,Vths):
    if lab == 'sliprate':
        cb_label = 'Slip Rate [m/s]'
        cm = mpl.colormaps['RdYlBu_r']
        col_list = [cm(i)[0:3] for i in [0.15,0.5,0.8,0.9]]
        col_list = [cm(0.15)[0:3],mpl.colormaps['jet'](0.67),cm(0.8)[0:3],cm(0.9)[0:3]]
        col_list.insert(0,(0,0,0))
        col_list.insert(3,mpl.colors.to_rgb('w'))
        col_list.append(mpl.colormaps['turbo'](0))
        if params is None:
            float_list = [0,mpl.colors.LogNorm(vmin,vmax)(1e-9),mpl.colors.LogNorm(vmin,vmax)(1e-6)]
        else:
            float_list = [0,mpl.colors.LogNorm(vmin,vmax)(params.item().get('Vp')),mpl.colors.LogNorm(vmin,vmax)(params.item().get('V0'))]
        [float_list.append(k) for k in np.linspace(mpl.colors.LogNorm(vmin,vmax)(Vths),1,4)]
        cmap_n = get_continuous_cmap(col_list,input_hex=False,float_list=float_list)
    elif lab == 'shearT':
        cb_label = 'Shear Traction [MPa]'
        # ---- plotting total stress
        cmap_n = cram.davos
        # ---- plotting stress change
        # cb_label = 'Shear Traction Change [MPa]'
        # cm = cram.vik
        # col_list = [cm(i) for i in np.linspace(0,1,6)]
        # col_list.insert(3,mpl.colors.to_rgb('w'))
        # float_list = [mpl.colors.Normalize(vmin,vmax)(i) for i in np.linspace(vmin,0,4)]
        # [float_list.append(mpl.colors.Normalize(vmin,vmax)(k)) for k in np.linspace(0,vmax,4)[1:]]
        # cmap_n = get_continuous_cmap(col_list,input_hex=False,float_list=float_list)
    elif lab == 'normalT':
        cb_label = 'Normal Stress [MPa]'
        cmap_n = cram.davos
    elif lab == 'statevar':
        cb_label = 'State Variable [1/s]'
        cmap_n = 'magma'
    return cmap_n,cb_label

def class_figtype(zoom_frame,outputs,cumslip_outputs,publish,print_on=True):
    tstart,tend,evdep = cumslip_outputs[0][0],cumslip_outputs[0][1],cumslip_outputs[1][1]
    its_all = np.array([np.argmin(abs(outputs[0][:,0]-t)) for t in tstart])
    ite_all = np.array([np.argmin(abs(outputs[0][:,0]-t)) for t in tend])
    if len(zoom_frame) == 0: # Full image
        if print_on: print('Full image')
        processed_outputs = outputs
        tsmin,tsmax,its,ite,buffer1,buffer2,iev1,iev2 = [],[],[],[],[],[],[],[]
        xl_opt = 1
        ver_info_opt = True
        scatter_opt = 1
        vlines_opt = 0
        txt_opt = 0
        xlab_opt = 1
        name_opt = 1
    elif len(zoom_frame) == 2: # Zoom in of full image
        if print_on: print('Zoom in of full image')
        tsmin,tsmax = int(zoom_frame[0]),int(zoom_frame[1])
        if tsmin > outputs.shape[1]:
            ValueError('tmin > max. timestep - check input')
        elif tsmax > outputs.shape[1]:
            warnings.warn('tsmax > max. timestep')
        processed_outputs = outputs[:,tsmin:tsmax,:]
        its,ite,buffer1,buffer2,iev1,iev2 = [],[],[],[],[],[]
        xl_opt = 2
        ver_info_opt = True
        scatter_opt = 2
        vlines_opt = 0
        txt_opt = 1
        xlab_opt = 2
        name_opt = 2
    else:
        buffer1,buffer2 = int(zoom_frame[-2]),int(zoom_frame[-1])
        xl_opt = 3
        ver_info_opt = False
        scatter_opt = 3
        xlab_opt = 1
        if len(zoom_frame) == 3: # Single coseismic event
            if print_on: print('Single coseismic event')
            iev1,iev2 = int(zoom_frame[0]),[]
            its,ite = its_all[iev1],ite_all[iev1]
            vlines_opt = 1
            txt_opt = 2
            name_opt = 3
        elif len(zoom_frame) == 4 and zoom_frame[0]>=0: # Multiple coseismic event
            if print_on: print('Multiple coseismic events')
            iev1,iev2 = int(zoom_frame[0]),int(zoom_frame[1])
            its,ite = its_all[iev1],ite_all[iev2]
            vlines_opt = 2
            txt_opt = 1
            name_opt = 4
        elif len(zoom_frame) == 4 and zoom_frame[0]<0: # Interseismic event
            if print_on: print('Interseismic period')
            iev1,iev2 = abs(int(zoom_frame[0])),abs(int(zoom_frame[1]))
            if iev2 == len(its_all):
                its,ite = ite_all[iev1],outputs.shape[1]-1
            else:
                its,ite = ite_all[iev1],its_all[iev2]
            vlines_opt = 1
            txt_opt = 3
            name_opt = 5
        processed_outputs = outputs[:,its-buffer1:ite+buffer2,:]
        tsmin,tsmax = [],[]
    if publish:
        ver_info_opt = False
    fig_opts = [xl_opt,ver_info_opt,scatter_opt,vlines_opt,txt_opt,xlab_opt,name_opt]
    return processed_outputs,evdep,its_all,ite_all,tsmin,tsmax,its,ite,buffer1,buffer2,iev1,iev2,fig_opts

def decoration(time,zoom_frame,outputs,cumslip_outputs,ver_info,Hs,acolor,rths,plot_in_timestep,plot_in_sec,publish):
    evdep,its_all,ite_all,tsmin,tsmax,its,ite,buffer1,buffer2,iev1,iev2,fig_opts = class_figtype(zoom_frame,outputs,cumslip_outputs,publish,print_on=False)[1:]
    system_wide,partial_rupture,event_cluster,lead_fs = analyze_events(cumslip_outputs,rths)[2:6]
    xl_opt,ver_info_opt,scatter_opt,vlines_opt,txt_opt,xlab_opt,name_opt = fig_opts

    if xl_opt == 1 or xl_opt == 2:
        xl = plt.gca().get_xlim()
    elif xl_opt == 3:
        width = np.round(time[ite]-time[its])
        xl = [time[its]-width/6,time[ite]+width/6]

    if ver_info_opt and len(ver_info) > 0:
        if ver_info[:2] == '+ ':
            ver_info = ver_info[2:]
        plt.text(xl[0]+(xl[1]-xl[0])*0.025,Hs[0]*0.975,ver_info,color=acolor,fontsize=35,fontweight='bold',ha='left',va='bottom')

    if scatter_opt == 2:
        lim_s, lim_e = tsmin, tsmax
    elif scatter_opt == 3:
        lim_s, lim_e = its, ite

    if len(its_all) > 0:
        if plot_in_timestep:
            if scatter_opt == 1:
                xs = its_all
            else:
                xs = its_all - lim_s
        else:
            xs = time[its_all]

    if scatter_opt == 1:
        if len(system_wide) > 0:
            plt.scatter(xs[system_wide],evdep[system_wide],s=300,marker='*',facecolor=mydarkviolet,edgecolor='k',lw=1.5,zorder=3,label='System-size events')
        if len(lead_fs) > 0:
            plt.scatter(xs[lead_fs],evdep[lead_fs],s=100,marker='d',facecolor=mydarkviolet,edgecolor='k',lw=1.5,zorder=3,label='Leading foreshocks')
        if len(partial_rupture) > 0:
            plt.scatter(xs[partial_rupture],evdep[partial_rupture],s=100,marker='d',facecolor='w',edgecolor='k',lw=1.5,zorder=2,label='Partial rupture events')
    else:
        isys = np.where(np.logical_and(its_all[system_wide]>=lim_s,its_all[system_wide]<=lim_e))[0]
        if len(lead_fs) > 0:
            ilfs = np.where(np.logical_and(its_all[lead_fs]>=lim_s,its_all[lead_fs]<=lim_e))[0]
        if len(partial_rupture) > 0:
            ipart = np.where(np.logical_and(its_all[partial_rupture]>=lim_s,its_all[partial_rupture]<=lim_e))[0]
        else:
            ipart = []
        
        if len(system_wide[isys]) > 0:
            plt.scatter(xs[system_wide][isys],evdep[system_wide][isys],s=300,marker='*',facecolor=mydarkviolet,edgecolor='k',lw=1.5,zorder=3,label='System-size events')
        if len(lead_fs) > 0:
            plt.scatter(xs[lead_fs][ilfs],evdep[lead_fs][ilfs],s=100,marker='d',facecolor=mydarkviolet,edgecolor='k',lw=1.5,zorder=3,label='Leading foreshocks')
        if len(partial_rupture)>0 and len(partial_rupture[ipart]) > 0:
            plt.scatter(xs[partial_rupture][ipart],evdep[partial_rupture][ipart],s=100,marker='d',facecolor='w',edgecolor='k',lw=1.5,zorder=2,label='Partial rupture events')
    plt.legend(fontsize=20,framealpha=1,loc='lower right')

    plt.hlines(y=Hs[1],xmin=xl[0],xmax=xl[1],linestyles='--',color=acolor,lw=1.5)
    plt.hlines(y=Hs[-1],xmin=xl[0],xmax=xl[1],linestyles='--',color=acolor,lw=1.5)

    if vlines_opt == 1:
        if plot_in_timestep:
            plt.vlines(x=[0,ite-its],ymin=0,ymax=Hs[0],linestyles='--',color=acolor,lw=1.5)
        else:
            plt.vlines(x=[time[its],time[ite]],ymin=0,ymax=Hs[0],linestyles='--',color=acolor,lw=1.5)
    elif vlines_opt == 2:
        if len(its_all) > 0:
            plt.vlines(x=xs[iev1:iev2+1],ymin=0,ymax=Hs[0],linestyles='--',color=acolor,lw=1.5)

    if len(its_all) > 0:
        if txt_opt == 1:
            for k in range(len(its_all)):
                if its_all[k]>=lim_s and its_all[k]<=lim_e:
                    plt.text(xs[k],evdep[k]-0.2,'%d'%(k),color=acolor,fontsize=20,ha='right',va='bottom')
        elif txt_opt == 2:
            plt.text(xs[iev1]+width/18,23,'Event %d'%(iev1),fontsize=30,fontweight='bold',color=acolor,ha='left',va='bottom') # coseismic
        elif txt_opt == 3:
            plt.text(xs[iev1]-width/150,12,'Event %d'%(iev1),fontsize=30,fontweight='bold',color=acolor,ha='right',va='bottom',rotation=90)
            if plot_in_timestep:
                plt.text(ite-its+width/150,12,'Event %d'%(iev2),fontsize=30,fontweight='bold',color=acolor,ha='left',va='bottom',rotation=90)
            else:
                plt.text(time[ite]+width/150,12,'Event %d'%(iev2),fontsize=30,fontweight='bold',color=acolor,ha='left',va='bottom',rotation=90)

    if xl_opt == 1:
        plt.xlim(0,xl[1])
    elif xl_opt == 2:
        if plot_in_timestep:
            plt.xlim(0,xl[1])
        else:
            plt.xlim(time[tsmin],xl[1])
    elif xl_opt == 3:
        plt.xlim(xl)
        
    plt.ylim(0,Hs[0])
    plt.gca().invert_yaxis()
    plt.ylabel('Depth [km]',fontsize=30)

    if xlab_opt == 1:
        if plot_in_timestep:
            plt.xlabel('Timesteps',fontsize=30)
        elif plot_in_sec:        
            plt.xlabel('Time [s]',fontsize=30)
        else:        
            plt.xlabel('Time [yrs]',fontsize=30)
    elif xlab_opt == 2:
        if plot_in_timestep:
            if tsmin > 1e-3:
                xt = plt.gca().get_xticks()
                xt_lab = ['%d'%(ticks+tsmin) for ticks in xt]
                plt.gca().set_xticks(xt)
                plt.gca().set_xticklabels(xt_lab)
            plt.xlabel('Timesteps',fontsize=30)
        elif plot_in_sec:
            plt.xlabel('Time [s]',fontsize=30)
        else:        
            plt.xlabel('Time [yrs]',fontsize=30)

    if name_opt == 1:
        fig_name = '_image'
    elif name_opt == 2:
        if np.mod(tsmin,tsmax-tsmin) == 0:
            fig_name = '_zoom_image_%d_%d'%(tsmin/(tsmax-tsmin),tsmax/(tsmax-tsmin))
        else:
            fig_name = '_zoom_image_%d_%d'%(tsmin/1000,tsmax/1000)
    elif name_opt == 3:
        fig_name = '_zoom_image_ev%d'%(iev1)
    elif name_opt == 4:
        fig_name = '_zoom_image_ev%dto%d'%(iev1,iev2)
    elif name_opt == 5:
        fig_name = '_zoom_image_interevent_%d_%d'%(iev1,iev2)

    return fig_name

def plot_image(X,Y,var,lab,outputs,cumslip_outputs,save_dir,prefix,params,zoom_frame,rths,vmin,vmax,Vths,plot_in_timestep,plot_in_sec,cb_on,publish,save_on):
    Hs = ch.load_parameter(prefix)[1]

    plt.rcParams['font.size'] = '27'
    fig,ax=plt.subplots(figsize=(20.6,11))

    cmap_n,cb_label = gen_cmap(lab,params,vmin,vmax,Vths)
    if lab == 'sliprate':
        cb = plt.pcolormesh(X,Y,var,cmap=cmap_n,norm=mpl.colors.LogNorm(vmin,vmax))
        acolor = 'w'
    elif lab == 'shearT':
        # ---- Stress change
        # var = np.array([var[i,:]-var[i,0] for i in range(var.shape[0])])
        # ---- Total stress
        var = np.array([var[i,:] for i in range(var.shape[0])])
        cb = plt.pcolormesh(X,Y,var,cmap=cmap_n,vmin=vmin,vmax=vmax)
        acolor = 'k'
    else:
        cb = plt.pcolormesh(X,Y,var,cmap=cmap_n,vmin=vmin,vmax=vmax)
        acolor = 'w'
    if cb_on:
        plt.colorbar(cb,extend='both').set_label(cb_label,fontsize=30,rotation=270,labelpad=30) # vertical colorbar
        # plt.colorbar(cb,extend='both',location='bottom',shrink=0.5).set_label(cb_label,fontsize=30,labelpad=15) # horizonta colorbar

    if plot_in_sec:
        time = np.array(outputs[0,:,0])
    else:
        time = np.array(outputs[0,:,0])/ch.yr2sec
    ver_info = ch.version_info(prefix)
    fig_name = decoration(time,zoom_frame,outputs,cumslip_outputs,ver_info,Hs,acolor,rths,plot_in_timestep,plot_in_sec,publish)

    plt.tight_layout()
    if save_on:
        if plot_in_timestep:
            if publish:
                plt.savefig('%s/publish_%s%s_timesteps.png'%(save_dir,lab,fig_name),dpi=300)
            else:
                plt.savefig('%s/%s%s_timesteps.png'%(save_dir,lab,fig_name),dpi=300)
        else:
            if publish:
                plt.savefig('%s/publish_%s%s.png'%(save_dir,lab,fig_name),dpi=300)
            else:
                plt.savefig('%s/%s%s.png'%(save_dir,lab,fig_name),dpi=300)
    return ax

def get_continuous_cmap(col_list,input_hex=False,float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in col_list.
        If float_list is provided, each color in col_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        col_list: list of color code strings
        float_list: list of floats between 0 and 1, same length as col_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    if input_hex:
        rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in col_list]
    else:
        rgb_list = col_list.copy()

    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]
