#!/usr/bin/env python3
"""
Collection of customized plot-related functions and constants
Last modification: 2023.05.25.
by Jeena Yun
"""
import numpy as np
import matplotlib.colors as mcolors

class Figpref:
    def __init__(self):
        # My colors
        self.mypink = (230/255,128/255,128/255)
        self.mypalepink = (235/255,180/255,180/255)
        self.mygrey = (158/255,158/255,158/255)
        self.mygreen = (102/255,153/255,26/255)
        self.myblue = (118/255,177/255,230/255)
        self.myburgundy = (214/255,0,0)
        self.mydarkgrey = (89/255,89/255,89/255)
        self.myyellow = (1,221/255,51/255)
        self.myviolet = (130/255,6/255,214/255)
        self.myorange = (1,149/255,33/255)
        self.mynavy = (17/255,34/255,133/255)
        self.mybrown = (104/255,54/255,46/255)
        self.mypink = (230/255,128/255,128/255)
        self.mydarkpink = (200/255,110/255,110/255)
        # self.mydarkpink = (210/255,115/255,115/255)
        self.myblue = (118/255,177/255,230/255)
        self.myburgundy = (214/255,0,0)
        self.mynavy = (17/255,34/255,133/255)
        self.mylightblue = (218/255,230/255,240/255)
        self.myygreen = (120/255,180/255,30/255)
        self.mylavender = (170/255,100/255,215/255)
        # self.mydarkviolet = (120/255,55/255,145/255)
        self.mydarkviolet = (145/255,80/255,180/255)
        pptyellow = (255/255,217/255,102/255)

        # Trial info
        self.horz = [11,12,15,18]
        self.vert = [13,14,16,17,19,22,23,24,25,26,27,28,29]

        # Text box properties
        self.tboxprop = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor = 'k', lw=2)

    def plot_CFS(self,ax,xy,tri,dat,lb,ub,Mp,CFS_type,model_n,print_on):
        from cmcrameri import cm
        if CFS_type == 'd':
            cmap = cm.lajolla
            if not lb and not ub:
                p = ax.tripcolor(xy[:,0],xy[:,1],tri,dat,cmap=cmap)
            else:
                p = ax.tripcolor(xy[:,0],xy[:,1],tri,dat,cmap=cmap,vmin=lb,vmax=ub)
        elif CFS_type == 's':
            cmap = cm.vik
            if not lb and not ub:
                p=ax.tripcolor(xy[:,0],xy[:,1],tri,dat,cmap=cmap)
            else:
                p=ax.tripcolor(xy[:,0],xy[:,1],tri,dat,cmap=cmap,vmin=lb,vmax=ub)
        
        if np.isin(model_n,self.horz):
            ax.plot([-1.5,1.5],[0,0],'--',color='w',lw=5,dashes=(3,1))
            ax.scatter(Mp[0],Mp[1],s=81,edgecolor='w',lw=3,facecolors='none')
            ax.set_xlim(-6,6); ax.set_ylim(-6,6)
        else:
            ax.set_xlim(-1.5,1.5); ax.set_ylim(-8,-2)
            ax.axhline(y=-3,linestyle='--',color='w',lw=3)
            ax.axhline(y=-5,linestyle='--',color='w',lw=3)
            ax.axhline(y=-7,linestyle='--',color='w',lw=3)
            ax.axvline(x=0,linestyle='--',color='w',lw=3)

        if print_on:
            import statistics
            print('Max. = %4.4f'%max(dat))
            print('3rd Q. = %4.4f'%np.percentile(dat,75))
            print('Median = %4.4f'%statistics.median(dat))
            print('1st Q. = %4.4f'%np.percentile(dat,25))
            print('Min. = %4.4f'%min(dat))

        return p

    def plot_waveform(self,ax,t,dat,normalize,col,txt_str,txt_pos,xtick_on,lgd):
        if normalize:
            dat = dat/max(abs(np.array(dat)))
        if not len(lgd)==0:
            ax.plot(t,dat,color=col,label=lgd,lw=2,zorder=3)
        else:
            ax.plot(t,dat,color=col,lw=2,zorder=3)
        if normalize:
            ax.set_ylim(-1.1,1.1)
        yl = ax.get_ylim()
        xl = ax.get_xlim()
        if not len(txt_str) == 0:
            ax.text(min(xl)+0.3,yl[0]+txt_pos*(yl[1]-yl[0]),txt_str,fontsize=15,fontweight='bold',ha='left',va='top')       
        if not isinstance(xtick_on,str):
            if not xtick_on:
                ax.axes.xaxis.set_ticklabels([])
            else:
                if max(xl) > 9:
                    ax.set_xticks(np.linspace(0,10,6))
                    ax.set_xticklabels(['0','2','4','6','8','10'])
                else:
                    ax.set_xticks(np.linspace(3,7,10))
                    ax.set_xticklabels(['3.0','3.5','4.0','4.5','5.0','5.5','6.0','6.5','7.0','7.5'])
                ax.set_xlabel('Time from the Origin [s]',fontsize=17)
        else:
            ax.set_xlabel(xtick_on,fontsize=17)
        return yl, xl

    def get_continuous_cmap(self,col_list,input_hex=False,float_list=None):
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
            rgb_list = [self.rgb_to_dec(self.hex_to_rgb(i)) for i in col_list]
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
        cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp

    def hex_to_rgb(self,value):
        '''
        Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values'''
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def rgb_to_dec(self,value):
        '''
        Converts rgb to decimal colours (i.e. divides each value by 256)
        value: list (length 3) of RGB values
        Returns: list (length 3) of decimal values'''
        return [v/256 for v in value]