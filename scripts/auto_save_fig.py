#!/usr/bin/env python3
import numpy as np
from faultoutputs_image import *
from cumslip_compute import *
import os

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

print('Load saved data: %s/outputs'%(save_dir))
outputs = np.load('%s/outputs.npy'%(save_dir))
print('Load saved data: %s/outputs_depthinfo'%(save_dir))
dep = np.load('%s/outputs_depthinfo.npy'%(save_dir))
print('Load saved data: %s/const_params.npy'%(save_dir))
params = np.load('%s/const_params.npy'%(save_dir),allow_pickle=True)

# ----------
Vths = 2e-1
Vlb = 0
intv = 0.15
dt_interm = 0
cuttime = 0
rths = 10
dt_creep = 2*ch.yr2sec
dt_coseismic = 0.5

cumslip_outputs = compute_cumslip(outputs,dep,cuttime,Vlb,Vths,dt_creep,dt_coseismic,dt_interm,intv)

# ----------
image = 'sliprate'
# image = 'shearT'
vmin=1e-12;vmax=1e1
# zoom_frame_in = '[200000,300000]'
# zoom_frame_in = '[136,1000,1000]'
# zoom_frame_in = '[-67,-68,20000,20000]'
# zoom_frame_in = '[77,81,1000,1000]'
# zoom_frame_in = '[69,72,1000,1000]'
# plot_in_timestep = False; plot_in_sec = True
plot_in_timestep = True; plot_in_sec = False
c = 0
while c <= outputs.shape[1]:
    print('[%d,%d]'%(c,c+1e5))
    zoom_frame_in = [int(c),int(c+1e5)]
    fout_image(image,outputs,dep,params,cumslip_outputs,save_dir,prefix,rths,vmin,vmax,Vths,zoom_frame_in,plot_in_timestep,plot_in_sec,save_on=True)
    c += 1e5