#!/usr/bin/env python3
import numpy as np
from faultoutputs_image import *
from cumslip_compute import *

prefix = 'Thakur20_various_fractal_profiles/v6_ab2_Dc2'

# ----------
save_dir = 'models/'+prefix

print('Load saved data: %s/outputs'%(save_dir))
outputs = np.load('%s/outputs.npy'%(save_dir))
print('Load saved data: %s/outputs_depthinfo'%(save_dir))
dep = np.load('%s/outputs_depthinfo.npy'%(save_dir))
print('Load saved data: %s/const_params.npy'%(save_dir))
params = np.load('%s/const_params.npy'%(save_dir),allow_pickle=True)

# ----------
Vths = 1e-1
Vlb = 0
dt_interm = 0
cuttime = 0
rths = 10
dt_creep = 2*ch.yr2sec
dt_coseismic = 0.5

cumslip_outputs = compute_cumslip(outputs,dep,cuttime,Vlb,Vths,dt_creep,dt_coseismic,dt_interm)

# ----------
lab = 'sliprate'
vmin=1e-12;vmax=1e1
# zoom_frame_in = '[200000,300000]'
# plot_in_timestep = True; plot_in_sec = False
# zoom_frame_in = '[136,1000,1000]'
# zoom_frame_in = '[-67,-68,20000,20000]'
# zoom_frame_in = '[77,81,1000,1000]'
zoom_frame_in = '[69,72,1000,1000]'
plot_in_timestep = False; plot_in_sec = True

fout_image(lab,outputs,dep,params,cumslip_outputs,save_dir,prefix,rths,vmin,vmax,Vths,zoom_frame_in,plot_in_timestep,plot_in_sec,save_on=True)
