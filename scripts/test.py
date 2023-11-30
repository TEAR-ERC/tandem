#!/usr/bin/env python3
import numpy as np
import os
import change_params
import setup_shortcut
ch = change_params.variate()
sc = setup_shortcut.setups()

model_n = 'perturb_stress'
output_branch_n = 'reference'
write_on = False

# Perturbation related parameters
seissol_model_n = 'vert_slow'
strike = 340
fcoeff = 0.4
dt = 0.01
target_sys_evID = 6        # Between 0 to 63
time_diff_in_sec = 58320   # 16 h 12 m

# Execution file parameters
n_node = 40
ckp_freq_step = 50
ckp_freq_ptime = 1000000000
ckp_freq_cputime = 60
dstep = 50000

# Path and file names
output_save_dir = '/export/dump/jyun/%s/%s'%(model_n,output_branch_n)
matched_save_dir = '/export/dump/jyun/%s/match%d'%(model_n,target_sys_evID)
run_branch_n = 'pert%d_%s%d'%(target_sys_evID,sc.model_code(seissol_model_n),strike)
fname_lua = '/home/jyun/Tandem/%s/scenario_perturb.lua'%(model_n)
fname_toml = '/home/jyun/Tandem/%s/parameters_perturb_scenario.toml'%(model_n)
fname_shell = '/home/jyun/Tandem/perturb_routine.sh'

print('====================== Summary of Input Parameters =======================')
print('output_save_dir = %s'%(output_save_dir))
print('matched_save_dir = %s'%(matched_save_dir))
print('run_branch_n = %s'%(run_branch_n))
print('seissol_model_n = %s'%(seissol_model_n))
print('strike = %d'%(strike))
print('fcoeff = %1.1f'%(fcoeff))
print('dt = %1.2f'%(dt))
print('n_node = %d'%(n_node))
print('ckp_freq_step = %d'%(ckp_freq_step))
print('ckp_freq_ptime = %d'%(ckp_freq_ptime))
print('ckp_freq_cputime = %d'%(ckp_freq_cputime))
print('dstep = %d'%(dstep))

# 0. Check if the matched checkpoint exists - if not, run make_closer
if not os.path.exists(matched_save_dir):
    print('##### %s NOT FOUND - Need time matching #####'%(matched_save_dir))
    # fshell = open(fname_shell,'w')
    # fshell.write('# Run perturbation period\n')
    import subprocess
    # Syntax e.g., python /home/jyun/Tandem/make_closer.py perturb_stress reference 6 --write_on
    subprocess.run(["python","/home/jyun/Tandem/make_closer.py",model_n,output_branch_n,str(target_sys_evID),"--write_on"])
    subprocess.run(["/home/jyun/Tandem/match_time_run_tandem.sh"],shell=True)

# print('Hello from Python')

# import subprocess
# subprocess.run(["echo","This is an output"])
# subprocess.run(["./sptest.sh"],shell=True)
# print(result.stdout)

# import numpy as np
# from faultoutputs_image import *
# from cumslip_compute import *
# import os

# prefix = 'perturb_stress/reference'

# # ----------
# if 'j4yun/' in os.getcwd(): # local
#     print('local')
#     save_dir = 'models/'+prefix
# elif 'di75weg/' in os.getcwd(): # supermuc
#     print('supermuc')
#     save_dir = '/hppfs/scratch/06/di75weg/'+prefix
# elif 'jyun/' in os.getcwd(): # LMU server
#     print('LMU server')
#     save_dir = '/export/dump/jyun/'+prefix
# print(save_dir)

# print('Load saved data: %s/outputs'%(save_dir))
# outputs = np.load('%s/outputs.npy'%(save_dir))
# print('Load saved data: %s/outputs_depthinfo'%(save_dir))
# dep = np.load('%s/outputs_depthinfo.npy'%(save_dir))
# print('Load saved data: %s/const_params.npy'%(save_dir))
# params = np.load('%s/const_params.npy'%(save_dir),allow_pickle=True)

# # ----------
# Vths = 1e-1
# Vlb = 0
# intv = 0.15
# dt_interm = 0
# cuttime = 0
# rths = 10
# dt_creep = 2*ch.yr2sec
# dt_coseismic = 0.5

# cumslip_outputs = compute_cumslip(outputs,dep,cuttime,Vlb,Vths,dt_creep,dt_coseismic,dt_interm,intv)

# # ----------
# image = 'sliprate'
# # image = 'shearT'
# vmin=1e-12;vmax=1e1
# # zoom_frame_in = '[200000,300000]'
# # plot_in_timestep = True; plot_in_sec = False
# # zoom_frame_in = '[136,1000,1000]'
# # zoom_frame_in = '[-67,-68,20000,20000]'
# # zoom_frame_in = '[77,81,1000,1000]'
# # zoom_frame_in = '[69,72,1000,1000]'
# # plot_in_timestep = False; plot_in_sec = True
# plot_in_timestep = True; plot_in_sec = False
# c = 1e5
# while c <= outputs.shape[1]:
#     print('[%d,%d]'%(c,c+1e5))
#     zoom_frame_in = '[%d,%d]'%(c,c+1e5)
#     fout_image(image,outputs,dep,params,cumslip_outputs,save_dir,prefix,rths,vmin,vmax,Vths,zoom_frame_in,plot_in_timestep,plot_in_sec,save_on=True)
#     c += 1e5