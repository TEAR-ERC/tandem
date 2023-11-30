#!/usr/bin/env python3
'''
Automatically write scripts perturbation (ultimate compilation)
By Jeena Yun
Last modification: 2023.11.21.
'''
import numpy as np
import os
import change_params
import setup_shortcut
ch = change_params.variate()
sc = setup_shortcut.setups()

model_n = 'perturb_stress'
output_branch_n = 'reference'
write_on = True

# Perturbation related parameters
seissol_model_n = 'dipping_slow'
strike = 350
fcoeff = 0.4
dt = 0.01
target_sys_evID = 31        # Between 0 to 63
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
    fshell = open(fname_shell,'w')
    fshell.write('# Run perturbation period\n')
    import subprocess
    # Syntax e.g., python /home/jyun/Tandem/make_closer.py perturb_stress reference 6 --write_on
    subprocess.run(["python","/home/jyun/Tandem/make_closer.py",model_n,output_branch_n,target_sys_evID,"--write_on"],shell=True)
    subprocess.run(["/home/jyun/Tandem/match_time_run_tandem.sh"],shell=True)

# 1. Load event outputs
from cumslip_compute import analyze_events
if 'v6_ab2_Dc2' in output_save_dir:
    Vths = 1e-1
    intv = 0.15
elif 'perturb_stress' in output_save_dir:
    Vths = 2e-1
    intv = 0.15
else:
    Vths = 1e-2
    intv = 0.
Vlb = 0
dt_interm = 0
cuttime = 0
rths = 10
dt_creep = 2*ch.yr2sec
dt_coseismic = 0.5

cumslip_outputs = np.load('%s/cumslip_outputs_Vths_%1.0e_srvar_%03d_rths_%d_tcreep_%d_tseis_%02d.npy'%(output_save_dir,Vths,intv*100,rths,dt_creep/ch.yr2sec,dt_coseismic*10),allow_pickle=True)
spin_up_idx = np.load('%s/spin_up_idx_Vths_%1.0e_srvar_%03d_rths_%d_tcreep_%d_tseis_%02d.npy'%(output_save_dir,Vths,intv*100,rths,dt_creep/ch.yr2sec,dt_coseismic*10))
tstart,tend,evdep = cumslip_outputs[0][0],cumslip_outputs[0][1],cumslip_outputs[1][1]
system_wide = analyze_events(cumslip_outputs,rths)[2]
idx = system_wide[system_wide>=spin_up_idx][target_sys_evID]

# 2. Extract exact init time at the time of the checkpoint
from read_outputs import load_checkpoint_info
ckp_dat = load_checkpoint_info(matched_save_dir)
stepnum = int(np.sort(ckp_dat,axis=0)[-1][0])
init_time = np.sort(ckp_dat,axis=0)[-1][-1]
maxstep = int(stepnum + 15/dt + 1)
print('System-size Event Index = %d; Event %d; Hypocenter Depth: %1.2f [km]'%(target_sys_evID,idx,evdep[idx]))
print('Difference in time between real foreshock and the checkpoint: %1.4f s'%(tstart[idx]-time_diff_in_sec - init_time))
print('==========================================================================')

if write_on:
    # 3. Generate Lua scenario
    scenarios = np.genfromtxt('perturb_stress/scenario_perturb.lua',skip_header=4,delimiter=' =',usecols=0,dtype='str')
    scenario_name = '%s%d_%d'%(sc.model_code(seissol_model_n),strike,stepnum)
    if scenario_name in scenarios: # Check if the scenario is already there
        print('%s already exists - skip writing lua file'%(scenario_name))
    else:
        flua = open(fname_lua,'a')
        flua.write('\n%s = ridgecrest54.new{model_n=\'%s\',strike=%d,fcoeff=%1.1f,dt=%1.2f,init_time=%1.18e}'%(scenario_name,seissol_model_n,strike,fcoeff,dt,init_time))
        flua.close()

    # 4. Generate parameter file
    fpar = open(fname_toml,'w')
    fpar.write('final_time = 157680000000\n')
    fpar.write('mesh_file = "ridgecrest_hf25.msh"\n')
    fpar.write('mode = "QDGreen"\n')
    fpar.write('type = "poisson"\n')
    fpar.write('lib = "scenario_perturb.lua"\n')
    fpar.write('scenario = "%s"\n'%(scenario_name))
    fpar.write('ref_normal = [-1, 0]\n')
    fpar.write('boundary_linear = true\n\n')

    fpar.write('gf_checkpoint_prefix = "/home/jyun/Tandem/perturb_stress/greensfun/hf25"\n\n')

    fpar.write('[fault_probe_output]\n')
    fpar.write('prefix = "faultp_"\n')
    fpar.write('t_max = 0.009\n')
    sc.write_faultprobe_loc(ch.extract_prefix(output_save_dir),fpar,dmin=0.02,dmax=1.,dip=90,write_on=write_on)

    fpar.write('[domain_probe_output]\n')
    fpar.write('prefix = "domainp_"\n')
    fpar.write('t_max = 0.009\n')
    sc.write_domainprobe_loc(fpar,xmax=100,dx=5,write_on=write_on)
    fpar.close()

    # 5. Generate a shell file to operate everything
    fshell = open(fname_shell,'w')
    fshell.write('#!/bin/bash\n')
    # 5.0. Run a safety check
    if not os.path.exists('/export/dump/jyun/%s/match%d/outputs/checkpoint/step%d'%(model_n,target_sys_evID,stepnum)):
        fshell.close()
        FileExistsError('No such path /export/dump/jyun/%s/match%d/outputs/checkpoint/step%d'%(model_n,target_sys_evID,stepnum))
    else:
        fshell.write('# Define some useful functions\n')
        fshell.write('process_output_full() { echo "/export/dump/jyun/$1/$2"; '
                     'mkdir -p "/export/dump/jyun/$1/$2"; '
                     'mv "/export/dump/jyun/$1/outputs_$2" "/export/dump/jyun/$1/$2"; '
                     'mv "/export/dump/jyun/$1/$2/outputs_$2" "/export/dump/jyun/$1/$2/outputs"; '
                     'python /home/jyun/Tandem/get_plots.py /export/dump/jyun/$1/$2 -c; }\n')
        fshell.write('read_time_full() { /home/jyun/Tandem/read_time_recursive "/export/dump/jyun/$1/$2"; }\n\n')
    
    # 5.1. Run perturbation period
    fshell.write('# Run perturbation period\n')
    fshell.write('model_n=%s\n'%(model_n))
    fshell.write('branch_n=%s\n'%(run_branch_n))
    fshell.write('tdhome=/home/jyun/Tandem\n')
    fshell.write('setup_dir=$tdhome/$model_n\n')
    fshell.write('mkdir -p /export/dump/jyun/$model_n\n')
    fshell.write('cd /export/dump/jyun/$model_n\n')
    fshell.write('mkdir -p outputs_$branch_n\n')
    fshell.write('cd outputs_$branch_n\n')
    fshell.write('echo "Tandem running in a directory: " $setup_dir\n')
    fshell.write('mpiexec -bind-to core -n %d tandem $setup_dir/%s --petsc -ts_checkpoint_load ../match%d/outputs/checkpoint/step%d '
                '-ts_adapt_type none -ts_dt %.2f -ts_max_steps %d '
                '-ts_checkpoint_freq_step 1 -ts_checkpoint_freq_physical_time %d -ts_checkpoint_freq_cputime %d '
                '-options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log\n\n'\
                %(n_node,fname_toml.split('/')[-1],target_sys_evID,stepnum,dt,maxstep,ckp_freq_ptime,ckp_freq_cputime))

    # 5.2. Process the perturation period output and generate checkpoint time info
    fshell.write('# Process the perturbation period output, change the directory name, and generate checkpoint time info\n')
    fshell.write('conda activate ridgecrest\n')
    fshell.write('process_output_full $model_n $branch_n\n')
    fshell.write('read_time_full $model_n $branch_n\n\n')

    # 5.3. Run after perturbation period
    if not os.path.exists('/export/dump/jyun/%s/%s/outputs/checkpoint/step%d'%(model_n,run_branch_n,maxstep)):
        fshell.close()
        FileExistsError('No such path /export/dump/jyun/%s/%s/outputs/checkpoint/step%d'%(model_n,run_branch_n,maxstep))
    fshell.write('# Run after perturbation period\n')
    fshell.write('branch_n=after_%s\n'%(run_branch_n))
    fshell.write('cd /export/dump/jyun/$model_n\n')
    fshell.write('mkdir -p outputs_$branch_n\n')
    fshell.write('cd outputs_$branch_n\n')
    fshell.write('echo "Tandem running in a directory: " $setup_dir\n')
    fshell.write('mpiexec -bind-to core -n %d tandem $setup_dir/parameters_reference.toml --petsc -ts_checkpoint_load ../%s/outputs/checkpoint/step%d '
                '-ts_adapt_type basic -ts_max_steps %d '
                '-ts_checkpoint_freq_step %d -ts_checkpoint_freq_physical_time %d -ts_checkpoint_freq_cputime %d '
                '-options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log\n\n'\
                %(n_node,run_branch_n,maxstep,maxstep+dstep,ckp_freq_step,ckp_freq_ptime,ckp_freq_cputime))
    
    # 5.4. Finally, process the after perturation period output
    fshell.write('# Process the after perturbation period output, change the directory name, and generate checkpoint time info\n')
    fshell.write('process_output_full $model_n $branch_n\n')
    fshell.write('read_time_full $model_n $branch_n\n\n')
    fshell.close()
