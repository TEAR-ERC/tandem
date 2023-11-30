
#!/usr/bin/env python3
'''
A script to match the time of nearest checkpoint to be the nearest foreshock point
By Jeena Yun
Last modification: 2023.11.21.
'''
import argparse
import os
import numpy as np
import change_params
import setup_shortcut
ch = change_params.variate()
sc = setup_shortcut.setups()

# ---------------------- Set input parameters
parser = argparse.ArgumentParser()
parser.add_argument("model_n",type=str.lower,help=": Name of big group of the model")
parser.add_argument("output_branch_n",type=str.lower,help=": Name of the branch where outputs reside")
parser.add_argument("target_sys_evID",type=int,help=": System-wide event index")
parser.add_argument("--write_on",action="store_true",help=": Write lua, toml, and shell script?",default=False)
parser.add_argument("--n_node",type=int,help=": Number of nodes for tandem simulation",default=40)
parser.add_argument("--time_diff_in_sec",type=float,help=": If given, time difference between the perturbation point and the mainshock",default=58320)
parser.add_argument("--ckp_freq_ptime",type=int,help=": If given, physical time interval for checkpointing",default=1000000000)
parser.add_argument("--ckp_freq_cputime",type=int,help=": If given, CPU time interval for checkpointing",default=60)
parser.add_argument("--background_on",action="store_true",help=": If given, the process will run in a background",default=False)
args = parser.parse_args()

if args.background_on:
    fin = '&'
else:
    fin = ''

# ---------------------- 
# Path and file names
run_branch_n = 'match%d'%(args.target_sys_evID)
output_save_dir = '/export/dump/jyun/%s/%s'%(args.model_n,args.output_branch_n)
fname_toml = '/home/jyun/Tandem/%s/parameters_match_time.toml'%(args.model_n)
fname_shell = '/home/jyun/Tandem/match_time_run_tandem.sh'

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
idx = system_wide[system_wide>=spin_up_idx][args.target_sys_evID]

# 2. Extract exact init time at the time of the checkpoint
from read_outputs import load_checkpoint_info
ckp_dat = load_checkpoint_info(output_save_dir)
T_foreshock = tstart[idx]-args.time_diff_in_sec
ckp_idx = np.where(ckp_dat[:,1]>=T_foreshock)[0][0]-1
stepnum = int(ckp_dat[ckp_idx][0])
init_time = ckp_dat[ckp_idx][1]
print('Event %d; System-size Event Index %d; Hypocenter Depth: %1.2f [km]'%(idx,args.target_sys_evID,evdep[idx]))
print('Difference in time between real foreshock and the checkpoint: %1.4f s'%(T_foreshock - init_time))

print('Spinned-up system-size events index: %d / Event index: %d'%(args.target_sys_evID,idx))
print('Nearest checkpoint #: %d'%(stepnum))
print('Nearest checkpoint time: %1.18f'%(init_time))
print('Foreshock start time: %1.18f'%(T_foreshock))

if args.write_on:
    # 3. Generate parameter file
    fpar = open(fname_toml,'w')
    fpar.write('final_time = %1.18f\n'%(T_foreshock))
    fpar.write('mesh_file = "ridgecrest_hf25.msh"\n')
    fpar.write('mode = "QDGreen"\n')
    fpar.write('type = "poisson"\n')
    fpar.write('lib = "matfric_Fourier_main_reference.lua"\n')
    fpar.write('scenario = "bp1_sym"\n')
    fpar.write('ref_normal = [-1, 0]\n')
    fpar.write('boundary_linear = true\n\n')

    fpar.write('gf_checkpoint_prefix = "/home/jyun/Tandem/perturb_stress/greensfun/hf25"\n\n')

    fpar.write('[fault_probe_output]\n')
    fpar.write('prefix = "faultp_"\n')
    fpar.write('t_max = 3600\n')
    sc.write_faultprobe_loc(ch.extract_prefix(output_save_dir),fpar,dmin=0.02,dmax=1.,dip=90,write_on=args.write_on)
    fpar.close()

    # 4. Generate a shell file for execution
    fshell = open(fname_shell,'w')
    fshell.write('#!/bin/bash\n')
    # 4.0. Run a safety check
    if not os.path.exists('/export/dump/jyun/%s/%s/outputs/checkpoint/step%d'%(args.model_n,args.output_branch_n,stepnum)):
        fshell.close()
        FileExistsError('No such path /export/dump/jyun/%s/%s/outputs/checkpoint/step%d'%(args.model_n,args.output_branch_n,stepnum))
    else:
        fshell.write('# Define some useful functions\n')
        fshell.write('process_output_full() { echo "/export/dump/jyun/$1/$2"; '
                     'mkdir -p "/export/dump/jyun/$1/$2"; '
                     'mv "/export/dump/jyun/$1/outputs_$2" "/export/dump/jyun/$1/$2"; '
                     'mv "/export/dump/jyun/$1/$2/outputs_$2" "/export/dump/jyun/$1/$2/outputs"; '
                     'python /home/jyun/Tandem/get_plots.py /export/dump/jyun/$1/$2 -c; }\n')
        fshell.write('read_time_full() { /home/jyun/Tandem/read_time_recursive "/export/dump/jyun/$1/$2"; }\n\n')

    # 4.1. Run tandem model
    fshell.write('model_n=%s\n'%(args.model_n))
    fshell.write('branch_n=%s\n'%(run_branch_n))
    fshell.write('tdhome=/home/jyun/Tandem\n')
    fshell.write('setup_dir=$tdhome/$model_n\n')
    fshell.write('rm -rf $setup_dir/*profile_$branch_n\n')
    fshell.write('mkdir -p /export/dump/jyun/$model_n\n')
    fshell.write('cd /export/dump/jyun/$model_n\n')
    fshell.write('mkdir -p outputs_$branch_n\n')
    fshell.write('cd outputs_$branch_n\n')
    fshell.write('echo "Tandem running in a directory: " $setup_dir\n')
    fshell.write('mpiexec -bind-to core -n %d tandem $setup_dir/%s --petsc -ts_checkpoint_load ../%s/outputs/checkpoint/step%d '
                '-ts_checkpoint_freq_step 1 -ts_checkpoint_freq_physical_time %d -ts_checkpoint_freq_cputime %d '
                '-options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log %s\n\n'\
                %(args.n_node,fname_toml.split('/')[-1],args.output_branch_n,stepnum,args.ckp_freq_ptime,args.ckp_freq_cputime,fin))

    # 4.2. Process the perturation period output
    fshell.write('# Process the output, change the directory name, and generate checkpoint time info\n')
    fshell.write('conda activate ridgecrest\n')
    fshell.write('process_output_full $model_n $branch_n\n')
    fshell.write('read_time_full $model_n $branch_n\n\n')
    fshell.close()
