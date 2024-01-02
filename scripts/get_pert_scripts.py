#!/usr/bin/env python3
'''
Automatically write scripts perturbation (ultimate compilation) - argparse ver.
By Jeena Yun
Last modification: 2023.12.01.
'''
import numpy as np
import argparse
import os
import change_params
import setup_shortcut
ch = change_params.variate()
sc = setup_shortcut.setups()

# ---------------------- Set input parameters
parser = argparse.ArgumentParser()
parser.add_argument("model_n",type=str.lower,help=": Name of big group of the model")
parser.add_argument("output_branch_n",type=str.lower,help=": Name of the branch where outputs reside")
parser.add_argument("target_sys_evID",type=int,help=": System-wide event index")
parser.add_argument("seissol_model_n",type=str.lower,help=": Name of the SeisSol model")
parser.add_argument("strike",type=int,help=": Strike of the SeisSol model")
parser.add_argument("--write_on",action="store_true",help=": Write lua, toml, and shell script?",default=False)
parser.add_argument("--dt",type=float,help=": If given, time interval of the SeisSol output",default=0.01)
parser.add_argument("--n_node",type=int,help=": Number of nodes for tandem simulation",default=40)
parser.add_argument("--time_diff_in_sec",type=float,help=": If given, time difference between the perturbation point and the mainshock",default=58320)
parser.add_argument("--ckp_freq_step",type=int,help=": If given, step interval for checkpointing",default=50)
parser.add_argument("--ckp_freq_ptime",type=int,help=": If given, physical time interval for checkpointing",default=1000000000)
parser.add_argument("--ckp_freq_cputime",type=int,help=": If given, CPU time interval for checkpointing",default=60)
parser.add_argument("--dstep",type=int,help=": If given, number of steps to run for the after perturbation model",default=50000)
parser.add_argument("--multiply",type=int,help=": If given, the integer is multiplied to the given perturbation model",default=1)
# parser.add_argument("--background_on",action="store_true",help=": If given, the process will run in a background",default=False)
args = parser.parse_args()

fcoeff = 0.4 # Perturbation related parameters

# -------- 0. Generate multiplied model if given
if args.multiply != 1:
    print('MULTIPLY ON: SeisSol model %s multiplied by %d'%(args.seissol_model_n,args.multiply))
    seissol_model_n = args.seissol_model_n + '_X%d'%(args.multiply)
    if args.write_on and not os.path.exists("/home/jyun/Tandem/%s/seissol_outputs/ssaf_%s_Pn_pert_mu%02d_%d.dat"%(args.model_n,seissol_model_n,int(fcoeff*10),args.strike)):
        print('XXX Writing multiplied model %s XXX'%(seissol_model_n))
        delPn = np.loadtxt("/home/jyun/Tandem/%s/seissol_outputs/ssaf_%s_Pn_pert_mu%02d_%d.dat"%(args.model_n,args.seissol_model_n,int(fcoeff*10),args.strike))
        delTs = np.loadtxt("/home/jyun/Tandem/%s/seissol_outputs/ssaf_%s_Ts_pert_mu%02d_%d.dat"%(args.model_n,args.seissol_model_n,int(fcoeff*10),args.strike))
        depth_range = np.loadtxt("/home/jyun/Tandem/%s/seissol_outputs/ssaf_%s_dep_stress_pert_mu%02d_%d.dat"%(args.model_n,args.seissol_model_n,int(fcoeff*10),args.strike))
        np.savetxt("/home/jyun/Tandem/%s/seissol_outputs/ssaf_%s_Pn_pert_mu%02d_%d.dat"%(args.model_n,seissol_model_n,int(fcoeff*10),args.strike),X=delPn*args.multiply,fmt='%.20f',delimiter='\t',newline='\n')
        np.savetxt("/home/jyun/Tandem/%s/seissol_outputs/ssaf_%s_Ts_pert_mu%02d_%d.dat"%(args.model_n,seissol_model_n,int(fcoeff*10),args.strike),X=delTs*args.multiply,fmt='%.20f',delimiter='\t',newline='\n')
        np.savetxt("/home/jyun/Tandem/%s/seissol_outputs/ssaf_%s_dep_stress_pert_mu%02d_%d.dat"%(args.model_n,seissol_model_n,int(fcoeff*10),args.strike),X=depth_range,fmt='%.20f',newline='\n')
else:
    seissol_model_n = args.seissol_model_n

# Set path and file names
output_save_dir = '/export/dump/jyun/%s/%s'%(args.model_n,args.output_branch_n)
matched_save_dir = '/export/dump/jyun/%s/match%d'%(args.model_n,args.target_sys_evID)
run_branch_n = 'pert%d_%s%d'%(args.target_sys_evID,sc.model_code(seissol_model_n),args.strike)
fname_lua = '/home/jyun/Tandem/%s/scenario_perturb.lua'%(args.model_n)
fname_toml = '/home/jyun/Tandem/%s/parameters_perturb_scenario.toml'%(args.model_n)
# fname_toml_after = '/home/jyun/Tandem/%s/parameters_after_perturb.toml'%(args.model_n)
fname_shell = '/home/jyun/Tandem/routine_perturb.sh'

print('====================== Summary of Input Parameters =======================')
print('output_save_dir = %s'%(output_save_dir))
print('matched_save_dir = %s'%(matched_save_dir))
print('run_branch_n = %s'%(run_branch_n))
print('seissol_model_n = %s'%(seissol_model_n))
print('strike = %d'%(args.strike))
print('fcoeff = %1.1f'%(fcoeff))
print('dt = %1.2f'%(args.dt))
print('n_node = %d'%(args.n_node))
print('ckp_freq_step = %d'%(args.ckp_freq_step))
print('ckp_freq_ptime = %d'%(args.ckp_freq_ptime))
print('ckp_freq_cputime = %d'%(args.ckp_freq_cputime))
print('dstep = %d'%(args.dstep))

# -------- 1. Check if the matched checkpoint exists - if not, run make_closer
if not os.path.exists(matched_save_dir):
    print('##### %s NOT FOUND - Need time matching #####'%(matched_save_dir))
    if args.write_on:
        import subprocess
        print('XXX Writing file %s XXX'%(fname_shell))
        fshell = open(fname_shell,'w')
        fshell.write('# Run perturbation period\n')
        # Syntax e.g., python /home/jyun/Tandem/make_closer.py perturb_stress reference 6 --write_on
        subprocess.run(["python","/home/jyun/Tandem/make_closer.py",args.model_n,args.output_branch_n,"%d"%(args.target_sys_evID),"--write_on"])
        subprocess.run(["/home/jyun/Tandem/match_time_run_tandem.sh"],shell=True)

# -------- 2. Load event outputs
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

# -------- 3. Extract exact init time at the time of the checkpoint
from read_outputs import load_checkpoint_info
ckp_dat = load_checkpoint_info(matched_save_dir)
stepnum = int(np.sort(ckp_dat,axis=0)[-1][0])
init_time = np.sort(ckp_dat,axis=0)[-1][-1]
maxstep = int(stepnum + 15/args.dt + 1)
print('System-size Event Index = %d; Event %d; Hypocenter Depth: %1.2f [km]'%(args.target_sys_evID,idx,evdep[idx]))
print('Difference in time between real foreshock and the checkpoint: %1.4f s'%(tstart[idx]-args.time_diff_in_sec - init_time))
print('==========================================================================')

if args.write_on:
    # -------- 4. Generate Lua scenario
    scenarios = np.genfromtxt('perturb_stress/scenario_perturb.lua',skip_header=4,delimiter=' =',usecols=0,dtype='str')
    scenario_name = '%s%d_%d'%(sc.model_code(seissol_model_n),args.strike,stepnum)
    if scenario_name in scenarios: # Check if the BEFORE scenario is already there
        print('%s already exists - skip writing lua file'%(scenario_name))
    else:
        print('XXX Writing file %s XXX'%(fname_lua))
        flua = open(fname_lua,'a')
        # flua.write('\n%s = ridgecrest54.new{model_n=\'%s\',strike=%d,fcoeff=%1.1f,dt=%1.2f,init_time=%1.18e,fname_final_stress=\'\'}'%(scenario_name,seissol_model_n,args.strike,fcoeff,args.dt,init_time))
        flua.write('\n%s = ridgecrest54.new{model_n=\'%s\',strike=%d,fcoeff=%1.1f,dt=%1.2f,init_time=%1.18e}'%(scenario_name,seissol_model_n,args.strike,fcoeff,args.dt,init_time))
        flua.close()    
    # if 'after_'+scenario_name in scenarios: # Check if the AFTER scenario is already there
    #     print('%s already exists - skip writing lua file'%('after_'+scenario_name))
    # else:
    #     flua = open(fname_lua,'a')
    #     flua.write('\nafter_%s = ridgecrest54.new{model_n=\'%s\',strike=%d,fcoeff=%1.1f,dt=%1.2f,init_time=-1,fname_final_stress=\'final_stress_%s\'}'%(scenario_name,seissol_model_n,args.strike,fcoeff,args.dt,run_branch_n))
    #     flua.close()

    # -------- 5. Generate parameter file
    # 5.1. Before perturbation
    print('XXX Writing file %s XXX'%(fname_toml))
    fpar = open(fname_toml,'w')
    fpar.write('final_time = 157680000000\n')
    fpar.write('mesh_file = "ridgecrest_hf25.msh"\n')
    fpar.write('mode = "QDGreen"\n')
    fpar.write('type = "poisson"\n')
    fpar.write('lib = "scenario_perturb.lua"\n')
    fpar.write('scenario = "%s"\n'%(scenario_name))
    fpar.write('ref_normal = [-1, 0]\n')
    fpar.write('boundary_linear = true\n\n')

    fpar.write('gf_checkpoint_prefix = "/export/dump/jyun/GreensFunctions/ridgecrest_hf25"\n\n')

    fpar.write('[fault_probe_output]\n')
    fpar.write('prefix = "faultp_"\n')
    fpar.write('t_max = 0.009\n')
    sc.write_faultprobe_loc(ch.extract_prefix(output_save_dir),fpar,dmin=0.02,dmax=1.,dip=90,write_on=args.write_on)

    fpar.write('[domain_probe_output]\n')
    fpar.write('prefix = "domainp_"\n')
    fpar.write('t_max = 0.009\n')
    sc.write_domainprobe_loc(fpar,xmax=100,dx=5,write_on=args.write_on)
    fpar.close()

    # # 5.2. After perturbation
    # fpar = open(fname_toml_after,'w')
    # fpar.write('final_time = 157680000000\n')
    # fpar.write('mesh_file = "ridgecrest_hf25.msh"\n')
    # fpar.write('mode = "QDGreen"\n')
    # fpar.write('type = "poisson"\n')
    # fpar.write('lib = "scenario_perturb.lua"\n')
    # fpar.write('scenario = "after_%s"\n'%(scenario_name))
    # fpar.write('ref_normal = [-1, 0]\n')
    # fpar.write('boundary_linear = true\n\n')

    # fpar.write('gf_checkpoint_prefix = "/export/dump/jyun/GreensFunctions/ridgecrest_hf25"\n\n')

    # fpar.write('[fault_probe_output]\n')
    # fpar.write('prefix = "faultp_"\n')
    # fpar.write('t_max = 0.009\n')
    # sc.write_faultprobe_loc(ch.extract_prefix(output_save_dir),fpar,dmin=0.02,dmax=1.,dip=90,write_on=args.write_on)

    # fpar.write('[domain_probe_output]\n')
    # fpar.write('prefix = "domainp_"\n')
    # fpar.write('t_max = 0.009\n')
    # sc.write_domainprobe_loc(fpar,xmax=100,dx=5,write_on=args.write_on)
    # fpar.close()

    # -------- 6. Generate a shell file to operate everything
    print('XXX Writing file %s XXX'%(fname_shell))
    fshell = open(fname_shell,'w')
    fshell.write('#!/bin/bash\n')
    # 6.0. Define some useful functions
    fshell.write('# Define some useful functions\n')
    fshell.write('process_output_full() { echo "/export/dump/jyun/$1/$2"; '
                    'mkdir -p "/export/dump/jyun/$1/$2"; '
                    'mv "/export/dump/jyun/$1/outputs_$2" "/export/dump/jyun/$1/$2"; '
                    'mv "/export/dump/jyun/$1/$2/outputs_$2" "/export/dump/jyun/$1/$2/outputs"; '
                    'python /home/jyun/Tandem/get_plots.py /export/dump/jyun/$1/$2 -c; }\n')
    fshell.write('read_time_full() { /home/jyun/Tandem/read_time_recursive "/export/dump/jyun/$1/$2"; }\n')
    fshell.write('existckp_full() { ls "/export/dump/jyun/$1/$2"; }\n\n')
    
    # 6.1. Run the perturbation period
    fshell.write('# Run the perturbation period\n')
    fshell.write('model_n=%s\n'%(args.model_n))
    # fshell.write('branch_n=%s\n'%(run_branch_n))
    fshell.write('tdhome=/home/jyun/Tandem\n')
    fshell.write('setup_dir=$tdhome/$model_n\n')
    # fshell.write('mkdir -p /export/dump/jyun/$model_n\n')
    # fshell.write('cd /export/dump/jyun/$model_n\n')
    # fshell.write('mkdir -p outputs_$branch_n\n')
    # fshell.write('cd outputs_$branch_n\n')
    # fshell.write('echo "Tandem running in a directory: " $setup_dir\n\n')

    # # 6.1.0. Run a safety check
    # fshell.write('# Safety check\n')
    # # fshell.write('python /home/jyun/Tandem/check_exist.py /export/dump/jyun/%s/match%d/outputs/checkpoint/step%d\n\n'%(args.model_n,args.target_sys_evID,stepnum))
    # fshell.write('existckp_full $model_n match%d/outputs/checkpoint/step%d\n\n'%(args.target_sys_evID,stepnum))

    # # 6.1.2. If safe, proceed
    # fshell.write('# If safe, proceed\n')
    # fshell.write('mpiexec -bind-to core -n %d tandem $setup_dir/%s --petsc -ts_checkpoint_load ../match%d/outputs/checkpoint/step%d '
    #             '-ts_adapt_type none -ts_dt %.2f -ts_max_steps %d '
    #             '-ts_checkpoint_freq_step 1 -ts_checkpoint_freq_physical_time %d -ts_checkpoint_freq_cputime %d '
    #             '-options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log\n\n'\
    #             %(args.n_node,fname_toml.split('/')[-1],args.target_sys_evID,stepnum,args.dt,maxstep,args.ckp_freq_ptime,args.ckp_freq_cputime))
    # # fshell.write('mpiexec -bind-to core -n %d tandem $setup_dir/%s --petsc -ts_checkpoint_load ../match%d/outputs/checkpoint/step%d '
    # #             '-ts_adapt_dt_max %.2f -ts_max_steps %d '
    # #             '-ts_checkpoint_freq_step 1 -ts_checkpoint_freq_physical_time %d -ts_checkpoint_freq_cputime %d '
    # #             '-options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log\n\n'\
    # #             %(args.n_node,fname_toml.split('/')[-1],args.target_sys_evID,stepnum,args.dt,maxstep,args.ckp_freq_ptime,args.ckp_freq_cputime))

    # # 6.2. Process the perturation period output and generate checkpoint time info
    # fshell.write('# Process the perturbation period output, change the directory name, and generate checkpoint time info\n')
    # fshell.write('process_output_full $model_n $branch_n\n')
    # fshell.write('read_time_full $model_n $branch_n\n')
    # fshell.write('python extract_final.py $model_n %s --save_on\n\n'%(run_branch_n))
    
    # 6.3. Run the after perturbation period
    fshell.write('# Run the after perturbation period\n')
    fshell.write('branch_n=after_%s\n'%(run_branch_n))
    fshell.write('cd /export/dump/jyun/$model_n\n')
    fshell.write('mkdir -p outputs_$branch_n\n')
    fshell.write('cd outputs_$branch_n\n')
    fshell.write('echo "Tandem running in a directory: " $setup_dir\n\n')

    # 6.3.0. Run a safety check
    fshell.write('# Safety check\n')
    # fshell.write('python /home/jyun/Tandem/check_exist.py /export/dump/jyun/%s/%s/outputs/checkpoint/step%d\n\n'%(args.model_n,run_branch_n,maxstep))
    fshell.write('existckp_full $model_n %s/outputs/checkpoint/step%d\n\n'%(run_branch_n,maxstep))
    
    # 6.3.1. If safe, proceed
    fshell.write('# If safe, proceed\n')
    fshell.write('mpiexec -bind-to core -n %d tandem $setup_dir/%s --petsc -ts_checkpoint_load ../%s/outputs/checkpoint/step%d '
                '-ts_adapt_type basic -ts_max_steps %d '
                '-ts_checkpoint_freq_step %d -ts_checkpoint_freq_physical_time %d -ts_checkpoint_freq_cputime %d '
                '-options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log\n\n'\
                %(args.n_node,fname_toml.split('/')[-1],run_branch_n,maxstep,maxstep+args.dstep,args.ckp_freq_step,args.ckp_freq_ptime,args.ckp_freq_cputime))
    # fshell.write('mpiexec -bind-to core -n %d tandem $setup_dir/%s --petsc -ts_checkpoint_load ../%s/outputs/checkpoint/step%d '
    #             '-ts_adapt_type basic -ts_max_steps %d '
    #             '-ts_checkpoint_freq_step %d -ts_checkpoint_freq_physical_time %d -ts_checkpoint_freq_cputime %d '
    #             '-options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log\n\n'\
    #             %(args.n_node,fname_toml_after.split('/')[-1],run_branch_n,maxstep,maxstep+args.dstep,args.ckp_freq_step,args.ckp_freq_ptime,args.ckp_freq_cputime))
    # fshell.write('mpiexec -bind-to core -n %d tandem $setup_dir/parameters_reference.toml --petsc -ts_checkpoint_load ../%s/outputs/checkpoint/step%d '
    #             '-ts_max_steps %d '
    #             '-ts_checkpoint_freq_step %d -ts_checkpoint_freq_physical_time %d -ts_checkpoint_freq_cputime %d '
    #             '-options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log\n\n'\
    #             %(args.n_node,run_branch_n,maxstep,maxstep+args.dstep,args.ckp_freq_step,args.ckp_freq_ptime,args.ckp_freq_cputime))
    
    # 6.4. Finally, process the after perturation period output
    fshell.write('# Finally, Process the after perturbation period output, change the directory name, and generate checkpoint time info\n')
    fshell.write('process_output_full $model_n $branch_n\n')
    fshell.write('read_time_full $model_n $branch_n\n\n')
    fshell.close()
