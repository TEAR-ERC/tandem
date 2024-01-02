#!/bin/bash
model_n=perturb_stress
branch_n=hf10_reference
tdhome=/home/jyun/Tandem
setup_dir=$tdhome/$model_n
rm -rf $setup_dir/*profile_$branch_n
mkdir -p /export/dump/jyun/$model_n
cd /export/dump/jyun/$model_n

# ####################################
# # DON'T FORGET TO REMOVE THIS LINE #
# echo "Remove directory: " /export/dump/jyun/$model_n/$branch_n
# rm -rf $branch_n
# ####################################

echo "Create directory: " /export/dump/jyun/$model_n/outputs_$branch_n
mkdir -p outputs_$branch_n
cd outputs_$branch_n
# mkdir -p domain_output
echo "Tandem running in a directory: " $setup_dir

# --- With checkpointing every certain time steps, physical time & cpu time
# mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_reference.toml --petsc -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor -ts_max_steps 70000 -ts_checkpoint_freq_step 50 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 60 -ts_checkpoint_path ckp_$branch_n > $setup_dir/messages_$branch_n.log &
# mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_reference.toml --petsc -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor -ts_max_steps 70000 -ts_checkpoint_freq_step 50 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 60 -ts_checkpoint_path checkpoint > $setup_dir/messages_$branch_n.log &
mpiexec -bind-to core -n 60 tandem $setup_dir/parameters_hf10_reference.toml --petsc -ts_checkpoint_freq_step 50 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 60 -ts_checkpoint_path checkpoint -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &
# mpiexec -bind-to core -n 60 tandem $setup_dir/p10Dc2.toml --petsc -ts_checkpoint_freq_step 50 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 60 -ts_checkpoint_path checkpoint -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &

# --- Load checkpointing
# mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_reference.toml --petsc -ts_checkpoint_load ../reference/outputs/checkpoint/step1555650 -ts_max_steps 1555660 -ts_checkpoint_freq_step 50 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 120 -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &
# With a fixed time step:
# mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_reference.toml --petsc -ts_checkpoint_load ../reference/outputs/checkpoint/step1555650 -ts_adapt_type none -ts_dt 0.01 -ts_max_steps 1555660 -ts_checkpoint_freq_step 50 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 120 -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &

# mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_nopert.toml --petsc -ts_checkpoint_load ../match31/outputs/checkpoint/step4915642 -ts_max_steps 4915672 -ts_checkpoint_freq_step 1000000 -ts_checkpoint_freq_cputime 10000000000 -ts_checkpoint_freq_physical_time 1000000000000 -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &
# mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_reference.toml --petsc -ts_checkpoint_load ../diverge_test_ref1/outputs/checkpoint/step4650 -ts_max_steps 5000 -ts_checkpoint_freq_step 1000000 -ts_checkpoint_freq_cputime 10000000000 -ts_checkpoint_freq_physical_time 1000000000000 -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &

# --- After perturbation
# mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_reference.toml --petsc -ts_checkpoint_load ../pert8/outputs/checkpoint/step1516951 -ts_adapt_type basic -ts_max_steps 1566951 -ts_checkpoint_freq_step 50 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 120 -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &

# --- Turn off checkpointing
# mpiexec -bind-to core -n 60 tandem $setup_dir/build_GF.toml --petsc -ts_checkpoint_storage_type none -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &

# --- Stale) Base command
# mpiexec -bind-to core -n 5 tandem $setup_dir/parameters_$branch_n.toml --petsc -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &
# --- Stale) MTMOD model
# mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_reference.toml --petsc -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor -ts_adapt_dt_max 5e5 > $setup_dir/messages_$branch_n.log &