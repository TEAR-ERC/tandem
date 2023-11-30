#!/bin/bash
# Define some useful functions
process_output_full() { echo "/export/dump/jyun/$1/$2"; mkdir -p "/export/dump/jyun/$1/$2"; mv "/export/dump/jyun/$1/outputs_$2" "/export/dump/jyun/$1/$2"; mv "/export/dump/jyun/$1/$2/outputs_$2" "/export/dump/jyun/$1/$2/outputs"; python /home/jyun/Tandem/get_plots.py /export/dump/jyun/$1/$2 -c; }
read_time_full() { /home/jyun/Tandem/read_time_recursive "/export/dump/jyun/$1/$2"; }

# Run the perturbation period
model_n=perturb_stress
branch_n=pert8_vsX30_340
tdhome=/home/jyun/Tandem
setup_dir=$tdhome/$model_n
mkdir -p /export/dump/jyun/$model_n
cd /export/dump/jyun/$model_n
mkdir -p outputs_$branch_n
cd outputs_$branch_n
echo "Tandem running in a directory: " $setup_dir

# Safety check
python /home/jyun/Tandem/check_exist.py /export/dump/jyun/perturb_stress/match8/outputs/checkpoint/step1515453

# If safe, proceed
mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_perturb_scenario.toml --petsc -ts_checkpoint_load ../match8/outputs/checkpoint/step1515453 -ts_adapt_type none -ts_dt 0.01 -ts_max_steps 1516954 -ts_checkpoint_freq_step 1 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 60 -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log

# Process the perturbation period output, change the directory name, and generate checkpoint time info
process_output_full $model_n $branch_n
read_time_full $model_n $branch_n

# Run the after perturbation period
branch_n=after_pert8_vsX30_340
cd /export/dump/jyun/$model_n
mkdir -p outputs_$branch_n
cd outputs_$branch_n
echo "Tandem running in a directory: " $setup_dir

# Safety check
python /home/jyun/Tandem/check_exist.py /export/dump/jyun/perturb_stress/pert8_vsX30_340/outputs/checkpoint/step1516954

# If safe, proceed
mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_reference.toml --petsc -ts_checkpoint_load ../pert8_vsX30_340/outputs/checkpoint/step1516954 -ts_adapt_type basic -ts_max_steps 1566954 -ts_checkpoint_freq_step 50 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 60 -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log

# Finally, Process the after perturbation period output, change the directory name, and generate checkpoint time info
process_output_full $model_n $branch_n
read_time_full $model_n $branch_n

