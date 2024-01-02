#!/bin/bash
# Define some useful functions
process_output_full() { echo "/export/dump/jyun/$1/$2"; mkdir -p "/export/dump/jyun/$1/$2"; mv "/export/dump/jyun/$1/outputs_$2" "/export/dump/jyun/$1/$2"; mv "/export/dump/jyun/$1/$2/outputs_$2" "/export/dump/jyun/$1/$2/outputs"; python /home/jyun/Tandem/get_plots.py /export/dump/jyun/$1/$2 -c; }
read_time_full() { /home/jyun/Tandem/read_time_recursive "/export/dump/jyun/$1/$2"; }

model_n=perturb_stress
branch_n=match8_5h
tdhome=/home/jyun/Tandem
setup_dir=$tdhome/$model_n
rm -rf $setup_dir/*profile_$branch_n
mkdir -p /export/dump/jyun/$model_n
cd /export/dump/jyun/$model_n
mkdir -p outputs_$branch_n
cd outputs_$branch_n
echo "Tandem running in a directory: " $setup_dir
mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_match_time.toml --petsc -ts_checkpoint_load ../reference/outputs/checkpoint/step1516350 -ts_checkpoint_freq_step 1 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 60 -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &

# Process the output, change the directory name, and generate checkpoint time info
process_output_full $model_n $branch_n
read_time_full $model_n $branch_n

