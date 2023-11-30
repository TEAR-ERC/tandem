# Generate output and change directory name
model_n=perturb_stress
pert_branch_n=pert31_vs340
save_dir=/export/dump/jyun/$model_n/$pert_branch_n
echo $save_dir
mkdir -p $save_dir
mv /export/dump/jyun/$model_n/outputs_$pert_branch_n $save_dir
mv $save_dir/outputs_$pert_branch_n $save_dir/outputs
python get_plots.py $save_dir -c

# Generate checkpoint time info
./read_time_recursive /export/dump/jyun/$model_n/$pert_branch_n

branch_n=after_pert31_vs340
tdhome=/home/jyun/Tandem
setup_dir=$tdhome/$model_n
rm -rf $setup_dir/*profile_$branch_n
mkdir -p /export/dump/jyun/$model_n
cd /export/dump/jyun/$model_n
mkdir -p outputs_$branch_n
cd outputs_$branch_n
echo "Tandem running in a directory: " $setup_dir
mpiexec -bind-to core -n 40 tandem $setup_dir/parameters.toml --petsc -ts_checkpoint_load ../pert31_vs340/outputs/checkpoint/step4917143 -ts_adapt_type basic -ts_max_steps 4967143 -ts_checkpoint_freq_step 50 -ts_checkpoint_freq_physical_time 1000000000 -ts_checkpoint_freq_cputime 60 -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &
