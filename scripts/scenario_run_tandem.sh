model_n=perturb_stress
branch_n=nopert31
tdhome=/home/jyun/Tandem
setup_dir=$tdhome/$model_n
rm -rf $setup_dir/*profile_$branch_n
mkdir -p /export/dump/jyun/$model_n
cd /export/dump/jyun/$model_n
mkdir -p outputs_$branch_n
cd outputs_$branch_n
echo "Tandem running in a directory: " $setup_dir
mpiexec -bind-to core -n 40 tandem $setup_dir/parameters_nopert.toml --petsc -ts_checkpoint_load ../match31/outputs/checkpoint/step4915642 -ts_adapt_type none -ts_dt 0.01 -ts_max_steps 4917143 -ts_checkpoint_freq_step 1 -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_$branch_n.log &
