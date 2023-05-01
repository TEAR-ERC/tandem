model_n=Thakur20_various_fractal_profiles
tdhome=/home/jyun/Tandem
setup_dir=$tdhome/$model_n
mkdir -p /export/dump/jyun/$model_n
cd /export/dump/jyun/$model_n
mkdir -p outputs_ab2
cd outputs_ab2
echo "Tandem running in a directory: " $setup_dir
mpiexec -bind-to core -n 64 tandem $setup_dir/p65ab2.toml --petsc -options_file $tdhome/options/lu_mumps.cfg -options_file $tdhome/options/rk45.cfg -ts_monitor > $setup_dir/messages_ab2.log &
