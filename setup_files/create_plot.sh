model_n=Thakur20_various_fractal_profiles
# model_n=DZ_hetero_stress
# model_n=Thakur20_hetero_stress
branch_n=ab2_Dc1
save_dir=/export/dump/jyun/$model_n/$branch_n
mkdir -p $save_dir
mv /export/dump/jyun/$model_n/outputs_$branch_n $save_dir
mv $save_dir/outputs_$branch_n $save_dir/outputs
# python get_plots.py $save_dir -sr 2 -sl 2 -ab -st 2 -ist -csl -dd -dtcr 2 -dtco 0.5
python get_plots.py $save_dir -c -csl -dtcr 2 -dtco 0.5 -abio -dcio
# python get_plots.py $save_dir -csl -dtcr 2 -dtco 0.5 -dcio -stio
# python get_plots.py $save_dir -csl -dd -dtcr 2 -dtco 0.5
# python get_plots.py $save_dir -csl -dd -spup 5 -dtcr 2 -dtco 0.5
