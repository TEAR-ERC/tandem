model_n=perturb_stress
branch_n=pert8_vsX30_340
save_dir=/export/dump/jyun/$model_n/$branch_n
echo $save_dir
mkdir -p $save_dir
mv /export/dump/jyun/$model_n/outputs_$branch_n $save_dir
mv $save_dir/outputs_$branch_n $save_dir/outputs

# python get_plots.py $save_dir -c #> $model_n/getplot_$branch_n.log &
# python get_plots.py $save_dir -dtcr 2 -dtco 0.5 -Vths 0.2 -im delnormalT -sec
# python get_plots.py $save_dir -dtcr 2 -dtco 0.5 -Vths 0.2 -ist -ab -dc -gr 0 -u t > $model_n/getplot_$branch_n.log &
# python get_plots.py $save_dir -dtcr 2 -dtco 0.5 -Vths 0.2 -im sliprate -ts -zf 560000 640000 -pub
python get_plots.py $save_dir -dtcr 2 -dtco 0.5 -Vths 0.2 -im sliprate -ts