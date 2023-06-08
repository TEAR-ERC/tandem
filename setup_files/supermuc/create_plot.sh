plottool_dir=$WORK/jeena-tandem/plot_tools
save_dir=$SCRATCH/$model_n/$branch_n
mv $output_dir $save_dir/outputs

# python $plottool_dir/get_plots.py $save_dir -c -csl -dtcr 2 -dtco 0.5 -abio -stio -imsr -ts
# python $plottool_dir/get_plots.py $save_dir -csl -dtcr 2 -dtco 0.5 -dcio -dd
python $plottool_dir/get_plots.py $save_dir -csl -dtcr 2 -dtco 0.5
# python $plottool_dir/get_plots.py $save_dir -csl -dd -spup 5 -dtcr 2 -dtco 0.5
