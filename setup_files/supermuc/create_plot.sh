#!/bin/bash
# Job Name and Files (also --job-name)

#SBATCH -J plot
#Output and error (also --output, --error):
#SBATCH -o ./messages_%x.out
#SBATCH -e ./messages_%x.err

#Initial working directory:
#SBATCH --chdir=/hppfs/work/pn49ha/di75weg/jeena-tandem/setup_files/supermuc

#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=j4yun@ucsd.edu

# Wall clock limit:
#SBATCH --time=00:30:00
#SBATCH --no-requeue

#Setup of execution environment
#SBATCH --export=ALL
#SBATCH --account=pn49ha
#constraints are optional
#--constraint="scratch&work"
#SBATCH --partition=test

#SBATCH --ear=off

#Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
module load slurm_setup

export OMP_NUM_THREADS=1

source /etc/profile.d/modules.sh
echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS 'cpus_per_task:' $SLURM_CPUS_PER_TASK
ulimit -Ss 2097152

model_n=lithostatic_sn
branch_n=v2_ab2_Dc2_adjusted
output_dir=$SCRATCH/$model_n/$branch_n/outputs_$branch_n
plottool_dir=$WORK/jeena-tandem/plot_tools
save_dir=$SCRATCH/$model_n/$branch_n
mv $output_dir $save_dir/outputs

echo $model_n'/'$branch_n

source $SCRATCH/$envn/bin/activate
mpiexec -n $SLURM_NTASKS python3.8 $plottool_dir/get_plots.py $save_dir -csl -dtcr 2 -dtco 0.5 -stio -abio -imsr -ts -stf -evan
# mpiexec -n $SLURM_NTASKS python3.8 $plottool_dir/get_plots.py $save_dir -csl -dtcr 2 -dtco 0.5 -abio -dcio
