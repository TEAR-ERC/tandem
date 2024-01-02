import numpy as np
import argparse
import os
import change_params
import setup_shortcut
from read_outputs import load_fault_probe_outputs
sc = setup_shortcut.setups()
ch = change_params.variate()

setup_dir = '/home/jyun/Tandem'
output_dir = '/export/dump/jyun'

#  Set input parameters
parser = argparse.ArgumentParser()
parser.add_argument("model_n",type=str,help=": Name of big group of the model")
parser.add_argument("ver_name",type=str,help=": Version name for the perturbation model")
parser.add_argument("--save_on",action="store_true",help=": Save on?",default=False)
args = parser.parse_args()

if args.save_on:
    if not os.path.exists('%s/%s/final_stress_%s.dat'%(setup_dir,args.model_n,args.ver_name)):
        outputs,dep,params = load_fault_probe_outputs('%s/%s/%s'%(output_dir,args.model_n,args.ver_name))
        Ts = outputs[:,-1,3]
        Pn = outputs[:,-1,5]
        ii = np.argsort(dep)
        dep = dep[ii]
        Ts = Ts[ii]
        Pn = Pn[ii]
        fid = open('%s/%s/final_stress_%s.dat'%(setup_dir,args.model_n,args.ver_name),'w')
        [fid.write('%1.2f\t%1.18e\t%1.18e\n'%(dep[i],Ts[i],Pn[i])) for i in range(len(dep))]
        fid.close()
    else:
        print('File %s/%s/final_stress_%s.dat already exists - skip'%(setup_dir,args.model_n,args.ver_name))
else:
    if not os.path.exists('%s/%s/%s'%(output_dir,args.model_n,args.ver_name)):
        print('%s/%s/%s'%(output_dir,args.model_n,args.ver_name))
        raise LookupError()
    else:
        print('%s/%s/final_stress_%s.dat'%(setup_dir,args.model_n,args.ver_name))


# -------- old version for all existing files
# fnames = glob.glob('%s/%s/pert*'%(output_dir,model_n))
# for fn in fnames:
#     ver_name = fn.split('%s/%s/'%(output_dir,model_n))[-1]
#     if save_on:
#         outputs,dep,params = load_fault_probe_outputs('%s/%s/%s'%(output_dir,model_n,ver_name))
#         Ts = outputs[:,-1,3]
#         Pn = outputs[:,-1,5]
#         ii = np.argsort(dep)
#         dep = dep[ii]
#         Ts = Ts[ii]
#         Pn = Pn[ii]

#         fid = open('%s/%s/final_stress_%s.dat'%(setup_dir,model_n,ver_name),'w')
#         [fid.write('%1.2f\t%1.18e\t%1.18e\n'%(dep[i],Ts[i],Pn[i])) for i in range(len(dep))]
#         fid.close()
#     else:
#         if not os.path.exists('%s/%s/%s'%(output_dir,model_n,ver_name)):
#             print('%s/%s/%s'%(output_dir,model_n,ver_name))
#             raise LookupError()
#         else:
#             print('%s/%s/final_stress_%s.dat'%(setup_dir,model_n,ver_name))

# -------- old version for adjusted lithostatic models
# based_on_mesh = True
# mesh_y = ch.generate_mesh_points(based_on_mesh,start_y=0,end_y=24,npoints=2000)
# final_tau = ch.same_length(dep,Ts,mesh_y)
# if save_on:
#     wfid = open('%s/fractal_litho_finaltau_%02d'%(setup_dir+prefix,litho_model),'w')
#     for i in np.argsort(mesh_y):
#         wfid.write('%4.20f\t%4.20f\n'%(mesh_y[i],final_tau[i]))
#     wfid.close()
