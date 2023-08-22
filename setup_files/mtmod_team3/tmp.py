import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("-spup","--spin_up", nargs=2, type=str, help=": Plot with spin-up after given amount of quantity",default=[])
args = parser.parse_args()

print(len(args.spin_up))
# print(args.spin_up)
# if len(args.spin_up) == 1:
#     print('Field npts not defined - using default value 50')
#     args.spin_up.append(50)
# print(args.spin_up)

# ---
# import numpy as np
# import matplotlib.pylab as plt
# import myplots
# from read_outputs import *

# mp = myplots.Figpref()

# # prefix = 'Thakur20_various_fractal_profiles/v6_Dc2_DZ_long'
# prefix = 'scaling_test'
# save_dir = 'models/'+prefix
# compute_and_save = 1

# if compute_and_save:
#     read_fault_outputs(save_dir,save_on=True)
# else:
#     outputs = load_fault_outputs(save_dir)


# V0 = float(input('Reference slip velocity V0 [m/s]: '))
# print(V0)

# ---
# import numpy as np
# import matplotlib.pyplot as plt
# from cumslip_compute import *
# from misc_plots import *
# import myplots
# import change_params
# import setup_shortcut

# sc = setup_shortcut.setups()
# mp = myplots.Figpref()
# ch = change_params.variate()

# prefix_list = ['Thakur20_hetero_stress/n8',
#                'Thakur20_various_fractal_profiles/ab2',
#                'Thakur20_various_fractal_profiles/Dc1',
#                'Thakur20_various_fractal_profiles/v6_Dc1_long',
#                'Thakur20_various_fractal_profiles/v6_ab2',
#                'Thakur20_various_fractal_profiles/ab2_Dc1']
# dir = '/Users/j4yun/Library/CloudStorage/Dropbox/Codes/Ridgecrest_CSC/Tandem'

# Vths = 1e-2
# Vlb = 0
# dt_interm = 0
# cuttime = 0
# mingap = 60
# rths = 10
# dt_creep = 2*ch.yr2sec
# dt_coseismic = 0.5

# for uu,prefix in enumerate(prefix_list):
#     print(prefix)
#     save_dir = dir + '/models/'+prefix
#     plot_dir = 'plots/' + prefix

#     # ---------- Load outputs
#     print('Load saved data: %s/outputs'%(save_dir))
#     outputs = np.load('%s/outputs.npy'%(save_dir))
#     print('Load saved data: %s/outputs_depthinfo'%(save_dir))
#     dep = np.load('%s/outputs_depthinfo.npy'%(save_dir))
#     params = sc.extract_from_lua(prefix,save_on=True)

#     cumslip_outputs = compute_cumslip(outputs,dep,cuttime,Vlb,Vths,dt_creep,dt_coseismic,dt_interm,mingap)
#     if np.max(cumslip_outputs[3][0]) > 35:
#         spin_up = 10
#     else:
#         spin_up = 2.5
#     spin_up_idx = compute_spinup(outputs,dep,cuttime,cumslip_outputs,spin_up)[-1]
#     plot_STF(save_dir,outputs,dep,cumslip_outputs,spin_up_idx,rths=10,save_on=True)

