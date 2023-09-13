#!/usr/bin/env python3
'''
An executable plotting script for Tandem to save figures directly from a remote server
By Jeena Yun
Update note: added Gutenberg-Richter relation plot
Last modification: 2023.07.26.
'''
import argparse

yr2sec = 365*24*60*60
wk2sec = 7*24*60*60

# Set input parameters -------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("save_dir", help=": directory to output files and to save plots")
parser.add_argument("-c","--compute", action="store_true", help=": Activate only when you want to compute")
parser.add_argument("-ot","--output_type", type=str.lower, choices=['fault_probe','fault','domain'], help=": Type of output to process ['fault_probe','fault','domain']",default='fault_probe')

# Fault output vs. time at certain depth
parser.add_argument("-sr","--sliprate", type=float, help=": If used, depth of slip rate vs. time plot [km]", default=0)
parser.add_argument("-sl","--slip", type=float, help=": If used, depth of slip vs. time plot [km]", default=0)
parser.add_argument("-st","--stress", type=float, help=": If used, depth of stress vs. time plot [km]", default=0)
parser.add_argument("-sv","--state_var", type=float, help=": If used, depth of state variable vs. time plot [km]", default=0)
parser.add_argument("-sec","--plot_in_sec", action="store_true", help=": Time axis in seconds",default=False)

# Input variable profile
parser.add_argument("-ist","--stressprof", action="store_true", help=": ON/OFF in & out stress profile")
parser.add_argument("-ab","--abprof", action="store_true", help=": ON/OFF in & out a-b profile")
parser.add_argument("-dc","--dcprof", action="store_true", help=": ON/OFF in & out Dc profile")

# Fault output image
parser.add_argument("-im","--image", type=str, choices=['sliprate','shearT','normalT','state_var'], help=": Type of image plot ['sliprate','shearT','normalT','state_var']")
parser.add_argument("-ts","--plot_in_timestep", action="store_true", help=": Time axis in timesteps",default=False)
parser.add_argument("-zf","--zoom_frame", nargs='+', type=int, help=": When used, event indexes or timestep ranges you want to zoom in",default=[])
parser.add_argument("-vmin","--vmin", type=float, help=": vmin for the plot")
parser.add_argument("-vmax","--vmax", type=float, help=": vmax for the plot")

# Cumulative slip profile related paramters
parser.add_argument("-csl","--cumslip", action="store_true", help=": ON/OFF cumulative slip profile")
parser.add_argument("-dtcr","--dt_creep", type=float, help=": Contour interval for CREEPING section [yr]")
parser.add_argument("-dtco","--dt_coseismic", type=float, help=": Contour interval for COSEISMIC section [s]")
parser.add_argument("-dtint","--dt_interm", type=float, help=": Contour interval for INTERMEDIATE section [wk]")
parser.add_argument("-Vths","--Vths", type=float, help=": Slip-rate threshold to define coseismic section [m/s]")
parser.add_argument("-Vlb","--Vlb", type=float, help=": When used with --Vths becomes lower bound of slip rate of intermediate section [m/s]")
parser.add_argument("-srvar","--SRvar", type=float, help=": Criterion for SR variation within a detected event")
parser.add_argument("-dd","--depth_dist", action="store_true", help=": Plot cumslip plot with hypocenter depth distribution",default=False)
parser.add_argument("-abio","--ab_inout", action="store_true", help=": Plot cumslip plot with a-b profile",default=False)
parser.add_argument("-stio","--stress_inout", action="store_true", help=": Plot cumslip plot with stress profile",default=False)
parser.add_argument("-dcio","--dc_inout", action="store_true", help=": Plot cumslip plot with Dc profile",default=False)
# parser.add_argument("-spup","--spin_up", type=float, help=": Plot with spin-up after given slip amount",default=0)
parser.add_argument("-spup","--spin_up", nargs=2, type=str, help=": Plot with spin-up after given amount of quantity",default=[])
parser.add_argument("-rths","--rths", type=float, help=": Rupture length threshold to define system wide event [m]",default=10)
parser.add_argument("-ct","--cuttime", type=float, help=": Show result up until to the given time to save computation time [yr]", default=0)
parser.add_argument("-mg","--mingap", type=float, help=": Minimum seperation time between two different events [s]", default=60)

# Miscellaneous plots
parser.add_argument("-evan","--ev_anal", action="store_true", help=": ON/OFF event analyzation plot")
parser.add_argument("-stf","--STF", action="store_true", help=": ON/OFF STF plot")
parser.add_argument("-M0","--M0", type=str.lower, choices=['1d','approx2d'], help=": Produce M0 plot of your choice ['1d','approx2d']")
parser.add_argument("-Mw","--Mw", action="store_true", help="When used with --M0, plots output in Mw scale")
parser.add_argument("-gr","--GR", nargs='+', type=float, help="Returns Gutenberg-Richter relation. Cutoff magnitude and number of points are required")

args = parser.parse_args()

# --- Check dependencies
if args.cumslip or args.ev_anal or args.STF or args.image or args.M0 or args.GR:        # When args.cumslip are true
    if not args.dt_creep:
        parser.error('Required field \'dt_creep\' is not defined - check again')
    if not args.dt_coseismic:
        parser.error('Required field \'dt_coseismic\' is not defined - check again')
    if not args.Vths:
        print('Required field \'Vths\' not defined - using default value 1e-2 m/s')
        args.Vths = 1e-2
    if args.SRvar is None:
        print('Field \'SRvar\' not defined - using default value 0.15')
        args.SRvar = 0.15
    dt_creep = args.dt_creep*yr2sec
    dt_coseismic = args.dt_coseismic
    if args.dt_interm:
        dt_interm = args.dt_interm*wk2sec
        if not args.Vlb:
            print('Required field \'Vlb\' not defined - using default value 1e-8 m/s')
            args.Vlb = 1e-8
    else:
        dt_interm = 0
        args.Vlb = 0

save_dir = args.save_dir
if 'models' in save_dir: # local
    prefix = save_dir.split('models/')[-1]
elif 'di75weg' in save_dir: # supermuc
    prefix = save_dir.split('di75weg/')[-1]
elif 'jyun' in save_dir: # LMU server
    prefix = save_dir.split('jyun/')[-1]

cuttime = args.cuttime*yr2sec

# Extract data ---------------------------------------------------------------------------------------------------------------------------
from read_outputs import *
if args.compute:
    print('Compute on - extract outputs')
    if args.output_type == 'fault_probe':
        outputs,dep = read_fault_probe_outputs(save_dir)
    elif args.output_type == 'fault':
        outputs,dep = read_fault_outputs(save_dir)
    params = extract_from_lua(save_dir,prefix)
else:
    if args.output_type == 'fault_probe':
        outputs,dep = load_fault_probe_outputs(save_dir)
    elif args.output_type == 'fault':
        outputs,dep = load_fault_outputs(save_dir)
    params = load_params(save_dir)

# Cumslip vs. Depth ----------------------------------------------------------------------------------------------------------------------
if args.cumslip:
    from cumslip_compute import *
    from cumslip_plot import *
    cumslip_outputs = compute_cumslip(outputs,dep,cuttime,args.Vlb,args.Vths,dt_creep,dt_coseismic,dt_interm,args.SRvar)

    # --- Plot the result
    if len(args.spin_up) > 0:
        spup_cumslip_outputs = compute_spinup(outputs,dep,cuttime,cumslip_outputs,args.spin_up)
        spup_where(save_dir,prefix,cumslip_outputs,spup_cumslip_outputs,args.Vths,dt_coseismic,args.rths)
    else:
        spup_cumslip_outputs = None

    if sum([args.ab_inout,args.stress_inout,args.dc_inout,args.depth_dist]) == 0:
        only_cumslip(save_dir,prefix,cumslip_outputs,args.Vths,dt_coseismic,args.rths,spup_cumslip_outputs)
    elif sum([args.ab_inout,args.stress_inout,args.dc_inout,args.depth_dist]) == 1:
        two_set(save_dir,prefix,outputs,dep,cumslip_outputs,args.Vths,dt_coseismic,args.depth_dist,args.ab_inout,args.stress_inout,args.dc_inout,args.rths,spup_cumslip_outputs)
    elif sum([args.ab_inout,args.stress_inout,args.dc_inout,args.depth_dist]) == 2:
        three_set(save_dir,prefix,outputs,dep,cumslip_outputs,args.Vths,dt_coseismic,args.depth_dist,args.ab_inout,args.stress_inout,args.dc_inout,args.rths,spup_cumslip_outputs)

# Fault output image ---------------------------------------------------------------------------------------------------------------------
if args.image:
    from faultoutputs_image import fout_image
    if not args.vmin:                       # No vmin defined
        if args.image == 'sliprate':
            vmin = 1e-12
        # elif args.image == 'shearT':
        #     vmin = -5
        else:
            vmin = None
    else:
        vmin = args.vmin
    if not args.vmax:                       # No vmax defined
        if args.image == 'sliprate':
            vmax = 1e1
        # elif args.image == 'shearT':
        #     vmax = 5
        else:
            vmax = None
    else:
        vmax = args.vmax
    if not 'cumslip_outputs' in locals():   # No event outputs computed
        from cumslip_compute import *
        cumslip_outputs = compute_cumslip(outputs,dep,cuttime,args.Vlb,args.Vths,dt_creep,dt_coseismic,dt_interm,args.SRvar)
    fout_image(args.image,outputs,dep,params,cumslip_outputs,save_dir,prefix,args.rths,vmin,vmax,args.Vths,args.zoom_frame,args.plot_in_timestep,args.plot_in_sec)

# Miscellaneous --------------------------------------------------------------------------------------------------------------------------
if args.ev_anal:
    from misc_plots import plot_event_analyze
    if not 'cumslip_outputs' in locals():
        from cumslip_compute import *
        cumslip_outputs = compute_cumslip(outputs,dep,cuttime,args.Vlb,args.Vths,dt_creep,dt_coseismic,dt_interm,args.SRvar)
    plot_event_analyze(save_dir,prefix,cumslip_outputs,args.rths)

if args.STF:
    from misc_plots import plot_STF
    if not 'cumslip_outputs' in locals():
        from cumslip_compute import *
        cumslip_outputs = compute_cumslip(outputs,dep,cuttime,args.Vlb,args.Vths,dt_creep,dt_coseismic,dt_interm,args.SRvar)

    if len(args.spin_up) > 0:
        if not 'spin_up_idx' in locals():
            spin_up_idx = compute_spinup(outputs,dep,cuttime,cumslip_outputs,args.spin_up)[-1]
    else:
        spin_up_idx = 0
    plot_STF(save_dir,outputs,dep,cumslip_outputs,spin_up_idx,args.rths)

if args.M0:
    from misc_plots import plot_M0
    if not 'cumslip_outputs' in locals():
        from cumslip_compute import *
        cumslip_outputs = compute_cumslip(outputs,dep,cuttime,args.Vlb,args.Vths,dt_creep,dt_coseismic,dt_interm,args.SRvar)

    if len(args.spin_up) > 0:
        if not 'spin_up_idx' in locals():
            spin_up_idx = compute_spinup(outputs,dep,cuttime,cumslip_outputs,args.spin_up)[-1]
    else:
        spin_up_idx = 0
    plot_M0(save_dir,cumslip_outputs,spin_up_idx,args.rths,args.M0,args.Mw)

if args.GR:
    cutoff_Mw = args.GR[0]
    if len(args.GR) == 1:
        print('Field npts not defined - using default value 50')
        npts = 50
    else:
        npts = args.GR[1]
    from misc_plots import plot_GR
    if not 'cumslip_outputs' in locals():
        from cumslip_compute import *
        cumslip_outputs = compute_cumslip(outputs,dep,cuttime,args.Vlb,args.Vths,dt_creep,dt_coseismic,dt_interm,args.SRvar)

    if len(args.spin_up) > 0:
        if not 'spin_up_idx' in locals():
            spin_up_idx = compute_spinup(outputs,dep,cuttime,args.rths,cumslip_outputs,args.spin_up)[-1]
    else:
        spin_up_idx = 0
    plot_GR(save_dir,prefix,cumslip_outputs,spin_up_idx,args.rths,cutoff_Mw,npts)

# Input variable profile -----------------------------------------------------------------------------------------------------------------
if args.stressprof:
    from stress_profile import plot_stress_vs_depth
    plot_stress_vs_depth(save_dir,prefix,outputs,dep)

if args.abprof:
    from ab_profile import plot_ab_vs_depth
    plot_ab_vs_depth(save_dir,prefix)

if args.dcprof:
    from Dc_profile import plot_Dc_vs_depth
    plot_Dc_vs_depth(save_dir,prefix)

# Fault output vs. time at certain depth -------------------------------------------------------------------------------------------------
if abs(args.sliprate)>0:
    from faultoutputs_vs_time import sliprate_time
    sliprate_time(save_dir,outputs,dep,args.sliprate,args.plot_in_sec)
    
if abs(args.slip)>0:
    from faultoutputs_vs_time import slip_time
    slip_time(save_dir,outputs,dep,args.slip,args.plot_in_sec)
    
if abs(args.stress)>0:
    from faultoutputs_vs_time import stress_time
    stress_time(save_dir,outputs,dep,args.stress,args.plot_in_sec)

if abs(args.state_var)>0:
    from faultoutputs_vs_time import state_time
    state_time(save_dir,outputs,dep,args.state_var,args.plot_in_sec)