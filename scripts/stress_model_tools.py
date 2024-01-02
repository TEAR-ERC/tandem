from scipy import interpolate
import numpy as np
import argparse
import change_params
import setup_shortcut
ch = change_params.variate()
sc = setup_shortcut.setups()

def fixed_model(save_dir,model_n,receivef_strike,target_depths,multiplies,mu,print_off):
    if not print_off: print('Fixed model: %s%d'%(sc.model_code(model_n),receivef_strike))
    if not print_off: print('Depth | Multiply | Peak dynamic dCFS [MPa] | Static dynamic dCFS [MPa]')
    delPn = np.load('%s/ssaf_%s_Pn_pert_mu%02d_%d.npy'%(save_dir,model_n,int(mu*10),receivef_strike))
    delTs = np.load('%s/ssaf_%s_Ts_pert_mu%02d_%d.npy'%(save_dir,model_n,int(mu*10),receivef_strike))
    depth_range = np.load('%s/ssaf_%s_dep_stress_pert_mu%02d_%d.npy'%(save_dir,model_n,int(mu*10),receivef_strike))

    mlen = max([len(target_depths),len(multiplies)])
    peak_dynamic,static = np.zeros(mlen),np.zeros(mlen)
    if len(target_depths) == len(multiplies):
        for it in range(len(target_depths)):
            target_depth = target_depths[it]
            multiply = multiplies[it]
            dCFSt = delTs*multiply + mu*(delPn*multiply)
            dcfs_at_D = [interpolate.interp1d(depth_range,dCFSt[ti])(-target_depth) for ti in range(dCFSt.shape[0])]
            if not print_off: print('%1.2f\t|\tX%d\t|\t%1.4f\t\t|\t%1.4f'%(target_depth,multiply,np.max(dcfs_at_D),np.mean(dcfs_at_D[-10:])))
            peak_dynamic[it] = np.max(dcfs_at_D)
            static[it] = np.mean(dcfs_at_D[-10:])
    elif len(target_depths) == 1:
        target_depth = target_depths[0]
        for it in range(len(multiplies)):
            multiply = multiplies[it]
            dCFSt = delTs*multiply + mu*(delPn*multiply)
            dcfs_at_D = [interpolate.interp1d(depth_range,dCFSt[ti])(-target_depth) for ti in range(dCFSt.shape[0])]
            if not print_off: print('%1.2f\t|\tX%d\t|\t%1.4f\t\t|\t%1.4f'%(target_depth,multiply,np.max(dcfs_at_D),np.mean(dcfs_at_D[-10:])))
            peak_dynamic[it] = np.max(dcfs_at_D)
            static[it] = np.mean(dcfs_at_D[-10:])
    elif len(multiplies) == 1:
        multiply = multiplies[0]
        for it in range(len(target_depths)):
            target_depth = target_depths[it]
            dCFSt = delTs*multiply + mu*(delPn*multiply)
            dcfs_at_D = [interpolate.interp1d(depth_range,dCFSt[ti])(-target_depth) for ti in range(dCFSt.shape[0])]
            if not print_off: print('%1.2f\t|\tX%d\t|\t%1.4f\t\t|\t%1.4f'%(target_depth,multiply,np.max(dcfs_at_D),np.mean(dcfs_at_D[-10:])))
            peak_dynamic[it] = np.max(dcfs_at_D)
            static[it] = np.mean(dcfs_at_D[-10:])
    return peak_dynamic,static

def fixed_event(save_dir,model_ns,receivef_strikes,target_depth,multiplies,mu,print_off):
    if not print_off: print('Fixed event at depth = %1.2f km'%(target_depth))
    if not print_off: print('Model Name | Strike | Multiply | Peak dynamic dCFS [MPa] | Static dynamic dCFS [MPa]')
    if len(model_ns) != len(receivef_strikes):
        raise SyntaxError('Lengthx of models do not match!')
    peak_dynamic,static = np.zeros(len(model_ns)),np.zeros(len(model_ns))
    for it in range(len(model_ns)):
        model_n = model_ns[it]
        receivef_strike = receivef_strikes[it]
        multiply = multiplies[it]
        delPn = np.load('%s/ssaf_%s_Pn_pert_mu%02d_%d.npy'%(save_dir,model_n,int(mu*10),receivef_strike))
        delTs = np.load('%s/ssaf_%s_Ts_pert_mu%02d_%d.npy'%(save_dir,model_n,int(mu*10),receivef_strike))
        depth_range = np.load('%s/ssaf_%s_dep_stress_pert_mu%02d_%d.npy'%(save_dir,model_n,int(mu*10),receivef_strike))
        dCFSt = delTs*multiply + mu*(delPn*multiply)
        dcfs_at_D = [interpolate.interp1d(depth_range,dCFSt[ti])(-target_depth) for ti in range(dCFSt.shape[0])]
        if not print_off: print('%s\t|\t%d\t|\tX%d\t|\t%1.4f\t\t|\t%1.4f'%(model_n,receivef_strike,multiply,np.max(dcfs_at_D),np.mean(dcfs_at_D[-10:])))
        peak_dynamic[it] = np.max(dcfs_at_D)
        static[it] = np.mean(dcfs_at_D[-10:])
    return peak_dynamic,static

# ---------------------- Set input parameters
def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seissol_model_n",nargs='+',type=str.lower,help=": Name of the SeisSol model")
    parser.add_argument("--strike",nargs='+',type=int,help=": Strike of the SeisSol model")
    parser.add_argument("--target_depths",nargs='+',type=float,help=": Depth of interest in km")
    parser.add_argument("--multiply",nargs='+',type=int,help=": If given, the integer is multiplied to the given perturbation model")
    parser.add_argument("--save_dir",type=str,help=": If given, directory where stress models are saved",default='perturb_stress/seissol_outputs')
    parser.add_argument("--mu",type=float,help=": If given, friction coefficient used for stress computation",default=0.4)
    parser.add_argument("--print_off", action="store_true", help=": ON/OFF cumulative slip profile",default=False)
    args = parser.parse_args()
    if args.multiply is None:
        multiply = np.ones(len(args.seissol_model_n))
    else:
        multiply = args.multiply
    if len(args.seissol_model_n) == 1:
        fixed_model(args.save_dir,args.seissol_model_n[0],args.strike[0],args.target_depths,multiply,args.mu,args.print_off)
    elif len(args.target_depths) == 1:
        fixed_event(args.save_dir,args.seissol_model_n,args.strike,args.target_depths[0],multiply,args.mu,args.print_off)

if __name__ == '__main__':
    main()