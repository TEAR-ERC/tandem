#!/usr/bin/env python3
'''
Functions related to reading tandem outputs: domain/fault/fault_probe
By Jeena Yun
Last modification: 2023.07.18.
'''
import numpy as np
from vtk import vtkXMLUnstructuredGridReader
from glob import glob
from csv import reader
import os
import change_params

ch = change_params.variate()

def read_pvd(fname):
    if not os.path.exists(fname):
        raise NameError('No such file found - check the input')
    fid = open(fname,'r')
    lines = fid.readlines()
    time = []
    for line in lines:
        if line.split('<')[1].split()[0] == 'DataSet':
            time.append(float(line.split('\"')[1]))
    fid.close()
    time = np.array(time)
    return time

def extract_from_lua(save_dir,prefix,save_on=True):
    fname = 'matfric_Fourier_main'
    if len(prefix.split('/')) == 1:
        fname = prefix + '/' + fname + '.lua'
    elif 'hetero_stress' in prefix and ch.get_model_n(prefix,'v') == 0:
        fname = prefix.split('/')[0] + '/' + fname + '.lua'
    elif '_long' in prefix.split('/')[-1]:
        strr = prefix.split('/')[-1].split('_long')
        fname = prefix.split('/')[0] + '/' + fname + '_'+strr[0]+'.lua'
    elif 'hetero_stress' in prefix and 'n' in prefix.split('/')[-1]:
        strr = prefix.split('/')[-1].split('_')
        tails = ''
        for k in range(1,len(strr)):
            tails += '_%s'%(strr[k])
        fname = prefix.split('/')[0] + '/' + fname + tails +'.lua'
    elif 'mtmod' in prefix:
        fname = prefix.split('/')[0] + '/' + fname + '.lua'
    elif 'BP1' in prefix:
        if 'delsn' in prefix.split('/')[-1]:
            fname = prefix.split('/')[0] + '/' + 'bp1_deltasn.lua'
        else:
            fname = prefix.split('/')[0] + '/' + 'bp1.lua'
    else:
        fname = prefix.split('/')[0] + '/' + fname + '_'+prefix.split('/')[-1]+'.lua'
    fname = ch.get_setup_dir() + '/' + fname
    print(fname)

    here = False
    try:
        fid = open(fname,'r')
    except FileNotFoundError:
        new_fname_list = fname.split('/supermuc')
        fname = new_fname_list[0] + new_fname_list[1]
        fid = open(fname,'r')
    lines = fid.readlines()
    params = {}
    for line in lines:
        if here:
            var = line.split('return ')[-1]
            params['mu'] = params['cs']**2 * params['rho0']
            here = False
        if 'mtmod.' in line and 'index' not in line and 'new' not in line:
            var = line.split('mtmod.')[1].split(' = ')
            if len(var[1].split('--')) > 1:
                if var[0] == 'dip':
                    params[var[0]] = float(var[1].split('--')[0].split('*')[0])
                elif var[0].lower() == 'h' or var[0].lower() == 'h2':
                    params[var[0]] = float(var[1].split('--')[0].split('*')[0]) * np.sin(params['dip'])
                else:
                    params[var[0]] = float(var[1].split('--')[0])            
            else:
                if var[0] == 'dip':
                    params[var[0]] = float(var[1].split('*')[0])
                elif var[0].lower() == 'h' or var[0].lower() == 'h2':
                    params[var[0]] = float(var[1].split('*')[0]) * np.sin(params['dip'])
                else:
                    params[var[0]] = float(var[1])
        elif 'mtmod:mu' in line and 'DZ' not in prefix:
            here = True
        elif 'BP1.' in line:
            var = line.split('BP1.')[1].split(' = ')
            if len(var[1].split('--')) > 1:
                params[var[0]] = float(var[1].split('--')[0])            
            else:
                params[var[0]] = float(var[1])
    fid.close()
    if save_on:
        print('Save data...',end=' ')
        np.save('%s/const_params'%(save_dir),params)
        print('done!')
    return np.array(params)
    
def read_fault_probe_outputs(save_dir,save_on=True):
    fnames = glob('%s/outputs/*.csv'%(save_dir))
    if len(fnames) == 0:
        raise NameError('No such file found - check the input')
    outputs = ()
    dep = []
    for fn in fnames:
        with open(fn, 'r') as csvfile:
            csvreader = reader(csvfile)
            stloc = next(csvreader)
            r_z = float(stloc[1].split(']')[0])
            dep.append(r_z)
            next(csvreader)
            dat = []
            for row in csvreader:
                dat.append(np.asarray(row).astype(float))
        outputs = outputs + (dat,)
    outputs = np.array(outputs)
    dep = np.array(dep)
    if save_on:
        print('Save data...',end=' ')
        np.save('%s/outputs'%(save_dir),outputs)
        np.save('%s/outputs_depthinfo'%(save_dir),dep)
        print('done!')
    return outputs,dep

def read_fault_outputs(save_dir,save_on=True):
    # --- Read time info from pvd file
    time = read_pvd('%s/outputs/fault.pvd'%(save_dir))

    # --- Load individual vtu files and extract fault outputs
    fnames = glob('%s/outputs/fault_*.vtu'%(save_dir))
    if len(fnames) == 0:
        raise NameError('No such file found - check the input')
    sliprate,slip,shearT,normalT,state_var,dep = \
    [np.array([]) for k in range(time.shape[0])],[np.array([]) for k in range(time.shape[0])],[np.array([]) for k in range(time.shape[0])],\
    [np.array([]) for k in range(time.shape[0])],[np.array([]) for k in range(time.shape[0])],[np.array([]) for k in range(time.shape[0])]
    for file_name in fnames:
        if 'static' in file_name:
            continue
        # --- Read the source file
        reader = vtkXMLUnstructuredGridReader()
        reader.SetFileName(file_name)
        reader.Update()  # Needed because of GetScalarRange
        output = reader.GetOutput()
        k = int(file_name.split('fault_')[-1].split('_')[0])
        # --- Save into local variable
        sliprate[k] = np.hstack((sliprate[k],np.array(output.GetPointData().GetArray('slip-rate0'))))
        slip[k] = np.hstack((slip[k],np.array(output.GetPointData().GetArray('slip0'))))
        shearT[k] = np.hstack((shearT[k],np.array(output.GetPointData().GetArray('traction0'))))
        normalT[k] = np.hstack((normalT[k],np.array(output.GetPointData().GetArray('normal-stress'))))
        state_var[k] = np.hstack((state_var[k],np.array(output.GetPointData().GetArray('state'))))
        dep[k] = np.hstack((dep[k],np.array([output.GetPoint(k)[1] for k in range(output.GetNumberOfPoints())])))

    # --- Convert the outputs into numpy array
    sliprate = np.array(sliprate)
    slip = np.array(slip)
    shearT = np.array(shearT)
    normalT = np.array(normalT)
    state_var = np.array(state_var)
    dep = np.array(dep)
    print(sliprate.shape,slip.shape,shearT.shape,normalT.shape,state_var.shape,dep.shape)

    # --- Sort them along depth
    ind = np.argsort(dep,axis=1)
    sliprate = np.take_along_axis(sliprate, ind, axis=1)
    slip = np.take_along_axis(slip, ind, axis=1)
    shearT = np.take_along_axis(shearT, ind, axis=1)
    normalT = np.take_along_axis(normalT, ind, axis=1)
    state_var = np.take_along_axis(state_var, ind, axis=1)
    dep = np.sort(dep,axis=1)

    # --- Finally, arrange their shape to match with the fault probe output
    outputs = np.array([np.vstack((time,state_var.T[dp],slip.T[dp],shearT.T[dp],sliprate.T[dp],normalT.T[dp])).T for dp in range(dep.shape[1])])

    if save_on:
        print('Save data...',end=' ')
        np.save('%s/fault_outputs'%(save_dir),outputs)
        np.save('%s/fault_outputs_depthinfo'%(save_dir),dep)
        print('done!')
    return outputs,dep

def load_params(save_dir):
    print('Load saved data: %s/const_params.npy'%(save_dir))
    params = np.load('%s/const_params.npy'%(save_dir),allow_pickle=True)
    return params

def load_fault_probe_outputs(save_dir):
    print('Load saved data: %s/outputs.npy'%(save_dir))
    outputs = np.load('%s/outputs.npy'%(save_dir))
    print('Load saved data: %s/outputs_depthinfo.npy'%(save_dir))
    dep = np.load('%s/outputs_depthinfo.npy'%(save_dir))
    return outputs,dep

def load_fault_outputs(save_dir):
    print('Load saved data: %s/fault_outputs.npy'%(save_dir))
    outputs = np.load('%s/fault_outputs.npy'%(save_dir))
    print('Load saved data: %s/fault_outputs_depthinfo.npy'%(save_dir))
    dep = np.load('%s/fault_outputs_depthinfo.npy'%(save_dir))
    return outputs,dep