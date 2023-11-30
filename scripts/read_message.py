#!/usr/bin/env python3
import numpy as np

def read_two_output(fname):
    fid = open(fname,'r')
    lines = fid.readlines()
    mesh_y = []
    a = []
    b = []
    c = 0
    for line in lines:
        if line[0].isspace():
            continue
        if line[0:4] == 'Comp':
            break     
        try:
            _y, _a, _b = line.split('\t')
        except:
            print('skip line %d:'%(c),line.strip())
            continue
        try:
            mesh_y.append(float(_y.strip())); a.append(float(_a.strip())); b.append(float(_b.strip()))
        except:
            print('skip line %d:'%(c),line.strip())
            continue
        c += 1
    print('Total %d points'%c)
    fid.close()
    idx = np.argsort(np.array(mesh_y))
    mesh_y = np.array(mesh_y)[idx]; a = np.array(a)[idx]; b = np.array(b)[idx]
    return mesh_y,a,b

def read_one_output(fname):
    fid = open(fname,'r')
    lines = fid.readlines()
    mesh_y = []
    var = []
    c = 0
    for line in lines:
        if line[0].isspace():
            continue
        if line[0:4] == 'Comp':
            break     
        try:
            _y, _var = line.split('\t')
        except:
            print('skip line %d:'%(c),line.strip())
            continue
        try:
            mesh_y.append(float(_y.strip())); var.append(float(_var.strip()))
        except:
            print('skip line %d:'%(c),line.strip())
            continue
        c += 1
    print('Total %d points'%c)
    fid.close()
    idx = np.argsort(np.array(mesh_y))
    mesh_y = np.array(mesh_y)[idx]; var = np.array(var)[idx]
    return mesh_y,var

