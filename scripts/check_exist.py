#!/usr/bin/env python3
'''
Check whether the checkpoint file exists
By Jeena Yun
Last modification: 2023.11.21.
'''
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("path_name",type=str,help=": Path name to the checkpoint")
args = parser.parse_args()

if not os.path.exists(args.path_name):
    # print('No such path %s'%(args.path_name))
    # os.system('exit 1')
    print(1)
    # subprocess.run(['exit','1'])
    # raise FileNotFoundError('No such path %s'%(args.path_name))
else:
    # print('Checkpoint at %s - Found'%(args.path_name))
    print(0)
