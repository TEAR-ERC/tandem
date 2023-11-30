#!/usr/bin/env python3
'''
Check whether the checkpoint file exists
By Jeena Yun
Last modification: 2023.11.21.
'''
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path_name",type=str,help=": Path name to the checkpoint")
args = parser.parse_args()

if not os.path.exists(args.path_name):
    raise FileNotFoundError('No such path %s'%(args.path_name))
else:
    print('Checkpoint at %s - Found'%(args.path_name))
