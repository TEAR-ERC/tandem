#!/usr/bin/env python3
'''
Concatenate truncated outputs due to the checkpointing feature
By Jeena Yun
Last modification: 2023.11.03.
'''

import numpy as np
import pandas as pd
import time
from glob import glob

dir1 = '/export/dump/jyun/perturb_stress/outputs_reference_1'
dir2 = '/export/dump/jyun/perturb_stress/outputs_reference_2'
dir3 = '/export/dump/jyun/perturb_stress/outputs_reference_3'
save_dir = '/export/dump/jyun/perturb_stress/outputs_reference'

fnames = glob('%s/faultp_*.csv'%(dir3))

flog = open('messages_faultp_concat.log','w')
ti = time.time()
for fname3 in np.sort(fnames):
    csv_name = fname3.split('outputs_reference_3')[-1]
    flog.write('Processing file %s\n'%(csv_name[1:]))
    fname1 = dir1+csv_name
    fname2 = dir2+csv_name

    dat1 = pd.read_csv(fname1,delimiter=',',skiprows=1)
    dat2 = pd.read_csv(fname2,delimiter=',',skiprows=1)
    dat3 = pd.read_csv(fname3,delimiter=',',skiprows=1)
    comment = pd.read_csv(fname1,nrows=0,sep='\t')

    dat123 = pd.concat([dat1,dat2,dat3],axis=0)
    # if fname3 == fnames[0]:
    #     print('dat1.shape:',dat1.shape)
    #     print('dat2.shape:',dat2.shape)
    #     print('dat3.shape:',dat3.shape)
    #     print('dat123.shape:',dat123.shape)
    if dat1.shape[0]+dat2.shape[0]+dat3.shape[0] != dat123.shape[0]:
        print('Something wrong!!')
        break

    fid = open(save_dir+csv_name,'a')
    fid.write(comment.columns[0]+'\n')
    dat123.to_csv(fid,index=False,float_format='%1.15e')
    fid.close()

flog.close()
print('============ SUMMARY ============')
print('Run time: %2.4f s'%(time.time()-ti))
print('=================================')