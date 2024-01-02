import glob
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_n",type=str,help=": Name of big group of the model")
parser.add_argument("--exception",nargs='+',type=str,help=": If given, process all files except this file")
parser.add_argument("--safe_mode",action="store_true",help=": If given, test run without real execution",default=False)
args = parser.parse_args()

print('Working directory: %s'%(args.model_n))
if args.safe_mode: print('*** SAFE MODE ***')
fnames = glob.glob('%s/messages_*.log'%(args.model_n))
for fn in fnames:
    prefix = fn.split('messages_')[-1].split('.log')[0]
    # print(prefix)
    if args.exception is not None and prefix in args.exception:
        print('Skip file %s'%(prefix))
    else:
        print("tail -n 20 %s > %s/summary_%s.txt"%(fn,args.model_n,prefix))
        if not args.safe_mode: os.system("tail -n 20 %s > %s/summary_%s.txt"%(fn,args.model_n,prefix))