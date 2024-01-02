import glob
import os 

fnames = glob.glob('perturb_stress/messages_*.log')

for fn in fnames:
    prefix = fn.split('messages_')[-1].split('.log')[0]
    # print(prefix)
    if prefix == 'after_pert20_vs340':
        print('not now')
    else:
        # print("tail -n 20 %s > perturb_stress/summary_%s.txt"%(fn,prefix))
        os.system("tail -n 20 %s > perturb_stress/summary_%s.txt"%(fn,prefix))
    # subprocess.run(["gensum",prefix])
# subprocess.run(["tail","-n","20","perturb_stress/messages_match23.log",">","summary_match23.txt"])
# os.system("tail -n 20 perturb_stress/messages_match23.log > perturb_stress/summary_match23.txt")
# os.system("gensum match23")
