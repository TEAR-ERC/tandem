import numpy as np
import change_params

save_on = 1
fractal_model = 2
ch = change_params.variate()

prefix = 'lithostatic_sn/v2'
save_dir = '/export/dump/jyun/' + prefix
# work_dir = '/home/jyun/jeena-tandem/setup_files/%s'%(prefix.split('/')[0])
work_dir = '%s'%(prefix.split('/')[0])

outputs = np.load('%s/outputs.npy'%(save_dir))
dep = np.load('%s/outputs_depthinfo.npy'%(save_dir))

time = np.array([outputs[i][:,0] for i in np.argsort(abs(dep))])
shearT = np.array([outputs[i][:,3] for i in np.argsort(abs(dep))])
normalT = np.array([outputs[i][:,5] for i in np.argsort(abs(dep))])
zdep = np.sort(abs(dep))

based_on_mesh = True
mesh_y = ch.generate_mesh_points(based_on_mesh,start_y=0,end_y=24,npoints=2000)

final_tau = ch.same_length(-zdep,shearT[:,-1],mesh_y)

cc = 0
if save_on:
    wfid = open('%s/fractal_litho_finaltau_%02d'%(work_dir,fractal_model),'w')
    for i in np.argsort(mesh_y):
        wfid.write('%4.20f\t%4.20f\n'%(mesh_y[i],final_tau[i]))
    wfid.close()