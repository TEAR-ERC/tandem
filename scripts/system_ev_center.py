import numpy as np
import matplotlib.pylab as plt
from faultoutputs_image import *
import setup_shortcut
import change_params
import myplots
from read_outputs import load_short_fault_probe_outputs

sc = setup_shortcut.setups()
mp = myplots.Figpref()
ch = change_params.variate()

prefix = 'perturb_stress/reference'

# ----------
save_dir = '/export/dump/jyun/'+prefix

# ----------
y,Hs,a,b,a_b,tau0,sigma0,Dc,others = ch.load_parameter(prefix)

# ----------
from cumslip_compute import *
import os

image = 'sliprate'
print('Image %s figure'%(image))

if image == 'sliprate':
    vmin,vmax = 1e-12,1e1
elif image == 'shearT':
    vmin,vmax = -5,5
else:
    vmin,vamx = None,None

if 'v6_ab2_Dc2' in prefix:
    Vths = 1e-1
    intv = 0.15
elif 'perturb_stress' in prefix:
    Vths = 2e-1
    intv = 0.15
else:
    Vths = 1e-2
    intv = 0.
Vlb = 0
dt_interm = 0
cuttime = 0
rths = 10
dt_creep = 2*ch.yr2sec
dt_coseismic = 0.5

if os.path.exists('%s/cumslip_outputs_Vths_%1.0e_srvar_%03d_rths_%d_tcreep_%d_tseis_%02d.npy'%(save_dir,Vths,intv*100,rths,dt_creep/ch.yr2sec,dt_coseismic*10)):
    print('Load saved file')
    cumslip_outputs = np.load('%s/cumslip_outputs_Vths_%1.0e_srvar_%03d_rths_%d_tcreep_%d_tseis_%02d.npy'%(save_dir,Vths,intv*100,rths,dt_creep/ch.yr2sec,dt_coseismic*10),allow_pickle=True)

tstart,tend = cumslip_outputs[0][0],cumslip_outputs[0][1]
rupture_length,av_slip,system_wide,partial_rupture,event_cluster,lead_fs,major_pr,minor_pr = analyze_events(cumslip_outputs,rths)
if len(major_pr) > 0: major_pr = event_cluster[major_pr][:,1]
if len(minor_pr) > 0: minor_pr = event_cluster[minor_pr][:,1]
evdep = cumslip_outputs[1][1]
if os.path.exists('%s/spin_up_idx_Vths_%1.0e_srvar_%03d_rths_%d_tcreep_%d_tseis_%02d.npy'%(save_dir,Vths,intv*100,rths,dt_creep/ch.yr2sec,dt_coseismic*10)):
    print('Load saved file')
    spin_up_idx = np.load('%s/spin_up_idx_Vths_%1.0e_srvar_%03d_rths_%d_tcreep_%d_tseis_%02d.npy'%(save_dir,Vths,intv*100,rths,dt_creep/ch.yr2sec,dt_coseismic*10))

outputs,dep,params = load_short_fault_probe_outputs(save_dir,1)
print('Total number of events: %d / Spin-up index: %d'%(len(tstart),spin_up_idx))
print('System-size indexes:',system_wide)

# ----------
iev1,iev2 = 185,187
buffer1 = 1000
buffer2 = buffer1
zoom_frame = [iev1,iev2,buffer1,buffer2]
its_all,ite_all,tsmin,tsmax,its,ite = class_figtype(zoom_frame,outputs,cumslip_outputs,publish=True,print_on=True)[2:8]

ax = fout_image(image,outputs,dep,params,cumslip_outputs,save_dir,prefix,rths,vmin,vmax,Vths,zoom_frame,plot_in_timestep=False,plot_in_sec=True,cb_off=True,publish=True,save_on=False)
fig_name = '_zoom_image_ev%dto%d'%(iev1,iev2)
xl = ax.get_xlim()
xtcks = ax.get_xticks()
step = np.diff(xtcks)[0]
ts = tstart[system_wide[np.where(np.logical_and(system_wide>=iev1,system_wide<=iev2))[0][0]]]
xt_up = np.arange(ts,xtcks.max(),step)
xt_down = np.arange(ts,xtcks.min(),-step)
new_xtcks = np.hstack((xt_down[-1:0:-1],xt_up))
new_xtl = ['%d'%(xt-ts) for xt in new_xtcks]
ax.set_xticks(new_xtcks)
ax.set_xticklabels(new_xtl)
ax.set_xlim(xl)
ax.set_xlabel('Time to the system size event [s]')
plt.savefig('%s/publish_%s%s.png'%(save_dir,image,fig_name),dpi=300)