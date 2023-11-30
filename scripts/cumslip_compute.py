#!/usr/bin/env python3
'''
Functions related to plotting cumulative slip vs. depth plot
By Jeena Yun
Last modification: 2023.09.09.
'''
import numpy as np
from scipy import interpolate
from event_analyze import analyze_events

yr2sec = 365*24*60*60
wk2sec = 7*24*60*60

# cumslip_outputs = [timeout, evout, creepout, coseisout, intermout]
# cumslip_outputs[0] = [tstart_coseis,tend_coseis]
# cumslip_outputs[1] = [evslip,evdep,fault_slip]
# cumslip_outputs[2] = [cscreep,depcreep]
# cumslip_outputs[3] = [cscoseis,depcoseis]
# cumslip_outputs[4] = [csinterm,depinterm]

def event_times(dep,outputs,Vlb,Vths,cuttime,dt_coseismic,intv,print_on=True):
    time = np.array(outputs[0][:,0])
    sliprate = abs(np.array([outputs[i][:,4] for i in np.argsort(abs(dep))]))
    z = np.sort(abs(dep))

    if abs(cuttime) >= 1e-3:
        if cuttime > np.max(time):
            raise ValueError('Cuttime larger than total simulation time - check again')
        sliprate = sliprate[:,time <= cuttime]
        time = time[time <= cuttime]

    psr = np.max(sliprate,axis=0)
    pd = np.argmax(sliprate,axis=0)

    # ----- Define events by peak sliprate
    if Vlb > 0:
        events = np.where(np.logical_and(psr < Vths,psr > Vlb))[0]
    else:
        events = np.where(psr > Vths)[0]

    if len(events) > 0:
        jumps = np.where(np.diff(events)>1)[0]+1

        tmp_tstart = time[events][np.hstack(([0],jumps))]
        tmp_tend = time[events][np.hstack((jumps-1,len(events)-1))]
        tmp_evdep = pd[events][np.hstack(([0],jumps))]

        # ----- Remove events with too short duration
        ii = np.where(tmp_tend-tmp_tstart>=dt_coseismic)[0]
        tstart = tmp_tstart[ii]
        tend = tmp_tend[ii]
        amax = tmp_evdep[ii]
        evdep = z[tmp_evdep[ii]]

        # ----- Adjust start time to be closer to the peak
        its_all = np.array([np.argmin(abs(time-t)) for t in tstart])
        ite_all = np.array([np.argmin(abs(time-t)) for t in tend])
        diffcrit = np.quantile(abs(np.diff(np.log10(psr[events]))),0.98)
        new_its_all = its_all.copy()
        for k,ts in enumerate(its_all):
            psr_inc = abs(np.diff(np.log10(psr)))[ts-1]
            width = int((ite_all[k] - ts)*intv)
            large_diffs = np.where(abs(np.diff(np.log10(psr)))[ts-1:ts-1+width]>=diffcrit)[0]
            if psr_inc < diffcrit and len(large_diffs) > 0:
                new_its_all[k] = large_diffs[0] + ts
        evdep = z[pd[new_its_all]]
        tstart = time[new_its_all]

        # ----- Remove events whose maximum peak slip rate is lower than certain criterion
        varsr = np.array([np.log10(psr[new_its_all[k]:ite_all[k]+1]).max()-np.log10(psr[new_its_all[k]:ite_all[k]+1]).min() for k in range(len(tstart))])
        ii = np.where(varsr/abs(np.log10(Vths))>=0.1)[0]
        if len(ii) < len(tstart):
            print('Negligible events with SR variation < 0.1Vths:',np.where(varsr/abs(np.log10(Vths))<0.1)[0])
            tstart = tstart[ii]
            tend = tend[ii]
            evdep = evdep[ii]
        else:
            print('All safe from the SR variation criterion')

        # ----- Remove events if it is only activated at specific depth: likely to be unphysical
        num_active_dep = np.array([np.sum(np.sum(sliprate[:,new_its_all[k]:ite_all[k]]>Vths,axis=1)>0) for k in range(len(tstart))])
        if len(num_active_dep>1) > 0:
            print('Remove single-depth activated event:',np.where(num_active_dep==1)[0])
            tstart = tstart[num_active_dep>1]
            tend = tend[num_active_dep>1]
            evdep = evdep[num_active_dep>1]
        else:
            print('All events activate more than one depth')

    else:
        tstart, tend, evdep = [],[],[]
    return tstart, tend, evdep
    
def compute_cumslip(outputs,dep,cuttime,Vlb,Vths,dt_creep,dt_coseismic,dt_interm,intv,print_on=True):
    if print_on: print('Cumulative slip vs. Depth plot >>> ',end='')

    if print_on: 
        if abs(cuttime) < 1e-3:
            print('No cutting')
        else:
            print('Cut at %2.1f yr'%(cuttime/yr2sec))
        if Vlb > 0:
            print('%1.0e < Slip rate < %1.0e'%(Vlb,Vths))
        else:
            print('Slip rate > %1.0e'%(Vths))

    cscreep,depcreep = [],[]
    cscoseis,depcoseis = [],[]
    fault_slip = []
    if dt_interm > 0:        
        csinterm,depinterm = [],[]

    # Obtain globally min. event start times and max. event tend times
    if dt_interm > 0:
        tstart_interm, tend_interm, evdep = event_times(dep,outputs,Vlb,Vths,cuttime,dt_coseismic,intv,print_on)
    tstart_coseis, tend_coseis, evdep = event_times(dep,outputs,0,Vths,cuttime,dt_coseismic,intv,print_on)
    if len(tstart_coseis) > 0:
        evslip = np.zeros(tstart_coseis.shape)
    else:
        evslip = []

    # Now interpolate the cumulative slip using given event time ranges
    for i in np.argsort(abs(dep)):
        z = abs(dep[i])
        time = np.array(outputs[i])[:,0]
        sliprate = np.array(outputs[i])[:,4]
        cumslip = np.array(outputs[i])[:,2]

        if abs(cuttime) >= 1e-3:
            sliprate = sliprate[time <= cuttime]
            cumslip = cumslip[time <= cuttime]
            time = time[time <= cuttime]

        f = interpolate.interp1d(time,cumslip)

        # -------------------- Creep
        tcreep = np.arange(time[0],time[-1],dt_creep)
        cscreep.append(f(tcreep))
        depcreep.append(z*np.ones(len(tcreep)))
        
        # -------------------- Inter
        if dt_interm > 0:
            cs = []
            depth = []
            for j in range(len(tstart_interm)):
                tinterm = np.arange(tstart_interm[j],tend_interm[j],dt_interm)
                cs.append(f(tinterm))
                depth.append(z*np.ones(len(tinterm)))

            csinterm.append([item for sublist in cs for item in sublist])
            depinterm.append([item for sublist in depth for item in sublist])

        # -------------------- Coseismic
        if len(tstart_coseis) > 0:
            cs = []
            depth = []
            Dbar = []
            for j in range(len(tstart_coseis)):
                tcoseis = np.arange(tstart_coseis[j],tend_coseis[j],dt_coseismic)
                cs.append(f(tcoseis))
                depth.append(z*np.ones(len(tcoseis)))
                Dbar.append(f(tcoseis)[-1]-f(tcoseis)[0])

            cscoseis.append([item for sublist in cs for item in sublist])
            depcoseis.append([item for sublist in depth for item in sublist])
            fault_slip.append(Dbar)

            # -------------------- Event detph
            if np.isin(z,evdep):
                indx = np.where(z==evdep)[0]
                evslip[indx] = f(tstart_coseis)[indx]
        else:
            cscoseis,depcoseis,fault_slip = [],[],[]

    timeout = [tstart_coseis,tend_coseis]
    evout = [evslip,evdep,fault_slip]
    creepout = [cscreep,depcreep]
    coseisout = [cscoseis,depcoseis]
    if dt_interm > 0:
        intermout = [csinterm,depinterm]
    
    if dt_interm > 0:
        return [timeout, evout, creepout, coseisout, intermout]
    else:
        return [timeout, evout, creepout, coseisout]
    
def compute_spinup(outputs,dep,cuttime,cumslip_outputs,spin_up,rths,print_on=True):
    system_wide = analyze_events(cumslip_outputs,rths)[2]
    tstart = cumslip_outputs[0][0]
    evslip,evdep = cumslip_outputs[1][0:2]
    cscreep = cumslip_outputs[2][0]
    cscoseis = cumslip_outputs[3][0]

    var_mode = spin_up[0]
    spin_up = float(spin_up[1])
    interm = len(cumslip_outputs) > 4
    if var_mode == 'm':
        if print_on: print('Spin-up applied after slip > %2.2f m'%(spin_up))
        spin_up_idx = system_wide[np.where(evslip[system_wide]>=spin_up)[0][0]]
    elif var_mode == 'yrs':
        if print_on: print('Spin-up applied after %2.2f yrs'%(spin_up))
        spin_up_idx = system_wide[np.where(tstart[system_wide]/yr2sec>=spin_up)[0][0]]

    spup_cscreep = np.copy(cscreep)
    if interm:
        csinterm = cumslip_outputs[4][0]
        spup_csinterm = np.copy(csinterm)
    spup_cscoseis = np.copy(cscoseis)
    spup_evslip = np.copy(evslip)

    new_init_Sl = []
    new_init_dp = []
    c = 0
    for i in np.argsort(abs(dep)):
        z = abs(dep[i])        
        cumslip = np.array(outputs[i])[:,2]
        time = np.array(outputs[i])[:,0]

        if abs(cuttime) >= 1e-3:
            sliprate = sliprate[time <= cuttime]
            time = time[time <= cuttime]

        f = interpolate.interp1d(time,cumslip)
        new_init_Sl.append(f(tstart[spin_up_idx]))
        new_init_dp.append(z)
        
        spup_cscreep[c] = cscreep[c] - new_init_Sl[-1]
        if interm:
            spup_csinterm[c] = csinterm[c] - new_init_Sl[-1]
        spup_cscoseis[c] = cscoseis[c] - new_init_Sl[-1]

        if np.isin(z,evdep):
            indx = np.where(z==evdep)[0]
            spup_evslip[indx] = evslip[indx] - new_init_Sl[-1]
        c += 1
    
    new_inits = [new_init_Sl,new_init_dp]

    if interm:
        return [new_inits, spup_evslip, spup_cscreep, spup_cscoseis, spup_csinterm, spin_up_idx]
    else:
        return [new_inits, spup_evslip, spup_cscreep, spup_cscoseis, spin_up_idx]
