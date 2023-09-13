#!/usr/bin/env python3
'''
Functions related to plotting cumulative slip vs. depth plot
By Jeena Yun
Last modification: 2023.07.24.
'''
import numpy as np
from scipy import interpolate

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

    # Define events by peak sliprate
    if Vlb > 0:
        events = np.where(np.logical_and(psr < Vths,psr > Vlb))[0]
    else:
        events = np.where(psr > Vths)[0]

    if len(events) > 0:
        jumps = np.where(np.diff(events)>1)[0]+1

        if False:
            tmp_tstart = np.zeros(len(jumps)+1)
            tmp_tend = np.zeros(len(jumps)+1)
            tmp_evdep = np.zeros(len(jumps)+1)
            if len(jumps) > 0:
                for j in range(len(jumps)+1):
                    if j==0:
                        tmp_tstart[j] = time[events][0]
                        tmp_tend[j] = time[events][jumps[0]-1]
                        tmp_evdep[j] = pd[events][0]
                    elif j == len(jumps):
                        tmp_tstart[j] = time[events][jumps[j-1]]
                        tmp_tend[j] = time[events][len(events)-1]
                    else:
                        tmp_tstart[j] = time[events][jumps[j-1]]
                        tmp_tend[j] = time[events][jumps[j]-1]
        else:
            tmp_tstart = time[events][np.hstack(([0],jumps))]
            tmp_tend = time[events][np.hstack((jumps-1,len(events)-1))]
            tmp_evdep = pd[events][np.hstack(([0],jumps))]

        ii = np.where(tmp_tend-tmp_tstart>=dt_coseismic)[0]
        tstart = tmp_tstart[ii]
        tend = tmp_tend[ii]
        amax = tmp_evdep[ii]
        evdep = z[tmp_evdep[ii]]

        its_all = np.array([np.argmin(abs(time-t)) for t in tstart])
        ite_all = np.array([np.argmin(abs(time-t)) for t in tend])
        diffcrit = np.quantile(abs(np.diff(np.log10(psr[events]))),0.98)
        new_its_all = its_all.copy()
        if False:
            for k,ts in enumerate(its_all):
                psr_inc = abs(np.diff(np.log10(psr)))[ts-1]
                newts = ts.copy()
                while psr_inc < diffcrit:
                    newts += 1
                    psr_inc = abs(np.diff(np.log10(psr)))[newts-1]
                new_its_all[k] = newts
        else:
            for k,ts in enumerate(its_all):
                psr_inc = abs(np.diff(np.log10(psr)))[ts-1]
                width = int((ite_all[k] - ts)*intv)
                large_diffs = np.where(abs(np.diff(np.log10(psr)))[ts-1:ts-1+width]>=diffcrit)[0]
                if psr_inc < diffcrit and len(large_diffs) > 0:
                    new_its_all[k] = large_diffs[0] + ts
        evdep = z[pd[new_its_all]]
        tstart = time[new_its_all]

        varsr = np.array([np.log10(psr[its_all[k]:ite_all[k]+1]).max()-np.log10(psr[its_all[k]:ite_all[k]+1]).min() for k in range(len(tstart))])
        ii = np.where(varsr/abs(np.log10(Vths))>=0.1)[0]
        if len(ii) < len(tstart):
            print('Negligible events with SR variation < 0.1Vths:',np.where(varsr/abs(np.log10(Vths))<0.1)[0])
            tstart = tstart[ii]
            tend = tend[ii]
            evdep = evdep[ii]
        else:
            print('All safe from the SR variation criterion')
    else:
        tstart, tend, evdep = [],[],[]
    return tstart, tend, evdep

def old_event_times(dep,outputs,Vlb=0,Vths=1e-2,cuttime=0,mingap=60,print_on=True):
    c = 0
    for i in np.argsort(abs(dep)):
        z = abs(dep[i])
        time = np.array(outputs[i])[:,0]
        sliprate = np.array(outputs[i])[:,4]
        cumslip = np.array(outputs[i])[:,2]

        if abs(cuttime) >= 1e-3:
            if cuttime > np.max(time):
                raise ValueError('Cuttime larger than total simulation time - check again')
            sliprate = sliprate[time <= cuttime]
            cumslip = cumslip[time <= cuttime]
            time = time[time <= cuttime]

        # Define events by sliprate
        if Vlb > 0:
            events = np.where(np.logical_and(sliprate < Vths,sliprate > Vlb))[0]
        else:
            events = np.where(sliprate > Vths)[0]

        if len(events) == 0:
            if print_on: print('Depth',z,' - no events')
            continue
        else:
            # Get indexes for the dynamic rupture components
            jumps = np.where(np.diff(events)>1)[0]+1

            # Get event start/end time for current depth
            tmp_tstart = np.zeros(len(jumps)+1)
            tmp_tend = np.zeros(len(jumps)+1)
            tmp_evdep = z*np.ones(len(jumps)+1)
            if len(jumps) > 0:
                for j in range(len(jumps)+1):
                    if j==0:
                        tmp_tstart[j] = time[events][0]
                        tmp_tend[j] = time[events][jumps[0]-1]
                    elif j == len(jumps):
                        tmp_tstart[j] = time[events][jumps[j-1]]
                        tmp_tend[j] = time[events][len(events)-1]
                    else:
                        tmp_tstart[j] = time[events][jumps[j-1]]
                        tmp_tend[j] = time[events][jumps[j]-1]
            else:
                continue
            
            if c == 0:
                # When first depth, initiate tstart
                tstart = np.copy(tmp_tstart)
                tend = np.copy(tmp_tend)
                evdep = np.copy(tmp_evdep)
            else:
                if len(tmp_tstart) > len(tstart):
                    # More number of events
                    long_tstart = np.copy(tmp_tstart)
                    long_tend = np.copy(tmp_tend)
                    long_evdep = np.copy(tmp_evdep)
                    short_tstart = np.copy(tstart)
                    short_tend = np.copy(tend)
                    short_evdep = np.copy(evdep)
                elif len(tmp_tstart) <= len(tstart):
                    # Less number of events
                    long_tstart = np.copy(tstart)
                    long_tend = np.copy(tend)
                    long_evdep = np.copy(evdep)
                    short_tstart = np.copy(tmp_tstart)
                    short_tend = np.copy(tmp_tend)
                    short_evdep = np.copy(tmp_evdep)

                # Iteratively update current event start/end time
                new_tstart = np.copy(long_tstart)
                new_tend = np.copy(long_tend)
                new_evdep = np.copy(long_evdep)

                want_append = 0
                for k in range(len(short_tstart)):
                    # Cases when the event need to be inserted
                    if short_tstart[k] > max(long_tend):
                        want_append = 1
                    else:
                        same_event = np.argmin(abs(short_tstart[k] - long_tstart))
                        if long_tstart[same_event] > short_tstart[k]:
                            if same_event > 0:
                                if short_tstart[k] > long_tend[same_event-1] and short_tend[k] < long_tstart[same_event]:
                                    want_append = 1
                                elif short_tstart[k] < long_tend[same_event-1] and short_tend[k] > long_tstart[same_event]:
                                    want_append = 1
                            if same_event == 0 and short_tend[k] < long_tstart[0]:
                                want_append = 1
                        elif long_tstart[same_event] < short_tstart[k]:
                            if same_event+1 < len(long_tstart):
                                if short_tstart[k] > long_tend[same_event] and short_tend[k] < long_tstart[same_event+1]:
                                    want_append = 1
                                elif short_tstart[k] < long_tend[same_event] and short_tend[k] > long_tstart[same_event+1]:
                                    want_append = 1
                            if same_event+1 == len(long_tstart) and short_tstart[k] > long_tend[-1]:
                                want_append = 1            

                        if want_append:
                            new_tstart = np.append(new_tstart,short_tstart[k])
                            new_tend = np.append(new_tend,short_tend[k])
                            new_evdep = np.append(new_evdep,short_evdep[k])
                        else:
                            # Cases when the either start or end time needs to be adjusted
                            if long_tstart[same_event] > short_tstart[k]:
                                if same_event == 0:
                                    new_tstart[same_event] = short_tstart[k]
                                    new_evdep[same_event] = short_evdep[k]
                                elif short_tstart[k] > long_tend[same_event-1]:
                                    new_tstart[same_event] = short_tstart[k]
                                    new_evdep[same_event] = short_evdep[k]
                                    if short_tend[k] > long_tend[same_event]:
                                        new_tend[same_event] = short_tend[k]
                                elif short_tstart[k] < long_tend[same_event-1] and short_tend[k] > long_tend[same_event-1]:
                                    new_tend[same_event-1] = short_tend[k]

                            elif long_tstart[same_event] < short_tstart[k]:
                                if short_tstart[k] > long_tend[same_event]:
                                    new_tstart[same_event+1] = short_tstart[k]
                                    new_evdep[same_event+1] = short_evdep[k]
                                    if same_event+1 < len(long_tstart) and short_tend[k] > long_tend[same_event+1]:
                                        new_tend[same_event+1] = short_tend[k]
                                elif short_tstart[k] < long_tend[same_event] and short_tend[k] > long_tend[same_event]:
                                    new_tend[same_event] = short_tend[k]

                            elif long_tstart[same_event] == short_tstart[k]:
                                if short_tend[k] > long_tend[same_event]:
                                    new_tend[same_event] = short_tend[k]
                    want_append = 0
                
                # Sort and update start/end time
                ii = np.argsort(new_tstart)
                tstart = np.copy(new_tstart[ii])
                tend = np.copy(new_tend[ii])
                evdep = np.copy(new_evdep[ii])
        c += 1

    # If there are too close events, merge them as one
    new_tstart = []
    new_tend = []
    new_evdep = []
    u = 0
    while u < len(tstart):
        nearest = np.where(tstart[tstart > tstart[u]] - tstart[u] <= mingap)[0] + u + 1
        if len(nearest) != 0:
            new_tstart.append(tstart[u])
            new_tend.append(tend[max(nearest)])
            new_evdep.append(evdep[u])
            u += len(nearest)+1
        else:
            new_tstart.append(tstart[u])
            new_tend.append(tend[u])
            new_evdep.append(evdep[u])
            u += 1

    tstart = np.copy(new_tstart)
    tend = np.copy(new_tend)
    evdep = np.copy(new_evdep)

    return tstart, tend, evdep

def old_compute_cumslip(outputs,dep,cuttime,Vlb,Vths,dt_creep,dt_coseismic,dt_interm,mingap,print_on=True):
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

    cscreep = []
    depcreep = []
    cscoseis = []
    depcoseis = []
    fault_slip = []
    if dt_interm > 0:        
        csinterm = []
        depinterm = []

    # Obtain globally min. event start times and max. event tend times
    if dt_interm > 0:
        tstart_interm, tend_interm, evdep = old_event_times(dep,outputs,Vlb,Vths,cuttime,mingap,print_on)
    tstart_coseis, tend_coseis, evdep = old_event_times(dep,outputs,0,Vths,cuttime,mingap,print_on)
    evslip = np.zeros(tstart_coseis.shape)

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

    cscreep = []
    depcreep = []
    cscoseis = []
    depcoseis = []
    fault_slip = []
    if dt_interm > 0:        
        csinterm = []
        depinterm = []

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
    
def compute_spinup(outputs,dep,cuttime,cumslip_outputs,spin_up,print_on=True):
    tstart,tend = cumslip_outputs[0]
    evslip,evdep,fault_slip = cumslip_outputs[1]
    cscreep,depcreep = cumslip_outputs[2]
    cscoseis,depcoseis = cumslip_outputs[3]

    time = np.array(outputs[0])[:,0]
    var_mode = spin_up[0]
    spin_up = float(spin_up[1])
    interm = len(cumslip_outputs) > 4
    if var_mode == 'm':
        if print_on: print('Spin-up applied after slip > %2.2f m'%(spin_up))
        spin_up_idx = np.where(evslip>=spin_up)[0][0]
    elif var_mode == 'yrs':
        if print_on: print('Spin-up applied after %2.2f yrs'%(spin_up))
        spin_up_idx = np.where(tstart/yr2sec>=spin_up)[0][0]

    spup_cscreep = np.copy(cscreep)
    if interm:
        csinterm,depinterm = cumslip_outputs[4]
        spup_csinterm = np.copy(csinterm)
    spup_cscoseis = np.copy(cscoseis)
    spup_evslip = np.copy(evslip)

    new_init_Sl = []
    new_init_dp = []
    c = 0
    for i in np.argsort(abs(dep)):
        z = abs(dep[i])        
        cumslip = np.array(outputs[i])[:,2]

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

def old_compute_spinup(outputs,dep,cuttime,cumslip_outputs,spin_up,print_on=True):
    # cumslip_outputs = [timeout, evout, creepout, coseisout, intermout]
    # cumslip_outputs[0] = [tstart_coseis,tend_coseis]
    # cumslip_outputs[1] = [evslip,evdep,fault_slip]
    # cumslip_outputs[2] = [cscreep,depcreep]
    # cumslip_outputs[3] = [cscoseis,depcoseis]
    # cumslip_outputs[4] = [csinterm,depinterm]
    tstart,tend = cumslip_outputs[0]
    evslip,evdep,fault_slip = cumslip_outputs[1]
    cscreep,depcreep = cumslip_outputs[2]
    cscoseis,depcoseis = cumslip_outputs[3]
    if len(cumslip_outputs) > 4:
        interm = True
    else:
        interm = False
    if print_on: print('Spin-up applied after slip > %2.2f m'%(spin_up))
    spin_up_idx = np.where(evslip>spin_up)[0][0]
    spup_cscreep = np.copy(cscreep)
    if interm:
        csinterm,depinterm = cumslip_outputs[4]
        spup_csinterm = np.copy(csinterm)
    spup_cscoseis = np.copy(cumslip_outputs[3][0])
    spup_evslip = np.copy(evslip)

    new_init_Sl = []
    new_init_dp = []
    c = 0
    for i in np.argsort(abs(dep)):
        z = abs(dep[i])
        time = np.array(outputs[i])[:,0]
        cumslip = np.array(outputs[i])[:,2]

        if abs(cuttime) >= 1e-3:
            sliprate = sliprate[time <= cuttime]
            time = time[time <= cuttime]

        f = interpolate.interp1d(time,cumslip)
        new_init_Sl.append(f(cumslip_outputs[0][0][spin_up_idx]-1))
        new_init_dp.append(z)
        
        spup_cscreep[c] = cscreep[c] - new_init_Sl[-1]
        if interm:
            spup_csinterm[c] = csinterm[c] - new_init_Sl[-1]
        spup_cscoseis[c] = cumslip_outputs[3][0][c] - new_init_Sl[-1]

        if np.isin(z,evdep):
            indx = np.where(z==evdep)[0]
            spup_evslip[indx] = evslip[indx] - new_init_Sl[-1]
        c += 1
    
    new_inits = [new_init_Sl,new_init_dp]

    if interm:
        return [new_inits, spup_evslip, spup_cscreep, spup_cscoseis, spup_csinterm, spin_up_idx]
    else:
        return [new_inits, spup_evslip, spup_cscreep, spup_cscoseis, spin_up_idx]
    
def cluster_events(cumslip_outputs):
    tstart = cumslip_outputs[0][0]
    event_gap = np.diff(tstart)/yr2sec
    event_cluster = [[0,0]]
    ci = 0
    for k,eg in enumerate(event_gap):
        if eg > 1:
            event_cluster.append([k+1,k+1])
            ci += 1
        else:
            event_cluster[ci][1] = k+1
    return np.array(event_cluster)

def analyze_events(cumslip_outputs,rths):
    from scipy import integrate,interpolate
    rupture_length = []
    av_slip = []
    if len(cumslip_outputs[3][1]) > 0:
        fault_z = np.array(cumslip_outputs[3][1]).T[0]
        fault_slip = np.array(cumslip_outputs[1][2]).T
        event_cluster = cluster_events(cumslip_outputs)

        for ti in range(fault_slip.shape[0]):
            fs = fault_slip[ti]
            Sths = 1e-2
            ii = np.where(fs>Sths)[0]
            if min(ii) > 0:
                ii = np.hstack(([min(ii)-1],ii))
            if max(ii) < len(fs)-1:
                ii = np.hstack((ii,[max(ii)+1]))
            rl = max(fault_z[ii])-min(fault_z[ii])
            f = interpolate.interp1d(fault_z,fs)
            npts = 1000
            newz = np.linspace(min(fault_z),max(fault_z),npts)
            Dbar = integrate.simpson(f(newz),newz)/rl
            rupture_length.append(rl)
            av_slip.append(Dbar)

        rupture_length = np.array(rupture_length)
        partial_rupture = np.where(rupture_length<rths)[0]
        system_wide = np.where(rupture_length>=rths)[0]

        lead_fs,major_pr,minor_pr = [],[],[]
        for k,ec in enumerate(event_cluster):
            if sum([np.logical_and(sw>=ec[0],sw<=ec[1]) for sw in system_wide]) >= 1:
                if ec[0] not in system_wide:
                    lead_fs.append(ec[0])
            elif ec[1]-ec[0]<=4:
                # minor_pr.append(ec[0])
                minor_pr.append(k)
            else:
                # major_pr.append(ec[0])
                major_pr.append(k)
        lead_fs = np.array(lead_fs)
        major_pr = np.array(major_pr)
        minor_pr = np.array(minor_pr)
    else:
        print('No events')
        rupture_length,av_slip,system_wide,partial_rupture,event_cluster,lead_fs,major_pr,minor_pr = \
                [],[],[],[],[],[],[],[]
    return rupture_length,av_slip,system_wide,partial_rupture,event_cluster,lead_fs,major_pr,minor_pr