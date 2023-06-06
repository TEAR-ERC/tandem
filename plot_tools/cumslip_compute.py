#!/usr/bin/env python3
'''
Functions related to plotting cumulative slip vs. depth plot
By Jeena Yun
Last modification: 2023.05.18.
'''
import numpy as np
from scipy import interpolate

yr2sec = 365*24*60*60
wk2sec = 7*24*60*60

def event_times(dep,outputs,Vlb,Vths,cuttime,mingap,print_on=True):
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

def compute_cumslip(outputs,dep,cuttime,Vlb,Vths,dt_creep,dt_coseismic,dt_interm,mingap,print_on=True):
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
        tstart_interm, tend_interm, evdep = event_times(dep,outputs,Vlb,Vths,cuttime,mingap,print_on)
    tstart_coseis, tend_coseis, evdep = event_times(dep,outputs,0,Vths,cuttime,mingap,print_on)
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
    
def compute_spinup(outputs,dep,cuttime,cumslip_outputs,spin_up,print_on=True):
    # cumslip_outputs = [timeout, evout, creepout, coseisout, intermout]
    # cumslip_outputs[0] = [tstart_coseis,tend_coseis]
    # cumslip_outputs[1] = [evslip,evdep,fault_slip]
    # cumslip_outputs[2] = [cscreep,depcreep]
    # cumslip_outputs[3] = [cscoseis,depcoseis]
    # cumslip_outputs[4] = [csinterm,depinterm]
    if len(cumslip_outputs) > 4:
        interm = True
    else:
        interm = False
    if print_on: print('Spin-up applied after slip > %2.2f m'%(spin_up))
    spin_up_idx = np.where(cumslip_outputs[1][0]>spin_up)[0][0]
    spup_cscreep = np.copy(cumslip_outputs[2][0])
    if interm:
        spup_csinterm = np.copy(cumslip_outputs[4][0])
    spup_cscoseis = np.copy(cumslip_outputs[3][0])
    spup_evslip = np.copy(cumslip_outputs[1][0])

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
        
        spup_cscreep[c] = cumslip_outputs[2][0][c] - new_init_Sl[-1]
        if interm:
            spup_csinterm[c] = cumslip_outputs[4][0][c] - new_init_Sl[-1]
        spup_cscoseis[c] = cumslip_outputs[3][0][c] - new_init_Sl[-1]

        if np.isin(z,cumslip_outputs[1][1]):
            indx = np.where(z==cumslip_outputs[1][1])[0]
            spup_evslip[indx] = cumslip_outputs[1][0][indx] - new_init_Sl[-1]
        c += 1
    
    new_inits = [new_init_Sl,new_init_dp]

    if interm:
        return [new_inits, spup_evslip, spup_cscreep, spup_cscoseis, spup_csinterm, spin_up_idx]
    else:
        return [new_inits, spup_evslip, spup_cscreep, spup_cscoseis, spin_up_idx]
    
def analyze_events(cumslip_outputs,rths):
    from scipy import integrate
    rupture_length = []
    av_slip = []
    fault_z = np.array(cumslip_outputs[3][1]).T[0]
    fault_slip = np.array(cumslip_outputs[1][2]).T

    for ti in range(fault_slip.shape[0]):
        fs = fault_slip[ti]
        Sths = max(fs)*0.01
        rl = max(fault_z[fs>Sths])-min(fault_z[fs>Sths])        
        Dbar = integrate.simpson(fs[fs>Sths],fault_z[fs>Sths])/rl
        rupture_length.append(rl)
        av_slip.append(Dbar)

    rupture_length = np.array(rupture_length)
    partial_rupture = np.where(rupture_length<rths)[0]
    system_wide = np.where(rupture_length>=rths)[0]
    return rupture_length,av_slip,system_wide,partial_rupture