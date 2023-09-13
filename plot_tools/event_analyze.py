#!/usr/bin/env python3
'''
Functions related to plotting cumulative slip vs. depth plot
By Jeena Yun
Last modification: 2023.09.09.
'''
import numpy as np

yr2sec = 365*24*60*60
wk2sec = 7*24*60*60


def compute_STF(save_dir,outputs,dep,cumslip_outputs):
    from scipy import integrate, interpolate
    tstart, tend = cumslip_outputs[0]
    params = np.load('%s/const_params.npy'%(save_dir),allow_pickle=True)
    time = np.array([outputs[i][:,0] for i in np.argsort(abs(dep))])
    sr = abs(np.array([outputs[i][:,4] for i in np.argsort(abs(dep))]))
    z = np.sort(abs(dep))*1e3
    if 'DZ' in save_dir:
        mu = params.item().get('mu_damage')*1e9
    else:
        mu = params.item().get('mu')*1e9

    npoints = 500
    f = np.array([mu * integrate.simpson(sr[:,t],z) for t in range(sr.shape[1])])
    stf = interpolate.interp1d(time[0],f)
    Fdot=np.array([stf(np.linspace(tstart[iev],tend[iev],npoints)) for iev in range(len(tstart))])
    t = np.array([np.linspace(tstart[iev],tend[iev],npoints)-tstart[iev] for iev in range(len(tstart))])
    return t,Fdot

def compute_M0(save_dir,rupture_length,av_slip,mode,Mw):
    params = np.load('%s/const_params.npy'%(save_dir),allow_pickle=True)
    if 'DZ' in save_dir:
        mu = params.item().get('mu_damage')*1e9
    else:
        mu = params.item().get('mu')*1e9
    
    rupture_length *= 1e3
    if mode == '1d':
        print('1D: Moment per length')
        M0 = np.array([mu * rupture_length[iev] * av_slip[iev] for iev in range(len(av_slip))])
    elif mode == 'approx2d':
        print('Approximated 2D: Moment assuming a square fault patch')
        M0 = np.array([mu * (rupture_length[iev]**2) * av_slip[iev] for iev in range(len(av_slip))])
    if Mw:
        print('Output in moment magnitude (Mw) instead of moment (M0)')
        return 2/3*(np.log10(M0)-9.1)
    else:
        return M0
    
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
 
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
 
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
 
    # calculating regression coefficients
    b = SS_xy / SS_xx
    a = m_y - b*m_x
 
    return (a, b)

def compute_GR(save_dir,cumslip_outputs,spin_up_idx,rths,cutoff_Mw,npts):
    rupture_length,av_slip = analyze_events(cumslip_outputs,rths)[:2]
    Mw = compute_M0(save_dir,rupture_length,av_slip,mode='approx2d',Mw=True)
    Mw = Mw[spin_up_idx:]
    # baseval = np.linspace(min(Mw),max(Mw),npts)[:-1]
    # N = np.array([sum(Mw > mag) for mag in baseval])
    baseval = np.linspace(min(Mw),max(Mw),npts)
    # baseval = np.sort(Mw)
    N = np.array([sum(Mw >= mag) for mag in baseval])
    if cutoff_Mw == 0:
        x = baseval; y = np.log10(N)
    else:
        # ii = np.where(baseval>=cutoff_Mw)[0]; x = baseval[ii]; y = np.log10(N[ii])
        ii = np.where(baseval<=cutoff_Mw)[0]; x = baseval[ii]; y = np.log10(N[ii])
    a,b = estimate_coef(x,y)
    yN = np.power(10,a + b*x)
    return baseval,N,b,x,yN,a

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
    from scipy import integrate, interpolate
    rupture_length = []
    av_slip = []
    if len(cumslip_outputs[3][1]) > 0:
        fault_z = np.array(cumslip_outputs[3][1]).T[0]
        fault_slip = np.array(cumslip_outputs[1][2]).T
        event_cluster = cluster_events(cumslip_outputs)

        for ti in range(fault_slip.shape[0]):
            fs = fault_slip[ti]
            try:
                Sths = 1e-2
                ii = np.where(fs>Sths)[0]
                if min(ii) > 0:
                    ii = np.hstack(([min(ii)-1],ii))
                if max(ii) < len(fs)-1:
                    ii = np.hstack((ii,[max(ii)+1]))
            except ValueError:
                Sths = 1e-3
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
                minor_pr.append(k)
            else:
                major_pr.append(k)
        lead_fs = np.array(lead_fs)
        major_pr = np.array(major_pr)
        minor_pr = np.array(minor_pr)
    else:
        print('No events')
        rupture_length,av_slip,system_wide,partial_rupture,event_cluster,lead_fs,major_pr,minor_pr = \
                [],[],[],[],[],[],[],[]
    return rupture_length,av_slip,system_wide,partial_rupture,event_cluster,lead_fs,major_pr,minor_pr

