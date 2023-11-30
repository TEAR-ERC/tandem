# -------------------------------------------- Computing functions
import numpy as np
from scipy import integrate
from scipy import interpolate
from cumslip_compute import *
import pandas as pd

def compute_G(dep,time,tstart,tend,cumslip,shearT,quiet=True):
    G,Wr,D,ssd,dsd,Dtot = [],[],[],[],[],[]
    for iev in range(len(tstart)):
        if not quiet:
            print('Event',iev+1)
        Gev,Wev,Dev,ssdev,dsdev,dtotev = [],[],[],[],[],[]
        for idep in np.argsort(abs(dep)):
            its,ite = np.argmin(abs(time[idep]-tstart[iev]))-20, np.argmin(abs(time[idep]-tend[iev]))+20
            x=cumslip[idep][its:ite] - cumslip[idep][its]
            y=shearT[idep][its:ite]
            xb,yb,xr,yr,err_note = get_xy(x,y)
            if not quiet:
                print(err_note)

            if len(xb) == 0:
                Gev.append(np.nan)
                Wev.append(np.nan)
                Dev.append(np.nan)
                dsdev.append(np.nan)
                ssdev.append(np.nan)
                dtotev.append(np.nan)
            else:
                area_b = integrate.simpson(yb-np.min(yb),xb)
                area_r = integrate.simpson(yr-np.min(yr),xr)
                Gev.append(area_b*1e6)
                Wev.append(area_r*1e6)
                Dev.append(xb[-1]-xb[0])
                dsdev.append((yb[0]-np.min(yb))*1e6)
                dtotev.append(np.max(x))
                if len(yr)>0:
                    ssdev.append((yb[0]-yr[-1])*1e6)
                else:
                    ssdev.append((yb[0]-yb[-1])*1e6)
        Gev = np.array(Gev)
        Wev = np.array(Wev)
        Dev = np.array(Dev)
        dsdev = np.array(dsdev)
        ssdev = np.array(ssdev)
        dtotev = np.array(dtotev)

        if len(G) == 0:
            G = Gev.copy()
            Wr = Wev.copy()
            D = Dev.copy()
            dsd = dsdev.copy()
            ssd = ssdev.copy()
            Dtot = dtotev.copy()
        else:
            G = np.vstack((G,Gev))
            Wr = np.vstack((Wr,Wev))
            D = np.vstack((D,Dev))
            dsd = np.vstack((dsd,dsdev))
            ssd = np.vstack((ssd,ssdev))
            Dtot = np.vstack((Dtot,dtotev))
    return G,D,dsd,Wr,ssd,Dtot

def load_csv(fnames):
    G,D,L = [],[],[]
    for fn in fnames:
        if not 'reference' in fn:
            print(fn)
            df = pd.read_csv(fn,sep=';',encoding= 'unicode_escape',usecols=['value (J/m^2)','value (m)'],na_values= ['#DIV/0!',' '],dtype='float').dropna().to_numpy().T
            L.append(len(df[0]))
            Gi,Di = df[0],df[1]
            if len(G) == 0:
                G = Gi.copy()
                D = Di.copy()
            else:
                G = np.hstack((G,Gi))
                D = np.hstack((D,Di))
    return G,D,np.array(L)

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
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
 
    return (b_0, b_1)

def get_xy(x,y):
    err_note = ''
    if sum(x<0) > 0:
        err_note += 'negative cumslip adjusted '
        x = abs(x)
    while not np.all(np.diff(x) > 0.0):
        if len(np.where(np.diff(x)<0)[0]) > 0:
            err_note += '/ decreasing cumslip adjusted '
            x[np.where(np.diff(x)<0)[0]] = (x[np.where(np.diff(x)<0)[0]-1]+x[np.where(np.diff(x)<0)[0]+1])/2
        ix = np.where(np.diff(x) > 0)[0]+1
        x = x[ix]; y = y[ix]
    if len(np.where(np.diff(x) > 0)[0]) < 2:
        err_note += '/ no points left - discarded'
        xb,yb,xr,yr = [],[],[],[]
    else:
        if len(x) < 500:
            fx = interpolate.interp1d(np.arange(len(x)),x)
            fy = interpolate.interp1d(np.arange(len(y)),y)
            x = fx(np.linspace(0,len(x)-1,500))
            y = fy(np.linspace(0,len(y)-1,500))
        xb = x[:np.argmin(y)+1]; yb = y[:np.argmin(y)+1]
        xr = x[np.argmin(y):]; yr = y[np.argmin(y):]
        if len(np.where(np.diff(xb) > 0)[0]) < 2:
            err_note += '/ no points left2 - discarded2'
            xb,yb,xr,yr = [],[],[],[]
    if len(err_note) > 0 and err_note[0] == '/':
        err_note = err_note[2:]
    return xb,yb,xr,yr,err_note

def logxy(xr,b1,b0):
    y = np.power(10,b0 + b1*np.log10(xr))
    return y

def minmax(rmin,rmax,var,zero_out=False):
    if zero_out:
        varmin = np.nanmin(var[abs(var)>0])
    else:
        varmin = np.nanmin(var)
    if varmin < rmin:
        rmin = varmin
    if np.nanmax(var) > rmax:
        rmax = np.nanmax(var)
    return rmin,rmax