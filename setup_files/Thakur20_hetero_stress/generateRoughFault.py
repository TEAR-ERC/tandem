import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp, atan, cos, sin, log10, pi, tanh
import random

# random.seed()
from datetime import datetime
from numpy.random import rand
import configparser
from netCDF4 import Dataset

plt.rcParams.update({"font.size": 14})


def writeNetcdf(fname, x, y, aName, aData):
    "create a netcdf file"
    print("writing " + fname)
    nx = x.shape[0]
    ny = y.shape[0]
    with Dataset(fname, "w", format="NETCDF4") as rootgrp:
        rootgrp.createDimension("u", nx)
        rootgrp.createDimension("v", ny)

        vx = rootgrp.createVariable("u", "f4", ("u",))
        vx[:] = x
        vy = rootgrp.createVariable("v", "f4", ("v",))
        vy[:] = y
        for i in range(len(aName)):
            vTd = rootgrp.createVariable(aName[i], "f4", ("v", "u"))
            vTd[:, :] = aData[i][:, :]
    print("done writing " + fname)


def ComputeCosineTapper(t, width):
    cosinetapperS = np.ones(t.shape)
    # we tapper the data to have cleaner spectrograms
    width = 2.0
    ids = np.where(t < width)
    for ki in ids[0]:
        cosinetapperS[ki] = 0.5 * (1 - cos(pi * t[ki] / width))
    tf = t[-1]
    ids = np.where(t > (tf - width))
    for ki in ids[0]:
        cosinetapperS[ki] = 0.5 * (1 - cos(pi * (tf - t[ki]) / width))
    return cosinetapperS


def GenerateRoughTopography(N, Nmax, Pmax, lambdaMin, lambdaMaxS, lambdaMaxD, L, H):
    a = np.zeros(N * N, dtype=complex)
    a.shape = (N, N)
    beta = 2 * (H + 1.0)
    for i in range(0, Nmax + 1):
        # The loop over j is vectorialized
        randPhase = rand(Pmax + 1) * np.pi * 2.0
        # j index
        aJ = np.arange(0, Pmax + 1, 1, dtype=int)
        Kij = np.sqrt(i ** 2 + aJ ** 2)
        fac = np.power(Kij, -beta * 0.5)
        # remove lambda>lambdaMaxS
        val = (lambdaMaxS * i / L) ** 2 + (lambdaMaxD * aJ / L) ** 2
        fac[val < 1.0] = 0.0
        # remove lambda<lambdaMin
        val = (lambdaMin * i / L) ** 2 + (lambdaMin * aJ / L) ** 2
        fac[val > 1.0] = 0.0
        if i == 0:
            fac[0] = 0.0
        a[i, 0 : Pmax + 1] = fac * np.exp(randPhase * 1.0j)
        # h is real, therefore we have the following symetries
        # a[i,N-j] = np.conjugate(a[i,j])
        # a[N-i,j] = np.conjugate(a[i,j])
        # a[N-i,N-j] = a[i,j]
        if i != 0:
            a[N - i, 0 : Pmax + 1] = np.conjugate(a[i, 0 : Pmax + 1])
            a[i, N - 1 : N - 1 - (Pmax + 1) + 1 : -1] = np.conjugate(a[i, 1 : Pmax + 1])
            a[N - i, N - 1 : N - 1 - (Pmax + 1) + 1 : -1] = a[i, 1 : Pmax + 1]

    a = a * (N ** 2)
    h = np.real(np.fft.ifft2(a))
    return h


def GenerateRoughTopography2Loops(
    N, Nmax, Pmax, lambdaMin, lambdaMaxS, lambdaMaxD, L, H
):
    # This function is currently not in use
    vonKarman = False
    a = np.zeros(N * N, dtype=complex)
    a.shape = (N, N)
    beta = 2 * (H + 1.0)
    for i in range(0, Nmax + 1):
        randPhase = rand(2 * (Pmax + 1)) * np.pi * 2.0
        for j in range(0, Pmax + 1):
            if vonKarman:
                val = pow(lambdaMin * i / L, 2) + pow(lambdaMin * j / L, 2)
                if val > 1.0:
                    continue
                # k2=(lambdaMaxD*i/L)**2+(lambdaMaxS*j/L)**2
                # k2 = pow((1.*i)/Nmin,2)+pow((1.*j)/Pmin,2)
                # k2=i**2+j**2
                k2 = pow(lambdaMaxS * i / L, 2) + pow(lambdaMaxD * j / L, 2)
                fac = pow(1 + k2, -beta * 0.25)
            else:
                # Shi and Day procedure
                if max(i, j) == 0:
                    continue
                k = sqrt(i ** 2 + j ** 2)

                # dealing with anisotropy in lambdaMax:
                val = pow(lambdaMaxS * i / L, 2) + pow(lambdaMaxD * j / L, 2)
                if val < 1.0:
                    continue
                # same with lambdaMin:
                val = pow(lambdaMin * i / L, 2) + pow(lambdaMin * j / L, 2)
                if val > 1.0:
                    continue
                fac = pow(k, -beta * 0.5)

            a[i, j] = fac * np.exp(randPhase[2 * j] * 1.0j)

            # Copy conjugates to other quadrants, to get real values in spacial domain
            if i != 0:
                a[N - i, j] = np.conjugate(a[i, j])
            if j != 0:
                a[i, N - j] = np.conjugate(a[i, j])
            if min(i, j) > 0:
                a[N - i, N - j] = a[i, j]
    a = a * (N ** 2)
    h = np.real(np.fft.ifft2(a))
    return h


def generate_plot_gradient(x, y, normGradient, lambdaMin, lambdaMaxS, append2prefix):
    X, Y = np.meshgrid(x, y)
    xr = max(x) - min(x)
    yr = max(y) - min(y)
    fig, ax = plt.subplots(figsize=(10, 10 * yr / xr))
    m = plt.pcolormesh(X, -Y, normGradient, cmap="plasma")
    m.set_rasterized(True)
    plt.clim(0, 0.6)
    # plt.colorbar()
    plt.axis("equal")
    plt.gca().set_adjustable("box")
    plt.xlim([min(x), max(x)])
    fname = "output/Gradient%d_%d%s.svg" % (
        int(lambdaMin),
        int(lambdaMaxS),
        append2prefix,
    )
    plt.savefig(fname, bbox_inches="tight", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 5))
    bins = np.arange(0, 0.8, 0.025)
    plt.hist(normGradient.flatten(), bins)
    plt.xticks(np.arange(0, 0.8, 0.2))
    mytext = "10%%: %.2f\nmedian: %.2f\n90%%: %.2f" % (
        np.percentile(normGradient.flatten(), 10),
        np.median(normGradient.flatten()),
        np.percentile(normGradient.flatten(), 90),
    )
    plt.text(
        0.7,
        0.7,
        mytext,
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_xlabel("Slope [1/m]")

    fname = "output/Gradient_%s_hist.svg" % (append2prefix)
    plt.savefig(fname, bbox_inches="tight", dpi=150)


def GenerateRoughFault(Ls, Ld, fname):
    # Read roughness parameters
    config = configparser.ConfigParser()
    config.read(fname)
    lambdaMin = config.getfloat("roughness", "lambdaMin")
    lambdaMaxS = config.getfloat("roughness", "lambdaMaxS")
    lambdaMaxD = config.getfloat("roughness", "lambdaMaxD")
    alphaExp = config.getfloat("roughness", "alphaExp")
    distribution = config.get("roughness", "distribution")
    seed = config.getint("roughness", "seed")

    try:
        lambdaMaxSubSet = config.getfloat("roughness", "lambdaMaxSubSet")
        lambdaMinSubSet = config.getfloat("roughness", "lambdaMinSubSet")
        reverseSubset = config.getboolean("roughness", "reverseSubset")
        append2prefix = "subset_%d_%d_" % (int(lambdaMinSubSet), int(lambdaMaxSubSet))
        if reverseSubset:
            append2prefix = append2prefix + "_r"
        roughnessSubSet = True
    except configparser.NoOptionError:
        roughnessSubSet = False
        append2prefix = ""
    append2prefix = f"rough_{int(lambdaMin)}_{int(lambdaMaxS)}_seed{seed}_exp{alphaExp:.2f}_{append2prefix}"

    alpha = pow(10, alphaExp)
    np.random.seed(seed)

    startTime = datetime.now()
    L = max(Ls, Ld)

    # for a good discretisation of the largest wavelength (all directions)
    # (in any direction)
    # we want that 1 << k(lambdaMax)= (here 8)
    L = max(Ls, Ld, 8 * max(lambdaMaxS, lambdaMaxD))
    print("L", L)
    H = 1.0
    N = 512 * 8 * 2

    Nmax = int(L / lambdaMin) + 1
    Pmax = Nmax

    if max(Nmax, Pmax) >= N / 2:
        print(f"max(Nmax, Pmax)=max({Nmax},{Pmax})>=N/2({N/2}), increase N")
        quit()

    if distribution == "von Karman":
        vonKarman = True
        print("using von Karman")
    else:
        vonKarman = False
        print("band-limited self-similar fault")
    if vonKarman and (lambdaMaxS != lambdaMaxD):
        print("von Karman: this implementation requires lambdaMaxS==lambdaMaxD")

    dx = 1.0 * L / (N - 1)
    x = np.arange(0, Ls + dx, dx)
    y = np.arange(0.0, Ld + dx, dx)
    nx = x.shape[0]
    ny = y.shape[0]

    h = GenerateRoughTopography(N, Nmax, Pmax, lambdaMin, lambdaMaxS, lambdaMaxD, L, H)
    # Trim to fault dimensions
    h = h[0:ny, 0:nx]
    # compute rms roughness
    hrms = np.std(h)
    # scale to targeted Hrms
    targetHrms = alpha * max(lambdaMaxS, lambdaMaxD)
    print("Hrms: %f, target:%f" % (hrms, targetHrms))
    h = h * targetHrms / hrms
    print("done in:", datetime.now() - startTime)
    if roughnessSubSet:
        # In this case the first step was only meant to compute targetHrms/hrms
        np.random.seed(seed)
        hsub = GenerateRoughTopography(
            N, Nmax, Pmax, lambdaMinSubSet, lambdaMaxSubSet, lambdaMaxSubSet, L, H
        )
        if reverseSubset:
            h = h - hsub[0:ny, 0:nx] * targetHrms / hrms
            print("warning removing 100-200 (hardcoded)")
            hsub = GenerateRoughTopography(N, Nmax, Pmax, 100, 200, 200, L, H)
            h = h - hsub[0:ny, 0:nx] * targetHrms / hrms
        else:
            h = hsub[0:ny, 0:nx] * targetHrms / hrms
        print("done generating subset")

    if False:
        # cosinetapperS = ComputeCosineTapper(x, 1e-10*x[1])
        freq = np.fft.fftfreq(n=x.shape[-1], d=x[1] - x[0])
        N = int(np.shape(freq)[0] / 2)
        sp = np.zeros(np.shape(freq))
        for j in range(ny):
            # sp = sp + np.fft.fft(h[j,:]*cosinetapperS)
            sp = sp + np.fft.fft(h[j, :])
        sp = sp / ny
        plt.loglog(freq[0:N], np.abs(sp)[0:N], "b")
        # same in other dir
        # cosinetapperS = ComputeCosineTapper(y, 1e-10*y[1])
        freq = np.fft.fftfreq(n=y.shape[-1], d=y[1] - y[0])
        N = int(np.shape(freq)[0] / 2)
        sp = np.zeros(np.shape(freq))
        for i in range(nx):
            # sp = sp + np.fft.fft(h[:,i]*cosinetapperS)
            sp = sp + np.fft.fft(h[:, i])
        sp = sp / nx
        plt.loglog(freq[0:N], np.abs(sp)[0:N], "r")
        plt.loglog(
            freq[0:N],
            1e4 * np.power(1 + (freq[0:N] * lambdaMaxD) ** 2, -2.0 / 2),
            "k--",
        )
        plt.loglog(
            freq[0:N], 1e4 * np.power(1 + (freq[0:N] * lambdaMaxS) ** 2, -2.0 / 2), "k"
        )
        plt.legend(["dip", "strike", "dip2", "strike2"])
        if vonKarman:
            plt.title("vanKarman")
        else:
            plt.title("band-limited self-similar")
        plt.ylim(bottom=1e-4)
        plt.show()

    if False:
        X, Y = np.meshgrid(x, y)
        plt.pcolormesh(X, Y, h, cmap="PiYG")
        plt.colorbar()
        plt.show()

    # Might be an idea to keep
    SmoothedGeomNucleation = False
    print_h_nc = True
    print_gradient_nc = False

    if SmoothedGeomNucleation:
        # Smoothen geometry around nucleation
        Rnuc = 5e3
        print("Smoothening around nucleation")
        for i in range(0, nx):
            for j in range(0, ny):
                x1 = x[i]
                z1 = -y[j]
                dhypo = sqrt(pow(x1, 2) + pow(z1 + 10e3, 2))
                fac = 0.5 + 0.5 * tanh((dhypo - 0.70 * Rnuc) / (0.15 * Rnuc))
                h[j, i] = fac * h[j, i]

    if print_h_nc:
        print("printing h for latter postprocessing")
        fname = f"output/h_{append2prefix}.nc"
        writeNetcdf(fname, x, y, ["h"], [h])

    if print_gradient_nc:

        dy1, dy2 = np.gradient(h, dx)
        normGradient = np.sqrt(np.square(dy1) + np.square(dy2))

        fname = f"output/Gradient_{append2prefix}.nc"
        writeNetcdf(fname, x, y, ["grad"], [normGradient])

        generate_plot_gradient(x, y, normGradient, lambdaMin, lambdaMaxS, append2prefix)

    return (x, y, h)
