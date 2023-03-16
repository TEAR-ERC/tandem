#!/usr/bin/env python3
import numpy as np
from math import tan, pi, pow, cos, sin, atan2
import os
from datetime import datetime
from scipy.interpolate import interp1d, splprep, splev, RectBivariateSpline
import argparse
from enum import IntEnum


class dipType(IntEnum):
    CONSTANT = 0
    DEPTH_DEPENDENT = 1
    STRIKE_DEPENDENT = 2


parser = argparse.ArgumentParser(
    description="create curved fault geometry from pl file"
)
parser.add_argument(
    "filename", help="fault trace (*.pl) or ascii file (2 or 3 columns)"
)
parser.add_argument(
    "dipType",
    type=int,
    help="0: constant dip, 1: depth dependant dip, described by an ascii file, 2: dip variying along the length of the trace",
)
parser.add_argument(
    "dipDesc",
    help="dipType=0: dip value dipType=1 name of ascii file with 2 columns (depth, dip). dipType=2: idem with (relative length[0-1], dip)",
)
parser.add_argument(
    "--extrudeDir",
    nargs=1,
    metavar=("strike"),
    help="strike direction used to extrude the fault trace. Described by an ascii file: relative length[0-1] strike",
)
parser.add_argument(
    "--translate",
    nargs=2,
    metavar=("x0", "y0"),
    default=([0, 0]),
    help="translates all nodes by (x0,y0)",
    type=float,
)
parser.add_argument(
    "--dd",
    nargs=1,
    metavar=("dd"),
    default=([1e3]),
    help="sampling along depth (m)",
    type=float,
)
parser.add_argument(
    "--maxdepth",
    nargs=1,
    metavar=("maxdepth"),
    default=([20e3]),
    help="max depth (positive) of fault (m)",
    type=float,
)
parser.add_argument(
    "--extend",
    nargs=1,
    metavar=("extend"),
    default=([00e3]),
    help="extend toward z= extend (positive)",
    type=float,
)
parser.add_argument(
    "--first_node_ext",
    nargs=1,
    default=([0.0]),
    help="extend along strike trace before the first node (arg: length in km)",
    type=float,
)
parser.add_argument(
    "--last_node_ext",
    nargs=1,
    default=([0.0]),
    help="extend along strike trace after the last node (arg: length in km)",
    type=float,
)
parser.add_argument(
    "--proj",
    nargs=1,
    metavar=("projname"),
    help="transform vertex array to projected system.\
projname: name of the (projected) Coordinate Reference System (CRS) (e.g. EPSG:32646 for UTM46N)",
)
parser.add_argument(
    "--addRoughness",
    nargs=1,
    dest="addRoughness",
    help="add fault roughness to the fault, parametrized by roughness.ini",
)
parser.add_argument(
    "--smoothingParameter",
    nargs=1,
    metavar=("smoothingParameter"),
    default=([1e5]),
    help="smoothing parameter (the bigger the smoother)",
    type=float,
)
parser.add_argument(
    "--plotFaultTrace",
    dest="plotFaultTrace",
    action="store_true",
    default=False,
    help="plot resampled fault trace in matplotlib",
)
args = parser.parse_args()

if not os.path.exists("output"):
    os.makedirs("output")


def cross(left, right):
    # numpy cross is very inefficient
    x = (left[1] * right[2]) - (left[2] * right[1])
    y = (left[2] * right[0]) - (left[0] * right[2])
    z = (left[0] * right[1]) - (left[1] * right[0])
    return np.array([x, y, z])


def smooth(y, box_hpts):
    # modified from https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    box_pts = 2 * box_hpts + 1
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    y_smooth[0:box_hpts] = y[0:box_hpts]
    y_smooth[-box_hpts:] = y[-box_hpts:]
    return y_smooth


def Project(nodes):
    print("Projecting the nodes coordinates")
    # project the data to geocentric (lat, lon)
    from pyproj import Transformer

    transformer = Transformer.from_crs("epsg:4326", args.proj[0], always_xy=True)
    nodes[:, 0], nodes[:, 1] = transformer.transform(nodes[:, 0], nodes[:, 1])
    return nodes


def compute_rel_curvilinear_coordinate(nodes):
    # compute the relative curvilinear coordinate along strike
    dist = np.linalg.norm(nodes[1:nx] - nodes[0 : nx - 1], axis=1)
    distall = np.sum(dist)
    reldist_seg = dist / distall
    reldist = np.zeros(nx)
    for i in range(1, nx):
        reldist[i] = reldist[i - 1] + reldist_seg[i - 1]
    reldist[reldist > 1] = 1.0
    return reldist


def generate_vertices_constant_dip(maxdepth, sign=1):
    """
    extends 2d fault trace (nodes array) along constant dip
    returns a 3d array of vertices coordinates
    """
    nd = int(maxdepth / (sin(dip) * dx))
    vertices = np.zeros((nx, nd, 3))
    vertices[:, 0, :] = nodes
    one_over_tan_dip = 1.0 / tan(dip)
    for i in range(0, nx):
        ud = -(one_over_tan_dip * av1[i, :] - uz)
        ud = ud / np.linalg.norm(ud)
        for j in range(1, nd):
            vertices[i, j, :] = vertices[i, j - 1, :] - sign * dx * ud
    return vertices


def generate_vertices_dip_vary_along_strike(maxdepth, sign=1):
    """
    extends 2d fault trace (nodes array) along dip, with dip varying along strike
    In this case we use constant depth strides between along dip vertices
    that is the size of the mesh along dip will vary with dip
    returns a 3d array of vertices coordinates
    """
    nd = int(maxdepth / dx)
    vertices = np.zeros((nx, nd, 3))
    vertices[:, 0, :] = nodes
    for i in range(0, nx):
        one_over_tan_dip = 1.0 / tan(aDip[i])
        for j in range(1, nd):
            ud = -(one_over_tan_dip * av1[i, :] - uz)
            vertices[i, j, :] = vertices[i, j - 1, :] - sign * dx * ud
    return vertices


def generate_vertices_dip_vary_along_dip(maxdepth, sign=1):
    """
    extends 2d fault trace (nodes array) along depth-dependent dip
    returns a 3d array of vertices coordinates
    """
    # compute depth array
    current_depth = 0.0
    depth = []
    while current_depth <= maxdepth:
        depth.append(current_depth)
        current_depth += dx * sin(dipangle(-sign * current_depth))
    depth = np.array(depth)
    nd = depth.shape[0]
    vertices = np.zeros((nx, nd, 3))
    vertices[:, 0, :] = nodes
    for i in range(0, nx):
        for j in range(1, nd):
            mydip = dipangle(-sign * depth[j] + vertices[i, 0, 2])
            one_over_tan_dip = 1.0 / tan(mydip)
            ud = -(one_over_tan_dip * av1[i, :] - uz)
            ud = ud / np.linalg.norm(ud)
            vertices[i, j, :] = vertices[i, j - 1, :] - sign * dx * ud
    return vertices


def get_vertice_generator(mydipType):
    """
    creator component of the factory method
    """
    if mydipType == dipType.CONSTANT:
        return generate_vertices_constant_dip
    elif mydipType == dipType.DEPTH_DEPENDENT:
        return generate_vertices_dip_vary_along_dip
    else:
        return generate_vertices_dip_vary_along_strike


def generate_vertices(maxdepth, sign=1):
    vertice_getter = get_vertice_generator(args.dipType)
    return vertice_getter(maxdepth, sign)


dx = args.dd[0]

# Reading dip value
if args.dipType == dipType.CONSTANT:
    dip = float(args.dipDesc) * pi / 180.0
elif args.dipType == dipType.DEPTH_DEPENDENT:
    print(
        "depth dependant dip described (depth vs dip) by the file %s" % (args.dipDesc)
    )
    depthdip = np.loadtxt(args.dipDesc)
    deptha = depthdip[:, 0]
    dipa = depthdip[:, 1] * pi / 180.0
    dipangle = interp1d(deptha, dipa, kind="linear")
elif args.dipType == dipType.STRIKE_DEPENDENT:
    print(
        "along strike varying dip described (relative length along strike (0-1) vs dip) by file %s"
        % (args.dipDesc)
    )
    curviligndip = np.loadtxt(args.dipDesc)
    relD = curviligndip[:, 0]
    dipa = curviligndip[:, 1] * pi / 180.0
    dipangle = interp1d(relD, dipa, kind="linear")
else:
    raise ValueError("dipType not in 0-2", args.dipType)

if args.extrudeDir:
    print(
        f"strike direction used to extrude the fault trace, described by file {args.extrudeDir[0]}"
    )
    strike_extrude = np.loadtxt(args.extrudeDir[0])
    relD = strike_extrude[:, 0]
    aStrike = strike_extrude[:, 1]
    strike_extrude = interp1d(relD, aStrike, kind="linear")

# Reading fault trace
bn = os.path.basename(args.filename)
ext = bn.split(".")[1]
# Trace described by a pl file
if ext == "pl":
    nodes = []
    with open(args.filename) as fid:
        lines = fid.readlines()
    for li in lines:
        if li.startswith("VRTX"):
            lli = li.split()
            nodes.append([float(lli[2]), float(lli[3]), float(lli[4])])
    nodes = np.asarray(nodes)
# Trace described by an Ascii file
else:
    nodes = np.loadtxt(args.filename)
    ndim = nodes.shape[1]
    if ndim == 2:
        nx = np.shape(nodes)[0]
        b = np.zeros((nx, 1))
        nodes = np.append(nodes, b, axis=1)

if args.proj:
    nodes = Project(nodes)

if args.first_node_ext[0] > 0:
    u0 = nodes[0, :] - nodes[1, :]
    u0 = u0 / np.linalg.norm(u0)
    nodes = np.vstack([nodes[0, :] + u0 * args.first_node_ext[0], nodes])

if args.last_node_ext[0] > 0:
    u0 = nodes[-1, :] - nodes[-2, :]
    u0 = u0 / np.linalg.norm(u0)
    nodes = np.vstack([nodes, nodes[-1, :] + u0 * args.last_node_ext[0]])


nodes[:, 0] = nodes[:, 0] + args.translate[0]
nodes[:, 1] = nodes[:, 1] + args.translate[1]

# Compute Fault length and nx
diff = nodes[1:, :] - nodes[0:-1, :]
faultlength = np.sum(np.sqrt(np.square(diff[:, 0]) + np.square(diff[:, 1])))
print("faultlength = %.2f km" % (faultlength / 1e3))
nx = int(faultlength / dx)

# smooth and resample fault trace
nnodes = nodes.shape[0]
spline_deg = 3 if nnodes > 3 else (2 if nnodes > 2 else 1)
tck, u = splprep(
    [nodes[:, 0], nodes[:, 1], nodes[:, 2]], s=args.smoothingParameter[0], k=spline_deg
)
unew = np.linspace(0, 1, nx)
new_points = splev(unew, tck)
nNewNodes = np.shape(new_points[0])[0]
nodes = np.zeros((nNewNodes, 3))
nodes[:, 0] = new_points[0]
nodes[:, 1] = new_points[1]
nodes[:, 2] = new_points[2]

# Plot smooth fault trace over picked trace
if args.plotFaultTrace:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(nodes[:, 0], nodes[:, 1], "ro")
    plt.axis("equal")
    ax.plot(new_points[0], new_points[1], "rx-")
    plt.show()


# Compute x,y abscisse coordinates of each nodes
diff = nodes[1:, :] - nodes[0:-1, :]
distBetweenNodes = np.sqrt(np.square(diff[:, 0]) + np.square(diff[:, 1]))
print(
    "distance between nodes: 15, 50 and 85 percentiles",
    np.percentile(distBetweenNodes, 15),
    np.percentile(distBetweenNodes, 50),
    np.percentile(distBetweenNodes, 85),
)

xi = np.zeros((nNewNodes))
for i in range(1, nNewNodes):
    xi[i] = xi[i - 1] + distBetweenNodes[i - 1]

nx = np.shape(nodes)[0]
uz = np.array([0, 0, 1])

if args.dipType == dipType.STRIKE_DEPENDENT or args.extrudeDir:
    reldist = compute_rel_curvilinear_coordinate(nodes)

if args.dipType == dipType.STRIKE_DEPENDENT:
    # apply smoothing kernel to avoid sharp normal changes
    aDip = smooth(dipangle(reldist), box_hpts=2)

# v0: unit strike vector
# v1: normal to v0, in the horizontal plane
av0 = np.zeros((nx, 3))
av1 = np.zeros((nx, 3))
for i in range(0, nx):
    if not args.extrudeDir:
        v0 = nodes[min(i + 1, nx - 1), :] - nodes[max(i - 1, 0), :]
    else:
        strike = strike_extrude(reldist[i]) * pi / 180.0
        v0 = [-cos(strike), -sin(strike), 0.0]
    v0[2] = 0
    v0 = v0 / np.linalg.norm(v0)
    av0[i, :] = v0
    av1[i, :] = np.array([-v0[1], v0[0], 0])

# Create new vertex below 0
vertices = generate_vertices(args.maxdepth[0])
# Create new vertex above 0
if args.extend[0] > 0:
    # Extension toward z plus
    vertices2 = np.flip(generate_vertices(args.extend[0], sign=-1), axis=1)
    vertices = np.concatenate((vertices2, vertices[:, 1:, :]), axis=1)
nd = vertices.shape[1]

# Generate rough fault deviation
if args.addRoughness:
    ### TODO test this code and refactor this block!

    print("adding roughness")
    # ReadRoughness parameterisation
    import configparser
    from generateRoughFault import GenerateRoughFault

    config = configparser.ConfigParser()
    config.read(args.addRoughness[0])
    try:
        lambdaMaxSubSet = config.getfloat("roughness", "lambdaMaxSubSet")
        lambdaMinSubSet = config.getfloat("roughness", "lambdaMinSubSet")
        reverseSubset = config.getboolean("roughness", "reverseSubset")
        append2prefix = "subset_%d_%d_" % (int(lambdaMinSubSet), int(lambdaMaxSubSet))
        if reverseSubset:
            append2prefix = append2prefix + "_r"
    except configparser.NoOptionError:
        append2prefix = ""

    lambdaMin = config.getfloat("roughness", "lambdaMin")
    lambdaMaxS = config.getfloat("roughness", "lambdaMaxS")
    seed = config.getint("roughness", "seed")
    alphaExp = -config.getfloat("roughness", "alphaExp")
    append2prefix = f"rough_{int(lambdaMin)}_{int(lambdaMaxS)}_seed{seed}_exp{alphaExp:.2f}_{append2prefix}"

    # compute coordinates along y
    yi = np.zeros((nd))
    dist = np.linalg.norm(vertices[0, 1:nd, :] - vertices[0, 0 : nd - 1, :], axis=1)
    for i in range(1, nd):
        yi[i] = yi[i - 1] + dist[i - 1]

    X, Y, h = GenerateRoughFault(
        Ls=xi[nNewNodes - 1] + xi[1], Ld=yi[nd - 1] + yi[1], fname=args.addRoughness[0]
    )
    startTime = datetime.now()
    f = RectBivariateSpline(X, Y, np.transpose(h), kx=1, ky=1)
    print("interp done in:", datetime.now() - startTime)

    if args.dipType == dipType.CONSTANT:
        # In this case the loop can be vectorized
        for i in range(0, nx):
            ud = -(1.0 / tan(dip) * av1[i, :] - uz)
            normal = cross(v0, ud)
            normal = normal / np.linalg.norm(normal)
            normals = np.reshape(np.tile(normal, nd), (nd, 3))
            roughnessContrib = normals * np.transpose(f(xi[i], yi[:]))
            vertices[i, :, :] = vertices[i, :, :] + roughnessContrib
    else:
        for i in range(0, nx):
            for j in range(0, nd):
                if args.dipType == dipType.DEPTH_DEPENDENT:
                    mydip = dipangle(vertices[i, j, 2])
                else:
                    mydip = dipangle(reldist[i])
                ud = -(1.0 / tan(mydip) * av1[i, :] - uz)
                normal = cross(v0, ud)
                normal = normal / np.linalg.norm(normal)
                vertices[i, j, :] = vertices[i, j, :] + f(xi[i], yi[j]) * normal


### WRITE THE GOCAD TS FILE
bn = os.path.basename(args.filename)
prefix = bn.split(".")[0]
NX = nx
NY = nd

if args.addRoughness:
    prefix = append2prefix + prefix

fname = prefix + "0.ts"
with open(fname, "w") as fout:
    fout.write("GOCAD TSURF 1\nHEADER {\nname:" + prefix + "\n}\nTRIANGLES\n")
    for j in range(0, NY):
        for i in range(0, NX):
            fout.write(
                "VRTX "
                + str(i + j * NX + 1)
                + " %.10e %.10e %.10e\n" % tuple(vertices[i, j, :])
            )
    for j in range(NY - 1):
        for i in range(1, NX):
            fout.write(
                "TRGL %d %d %d\n" % (i + j * NX, i + 1 + j * NX, i + 1 + (j + 1) * NX)
            )
            fout.write(
                "TRGL %d %d %d\n" % (i + j * NX, i + 1 + (j + 1) * NX, i + (j + 1) * NX)
            )
    fout.write("END")
print(f"done writing {fname}")
