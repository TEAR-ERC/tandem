Mesh creation with Gmsh
=======================

`Gmsh <https://gmsh.info/>`_ allows CAD modelling as well as mesh generation.
It comes with its own scripting language that we use to build the geometry. 

Create a file called :code:`tutorial.geo` and open it with your favourite text editor.

We first define a few parameters.
These parameters can be either set from the Gmsh GUI or from the command line using
:code:`-setnumber`.

::

   DefineConstant[ res = {20.0, Min 0, Max 10, Name "Domain resolution" } ];
   DefineConstant[ res_f = {0.25, Min 0, Max 10, Name "Fault resolution" } ];
   DefineConstant[ dip = {60, Min 0, Max 90, Name "Dipping angle" } ];

   SetFactory("OpenCASCADE");

The last line enables the OpenCASCADE CAD kernel that we use to create our geometry.
The dip angle is converted from degrees to radians and a few constants to define the
bounding box are set:

::

   dip_rad = dip * Pi / 180.0;
   W = 40.0;
   H = 100.0;
   dX = 100.0;
   X0 = -dX;
   X1 = H * Cos(dip_rad) / Sin(dip_rad) + dX;
   Y0 = -H;

We create our domain :math:`[X_0,X_1] \times [Y_0, 0]`:

::

   box = news; Rectangle(box) = {X0, Y0, 0.0, X1-X0, -Y0};

.. seealso::

   The domain dimensions are given in kilometres. Thus, we are going to
   :doc:`scale the Lamé parameters <../reference/scaling>` accordingly.

We then insert a fault. As we are going to vary the *a*-parameter from 0 km to 40 km
depth, we split the fault to later set a higher resolution in the upper part of the fault.

::

   p1 = newp; Point(p1) = {0.0, 0.0, 0.0, res_f};
   p2 = newp; Point(p2) = {W * Cos(dip_rad) / Sin(dip_rad), -W, 0.0, res_f};
   p3 = newp; Point(p3) = {H * Cos(dip_rad) / Sin(dip_rad), -H, 0.0, res_f};

   fault1 = newl; Line(fault1) = {p1,p2};
   fault2 = newl; Line(fault2) = {p2,p3};

The mesh generator is currently unaware of the fault.
Hence, we intersect the fault with the domain:

::

   v[] = BooleanFragments{ Surface{box}; Delete; }{ Line{fault1, fault2}; Delete; };

The Line-IDs have changed in the above boolean operation.
We recover the individual lines by searching them inside bounding boxes:

::

   eps = 1e-3;
   top[] = Curve In BoundingBox{X0-eps, -eps, -eps, X1+eps, eps, eps};
   bottom[] = Curve In BoundingBox{X0-eps, Y0-eps, -eps, X1+eps, Y0+eps, eps};
   left[] = Curve In BoundingBox{X0-eps, Y0-eps, -eps, X0+eps, eps, eps};
   right[] = Curve In BoundingBox{X1-eps, Y0-eps, -eps, X1+eps, eps, eps};

Finally, we set resolution parameters, assign boundary conditions, and set the mesh
format to version 2.2.

::

   MeshSize{ PointsOf{Surface{:};} } = res;
   MeshSize{ PointsOf{Line{fault1};} } = res_f;

   Physical Curve(1) = {bottom(),top()};
   Physical Curve(3) = {fault1,fault2};
   Physical Curve(5) = {left[],right[]};
   Physical Surface(1) = {v[]};

   Mesh.MshFileVersion = 2.2;

The argument of :code:`Physical Curve` must be set to 1, 3, or 5.
A 1 stands for free surface, 3 for fault, and 5 for Dirichlet boundary condition.

We can now generate the mesh and adjust the resolution and dip angle from the command line.
E.g.

.. code:: console

   $ gmsh -2 tutorial.geo -setnumber res_f 0.5

Implementing curvilinear mesh
-------------------------------

As suggested by `Uphoff et al. (2023) <https://academic.oup.com/gji/article/233/1/586/6847163?login=false>`_, curvilinear meshing can suppress numerical artefacts (e.g., localization of strain and traction) that can be introduced near fault intersections/junctions. Curvilinear meshing can be implemented by setting the order of mesh to be 2 in Gmsh, using the option :code:`-order 2`.
E.g.

.. code:: console

   $ gmsh -2 tutorial.geo -setnumber res_f 0.5 -order 2
