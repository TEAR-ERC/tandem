// Verification geometry for sgf: three small faults in a large half space.
// Physical tags:  101,102,103 fault (BC::Fault)   100001 receiver surface
//                 100002 free surface (rest of top)   5 Dirichlet (sides, bottom)
// Build with:  gmsh -3 sgf_3_faults_example.geo -o sgf_3_faults_example.msh

L = 60.0;   // box half width
Z = 40.0;   // box depth
I = 10.0;   // receiver surface half width
hf = 0.25;  // element size on the faults
hs = 0.8;  // element size under the receiver surface
hF = 12.0; // element size at the far boundary

// ---- box corners -------------------------------------------------
Point(1)  = {-1*L, -1*L, 0, hF};
Point(2)  = {1*L, -1*L, 0, hF};
Point(3)  = {1*L, 1*L, 0, hF};
Point(4)  = {-1*L, 1*L, 0, hF};
Point(5)  = {-1*L, -1*L, -Z, hF};
Point(6)  = {1*L, -1*L, -Z, hF};
Point(7)  = {1*L, 1*L, -Z, hF};
Point(8)  = {-1*L, 1*L, -Z, hF};

// ---- inner receiver square --------------------------------------
Point(9)  = {-1*I, -1*I, 0, hs};
Point(10)  = {1*I, -1*I, 0, hs};
Point(11)  = {1*I, 1*I, 0, hs};
Point(12)  = {-1*I, 1*I, 0, hs};

// ---- lines -------------------------------------------------------
Line(1)  = {1, 2};
Line(2)  = {2, 3};
Line(3)  = {3, 4};
Line(4)  = {4, 1};
Line(5)  = {5, 6};
Line(6)  = {6, 7};
Line(7)  = {7, 8};
Line(8)  = {8, 5};
Line(9)  = {1, 5};
Line(10)  = {2, 6};
Line(11)  = {3, 7};
Line(12)  = {4, 8};
Line(13) = {9, 10};
Line(14) = {10, 11};
Line(15) = {11, 12};
Line(16) = {12, 9};

// ---- surfaces ----------------------------------------------------
Curve Loop(1) = {1, 2, 3, 4};           // top outer
Curve Loop(2) = {13, 14, 15, 16};       // receiver square
Curve Loop(3) = {5, 6, 7, 8};           // bottom
Plane Surface(1) = {2};                 // receiver surface
Plane Surface(2) = {1, 2};              // rest of the top
Plane Surface(3) = {3};                 // bottom
Curve Loop(4) = {1, 10, -5, -9};
Plane Surface(4) = {4};
Curve Loop(5) = {2, 11, -6, -10};
Plane Surface(5) = {5};
Curve Loop(6) = {3, 12, -7, -11};
Plane Surface(6) = {6};
Curve Loop(7) = {4, 9, -8, -12};
Plane Surface(7) = {7};

Surface Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Volume(1) = {1};

// ---- fault triangles, embedded as internal surfaces --------------
// fault 101
Point(101) = {-3.577350, -4.000000, -2.666667, hf};
Point(102) = {-3.577350, -2.000000, -2.666667, hf};
Point(103) = {-1.845299, -3.000000, -3.666667, hf};
Line(101) = {101, 102};
Line(102) = {102, 103};
Line(103) = {103, 101};
Curve Loop(101) = {101, 102, 103};
Plane Surface(101) = {101};

// fault 102
Point(104) = {1.898272, -2.091752, -3.028595, hf};
Point(105) = {3.630323, -1.091752, -3.028595, hf};
Point(106) = {3.471405, -2.816497, -4.442809, hf};
Line(104) = {104, 105};
Line(105) = {105, 106};
Line(106) = {106, 104};
Curve Loop(102) = {104, 105, 106};
Plane Surface(102) = {102};

// fault 103
Point(107) = {-0.699359, 3.788675, -2.422650, hf};
Point(108) = {1.032692, 2.788675, -2.422650, hf};
Point(109) = {-0.333333, 2.422650, -4.154701, hf};
Line(107) = {107, 108};
Line(108) = {108, 109};
Line(109) = {109, 107};
Curve Loop(103) = {107, 108, 109};
Plane Surface(103) = {103};

Surface{101, 102, 103} In Volume{1};

// ---- mesh size fields --------------------------------------------
Field[1] = Distance;
Field[1].SurfacesList = {101, 102, 103};
Field[1].Sampling = 60;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = hf;
Field[2].SizeMax = hF;
Field[2].DistMin = 1.5;
Field[2].DistMax = 25;

// Keep the volume under the receiver surface resolved, otherwise the
// displacement is sampled onto a fine grid from coarse elements.
Field[3] = Box;
Field[3].VIn = hs;
Field[3].VOut = hF;
Field[3].XMin = -12.0; Field[3].XMax = 12.0;
Field[3].YMin = -12.0; Field[3].YMax = 12.0;
Field[3].ZMin = -8.0;  Field[3].ZMax = 1;
Field[3].Thickness = 10.0;

Field[4] = Min;
Field[4].FieldsList = {2, 3};
Background Field = 4;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MshFileVersion = 2.2;   // tandem accepts 2.x only

// ---- physical groups: the tag value becomes FacetInfo::facetTag ---
Physical Surface(101) = {101};
Physical Surface(102) = {102};
Physical Surface(103) = {103};
Physical Surface(100001) = {1};      // receiver surface, BC::Natural
Physical Surface(100002) = {2};      // rest of the free surface, BC::Natural
Physical Surface(5) = {3, 4, 5, 6, 7};  // sides and bottom, BC::Dirichlet
Physical Volume(1) = {1};
