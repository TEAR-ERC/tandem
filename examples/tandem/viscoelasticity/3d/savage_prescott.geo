// Benchmark derived from https://pylith.readthedocs.io/en/latest/user/benchmarks/savage-prescott.html

X0 = -1000;
Xmid = 0;
X1 = 1000;
Y0 = -1000; 
Y1 = 1000;
Z0 = -400;
Zmid = -40;
Zlocked = -20;
Z1 = 0;

//Fault dimensions
fault_x = 0;
fault_y_min = -1000;
fault_y_max = 1000; 
fault_z_min = -40;  
//Start from Zmid
fault_z_max = 0;
eps = 1e-3;
SetFactory("OpenCASCADE");

//Create boxes
Box(1) = {X0, Y0, Z0, X1 - X0, Y1 - Y0, Zmid - Z0};
Box(2) = {X0, Y0, Zmid, X1 - X0, Y1 - Y0, Z1 - Zmid};

//Fault points
fault_p1 = newp; Point(fault_p1) = {fault_x, fault_y_min, fault_z_min};
fault_p2 = newp; Point(fault_p2) = {fault_x, fault_y_max, fault_z_min};
fault_p3 = newp; Point(fault_p3) = {fault_x, fault_y_max, fault_z_max};
fault_p4 = newp; Point(fault_p4) = {fault_x, fault_y_min, fault_z_max};

//fault lines
fault_l1 = newl; Line(fault_l1) = {fault_p1, fault_p2};
fault_l2 = newl; Line(fault_l2) = {fault_p2, fault_p3};
fault_l3 = newl; Line(fault_l3) = {fault_p3, fault_p4};
fault_l4 = newl; Line(fault_l4) = {fault_p4, fault_p1};

//fault plane
fault_ll = newll; Curve Loop(fault_ll) = {fault_l1, fault_l2, fault_l3, fault_l4};
fault_s = news; Plane Surface(fault_s) = {fault_ll};

limit = 6.7;

// patch points
patch_p1 = newp; Point(patch_p1) = {Xmid - limit, Y0, Zlocked};
patch_p2 = newp; Point(patch_p2) = {Xmid + limit, Y0, Zlocked};
patch_p3 = newp; Point(patch_p3) = {Xmid +limit, Y1, Zlocked};
patch_p4 = newp; Point(patch_p4) = {Xmid - limit, Y1, Zlocked};

// patch lines
patch_l1 = newl; Line(patch_l1) = {patch_p1, patch_p2};
patch_l2 = newl; Line(patch_l2) = {patch_p2, patch_p3};
patch_l3 = newl; Line(patch_l3) = {patch_p3, patch_p4};
patch_l4 = newl; Line(patch_l4) = {patch_p4, patch_p1};

// patch plane
patch_ll = newll; Curve Loop(patch_ll) = {patch_l1, patch_l2, patch_l3, patch_l4};
patch_s = news; Plane Surface(patch_s) = {patch_ll};

//Perform boolean fragments
v() = BooleanFragments{ Volume{1, 2}; Delete; }{ Surface{fault_s, patch_s}; Delete; };

//Boundary surfaces
Zp = Surface In BoundingBox{X0 - eps, Y0 - eps, Z1 - eps, X1 + eps, Y1 + eps, Z1 + eps};
Zm = Surface In BoundingBox{X0 - eps, Y0 - eps, Z0 - eps, X1 + eps, Y1 + eps, Z0 + eps};
Yp = Surface In BoundingBox{X0 - eps, Y0 - eps, Z0 - eps, X1 + eps, Y0 + eps, Z1 + eps};
Ym = Surface In BoundingBox{X0 - eps, Y1 - eps, Z0 - eps, X1 + eps, Y1 + eps, Z1 + eps};
Xm = Surface In BoundingBox{X0 - eps, Y0 - eps, Z0 - eps, X0 + eps, Y1 + eps, Z1 + eps};
Xp = Surface In BoundingBox{X1 - eps, Y0 - eps, Z0 - eps, X1 + eps, Y1 + eps, Z1 + eps};

//Single bounding box for all fault surfaces at x=0 from Zmid to Z1
fault() = Surface In BoundingBox{fault_x - eps, fault_y_min - eps, Zmid - eps, 
                                fault_x + eps, fault_y_max + eps, Z1 + eps};
patch() = Surface In BoundingBox{Xmid - limit - eps, Y0 - eps, Zlocked - eps, 
                                Xmid + limit + eps, Y1 + eps, Zlocked + eps};


//physical groups
free_surfaces() = {Zp(), Yp(), Ym()};
fault_surfaces() = {fault()};
patch_surfaces() = {patch()};
dirichlet_surfaces() = {Xp(), Xm()};
free_slip_surfaces() = {Zm()};
Physical Surface(1) = {free_surfaces()};
Physical Surface(5) = {dirichlet_surfaces(), fault_surfaces()};
Physical Surface(9) = {free_slip_surfaces()};
Physical Surface(101) = {patch_surfaces()};
Physical Volume(1) = {v()};

//distance field from fault surface
Field[1] = Distance;
Field[1].FacesList = {fault_surfaces()};

//mesh size away from fault
Field[2] = MathEval;
Field[2].F = Sprintf("F1 / 2 + 20");
Background Field = 2;
Mesh.MshFileVersion = 2.2;
