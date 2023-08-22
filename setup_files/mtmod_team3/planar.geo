DefineConstant[ res = {50.0, Min 0, Max 10000, Name "h domain" } ];
DefineConstant[ res_f = {0.2, Min 0, Max 1000, Name "h fault (up-dip)" } ];
DefineConstant[ res_f_dd = {10.0, Min 0, Max 10000, Name "h fault (down-dip)" } ];
DefineConstant[ dip = {30, Min 0, Max 90, Name "Dipping angle" } ];
DefineConstant[ S = {200, Min 0, Max 10000, Name "Domain size" } ];

// some constants, dip in radiant
dip_rad = dip * Pi / 180.0;
dX = 400;

// enable CAD modelling
SetFactory("OpenCASCADE");

X0 = -dX;
X1 = S * Cos(dip_rad) / Sin(dip_rad) + dX;
Y0 = -200;


Point(1) = {0.0, 0.0, 0.0, res_f};
//Point(2) = {H * Cos(dip_rad) / Sin(dip_rad), -H, 0.0, res_f};
//Point(3) = {H * Cos(dip_rad) / Sin(dip_rad) + (S - H) * Cos(deeper_slab_dip_rad) / Sin(deeper_slab_dip_rad), -S, 0.0, res};
Point(3) = {S * Cos(dip_rad) / Sin(dip_rad), -S, 0.0, res};
//Point(4) = {curve_offset + 0.5 * H * Cos(dip_rad) / Sin(dip_rad) + 100, -H / 2, 0.0, res_f};
//Point(4) = { Sqrt(H/(slab_curvature/1000)), -H, 0.0, res_f };
Point(5) = {20.0 * Cos(dip_rad) / Sin(dip_rad) + 0.001, -20.0, 0.0, res_f};

// Create main fault
main_fault = newl; Line(main_fault) = {1, 3};
//fault_extension = newl; Line(fault_extension) = {2, 3};

not_fault = newl; Line(not_fault) = {1, 5};

//MeshSize{ PointsOf{Line{main_fault()};} } = res_f;


/*// Dirichlet "fault"*/
/*diri = news;*/
/*Line(diri) = {2, 3};*/

box = news;
Rectangle(box) = {X0, Y0, 0.0, X1-X0, -Y0};
//Rectangle(box) = {X0, Y0, 0.0, X1, -Y0};

// Intersect domain with fault
v() = BooleanFragments{ Surface{box}; Delete; }{ Line{main_fault}; Delete; };

// Recover lines with bounding boxes
eps = 1e-3;
top() = Curve In BoundingBox{X0-eps, -eps, -eps, X1+eps, eps, eps};
bottom() = Curve In BoundingBox{X0-eps, Y0-eps, -eps, X1+eps, Y0+eps, eps};
left() = Curve In BoundingBox{X0-eps, Y0-eps, -eps, X0+eps, eps, eps};
right() = Curve In BoundingBox{X1-eps, Y0-eps, -eps, X1+eps, eps, eps};

faultS() = Curve In BoundingBox{X0+eps, Y0*0.5, -eps, X0-eps, 0.0+eps, eps};
faultSB() = Point In BoundingBox{X0+4.0*eps, Y0-eps, -eps, X1-4.0*eps, Y0+eps, eps};

// set mesh resolution
MeshSize{ PointsOf{Line{left()};} } = res;
MeshSize{ PointsOf{Line{right()};} } = res;

MeshSize{ PointsOf{Line{faultS()};} } = res_f;
MeshSize{ PointsOf{Point{faultSB()};} } = res_f_dd;


// 1 = free surface
Physical Curve(1) = {bottom(),top()};
// 3 = fault
Physical Curve(3) = {main_fault()};
// 5 = dirichlet
Physical Curve(5) = {left(),right()};
Physical Surface(1) = {v()};

grad_dist = res/5.0;

Field[1] = Attractor;
//Field[1].PointsList = {10};
Field[1].NNodesByEdge = 100;
Field[1].CurvesList = {2};

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = res_f;
Field[2].LcMax = res;
Field[2].DistMin = 1.0;
Field[2].DistMax = grad_dist;

Field[7] = Min;
Field[7].FieldsList = {2};
Background Field = 7;

Mesh.MshFileVersion = 2.2;
