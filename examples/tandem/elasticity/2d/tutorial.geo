DefineConstant[ res = {20.0, Min 0, Max 10, Name "Domain resolution" } ];
DefineConstant[ res_f = {0.25, Min 0, Max 10, Name "Fault resolution" } ];
DefineConstant[ dip = {60, Min 0, Max 90, Name "Dipping angle" } ];

SetFactory("OpenCASCADE");

dip_rad = dip * Pi / 180.0;
W = 40.0;
H = 100.0;
dX = 100.0;
X0 = -dX;
X1 = H * Cos(dip_rad) / Sin(dip_rad) + dX;
Y0 = -H;

box = news; Rectangle(box) = {X0, Y0, 0.0, X1-X0, -Y0};

p1 = newp; Point(p1) = {0.0, 0.0, 0.0, res_f};
p2 = newp; Point(p2) = {W * Cos(dip_rad) / Sin(dip_rad), -W, 0.0, res_f};
p3 = newp; Point(p3) = {H * Cos(dip_rad) / Sin(dip_rad), -H, 0.0, res_f};

fault1 = newl; Line(fault1) = {p1,p2};
fault2 = newl; Line(fault2) = {p2,p3};

v[] = BooleanFragments{ Surface{box}; Delete; }{ Line{fault1, fault2}; Delete; };

eps = 1e-3;
top[] = Curve In BoundingBox{X0-eps, -eps, -eps, X1+eps, eps, eps};
bottom[] = Curve In BoundingBox{X0-eps, Y0-eps, -eps, X1+eps, Y0+eps, eps};
left[] = Curve In BoundingBox{X0-eps, Y0-eps, -eps, X0+eps, eps, eps};
right[] = Curve In BoundingBox{X1-eps, Y0-eps, -eps, X1+eps, eps, eps};


MeshSize{ PointsOf{Surface{:};} } = res;
MeshSize{ PointsOf{Line{fault1};} } = res_f;

Physical Curve(1) = {bottom(),top()};
Physical Curve(3) = {fault1,fault2};
Physical Curve(5) = {left[],right[]};
Physical Surface(1) = {v[]};

Mesh.MshFileVersion = 2.2;
