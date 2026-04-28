DefineConstant[ res =   {40, Min 0, Max 1000, Name "Resolution" } ];
DefineConstant[ res_f = {1, Min 0, Max 1000, Name "Fault resolution" } ];

W_f = 18;
l_f = 36;

X0 = -50;
X1 = -X0;
Y0 = -50;
Y1 = -Y0;
Z0 = -50;

eps = 1e-3;

SetFactory("OpenCASCADE");
Box(1) = {X0, Y0, Z0, X1-X0, Y1-Y0, -Z0};

fault = news;
Rectangle(fault) = {-l_f/2, 0, 0, l_f, W_f};
Rotate{ {1, 0, 0}, {0, 0, 0}, -Pi/2} { Surface{fault}; }
pf = PointsOf{Surface{fault};};

v() = BooleanFragments{ Volume{1}; Delete; }{ Surface{fault}; Delete; }; 

fault() = Surface In BoundingBox{-l_f/2-eps, -eps, -W_f-eps, l_f/2+eps, eps, W_f+eps};
middle() = Surface In BoundingBox{X0-eps, -eps, Z0-eps, X1+eps, eps, eps};
top() = Surface In BoundingBox{X0-eps, Y0-eps, -eps, X1+eps, Y1+eps, eps};
bottom() = Surface In BoundingBox{X0-eps, Y0-eps, Z0-eps, X1+eps, Y1+eps, Z0+eps};
diri() = Surface{:};
diri() -= top();
diri() -= bottom();
diri() -= middle();

MeshSize{ PointsOf{Volume{:};} } = res;
MeshSize{ PointsOf{Surface{fault()};} } = res_f;

Physical Surface(1) = {bottom(),top()};
Physical Surface(3) = {fault()};
Physical Surface(5) = {diri()};
Physical Volume(1) = {v()};

Mesh.MshFileVersion = 2.2;
