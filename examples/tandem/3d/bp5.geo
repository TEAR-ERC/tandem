DefineConstant[ res =   {40, Min 0, Max 1000, Name "Resolution" } ];
DefineConstant[ res_f = {3, Min 0, Max 1000, Name "Fault resolution" } ];
DefineConstant[ size =  {100, Min 0, Max 1000, Name "Domain size" } ];

h_s = 2;
h_t = 2;
H = 12;
l = 60;
W_f = 40;
l_f = 100;
w = 12;

X0 = -size-l_f/2;
X1 = -X0;
Y0 = -size-l_f/4;
Y1 = -Y0;
Z0 = -size-W_f;

eps = 1e-3;

SetFactory("OpenCASCADE");
Box(1) = {X0,  0, Z0, X1-X0, -Y0, -Z0};
Box(2) = {X0,  0, Z0, X1-X0, Y0, -Z0};

fault = news;
Rectangle(fault) = {-l_f/2, 0, 0, l_f, W_f};
Rotate{ {1, 0, 0}, {0, 0, 0}, -Pi/2} { Surface{fault}; }
pf = PointsOf{Surface{fault};};

nuc1 = news;
Rectangle(nuc1) = {-l/2-h_t, 0, -h_s, l+2*h_t, 2*h_t + H};
Rotate{ {1, 0, 0}, {0, 0, -h_s}, -Pi/2} { Surface{nuc1}; }

nuc2 = news;
Rectangle(nuc2) = {-l/2, 0, -h_s-h_t, l, H};
Rotate{ {1, 0, 0}, {0, 0, -h_s-h_t}, -Pi/2} { Surface{nuc2}; }

nuc3 = news;
Rectangle(nuc3) = {-l/2, 0, -h_s-h_t, w, H};
Rotate{ {1, 0, 0}, {0, 0, -h_s-h_t}, -Pi/2} { Surface{nuc3}; }

v() = BooleanFragments{ Volume{1,2}; Delete; }{ Surface{fault,nuc1,nuc2,nuc3}; Delete; }; 

fault() = Surface In BoundingBox{-l_f/2-eps, -eps, -W_f-eps, l_f/2+eps, eps, W_f+eps};
top() = Surface In BoundingBox{X0-eps, Y0-eps, -eps, X1+eps, Y1+eps, eps};
bottom() = Surface In BoundingBox{X0-eps, Y0-eps, Z0-eps, X1+eps, Y1+eps, Z0+eps};

//top() = Surface In BoundingBox{X0-eps, -eps, -eps, X1+eps, 2*Y1+eps, eps};
//bottom() = Surface In BoundingBox{X0-eps, -eps, Z0-eps, X1+eps, 2*Y1+eps, Z0+eps};
//front() = Surface In BoundingBox{X1-eps, -eps, Z0-eps, X1+eps, 2*Y1+eps, 0+eps};
//back() = Surface In BoundingBox{X0-eps, -eps, Z0-eps, X0+eps, 2*Y1+eps, 0+eps}; 

diri() = Surface{:};
diri() -= top();
diri() -= bottom();
diri() -= fault();
//diri() -= front();
//diri() -= back();

MeshSize{ PointsOf{Volume{:};} } = res;
MeshSize{ PointsOf{Surface{fault()};} } = res_f;

//Physical Surface(1) = {top(),front(),back(),bottom()};
Physical Surface(1) = {top(),bottom()};
Physical Surface(3) = {fault()};
Physical Surface(5) = {diri()};
Physical Volume(1) = {v()};

Mesh.MshFileVersion = 2.2;
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;
