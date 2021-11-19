DefineConstant[ h = {0.1, Min 0, Max 1, Name "Resolution" } ];

SetFactory("OpenCASCADE");

Disk(1) = {0, 0, 0, 0.2};
Disk(2) = {0, 0, 0, 1};

v() = BooleanDifference{ Surface{2}; Delete; }{ Surface{1}; Delete; }; 

MeshSize{ PointsOf{Surface{:};} } = h;

Physical Curve(1) = {1};
Physical Curve(5) = {2};
Physical Surface(1) = {v()};
Mesh.MshFileVersion = 2.2;
