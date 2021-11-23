DefineConstant[ h = {0.1, Min 0, Max 1, Name "Resolution" } ];

SetFactory("OpenCASCADE");

Sphere(1) = {0, 0, 0, 0.5};
Sphere(2) = {0, 0, 0, 1};

v() = BooleanDifference{ Volume{2}; Delete; }{ Volume{1}; Delete; }; 

Characteristic Length{ PointsOf{Volume{:};} } = h;

Physical Surface(1) = {1};
Physical Surface(5) = {2};
Physical Volume(1) = {v()};
Mesh.MshFileVersion = 2.2;
