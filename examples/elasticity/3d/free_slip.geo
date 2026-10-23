DefineConstant[ h = {0.5, Min 0, Max 1, Name "Resolution" } ];

SetFactory("OpenCASCADE");

// Unit cube [0,1] x [0,1] x [0,1]
Box(1) = {0, 0, 0, 1, 1, 1};

// OpenCASCADE numbers the faces of a box as
// 1: x = 0, 2: x = 1, 3: y = 0, 4: y = 1, 5: z = 0, 6: z = 1

// Boundaries
Physical Surface(9) = {1, 2, 3, 4, 5}; // four sides + bottom (free slip)
Physical Surface(1) = {6};             // top (free surface)

Physical Volume(1) = {1};

MeshSize{ PointsOf{ Volume{1}; } } = h;
Mesh.MshFileVersion = 2.2;
