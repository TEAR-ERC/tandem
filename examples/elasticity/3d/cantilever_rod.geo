// Cantilever rod example 
// A 30mm x 30mm x 100mm rod

h_min = 0.0000009;   // 0.9 mm 
h_max = 0.00001;     // 10.0 mm
L = 0.0001;          // 100 mm 
W = 0.00003;         // 30 mm 
H = 0.00003;         // 30 mm 

SetFactory("OpenCASCADE");

// cantilever rod
Box(1) = {0, 0, 0, L, W, H};

// The surrounding side surfaces
// Natural/Free surface
Physical Surface(1) = {3, 4, 5, 6};

// Physical Groups
// The Fixed Surface (X = 0)
// Dirichlet
Physical Surface(5) = {1}; 

// The Free End (X = L) - to be refined
// Traction
Physical Surface(7) = {2}; 

// The Volume
Physical Volume(1) = {1};

// Mesh Refinement - Distance from Free End
Field[1] = Distance;
Field[1].SurfacesList = {2};

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_min;
Field[2].SizeMax = h_max;
Field[2].DistMin = 0;        
Field[2].DistMax = 0.00006;

Background Field = 2;

Mesh.CharacteristicLengthMin = h_min;
Mesh.CharacteristicLengthMax = h_max;
Mesh.MshFileVersion = 2.2;
