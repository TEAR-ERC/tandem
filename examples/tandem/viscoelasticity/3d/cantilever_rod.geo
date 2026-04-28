// Adapted from Marques, S.P.C. & Creus, G.J., 2012. Computational Viscoelasticity, Springer
// and Gharti et. al. https://academic.oup.com/gji/article/216/2/1364/5199199

h_min = 7.7e-5;             // minimum mesh size 
h_max = 1e-4;               // maximum mesh size away from the surface
SetFactory("OpenCASCADE");

// Initial geometry
Box(1) = {0, 0, 0, 100e-6, 30e-6, 30e-6};

// Tag faces
Physical Surface(5) = {1};
Physical Surface(7) = {2};    // Surface to be refined
Physical Surface(1) = {3,4,5,6};

Physical Volume("Rod") = {1};

// Distance field from Surface(7)
Field[1] = Distance;
Field[1].SurfacesList = {2};     // refine around face with tag 2
Field[1].Sampling = 100;

Field[2] = Threshold;
Field[2].InField = 1;      // use Distance field
Field[2].SizeMin = h_min;
Field[2].SizeMax = h_max;

Field[2].DistMin = 0;         // exactly at the surface
Field[2].DistMax = 30e-6;     // 30 micro meters away

Field[3] = Min;
Field[3].FieldsList = {2};

Background Field = 3;

// Mesh settings
Mesh.CharacteristicLengthMin = h_min;
Mesh.CharacteristicLengthMax = h_max;
Mesh.Algorithm3D = 1;
Mesh.MshFileVersion = 2.2;

