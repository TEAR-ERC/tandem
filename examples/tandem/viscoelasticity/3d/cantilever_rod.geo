// Adapted from Marques, S.P.C. & Creus, G.J., 2012. Computational Viscoelasticity, Springer
// and Gharti et. al. https://academic.oup.com/gji/article/216/2/1364/5199199

// clamped = 1 (default): the rod is fully clamped at x = 0 (Dirichlet, tag 5).
// clamped = 0: x = 0, y = 0 and z = 0 are symmetry planes (free slip with u.n = 0,
// tag 9). The rod is then in exactly uniaxial stress, the displacement is affine
// and the Zener closed form holds at every point; test/test_viscoelastic_cantilever.py
// uses this variant (gmsh -setnumber clamped 0).
DefineConstant[ clamped = {1, Choices{0, 1}, Name "Clamped end" } ];

h_min = 7.7e-5;             // minimum mesh size
h_max = 1e-4;               // maximum mesh size away from the surface
SetFactory("OpenCASCADE");

// Initial geometry
Box(1) = {0, 0, 0, 100e-6, 30e-6, 30e-6};

// Tag faces (OpenCASCADE box faces: 1: x = 0, 2: x = L, 3: y = 0, 4: y = W, 5: z = 0, 6: z = H)
If (clamped)
    Physical Surface(5) = {1};
    Physical Surface(1) = {3,4,5,6};
Else
    Physical Surface(9) = {1,3,5};
    Physical Surface(1) = {4,6};
EndIf
Physical Surface(7) = {2};    // Surface to be refined

Physical Volume("Rod") = {1};

// Distance field from Surface(7)
Field[1] = Distance;
Field[1].SurfacesList = {2};     // refine around face with tag 2

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

