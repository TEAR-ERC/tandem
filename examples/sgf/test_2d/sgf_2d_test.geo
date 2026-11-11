// 2D sgf test, adapted from tandem/examples/tandem/2d/bp3.geo.
// 60 deg dipping fault from the surface to 40 km down dip, split into four
// 10 km GF patches (tags 10-13). Below 40 km the fault line is an untagged
// interior line. Top = receiver surface (100001), sides + bottom Dirichlet (5).
// The u = 0 box shifts the surface displacement by O(1/d): about 4e-3 per unit
// slip at d = 400, 8e-4 at d = 2000 (gmsh -setnumber d 400 for the small box).
//   gmsh -2 sgf_2d_test.geo -o sgf_2d_test.msh

DefineConstant[ Lf = {0.5, Min 0, Max 10, Name "Fault resolution" } ];
DefineConstant[ Ls = {0.25, Min 0, Max 10, Name "Resolution near free surface" } ];
DefineConstant[ dip = {60, Min 0, Max 90, Name "Dipping angle" } ];

dip_rad = dip * Pi / 180.0;

h = 40;
DefineConstant[ d = {2000, Min 10, Max 10000, Name "Box depth (km)" } ];
w = d * Cos(dip_rad) / Sin(dip_rad);
w = w < d ? d : w;
d1 = 10;
d2 = 20;
d3 = 30;
d4 = 40;
Point(1) = {-w, 0, 0, h};
Point(2) = {w, 0, 0, h};
Point(3) = {w + d * Cos(dip_rad) / Sin(dip_rad), -d, 0, h};
Point(4) = {-w + d * Cos(dip_rad) / Sin(dip_rad), -d, 0, h};
Point(5) = {d * Cos(dip_rad) / Sin(dip_rad), -d, 0, h};
Point(6) = {0, 0, 0, h};
Point(7) = {d4 * Cos(dip_rad), -d4 * Sin(dip_rad), 0, h};
Point(8) = {d3 * Cos(dip_rad), -d3 * Sin(dip_rad), 0, h};
Point(9) = {d2 * Cos(dip_rad), -d2 * Sin(dip_rad), 0, h};
Point(10) = {d1 * Cos(dip_rad), -d1 * Sin(dip_rad), 0, h};
Line(1) = {1, 6};
Line(2) = {6, 2};
Line(3) = {2, 3};
Line(4) = {3, 5};
Line(5) = {5, 4};
Line(6) = {4, 1};
Line(7) = {7, 5};
Line(8) = {8, 7};
Line(9) = {9, 8};
Line(10) = {10, 9};
Line(11) = {6, 10};
Curve Loop(1) = {11, 10, 9, 8, 7, -4, -3, -2};
Curve Loop(2) = {11, 10, 9, 8, 7, 5, 6, 1};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Physical Curve(10) = {11};          // GF patch 0-10 km down dip
Physical Curve(11) = {10};          // 10-20 km
Physical Curve(12) = {9};           // 20-30 km
Physical Curve(13) = {8};           // 30-40 km
Physical Curve(100001) = {1, 2};    // receiver surface, BC::Natural
Physical Curve(5) = {3, 4, 5, 6};   // sides and bottom, BC::Dirichlet
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Field[1] = MathEval;
Field[1].F = Sprintf("%g + 3e-2*(x + y*%g)^2 + 2e-3*(min(0, y/%g+40))^2",
                        Lf, Cos(dip_rad)/Sin(dip_rad), Sin(dip_rad));
Field[2] = MathEval;
Field[2].F = Sprintf("%g + 0.1 * sqrt(x^2 + y^2)", Ls);
Field[3] = Min;
Field[3].FieldsList = {1, 2};
Background Field = 3;
Mesh.MshFileVersion = 2.2;
