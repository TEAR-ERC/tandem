DefineConstant[ Lf = {0.020, Min 0, Max 10, Name "Fault resolution" } ];
DefineConstant[ Ls = {0.020, Min 0, Max 10, Name "Resolution near free surface" } ];
DefineConstant[ dip = {60, Min 0, Max 90, Name "Dipping angle" } ];

dip_rad = dip * Pi / 180.0;

h = 40;
d = 400;
/*d = 180;*/
w = d * Cos(dip_rad) / Sin(dip_rad);
w = w < d ? d : w;
d1 = 15;
d2 = 16;
d3 = 18;
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
/* Bottom: Free-surface */ 
Physical Curve(3) = {8, 9, 10, 11};
Physical Curve(1) = {1, 2, 4, 5};
Physical Curve(5) = {3, 6, 7};
/* Bottom: Dirichlet */
/*Physical Curve(3) = {8, 9, 10, 11};*/
/*Physical Curve(1) = {1, 2};*/
/*Physical Curve(5) = {3, 4, 5, 6, 7};*/
/* All-fault */ 
/*Physical Curve(3) = {7, 8, 9, 10, 11};*/
/*Physical Curve(1) = {1, 2, 4, 5};*/
/*Physical Curve(5) = {3, 6};*/
Physical Surface(1) = {1};
Physical Surface(2) = {2};
Field[1] = MathEval;
/*Field[1].F = Sprintf("%g + (min(1, -y/40)*2e-2 + 1e-3)*(x + y*%g)^2 + 1e-3*(min(0, y/%g+180))^2",*/
                        /*Lf, Cos(dip_rad)/Sin(dip_rad), Sin(dip_rad));*/
Field[1].F = Sprintf("%g + 3e-2*(x + y*%g)^2 + 2e-3*(min(0, y/%g+40))^2",
                        Lf, Cos(dip_rad)/Sin(dip_rad), Sin(dip_rad));
Field[2] = MathEval;
Field[2].F = Sprintf("%g + 0.1 * sqrt(x^2 + y^2)", Ls);
Field[3] = Min;
Field[3].FieldsList = {1, 2};
Background Field = 3;
Mesh.MshFileVersion = 2.2;
