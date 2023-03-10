
DefineConstant[ res = {4, Min 0, Max 1, Name "cell resolution" } ];

// enable CAD modelling
SetFactory("OpenCASCADE");


//
// Mesh looks like this
//      line [3] = {3, 4} -> neumann
//  4 ------------ 3
//  | \          / |
//  |   \      /   |
//  |     \  /     |
//  |       5      | line [2] = {2, 3} -> dirichlet
//  |     /   \    |
//  |   /       \  |
//  | /           \|
//  1 ------------ 2
//      line [1] = {1, 2} -> dirichlet
//
//  line [4] = {4, 1} -> dirichlet
//
//  line [5] = {5, 4}
//  line [6] = {5, 2}
//  line [7] = {5, 3}
//  line [8] = {5, 1}


Point(1) = {-1.0, -1.0, 0.0, res};
Point(2) = {1.0, -1.0, 0.0, res};
Point(3) = {1.0, 1.0, 0.0, res};
Point(4) = {-1.0, 1.0, 0.0, res};
Point(5) = {0.0, 0.0, 0.0, res};


// Define 4 line segments
_line = newl; Line(_line) = {1, 2};
_line = newl; Line(_line) = {2, 3};
_line = newl; Line(_line) = {3, 4};
_line = newl; Line(_line) = {4, 1};

_line = newl; Line(_line) = {5, 4}; // diag, center -> upper left

_line = newl; Line(_line) = {5, 2}; // diag, center -> lower right
_line = newl; Line(_line) = {5, 3}; // diag, center -> upper right
_line = newl; Line(_line) = {5, 1}; // diag, center -> lower left


Physical Curve(1) = { 3 }; // [free surface] exterior: top
Physical Curve(5) = { 1, 2, 4 }; // [Dirichlet] exterior: bottom, right, left
// ignore three diagonal lines - these lines are not physical

// top triangle
tri[] = { 3, -5, 7 };
Line Loop(1) = tri[];
Plane Surface(1) = 1;

// left triangle
tri[] = { 4, -8, 5 };
Line Loop(2) = tri[];
Plane Surface(2) = 2;

// bottom triangle
tri[] = { 1, -6, 8 };
Line Loop(3) = tri[];
Plane Surface(3) = 3;

// right triangle
tri[] = { 6, 2, -7 };
Line Loop(4) = tri[];
Plane Surface(4) = 4;

Physical Surface(1) = { 1 };
Physical Surface(2) = { 2 };
Physical Surface(3) = { 3 };
Physical Surface(4) = { 4 };

Mesh.MshFileVersion = 2.2;
