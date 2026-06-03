
DefineConstant[ res = {0.1, Min 0, Max 1, Name "cell resolution" } ];

// enable CAD modelling
SetFactory("OpenCASCADE");



Point(1) = {-1.0, 0.0, 0.0, res};
Point(2) = {1.0, 0.0, 0.0, res};
Point(3) = {0.0, -1.0, 0.0, res};
Point(4) = {0.0, 1.0, 0.0, res};


// Define 2 line segments
h_line = newl; Line(h_line) = {1, 2};
v_line = newl; Line(v_line) = {3, 4};

box = news;
Rectangle(box) = {-1.0, -1.0, 0.0, 2.0, 2.0};

// This call below to BooleanFragments will result in 16 line segments being defined
// The original lines are deleted.
// It will also result in 4 volumes being defined - and the original volume will be deleted.
v() = BooleanFragments{ Surface{box}; Delete; }{ Line{h_line, v_line}; Delete; };


MeshSize{ PointsOf{Surface{:};} } = res;

Physical Curve(1) = { 9, 11, 7, 4 }; // top, bottom
Physical Curve(5) = { 1, 8, 6, 12 }; // left, right


Physical Surface(1) = { 1 };
Physical Surface(2) = { 2 };
Physical Surface(3) = { 3 };
Physical Surface(4) = { 4 };

Mesh.MshFileVersion = 2.2;
