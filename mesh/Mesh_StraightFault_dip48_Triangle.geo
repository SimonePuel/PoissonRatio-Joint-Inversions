//+
dx = 50.0;
dx_fault = 3.0;
dx_surface = 5.0;
//+
Point(1) = {-700, 0, 0, dx};
//+
Point(2) = {400, 0, 0, dx};
//+
Point(3) = {400, -500, 0, dx};
//+
Point(4) = {-700, -500, 0, dx};
//+
Point(5) = {0, 0, 0, dx_fault};
//+
Point(6) = {-56.777614710697804, -63.05792945985107, 0, dx_fault};
//+
Point(7) = {-500, -500, 0, dx};
//+
Point(8) = {-30, -5, 0, dx_fault};
//+
Point(9) = {-160, -5, 0, dx_fault};
//+
Point(10) = {-95, -55, 0, dx_fault};
//+
Point(11) = {-200, 0, 0, dx_surface};
//+
Line(1) = {1, 4};
//+
Line(2) = {4, 7};
//+
Line(3) = {7, 3};
//+
Line(4) = {3, 2};
//+
Line(5) = {2, 5};
//+
Line(6) = {5, 6};
//+
Line(7) = {6, 7};
//+
Line(8) = {1, 11};
//+
Line(9) = {11, 5};
//+
Line(10) = {8, 9};
//+
Line(11) = {9, 10};
//+
Line(12) = {10, 8};
//+
Curve Loop(1) = {5, 6, 7, 3, 4};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {8, 9, 6, 7, -2, -1};
//+
Curve Loop(3) = {11, 12, 10};
//+
Plane Surface(2) = {2, 3};
//+
Plane Surface(3) = {3};
//+
Physical Curve("surface", 1) = {8, 9, 5};
//+
Physical Curve("left", 2) = {1};
//+
Physical Curve("bottom", 3) = {2, 3};
//+
Physical Curve("right", 4) = {4};
//+
Physical Curve("fault", 5) = {6};
//+
Physical Curve("extended fault", 13) = {7};
//+
Physical Surface("blockleft", 7) = {2};
//+
Physical Surface("blockright", 8) = {1};
//+
Physical Surface("triangle_surface", 10) = {3};
