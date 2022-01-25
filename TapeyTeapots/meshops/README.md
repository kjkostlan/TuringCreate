Core to Turing-Create is mesh operations that transform points in 3d space, even for points which are off the mesh's surface slightly. Because of this extra requirement, a good library for mesh operations is lacking. Thus the need for hand-coding and taking code from other libraries.

What libraries are available:
pyMesh: Hard to install on windows, most code in C++.
MPI-IS/mesh: Simple C++/Python library https://github.com/MPI-IS/mesh
OpenMesh: C++ library https://www.graphics.rwth-aachen.de/software/openmesh/

**numba** is used for select hand-written codes, as it basically gives C++ runtime performance even when for-loops are used and is compatible with numpy.

