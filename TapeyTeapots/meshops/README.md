Mesh operations must be able to both transform the triangles and to transform points and directions in 3d space. That is why we are hand-writing them (at least for now) rather than using a library such as pyMesh. **Using a mesh library in the future will always be considered**

Also, pyMesh is hard to install so it would make portability much worse. Every other package is pip or pip with minimal modifications.

When possible, we use numpy and write functions that can operate on whole batches of points.
