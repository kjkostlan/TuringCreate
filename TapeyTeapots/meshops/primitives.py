# Simple shapes to get you started.
# Faces are counter-clockiwise from top.
from . import trimesh
import numpy as np

def _add_tri(mesh, ixs_ccw, face_ix, u0=0.0, v0=0.0, u1=1.0, v1=0.0, u2=0.5, v2=1.0):
    # Adds one triangle.
    mesh['faces'][:,face_ix] = ixs_ccw

    # mesh['uvs'] = [3,nFace,2,k], k>1 means multiple textures.
    nMat = mesh['uvs'].shape[3]
    mesh['uvs'][0,face_ix,0,:] = u0
    mesh['uvs'][1,face_ix,0,:] = u1
    mesh['uvs'][2,face_ix,0,:] = u2
    mesh['uvs'][0,face_ix,1,:] = v0
    mesh['uvs'][1,face_ix,1,:] = v1
    mesh['uvs'][2,face_ix,1,:] = v2

def _add_square(mesh, ixs_ccw, face_ix, u0=0.0, v0=0.0, u1=1.0, v1=1.0):
    ixs_ccw0 = [ixs_ccw[0],ixs_ccw[1],ixs_ccw[2]]
    ixs_ccw1 = [ixs_ccw[0],ixs_ccw[2],ixs_ccw[3]]
    _add_tri(mesh, ixs_ccw0, face_ix, u0=u0, v0=v0, u1=u1, v1=v0, u2=u1, v2=v1)
    _add_tri(mesh, ixs_ccw1, face_ix+1, u0=u0, v0=v0, u1=u1, v1=v1, u2=u0, v2=v1)

def cube():
    # A cube can become anything!
    mesh = {}
    mesh['verts'] = np.transpose([[-1,-1,-1],[+1,-1,-1],[-1,+1,-1],[+1,+1,-1],[-1,-1,+1],[+1,-1,+1],[-1,+1,+1],[+1,+1,+1]])
    mesh['faces'] = np.zeros([3,12], dtype=np.int64)
    mesh['uvs'] = np.zeros([3,12,2,1])
    face_pairs = [[[0, 4, 6, 2], 1], [[0, 1, 5, 4], 2], [[0, 2, 3, 1], 4]]
    for i in range(len(face_pairs)):
        _add_square(mesh, face_pairs[i][0], 4*i)
        _add_square(mesh, np.flip(face_pairs[i][0])+face_pairs[i][1], 4*i+2)
    return mesh

def sphere(resolution=32):
    # Latitude and longetude.
    resolution = max(resolution,3)
    fencepost = (resolution-1.0)/resolution
    theta = np.linspace(0,2.0*np.pi*fencepost,resolution)
    phi = np.linspace(-0.5*np.pi,0.5*np.pi,resolution)

    # Poles are at end.
    x = np.outer(np.cos(theta), np.sin(phi[1:-1]))
    y = np.outer(np.sin(theta), np.sin(phi[1:-1]))
    z = np.outer(np.ones_like(theta),np.sin(phi[1:-1]))

    mesh = {}
    mesh['verts'] = np.zeros([3,resolution*(resolution-2)+2])
    mesh['verts'][0,0:-2] = np.reshape(x, resolution*(resolution-2), order='C') # Phi will change faster than theta.
    mesh['verts'][1,0:-2] = np.reshape(y, resolution*(resolution-2), order='C')
    mesh['verts'][2,0:-2] = np.reshape(z, resolution*(resolution-2), order='C')
    mesh['verts'][:,-2] = [0,0,-1]; mesh['verts'][:,-1] = [0,0,1] # South pole.
    nFace = 2*resolution*(resolution-3)+2*resolution
    mesh['faces'] = np.zeros([3,nFace], dtype=np.int64)
    mesh['uvs'] = np.zeros([3, nFace,2,1])
    face_ix = 0
    stride = resolution-2
    for i in range(resolution): # thetas
        i1 = (i+1)%resolution
        # South pole:
        ixs_ccw = [resolution*(resolution-2), i1*stride+1, i*stride+1]
        _add_tri(mesh, ixs_ccw, face_ix, u0=0.5, v0=0.0, u1=theta[i1]/2.0*np.pi, v1=phi[1]/np.pi+0.5, u2=theta[i]/2.0*np.pi, v2=phi[1]/np.pi+0.5)
        face_ix = face_ix+1

        for j in range(1,resolution-2): # phis
            j1 = j+1
            ixs_ccw = [i*stride+j, i1*stride+j, i1*stride+j1, i*stride+j1]
            _add_square(mesh, ixs_ccw, face_ix, u0=theta[i]/2.0/np.pi, v0=phi[j]/np.pi+0.5, u1=theta[i1]/2.0*np.pi, v1=phi[j1]/np.pi+0.5)
            face_ix = face_ix+2

        # North pole:
        ixs_ccw = [resolution*(resolution-2)+1, i*stride+(resolution-2), i1*stride+(resolution-2)]
        _add_tri(mesh, ixs_ccw, face_ix, u0=0.5, v0=1.0, u1=theta[i]/2.0*np.pi, v1=phi[resolution-2]/np.pi+0.5, u2=theta[i1]/2.0*np.pi, v2=phi[resolution-2]/np.pi+0.5)
        face_ix = face_ix+1

    # Extra verts in the poles not attached to edges, oh well.
    return mesh

def cylinder(resolution=32, tallness=1.0):
    # Includes two central points.
    resolution = max(resolution,3)
    mesh = {}
    mesh['verts'] = np.zeros([3,2*resolution+2])
    mesh['verts'][:,0] = [0,0,-tallness] # Center of bottom face.
    mesh['verts'][:,1] = [0,0,tallness] # Center of top face.

    mesh['faces'] = np.zeros([3, resolution*4],dtype=np.int64)
    mesh['uvs'] = np.zeros([3,resolution*4,2,1])
    fencepost = (resolution-1.0)/resolution
    theta = np.linspace(0,2.0*np.pi*fencepost,resolution)
    x = np.cos(theta)
    y = np.sin(theta)
    mesh['verts'][0,2:resolution+2] = x
    mesh['verts'][1,2:resolution+2] = y
    mesh['verts'][2,2:resolution+2] = -tallness

    mesh['verts'][0,resolution+2:] = x
    mesh['verts'][1,resolution+2:] = y
    mesh['verts'][2,resolution+2:] = tallness

    for i in range(resolution): # bottom faces
        i1 = (i+1)%resolution
        ixs_ccw = [0,i1+2,i+2]
        _add_tri(mesh, ixs_ccw, i, u0=0.5, v0=0.0, u1=theta[i1]/2.0/np.pi, v1=0.333, u2=theta[i]/2.0/np.pi, v2=0.333)

    for i in range(resolution): # side faces
        i1 = (i+1)%resolution
        ixs_ccw = [i+2, i1+2, i1+resolution+2, i+resolution+2]
        _add_square(mesh, ixs_ccw, 2*i+resolution, u0=theta[i]/2.0/np.pi, v0=0.333, u1=theta[i1]/2.0/np.pi, v1=0.667)

    for i in range(resolution): # top faces
        i1 = (i+1)%resolution
        ixs_ccw = [1,i+resolution+2,i1+resolution+2]
        _add_tri(mesh, ixs_ccw, i+3*resolution, u0=0.5, v0=1.0, u1=theta[i]/2.0/np.pi, v1=0.667, u2=theta[i1]/2.0/np.pi, v2=0.667)
    return mesh

def cone(resolution=32, sharpness=1.0):
    # Includes a point at the base.
    resolution = max(resolution,3)
    mesh = {}
    mesh['verts'] = np.zeros([3,resolution+2])
    mesh['verts'][:,0] = [0,0,0] # Center of base.
    mesh['verts'][:,1] = [0,0,2.0*sharpness] # The tip.

    mesh['faces'] = np.zeros([3, resolution*2],dtype=np.int64)
    mesh['uvs'] = np.zeros([3,resolution*2,2,1])
    fencepost = (resolution-1.0)/resolution
    theta = np.linspace(0,2.0*np.pi*fencepost,resolution)
    x = np.cos(theta)
    y = np.sin(theta)
    mesh['verts'][0,2:] = x
    mesh['verts'][1,2:] = y

    for i in range(resolution): # bottom faces
        i1 = (i+1)%resolution
        ixs_ccw = [0,i1+2,i+2]
        _add_tri(mesh, ixs_ccw, i, u0=0.5, v0=0.0, u1=theta[i1]/2.0/np.pi, v1=0.5, u2=theta[i]/2.0/np.pi, v2=0.5)
    for i in range(resolution): # top faces
        i1 = (i+1)%resolution
        ixs_ccw = [1,i+2,i1+2]
        _add_tri(mesh, ixs_ccw, i+resolution, u0=0.5, v0=1.0, u1=theta[i]/2.0/np.pi, v1=0.5, u2=theta[i1]/2.0/np.pi, v2=0.5)
    return mesh

def torus(resolution=32, girth=0.25):
    # One of the more rewarding early CG shapes to code.
    resolution = max(resolution,3)
    fencepost = (resolution-1.0)/resolution
    theta = np.linspace(0,2.0*np.pi*fencepost,resolution)
    phi = np.linspace(0,2.0*np.pi*fencepost,resolution)

    # i is theta, j is phi.
    theta2D = np.tile(np.expand_dims(theta,1), [1, resolution])
    phi2D = np.transpose(np.tile(np.expand_dims(phi,1), [1, resolution]))

    unitX = np.cos(theta2D); unitY = np.sin(theta2D)
    x = unitX + girth*unitX*np.cos(phi2D)
    y = unitY + girth*unitY*np.cos(phi2D)
    z = girth*np.sin(phi2D)

    mesh = {}
    mesh['verts'] = np.zeros([3,resolution*resolution])
    mesh['verts'][0,:] = np.reshape(x, resolution*resolution, order='C') # Phi will change faster than theta.
    mesh['verts'][1,:] = np.reshape(y, resolution*resolution, order='C')
    mesh['verts'][2,:] = np.reshape(z, resolution*resolution, order='C')
    mesh['faces'] = np.zeros([3, 2*resolution*resolution], dtype=np.int64)
    mesh['uvs'] = np.zeros([3, 2*resolution*resolution,2,1])
    face_ix = 0
    for i in range(resolution): # thetas
        for j in range(resolution): # phis
            i1 = (i+1)%resolution
            j1 = (j+1)%resolution
            ixs_ccw = [i*resolution+j, i1*resolution+j, i1*resolution+j1, i*resolution+j1]
            _add_square(mesh, ixs_ccw, face_ix, u0=theta[i]/2.0/np.pi, v0=phi[j]/2.0/np.pi, u1=theta[i1]/2.0*np.pi, v1=phi[j1]/2.0/np.pi)
            face_ix = face_ix+2

    return mesh
