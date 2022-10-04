# Basic tools for quaterions, 3x3 matrixes, and 4x4 matrixes.
# Some functions are so simple as to not really need them.
# Most quaternion formulas are from:
  # https://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf
  # With quaternions as [r, i, j, k]
# Some 4x4 matrix stuff from: https://stackoverflow.com/questions/29079685/how-does-4x4-matrix-work-in-3d-graphic
import numpy as np
from . import coregeom

def _expand1(x):
    x = np.asarray(x)
    if len(x.shape) < 2:
        return np.expand_dims(x,1)
    return x

def _v1(x):
    x = np.asarray(x)
    n = np.linalg.norm(x)
    if n<1e-100:
        return np.asarray([0,0,1.0])
    else:
        return x/n

################ Quaternion-quaternion operations #############

def q1(q): #q = quaternion
    q = np.asarray(q)
    n = np.linalg.norm(q)
    if n<1e-100:
        return np.asarray([1.0,0,0,0])
    else:
        return q/n

def qq(p,q, normalize=False): #qq = q*q which multiplies quaterntions.
    if normalize:
        p = q1(p)
        q = q1(q)
    return np.asarray([p[0]*q[0] - (p[1]*q[1]+p[2]*q[2]+p[3]*q[3]),
                       p[0]*q[1] + q[0]*p[1] + p[2]*q[3] - p[3]*q[2],
                       p[0]*q[2] + q[0]*p[2] + p[3]*q[1] - p[1]*q[3],
                       p[0]*q[3] + q[0]*p[3] + p[1]*q[2] - p[2]*q[1]])

def q_inv(q):
    return np.asarray([q[0], -q[1], -q[2], -q[3]])/np.sum(q*q)

################ Transforming vectors #############

def qv(q, vectors_3xn, normalize=False): # v = vector.
    vectors_3xn = _expand1(vectors_3xn)
    if normalize:
        q = q1(q)
    q2 = np.sum(q[1:]*q[1:])
    q_ = np.expand_dims(q[1:],1)
    term_real = (q[0]*q[0]-q2)*vectors_3xn
    term_dot = 2.0*np.transpose(np.expand_dims(np.sum(q_*vectors_3xn,axis=0),1))*q_
    term_cross = 2.0*q[0]*np.cross(q[1:], vectors_3xn, axis=0)
    return term_real + term_dot + term_cross

def m33v(m33, vectors_3xn): # m33 = 3x3 rotation matrix.
    vectors_3xn = _expand1(vectors_3xn)
    return np.matmul(m33, vectors_3xn)

def m44v(m44, vectors_3xn): # m44 = 4x4 rotation + translation + affine.
    # Does NOT normalize the 4x4 matrix.
    vectors_3xn = _expand1(vectors_3xn)
    vectors_4xn = np.ones([4,vectors_3xn.shape[1]]); vectors_4xn[0:3,:] = vectors_3xn
    x = np.matmul(m44, vectors_4xn)
    return x[0:3,:]

################ Construction from simplier representations #############

def _v4_from_v(v):
    v4 = np.ones(4); v4[0:3] = v
    return v4

def q_from_polarshift(pole0, pole1): # Shifts pole0 into pole1
    pole0 = _v1(pole0); pole1 = _v1(pole1) # Normalize the 3-vectors.

    if np.sum(pole0*pole1) + 1.0 < 1e-8: # 180 degree singularity.
        if np.sum(np.abs(pole0-[0,0,-1])) < 1e-8:
            return np.asarray([0,1.0,0,0])
        zero_north = q_from_polarshift(pole0,np.asarray([0,0,1]))
        north_south = np.asarray([0,1.0,0,0])
        return qq(q_inv(zero_north),qq(np.asarray(north_south), zero_north))
    pole_mid = _v1(pole0+pole1) # half angle deluxe.
    imag = np.cross(pole0, pole_mid)
    real = np.sqrt(1.0-np.sum(imag*imag))
    return np.asarray([real, imag[0], imag[1], imag[2]])

def m44_from_m33(m33):
    out = np.identity(4)
    out[0:3, 0:3] = m33
    return out

def m33_from_q(q, normalize=False):
    if normalize:
        q = q1(q)
    return qv(q, np.identity(3))

def m44_from_q(q, normalize=False):
    return m44_from_m33(m33_from_q(q, normalize=normalize))

def m44_from_v(v):
    out = np.identity(4)
    out[0:3,3] = v
    return out

################ Convert between various forms, conversions are reversible #############

'''
def qTOq33(q, normalize=False): # Redundent with m33_from_q
    # Quaternion => 3x3 rotation matrix.
    # https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    if normalize:
        q = q1(q)
    out = np.zeros([3,3])
    q0 = q[0]; q1 = q[1]; q2 = q[2]; q3 = q[3]
    out[0,0] = 2.0*(q0*q0+q1*q1)-1.0; out[0,1] = 2.0*(q1*q2-q0*q3); out[0,2] = 2.0*(q1*q3+q0*q2)
    out[1,0] = 2.0*(q1*q2+q0*q3); out[1,1] = 2.0*(q0*q0+q2*q2)-1.0; out[1,2] = 2.0*(q2*q3-q0*q1)
    out[2,0] = 2.0*(q1*q3-q0*q2); out[2,1] = 2.0*(q2*q3+q0*q1); out[2,2] = 2.0*(q0*q0+q3*q3)-1.0
    return out
'''

def q33TOq(q33):
    # Going the other way is harder.
    # https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
    w,v = np.linalg.eig(q33)
    ix = np.argmin(np.abs(w-1.0))
    axis = _v1(np.real(v[:,ix]))
    cos_theta = 0.5*(np.trace(q33) - 1.0)
    #q = cos(theta/2)+usin(theta/2)
    #Note: cos(acos(x)/2) = sqrt(x+1)/sqrt(2)
    #      sin(acos(x)/2) = sqrt(1-x)/sqrt(2)
    real = np.sqrt(max(0.0, cos_theta+1.0))/np.sqrt(2.0) # Rounding errors may push cos_theta beyond 1.
    imag = np.sqrt(max(0.0, 1.0-cos_theta))/np.sqrt(2.0)*axis

    # Sign calculation: whichever one makes the best conversion matrix:
    q = np.asarray([real, imag[0], imag[1], imag[2]])
    q1 = np.asarray([-real, imag[0], imag[1], imag[2]])
    if np.sum(np.abs(m33_from_q(q1)-q33)) < np.sum(np.abs(m33_from_q(q)-q33)):
        return q1
    return q

def qrTOm33(q, r): # q-r decomposition, but q is quaternion.
    return np.matmul(m33_from_q(q),r)

def m33TOqr(m33): # q-r decomposition, but q is quaternion.
    q33,r = np.linalg.qr(m33)
    a = None # Cancel 2 negative signs so that r is close to the identity.
    if np.linalg.det(q33)<0: # Why do they not guarantee det=1?
        q33 = -q33; r = -r
    if r[0,0]<0 and r[1,1]<0:
        a = np.identity(3); a[0,0] = -1; a[1,1] = -1; # a^-1 = a
    if r[0,0]<0 and r[2,2]<0:
        a = np.identity(3); a[0,0] = -1; a[2,2] = -1; # a^-1 = a
    if r[1,1]<0 and r[2,2]<0:
        a = np.identity(3); a[1,1] = -1; a[2,2] = -1; # a^-1 = a
    if a is not None:
        q33 = np.matmul(q33,a); r = np.matmul(a,r) # put an a^-1 a in the middle, with a^-1 = a.
    return q33TOq(q33), r

def axisangleTOq(axis, radians):
    # Counter clockwise if the axis is pointed at you.
    axis = _v1(axis)
    half_c = np.cos(0.5*radians); half_s = np.sin(0.5*radians)
    out = np.zeros(4); out[0] = half_c; out[1:4] = half_s*axis
    if np.sum(out) < 0: # Sign convention.
        return -out
    return out

def qTOaxisangle(q):
    # Return [axis, radians], with axis normalized.
    q = q1(q)
    imag = q[1:]
    half_s = np.linalg.norm(imag)
    radians = 2.0*np.arcsin(min(half_s, 1.0))
    axis = _v1(imag)
    if q[0] < 0.0:
        radians = -radians
    return axis, radians

def m33vTOm44(m33, v=None):
    # Translation is applied after the m33.
    if v is None:
        v = np.asarray([0,0,0])
    out = np.identity(4)
    out[0:3, 0:3] = m33
    out[0:3,3] = v
    return out

def m44TOm33v(m44):
    #Returns m33, translation, with translation applied after the m33.
    m44 = np.copy(m44)
    return m44[0:3,0:3], m44[0:3,3]

def m44TOqrv(m44):
    # Returns quaternion, upper triangluar (scale + shear), and v.
    # Order of operations: shear, then rotate, then translate.
    m33, v = m44TOm33v(m44)
    q,r = m33TOqr(m33)
    return q,r,v

def qrvTOm44(q,r,v):
    m33 = qrTOm33(q, r)
    return m33vTOm44(m33,v)

################ Camera operations #############

def cam44v(cam44, vectors_3xn):
    # "Can a 4x4 matrix describe a camera's perspective"
    # Yes: m*v/(m*v)[3], where v is [x,y,z,1].
    # We use the conventions from: https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_model_view_projection
    # This means that the cube [-1,-1,-1] to [1,1,1] is rendered, and that z=-1 is in front.
    # Thus camera matrixed tend to have negative determinents.
    vectors_3xn = _expand1(vectors_3xn)
    vectors_4xn = np.ones([4, vectors_3xn.shape[1]]); vectors_4xn[0:3,:] = vectors_3xn
    x = np.matmul(cam44, vectors_4xn)
    x = x/x[[3],:]
    return x[0:3,:]

def cam44_invv(cam44, vectors_3xn):
    # From camera space to real space.
    # Process of making a projection:
    # 1. Add a w=1 component to the vectors.
    # 2. Multiply by cam44.
    # 3. Divide by w, and discard w.
    # Step 3 is scalar multiplication, since w/w is 1 but we don't yet know this scalar.
    # So we can add w=1 and than invert step 2. The right scalar to multiply that undoes things is w=1.
    # So it IS just inverting the camera matrix. But this function is if this nice fact gets forgotten!
    return cam44v(np.linalg.inv(cam44),vectors_3xn)

def solve_normalized(A):
    # Solves Ax=0, where A is underdetermined by one DOF and we constrain the solution to have norm 1.
    nEq = A.shape[0] # We have one more DOF than this.
    if A.shape[1] != A.shape[0]+1:
        raise Exception('A is not underdetermined by one DOF.')
    A1 = np.zeros([nEq+1,nEq+1]); A1[0:nEq,:] = A
    b1 = np.zeros(nEq+1); b1[nEq] = 1
    maxDetA = 0.0; ldetA1s = -1e100*np.ones(nEq+1)
    for i in range(nEq+1): # overkill to loop through all, could break on a threshold log_det.
        A1[nEq,:] = 0.0; A1[nEq,i] = 1.0
        _, ldetA1s[i] = np.linalg.slogdet(A1)
    best_ix = np.argmax(ldetA1s); A1[nEq,:] = 0.0; A1[nEq,best_ix] = 1.0
    x = np.linalg.solve(A1,b1)
    x = x/np.sqrt(np.sum(x*x)+1e-100)
    if np.sum(x)<0:
        x = -x
    return x

def solve43(cam44, b4_missing1):
    # Solves cam44*[x,y,z,1]^T = b4_missing1 for x,y, and z.
    # One element of v4_missing1 must be nan or None which we do not care what it becomes.
    row_ixs = []; b4_missing1 = np.asarray(b4_missing1, dtype=np.float64)

    isnans = np.isnan(b4_missing1)
    for i in range(4):
        if isnans[i] < 0.5:
            row_ixs.append(i)
    if len(row_ixs) != 3:
        raise Exception('Exactly one element must be None.')
    b3 = b4_missing1[row_ixs]
    m33 = cam44[row_ixs,0:3]
    b = -cam44[row_ixs,3] # cam44*[x,y,z,1] for x=y=z=0.
    return np.matmul(np.linalg.pinv(m33), b+b3) # Solve cam33*x = v3

def _proj_plane1(origin, normal, v): # Only one vector.
    return coregeom.project_to_plane(np.asarray(origin), np.asarray(normal), np.expand_dims(v,1))[:,0]

def _line_plane1(plane_origin, plane_normal, line_origin, line_direction): # Only one vector.
    return coregeom.line_plane_intersection(np.asarray(plane_origin), np.asarray(plane_normal), np.expand_dims(line_origin,1), np.expand_dims(line_direction,1))[:,0]

def magic_sign(): # Setting to +1 will break the unit tests.
    return -1.0

def cam44TOqvfcya(cam44):
    # Converts a perspective camera to a unit Quaternion, Camera location, and f-number.
    # q = rotation of camera. Identity q means camera camera points in -z direction (3 DOF)
    # v = location of center of camera (3 DOF)
    # f is kindof the f-number, frustum depth/half-width. A 90 degree FOV is an f of 1.0, telophotos are f>10. (1 DOF)
    # c is the [near, far] clipping plane. For very weird cameras far can be nearer. (2 DOF)
    # y = [y-stretch, y shear in x direction] of the camera image. Usually [1,0] (2 DOF)
    # a = Clipping plane shear slope (applied after y), + is away from camera [near-x, near-y, far-x, far-y]. Usually all zero. (4 DOF)
    # Total DOF: 3+3+1+2+2+4 = 15. Scaling cam44 is redundant, so all DOF accounted for.
    # Note: https://stackoverflow.com/questions/17650219/when-is-z-axis-flipped-opengl
    #       -1 in clip coords renders in front of 1, as it is left-handed.

    cam44 = cam44/cam44[3,3] # Redundant DOF.
    if np.abs(np.linalg.det(cam44)) < 1e-12:
        raise Exception('Cannot invert the camera matrix.')
    cam44_inv = np.linalg.inv(cam44) #"What point maps to here".

    # The origin of the camera maps to w=0.
    v = solve43(cam44, [0,0,None,0])

    # Map to the faces and edges of the unit cube in "clip coordinants"
    center0 = cam44v(cam44_inv, [0,0,-1])[:,0] # 0= near.
    center1 = cam44v(cam44_inv, [0,0,1])[:,0] # 1 = far.
    right0 = cam44v(cam44_inv, [1,0,-1])[:,0]
    right1 = cam44v(cam44_inv, [1,0,1])[:,0]
    top0 = cam44v(cam44_inv, [0,1,-1])[:,0]
    top1 = cam44v(cam44_inv, [0,1,1])[:,0]

    # Calculate the rotation matrix, ignoring the y-vector and a-shear:
    q33_z = magic_sign()*_v1(center0-v);
    right0a = _line_plane1(center0, q33_z, v, right0-v)
    q33_x = _v1(right0a-center0)
    q33_y = _v1(np.cross(q33_z, q33_x))
    q = q33TOq(np.stack([q33_x,q33_y,q33_z],axis=1))

    # Calculate the kind-of-f number
    f = np.linalg.norm(center0-v)/np.linalg.norm(right0a-center0)

    # Near and far clipping plane distances are based on the center of the screen:
    # c[0] is always positive.
    c = np.asarray([np.linalg.norm(center0-v), magic_sign()*np.dot(q33_z,center1-v)])

    # 2D shear of the y-vector:
    top0a = _line_plane1(center0, q33_z, v, top0-v)
    top0a_no_shear = coregeom.project_to_line(center0, q33_y, np.expand_dims(top0a,1))[:,0]
    looks_streched = np.dot(right0a-center0,q33_x)/np.dot(top0a_no_shear-center0,q33_y)
    looks_sheared = -np.dot(top0a-top0a_no_shear,q33_x)/np.linalg.norm(top0a_no_shear-center0)
    y = np.asarray([looks_streched, looks_sheared])

    # clipping plane shear toward-away from camera:
    right1a = _line_plane1(center1, q33_z, v, right1-v)
    top1a =  _line_plane1(center1, q33_z, v, top1-v)
    dir_right = _v1(right0a-v); dir_top = _v1(top0a-v)
    a0 = (np.dot(right0-v,dir_right)-np.dot(right0a-v,dir_right))/np.linalg.norm(right0a-center0)
    a1 = (np.dot(top0-v,dir_top)-np.dot(top0a-v,dir_top))/np.linalg.norm(top0a-center0)
    a2 = (np.dot(right1-v,dir_right)-np.dot(right1a-v,dir_right))/np.linalg.norm(right1a-center1)
    a3 = (np.dot(top1-v,dir_top)-np.dot(top1a-v,dir_top))/np.linalg.norm(top1a-center1)
    a = np.asarray([a0,a1,a2,a3])

    return q, v, f, c, y, a

def qvfcyaTOcam44(q,v,f=1.0,c=None,y=None,a=None):
    # Back to a 4x4 matrix.
    if c is None:
        c = np.asarray([1.0/512.0, 512.0])
    if y is None:
        y = np.asarray([1.0,0.0]) # no 2D shear.
    if a is None:
        a = np.asarray([0.0,0.0,0.0,0.0]) # No toward-away camera shear.

    # Undo the math to compute the frustum.
    q33 = m33_from_q(np.asarray(q)); q33[:,2] = magic_sign()*q33[:,2]
    center0 = v+q33[:,2]*c[0]; center1 = v+q33[:,2]*c[1]

    right0a = center0 + c[0]*q33[:,0]/f
    top0a_no_shear_no_stretch = center0 + c[0]*q33[:,1]/f
    top0a_no_shear = (top0a_no_shear_no_stretch-center0)/y[0]+center0 # Visual stretch means the vector actually shrinks.
    top0a = top0a_no_shear - y[1]*np.linalg.norm(top0a_no_shear-center0)*q33[:,0]
    right1a = center1 + c[1]*q33[:,0]/f
    top1a = (top0a-v)*c[1]/c[0] + v
    # Final undo step: add in the z-shear in the clipping plane.
    dir_right = _v1(right0a-v); dir_top = _v1(top0a-v)
    right0 = right0a + dir_right*a[0]*np.linalg.norm(right0a-center0)

    right1 = right1a + dir_right*a[2]*np.linalg.norm(right1a-center1)
    top0 = top0a + dir_top*a[1]*np.linalg.norm(top0a-center0)
    top1 = top1a + dir_top*a[3]*np.linalg.norm(top1a-center1)

    # Directly solving for the matrix is confusing due to divide by w.
    # However, we can use the frustom constraints to for the matrix as a 16-long vector.
    # Then get the 16 constraints on it and solve the 16x16 linear system.
    #cam44[i,j] = camu[4*i+j] is the 'C' reshape option we use.
    A = np.zeros([15,16]);
    # 16 constraints: (Hint: equations are kind of written backwards). 15/16 of the b-values stay zero.
    v4 = _v4_from_v(v); c04 = _v4_from_v(center0); c14 = _v4_from_v(center1);
    r04 = _v4_from_v(right0); r14 = _v4_from_v(right1);
    t04 = _v4_from_v(top0); t14 = _v4_from_v(top1);

    # There are 15 constraints that specify A up to a scalar constant, i.e. specify Ax=0
    cix = 0; # Cix = constraint ix. Each new constraint is a different row of the the 16x16 matrix.
    A[cix, 0*4+0:0*4+4] = v4; cix = cix+1 # Origin, x=0.
    A[cix, 1*4+0:1*4+4] = v4; cix = cix+1 # Origin, y=0.
    A[cix, 3*4+0:3*4+4] = v4; cix = cix+1 # Origin, w=0.
    A[cix, 0*4+0:0*4+4] = c04; cix = cix+1 # Center0, x=0
    A[cix, 1*4+0:1*4+4] = c04; cix = cix+1 # Center0, y=0
    A[cix, 2*4+0:2*4+4] = c04; A[cix, 3*4+0:3*4+4] = c04; cix = cix+1 # Center0, z+w = 0
    A[cix, 0*4+0:0*4+4] = r04; A[cix, 3*4+0:3*4+4] = -r04; cix = cix+1  # Right0, x-w = 0
    A[cix, 1*4+0:1*4+4] = r04; cix = cix+1 # Right0, y = 0
    A[cix, 2*4+0:2*4+4] = r04; A[cix, 3*4+0:3*4+4] = r04; cix = cix+1 # Right0, z+w = 0
    A[cix, 0*4+0:0*4+4] = t04; cix = cix+1 # Top0, x=0
    A[cix, 1*4+0:1*4+4] = t04; A[cix, 3*4+0:3*4+4] = -t04; cix = cix+1 # Top0, y-w=0
    A[cix, 2*4+0:2*4+4] = t04; A[cix, 3*4+0:3*4+4] = t04; cix = cix+1 # Top0, z+w = 0
    A[cix, 2*4+0:2*4+4] = c14; A[cix, 3*4+0:3*4+4] = -c14; cix = cix+1 # Center1, z-w = 0
    A[cix, 2*4+0:2*4+4] = t14; A[cix, 3*4+0:3*4+4] = -t14; cix = cix+1 # Top1, z-w=0.
    A[cix, 2*4+0:2*4+4] = r14; A[cix, 3*4+0:3*4+4] = -r14; cix = cix+1 # Right1, z-w=0.
    #best_ix = np.argmax(detAs); A[cix,:] = 0.0; A[cix,best_ix] = 1.0
    #cix = cix+1; # Now we got to all 16 constraints, making the linear system fully determined.
    # Note about some redundant constraints that are not needed: Center1, x=0 and center1 y=0 (constrained by origin and center0)
    # Top1, x=0 and y=w (constrained by origin and top0). Same idea with right1.
    # Note: There are 3 x-only, 3 y-only, 3 z-w, 3 z+w, 1 w, 1 x-w, 1 y-w, and the camw[3,3]=1
    # This means the 16x16 solve is overkill. Instead, we could get 1D solutions for
    # x,y,z-w, and z+w, and then solve the smaller 4x4 system.
    camu = solve_normalized(A)
    debug_show_A = False
    if debug_show_A:
        print('A:')
        for i in range(15):
            print('Row:'+str(i)+':',A[i,:])
        print('|A|=',np.linalg.det(A))
        print('Ax:',np.einsum('ij,j->i',A,camu))

    cam44 = np.reshape(camu, [4,4], order='C')
    debug_test_constraints = False
    if debug_test_constraints:
        print('Test center0:', cam44v(cam44, center0)[:,0], 'vs', [0,0,-1])
        print('Test center1:', cam44v(cam44, center1)[:,0], 'vs', [0,0,1])
        print('Test right0:', cam44v(cam44, right0)[:,0], 'vs', [1,0,-1])
        print('Test right1:', cam44v(cam44, right1)[:,0], 'vs', [1,0,1])
        print('Test top0:', cam44v(cam44, top0)[:,0], 'vs', [0,1,-1])
        print('Test top1:', cam44v(cam44, top1)[:,0], 'vs', [0,1,1])
    # Normalize:
    cam44 = cam44/np.sqrt(np.sum(cam44*cam44)+1e-100)
    if np.sum(cam44)<0:
        cam44 = -cam44
    return cam44

def cam_near_far(camera44, screenx, screeny, start_at_origin=False):
    # Returns [point at near clipping plane, point at far clipping plane]
    # 0 = center of screen, square screen from -1 to 1.
    # Set start_at_origin to true to override the start at near clipping plane.
    # (but will only work for perspective cameras).
    # camera(v) = (m*v)/(m*v)_w, with v a 4-vector. Doubling v (including the w term) does not change it.
    # Solve: camera44*[x,y,z,w] = [screenx, screeny, +-1, 1], then scale w to 1.

    def cam_solve(clip):
        xyzw = np.linalg.solve(camera44, clip)
        return xyzw[0:3]/xyzw[3]
    if start_at_origin:
        near_clip = cam_solve([screenx,screeny,0,1])
    else:
        near_clip = cam_solve([screenx,screeny,-1,1])
    far_clip = cam_solve([screenx,screeny,1,1])
    return near_clip, far_clip

def cam_plane_normal(camera44):
    # Points directly away from the camera.
    near_clip, far_clip = cam_near_far(camera, 0.0, 0.0)
    return _v1(far_clip-near_clip)

def camq_from_look(look, up=None):
    # Looks in the direction look, put tries to put "up" on the screen "up" in the world.
    if up is None:
        up = [0,0,1]
    look = _v1(np.asarray(look))
    # The default camera has it's three axis pointing in x,y,z, looking in -z.
    # The new three axes should be upX(-look), up, -look.
    q33_x = _v1(np.cross(up,-look)); q33_y = _v1(np.cross(-look,q33_x));
    q33_z = -look
    q33 = np.stack([q33_x, q33_y, q33_z],axis=1)
    q = q33TOq(q33)
    return q

def cam_from_look(v, look, up=None, f=1.0, c0=1.0/128, c1 = 1024):
    # Convience function.
    q = camq_from_look(look, up=up)
    return qvfcyaTOcam44(q,v,f=f,c=[c0, c1])

def cam_from_ortho(v, q, zoom_out=1.0, far_clip=1024.0):
    # The math for perspective cameras fails for orthographic (singular matrix).
    # Orthographic cameras do not affect w so no projection happens.
    cam44 = np.zeros([4,4])
    cam44[3,3] = 1.0
    q33 = m33_from_q(q)
    cam33 = np.transpose(q33)/zoom_out # The upper-left 3x3 is (change_in_clip)/(change_in_world).
    cam44[0:3,0:3] = cam33
    # The - on far_clip is b/c it is left-handed.
    cam44[2,0:3] = zoom_out*cam44[2,0:3]*2.0/(-far_clip) # Make changes in z less sensitive.
    go_to_zero = v+0.5*(-far_clip)*q33[:,2]
    offset = np.matmul(cam33, go_to_zero)
    offset[2] = zoom_out*offset[2]*2.0/(-far_clip)
    cam44[0:3,3] = -offset
    #print('matmul test:',np.matmul(cam44, [go_to_zero[0],go_to_zero[1],go_to_zero[2],1.0]))
    return cam44
