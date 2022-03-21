# Tests matrixes and quaternions.
import TapeyTeapots.meshops.quat34 as qmv
import TapeyTeapots.meshops.coregeom as coregeom
import numpy as np
from . import tmetric

def random_4x4_matrixes(n):
    m44s = []
    for i in range(n):
        m44 = np.random.randn(4,4)
        m44[3,0:3] = 0.0
        m44[3,3] = 1.0
        m44s.append(m44)
    return m44s

def random_qs(n):
    qs = []
    for i in range(n):
        q = np.random.randn(4)
        q = q/np.linalg.norm(q)
        qs.append(q)
    return qs

def matrix_tests(): # Matrixes and vectors.

    tests = []
    np.random.seed(999)
    for i in range(16):
        m44 = random_4x4_matrixes(1)[0]
        m33, v = qmv.m44TOm33v(m44)
        points = np.random.randn(3,8)
        points1A = qmv.m44v(m44, points)
        points1B = qmv.m33v(m33, points) + np.expand_dims(v,1)
        tests.append(tmetric.approx_eq(points1A,points1B))
        m44A = qmv.m33vTOm44(m33, v)
        tests.append(tmetric.approx_eq(m44A,m44))

    return tmetric.is_all_true(tests)

def quat_tests(): # No matrixes, only quaternions and vectors.

    tests = []
    np.random.seed(9999)
    for i in range(16):
        q1 = random_qs(1)[0]
        q2 = random_qs(1)[0]
        points = np.random.randn(3,8)
        if i==7: # Special cases may cause problems.
            q1 = np.asarray([1,0,0,0])
        elif i==8:
            q1 = np.asarray([0,1,0,0])
            points[:,0] = [1,1,0]
            points[:,1] = [-1,-1,0]
        elif i==2:
            points[:,0] = [0,0,1]
            points[:,1] = [0,0,-1]
        q1_inv = qmv.q_inv(q1)
        tests.append(tmetric.approx_eq(qmv.qq(q1_inv,q1),np.asarray([1,0,0,0])))
        points1 = qmv.qv(q1, points)
        tests.append(tmetric.approx_eq(np.sum(points*points),np.sum(points1*points1)))

        points12A = qmv.qv(q2, qmv.qv(q1, points)) # Test associative.
        points12B = qmv.qv(qmv.qq(q2,q1), points)
        tests.append(tmetric.approx_eq(points12A,points12B))

        axis, radians = qmv.qTOaxisangle(q1) # Test axis angle.
        q1A = qmv.axisangleTOq(axis, radians)
        tests.append(tmetric.approx_eq(q1,q1A) or tmetric.approx_eq(-q1,q1A))

        p0 = qmv._v1(points[:,0]); p1 = qmv._v1(points[:,1]) # Test polar shift.
        p0_ = qmv._expand1(p0); p1_ = qmv._expand1(p1)
        qp = qmv.q_from_polarshift(p0, p1)
        tests.append(tmetric.approx_eq(qmv.qv(qp,p0_),p1_))

    return tmetric.is_all_true(tests)

def matrix_quat_tests(): # The BIG ONE.
    tests = []
    # Quaternion to/from matrix conversion.
    np.random.seed(99999)
    for i in range(16):
        q = random_qs(1)[0]
        points = np.random.randn(3,8)
        q33 = qmv.m33_from_q(q)
        qA, r = qmv.m33TOqr(qmv.m33_from_q(q))
        points1 = qmv.m33v(q33, points)
        points1A = qmv.qv(qA, qmv.m33v(r, points))
        tests.append(tmetric.approx_eq(points1,points1A))
        # Note: qmv.qrTOm33(q,r) does not always equal q33 b/c r may have -1 instead of 1.

        # Use a random matrix, instead of quaternions. Internally uses 3x3 matrixes.
        m44 = random_4x4_matrixes(1)[0]
        q,r,v = qmv.m44TOqrv(m44)
        points2 = qmv.m44v(m44, points)
        points2A = qmv.qv(q, qmv.m33v(r, points))+np.expand_dims(v,1)
        tests.append(tmetric.approx_eq(points2,points2A))
        m44A = qmv.qrvTOm44(q,r,v)
        tests.append(tmetric.approx_eq(m44,m44A))

    return tmetric.is_all_true(tests)

def camera_tests():
    # Actually getting panda3D's camera to work properly will take a demo, however...
    np.random.seed(9999)
    tests = []

    # Test inversion:
    camera44 = np.random.randn(4,4); camera44[3,3] = 1.0
    v3 = np.random.randn(3,1)
    v3_green = qmv.cam44_invv(camera44,qmv.cam44v(camera44,v3))
    v3_green1 = qmv.cam44v(np.linalg.inv(camera44),qmv.cam44v(camera44,v3))
    tests.append(tmetric.approx_eq(v3,v3_green))
    tests.append(tmetric.approx_eq(v3,v3_green1))

    for i in range(4): # Test the solve 43.
        camera44 = np.random.randn(4,4); camera44[3,3] = 1.0
        b4_missing1 = np.random.randn(4); b4_missing1[i] = None
        v3_soln = qmv.solve43(camera44, b4_missing1)
        v4_soln = qmv._v4_from_v(v3_soln)
        b4_missing1_green = np.matmul(camera44, v4_soln); b4_missing1_green[i] = None
        tests.append(tmetric.approx_eq(b4_missing1,b4_missing1_green))

    # Test camera to-from (BIG TEST):
    for i in range(16):
        camera44 = np.random.randn(4,4); camera44[3,3] = 1.0
        q, v, f, c, y, a = qmv.cam44TOqvfcya(camera44)
        camera44_green = qmv.qvfcyaTOcam44(q,v,f,c=c,y=y,a=a)
        tests.append(tmetric.approx_eq(camera44,camera44_green))
        q_, v_, f_, c_, y_, a_ = qmv.cam44TOqvfcya(camera44_green)

        tests.append(tmetric.approx_eq([q, v, f, c, y, a],[q_, v_, f_, c_, y_, a_]))

    # Test the camera clipping planes:
    f = 10.0
    v = np.random.randn(3)
    q = qmv.q1(np.random.randn(4))
    look_dir = qmv.qv(q,np.expand_dims([0,0,-1],1))[:,0]
    c = [0.125,64]
    cam44 = qmv.qvfcyaTOcam44(q,v,f,c=c)
    center0_green = qmv.cam44v(cam44,np.expand_dims(v+look_dir*c[0],1))[:,0]
    tests.append(tmetric.approx_eq([0,0,-1],center0_green))
    center1_green = qmv.cam44v(cam44,np.expand_dims(v+look_dir*c[1],1))[:,0]
    tests.append(tmetric.approx_eq([0,0,1],center1_green))

    # Test raycasting for perspective cameras:
    for i in range(16):
        camera44 = np.random.randn(4,4)
        screenxy = np.random.randn(2)
        near_clip, direction = qmv.cam_ray(camera44, screenxy[0], screenxy[1])
        q, v, f, c, y, a = qmv.cam44TOqvfcya(camera44)
        cam_inv = np.linalg.inv(camera44) # invert is the gold standard.
        direction_gold = qmv._v1(qmv.cam44v(cam_inv, np.expand_dims([screenxy[0], screenxy[1], 1.0],1))[:,0]-
                                 qmv.cam44v(cam_inv, np.expand_dims([screenxy[0], screenxy[1],-1.0],1))[:,0])
        screen_near = qmv.cam44v(camera44, np.expand_dims(near_clip,1))[:,0]
        tests.append(tmetric.approx_eq(direction,direction_gold))
        tests.append(tmetric.approx_eq(screen_near,[screenxy[0], screenxy[1],-1.0]))

    # Test the camera look-in-direction:
    f = 10.0
    v = np.random.randn(3)
    look_dir = qmv._v1(np.random.randn(3))
    q = qmv.camq_from_look(look_dir, up=None)
    c = [0.25,64]
    cam44 = qmv.qvfcyaTOcam44(q,v,f,c=c)
    screen_center = v+look_dir*c[0]
    screen_up = qmv._v1(coregeom.project_to_plane(v, look_dir, np.expand_dims(v+[0,0,1],1))[:,0]-v)
    screen_right = np.cross(look_dir, screen_up)
    center_green = qmv.cam44v(cam44,np.expand_dims(screen_center,1))[:,0]
    tests.append(tmetric.approx_eq([0,0,-1],center_green))
    screenx = 0.7; screeny = 0.4;
    point_in_near_plane = screen_center+screenx*screen_right*c[0]/f+screeny*screen_up*c[0]/f
    onscreen_green = qmv.cam44v(cam44,np.expand_dims(point_in_near_plane,1))[:,0]
    tests.append(tmetric.approx_eq([screenx, screeny,-1],onscreen_green))

    # Test orthographic camera:
    zoom_out = 13.0
    far_clip = 27.0
    v = np.random.randn(3)
    q = qmv.q1(np.random.randn(4))
    cam44 = qmv.cam_from_ortho(v, q, zoom_out=zoom_out, far_clip=far_clip)

    look = qmv.qv(q,np.expand_dims([0,0,-1],1))[:,0]
    far_center_point = v+far_clip*look
    tests.append(tmetric.approx_eq([0,0,-1], qmv.cam44v(cam44,np.expand_dims(v,1))[:,0]))
    tests.append(tmetric.approx_eq([0,0,1], qmv.cam44v(cam44,np.expand_dims(far_center_point,1))[:,0]))
    point_in_clip_space = [0.4,0.5,0.6]
    up = qmv.qv(q,np.expand_dims([0,1,0],1))[:,0]
    right = qmv.qv(q,np.expand_dims([1,0,0],1))[:,0]
    point_in_real_space = v+point_in_clip_space[0]*zoom_out*right+point_in_clip_space[1]*zoom_out*up+\
                          far_clip*look*(point_in_clip_space[2]+1.0)*0.5
    tests.append(tmetric.approx_eq(point_in_clip_space, qmv.cam44v(cam44,np.expand_dims(point_in_real_space,1))[:,0]))

    return tmetric.is_all_true(tests)
