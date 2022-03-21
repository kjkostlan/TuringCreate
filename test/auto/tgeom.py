# Tests basic geometry operations.
import numpy as np
import math
import TapeyTeapots.meshops.coregeom as coregeom
from . import tmetric

def test_basics():
    # The regression line:
    geom_3xk_exact =  np.transpose(np.asarray([[1,2,3],[2,12,103],[3,22,203]]))# We know the line.
    origin_exact = [2,12,103]
    direction_exact = np.asarray([1,10,100])/np.linalg.norm([1,10,100])
    points_exact_green, direction_exact_green = coregeom.regression_line(geom_3xk_exact)

    linetest = tmetric.approx_eq([origin_exact, direction_exact],[points_exact_green, direction_exact_green])

    np.random.seed(123123) # This test fails on a set of measure zero.
    geom_3xk_random = np.random.randn(3, 16)
    trial_directions = [np.random.randn(3) for _ in range(64)]
    def eval_direction(d):
        d = d/(np.linalg.norm(d))
        scores = np.squeeze(np.sum(np.expand_dims(d,1)*geom_3xk_random, axis=0))
        return np.std(scores)
    _, best_dir_green = coregeom.regression_line(geom_3xk_random)
    max_score_trial = np.max([eval_direction(d) for d in trial_directions])
    linetest1 = (max_score_trial<eval_direction(best_dir_green))

    # The regression plane:
    normal_vec_plane = np.asarray([0.25, 0.5, -0.125])
    origin_vec_plane = [0,-2,-4]
    geom_3xk_random = np.random.randn(3, 16)
    #https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/least-squares-determinants-and-eigenvalues/projections-onto-subspaces/MIT18_06SCF11_Ses2.2sum.pdf
    proj_matrix = np.outer(normal_vec_plane,normal_vec_plane)/np.sum(normal_vec_plane*normal_vec_plane)
    geom_3xk_projected = np.matmul(proj_matrix,geom_3xk_random)
    geom_3xk_in_plane =geom_3xk_random - geom_3xk_projected + np.expand_dims(origin_vec_plane,1)
    _, normal_plane_green = coregeom.regression_plane(geom_3xk_in_plane) # origin vec will be different b/c or random points.
    planetest = tmetric.approx_eq(normal_plane_green, normal_vec_plane/np.linalg.norm(normal_vec_plane))

    # Projection fns:
    geom_3xk_random = np.random.randn(3, 16)
    normal_vec_plane = np.asarray([3.1416, 2.7818, 1.618])
    origin_vec_plane = [1.414, -1.2020569, 2.6854520010]

    geom_3xk_project_plane = coregeom.project_to_plane(origin_vec_plane, normal_vec_plane, geom_3xk_random)
    dots = np.sum(geom_3xk_project_plane*np.expand_dims(normal_vec_plane,1),axis=0)

    plane_proj_test = (np.std(dots)<1e-6) and (tmetric.approx_eq(dots[0],np.sum(origin_vec_plane*normal_vec_plane))) and (np.std(geom_3xk_project_plane) > 0.01)

    geom_3xk_project_line = coregeom.project_to_line(origin_vec_plane, normal_vec_plane, geom_3xk_random)
    geom_3xk_project_line0 = geom_3xk_project_line - np.expand_dims(origin_vec_plane,1)
    ratios = geom_3xk_project_line0[0,:]/geom_3xk_project_line0[1,:]
    line_proj_test = np.std(ratios)<1e-6

    # Line-line closest:
    pt0 = np.random.randn(3)
    dir0 = np.random.randn(3)
    pt1 = np.random.randn(3,7)
    dir1 = np.random.randn(3,7)

    [closest_0, closest_1] = coregeom.line_line_closest(pt0, dir0, pt1, dir1)
    closest_0 = closest_0[:,5]
    closest_1 = closest_1[:,5]
    pt1 = pt1[:,5]
    dir1 = dir1[:,5]
    min_norm_green = coregeom.dists(closest_0, closest_1)[0]
    eps = 1e-3 # must be > than sqrt(precision).
    norms = [coregeom.dists((closest_0-eps*dir0), closest_1)[0], coregeom.dists((closest_0+eps*dir0), closest_1)[0],
             coregeom.dists(closest_0, (closest_1-eps*dir1))[0], coregeom.dists(closest_0, (closest_1+eps*dir1))[0]]
    line_line_closest_test = np.min(norms) > min_norm_green

    # Line-plane intersection:
    plane_origin = np.random.randn(3)
    plane_normal = np.random.randn(3)
    line_origins = np.random.randn(3,16)
    line_directions = np.random.randn(3,16)
    intersects = coregeom.line_plane_intersection(plane_origin, plane_normal, line_origins, line_directions)
    line_plane_project = tmetric.approx_eq(coregeom.project_to_plane(plane_origin, plane_normal, intersects), intersects)
    line_plane_project = line_plane_project and tmetric.approx_eq([0.0,0.0,0.0],np.cross(line_origins[:,7]-intersects[:,7],line_directions[:,7]))

    return (linetest and linetest1 and planetest and plane_proj_test and line_proj_test and line_line_closest_test and line_plane_project)

def test_triangles():
    # Fun with triangles.
    tests = []

    np.random.seed(123456)
    random_ortho, _ = np.linalg.qr(np.random.randn(3, 3))

    horiz_right_triangle = np.expand_dims(np.transpose([[0,0,0],[1,0,0],[0,1,0]]),2)

    tests.append(tmetric.approx_eq(coregeom.triangle_areas(horiz_right_triangle),[0.5]))

    tests.append(tmetric.approx_eq(coregeom.triangle_scaled_normals(horiz_right_triangle),np.expand_dims([0,0,0.5],1)))

    rotated_triangle = np.einsum('iu,ujk->ijk',random_ortho, horiz_right_triangle)
    tests.append(tmetric.approx_eq(coregeom.triangle_areas(rotated_triangle),[0.5]))

    triangle_pair = np.concatenate([horiz_right_triangle, 2.5*rotated_triangle],axis=2)
    tests.append(tmetric.approx_eq(coregeom.triangle_areas(triangle_pair),[0.5, 0.5*2.5*2.5]))

    degenerate_triangle = np.expand_dims(np.transpose([[0,0,0],[1,0,0],[2,0,0]]),2)
    tests.append(tmetric.approx_eq(coregeom.triangle_areas(degenerate_triangle),[0.0]))

    simplex = np.transpose([[1,0,0],[0,1,0],[0,0,1]])
    center_of_simplex = np.asarray([1,1,1])/3.0
    edge_12 = np.asarray([0.5,0.5,0])
    edge_13 = np.asarray([0.5,0.0,0.5])
    edge_23 = np.asarray([0.0,0.5,0.5])
    vertex_2 = np.asarray([0.0,1.0,0.0])
    query = np.stack([center_of_simplex, edge_12, edge_13, edge_23, vertex_2], axis=1)
    result = np.zeros([4,query.shape[1]])
    result[0:3,:] = query
    tests.append(tmetric.approx_eq(coregeom.barycentric(simplex, query),result))
    tests.append(tmetric.approx_eq(coregeom.barycentric(2.5*simplex, 2.5*query),result))

    tests.append(tmetric.approx_eq(coregeom.barycentric(5.0*np.squeeze(horiz_right_triangle), np.transpose([[0,0,1]])),np.transpose([[1,0,0,0.2]])))

    rand_tri = np.random.randn(3,3)
    rand_pts = np.random.randn(3,16)
    bary = coregeom.barycentric(rand_tri, rand_pts)
    rand_pts_green = coregeom.unbarycentric(rand_tri, bary)
    tests.append(tmetric.approx_eq(rand_pts, rand_pts_green))

    # Triangle projection test:
    right_tri = np.transpose([[0,0,0],[1,0,0],[0,1,0]])
    pts0 = -np.random.random([3,16])
    pts1 = coregeom.triangle_project(right_tri, pts0)
    tests.append(tmetric.approx_eq(pts1, np.zeros_like(pts1)))

    return tmetric.is_all_true(tests)

def test_inside_loop():
    tests = []

    def rand_mat_no_zshear():
        rand_mat = np.random.random([3,3])
        rand_mat[2,:] = 0 # cannot shear orthogonal to the plane.
        rand_mat[:,2] = 0
        rand_mat[2,2] = 1
        random_ortho, _ = np.linalg.qr(np.random.randn(3, 3))
        return np.matmul(random_ortho, rand_mat)

    np.random.seed(4321)
    square = np.transpose([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
    pts = 0.414*np.random.randn(3,32)+np.expand_dims([0.5,0.5,0.0],axis=1)

    inside_gold = (pts[1,:]<1)*(pts[0,:]<1)*(pts[1,:]>0)*(pts[0,:]>0)
    inside_green = coregeom.is_inside_loop(square, pts)

    tests.append(tmetric.approx_eq(inside_gold+0.1, inside_green+0.1))

    # Apply a random transformation:
    for _ in range(8):
        rand_mat = rand_mat_no_zshear()
        square1 = np.matmul(rand_mat, square)
        pts1 = np.matmul(rand_mat, pts)
        tests.append(tmetric.approx_eq(inside_gold+0.1, coregeom.is_inside_loop(square1, pts1)+0.1))

    # Another random test:
    for _ in range(8):
        k = 12
        r = 0.125+np.random.random(k)
        theta = np.sort(np.concatenate([np.random.random(k-2)*2.0*np.pi, [0, np.pi]]))
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = np.random.randn(k)
        pts = np.stack([x, y, z*0],axis=0) # pts itself make completly planer.
        pts_inside = pts*0.99
        pts_inside[2,:] = z
        pts_outside = pts*1.01
        pts_outside[2,:] = z

        rand_mat = rand_mat_no_zshear()
        pts1 = np.matmul(rand_mat, pts)
        pts_inside1 = np.matmul(rand_mat, pts_inside)
        pts_outside1 = np.matmul(rand_mat, pts_outside)
        tests.append(tmetric.approx_eq(coregeom.is_inside_loop(pts1, pts_inside1)+0.1, 0.1+np.ones(k)))
        tests.append(tmetric.approx_eq(coregeom.is_inside_loop(pts1, pts_outside1)+0.1, 0.1+np.zeros(k)))

    return tmetric.is_all_true(tests)
