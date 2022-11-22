import numpy as np
import open3d as o3d 
from functools import reduce
from .functional import get_palm
from .hands import Hands
from .colors import Colors

def points3d_to_spheres(points3d_rig:np.ndarray, color=Colors.RED, radius=1.0, resolution=20):
    sphs = o3d.geometry.TriangleMesh()
    for point3d_rig in points3d_rig:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius,resolution)
        sph.paint_uniform_color(color)
        sph.translate(point3d_rig)
        sphs += sph
    return sphs

def pointcloud_cluster_maintain_by_plam(pcd:o3d.geometry.PointCloud, hands3d_rig:np.ndarray, eps=0.01, min_points=32):
    palm   = hands3d_rig.mean(0)
    points = np.array(pcd.points)
    labels = np.array(pcd.cluster_dbscan(eps=eps,min_points=min_points))
    min_label = None
    min_dist2 = np.inf        
    for label in np.unique(labels):
        if label >= 0:
            center = points[labels == label].mean(0)
            dist2  = ((center - palm)**2).sum()
            if dist2 < min_dist2:
                min_dist2 = dist2
                min_label = label 
    pcd = pcd.select_by_index(np.where(min_label == labels)[0])
    return pcd

def pointcloud_paint_by_joints(pcd:o3d.geometry.PointCloud, hands3d_rig:np.ndarray,r=Hands.RADIUS, color=Colors.ORANGE)->o3d.geometry.PointCloud: 
    if len(hands3d_rig) == 2 * Hands.N_JOINTS:
        r = np.tile(r,2)
    else:
        assert len(hands3d_rig) == Hands.N_JOINTS
    points3d_rig = np.array(pcd.points)
    colors       = np.array(pcd.colors)
    where_joints = (((points3d_rig[:, None, :] - hands3d_rig[None, :, :])**2).sum(-1) < r**2).any(-1)
    colors[where_joints] = color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def pointcloud_paint_by_palm(pcd:o3d.geometry.PointCloud, hands3d_rig:np.ndarray, r=Hands.FOCUS, color=Colors.GREEN)->o3d.geometry.PointCloud:
    palm = get_palm(hands3d_rig)
    points3d_rig = np.array(pcd.points)
    colors       = np.array(pcd.colors)
    where_not_focus = ((points3d_rig - palm)**2).sum(-1) > r ** 2
    colors[where_not_focus] = color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd



