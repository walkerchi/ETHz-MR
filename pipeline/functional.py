from typing import Union,Tuple
import open3d as o3d 
import numpy as np
from ..io import RGBFrame,\
                DepthFrame,\
                HumanFrame,\
                RigFrame
from .hands import Hands

def points4d_to_3d(points4d):
    assert points4d.shape[1] == 4, f"expected shape to be [n,4], but got {points4d.shape}"
    return points4d[:, :3] / points4d[:, 3:4]

def points3d_to_2d(points3d):
    assert points3d.shape[1] == 3, f"expected shape to be [n,3], but got {points3d.shape}"
    return points3d[:, :2] / points3d[:, 2:3]

def points3d_to_4d(points3d):
    assert points3d.shape[1] == 3, f"expected shape to be [n,3], but got {points3d.shape}"
    return np.concatenate([points3d, np.ones([len(points3d),1])], -1)

def depth_to_points4d(depth_frame:DepthFrame, rig_frame:RigFrame, dmin:float=0.2, dmax:float=0.7):
    cam2rig = np.linalg.inv(rig_frame.rig2cam)
    depth   = depth_frame.depth.flatten().astype(np.float32) / 1000
    mask    = (depth > dmin) & (depth < dmax)
    points3d_cam = depth[mask, None] * depth_frame.lut[mask]
    points4d_cam = points3d_to_4d(points3d_cam)
    points4d_rig = points4d_cam @ cam2rig.T
    return points4d_rig

def points4d_map_rgb(points4d_rig:np.ndarray, rgb_frame:RGBFrame, rig_frame:RigFrame, remove_black=True)->Tuple[np.ndarray,np.ndarray]:
    assert points4d_rig.shape[1] == 4, f"expected shape to be [n,4], but got {points4d_rig.shape}"
    world2rgb = np.linalg.inv(rgb_frame.rgb2world)
    h, w, _   = rgb_frame.rgb.shape
    points4d_world = points4d_rig @ rig_frame.rig2world.T 
    points4d_rgb   = points4d_world @ world2rgb.T
    points3d_rgb   = points4d_to_3d(points4d_rgb)
    uv3d           = points3d_rgb @ rgb_frame.intrinsic.T 
    uv2d           = points3d_to_2d(uv3d)
    uv2d           = uv2d.astype(int)
    where_within   = (0 <= uv2d[:, 0]) & (uv2d[:, 0] < w) & (0 <= uv2d[:, 1]) & (uv2d[:,1 ] < h)
    uv2d           = uv2d[where_within]
    colors         = rgb_frame.rgb[uv2d[:, 1], w - 1 - uv2d[:, 0]].astype(np.float32) / 255
    points4d_rig   = points4d_rig[where_within]
    if remove_black:
        where_not_black = (colors != 0).any(-1)
        colors          = colors[where_not_black]
        points4d_rig    = points4d_rig[where_not_black]
    return points4d_rig, colors

def points3d_to_pointcloud(points3d_rig:np.ndarray, colors:np.ndarray):
    assert points3d_rig.shape[1] == 3, f"expected shape to be [n,3], but got {points3d_rig.shape}"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d_rig)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals()
    return pcd

def points4d_to_pointcloud(points4d_rig:np.ndarray, colors:np.ndarray):
    points3d_rig = points4d_to_3d(points4d_rig)
    return points3d_to_pointcloud(points3d_rig, colors)

def get_hands3d(human_frame:HumanFrame, rig_frame:RigFrame)->Union[np.ndarray, None]:
    world2rig = np.linalg.inv(rig_frame.rig2world)
    hands = []
    if human_frame.lefthand is not None:
        hands.append(human_frame.lefthand)
    if human_frame.righthand is not None:
        hands.append(human_frame.righthand)
    if len(hands) == 0:
        return None
    hands4d_world = np.concatenate(hands, 0)
    hands4d_rig   = hands4d_world @ world2rig.T
    hands3d_rig   = points4d_to_3d(hands4d_rig)
    return hands3d_rig

def get_palm(hands3d_rig:np.ndarray)->np.ndarray:
    assert hands3d_rig.shape[1] == 3, f"expected shape to be [n,3], but got {hands3d_rig.shape}"
    if len(hands3d_rig) == Hands.N_JOINTS:
        palm = hands3d_rig[Hands.PALM]
    elif len(hands3d_rig) == Hands.N_JOINTS * 2:
        palm = hands3d_rig[Hands.PALM::Hands.N_JOINTS].mean(0)
    else:
        raise Exception(f"The shape of hands3d_rig is not correct, suppose to be [{Hands.N_JOINTS},3] ro [{Hands.N_JOINTS*2},3], but got {hands3d_rig.shape}")
    return palm
def points3d_remove_by_hands(points3d_rig:np.ndarray, hands3d_rig:np.ndarray,r=Hands.RADIUS)->np.ndarray: 
    assert points3d_rig.shape[1] == 3, f"expected shape to be [n,3], but got {points3d_rig.shape}"
    if len(hands3d_rig) == 2 * Hands.N_JOINTS:
        r = np.tile(r,2)
    else:
        assert len(hands3d_rig) == Hands.N_JOINTS
    where_joints = (((points3d_rig[:, None, :] - hands3d_rig[None, :, :])**2).sum(-1) < r**2).any(-1)
    points3d_rig  = points3d_rig[~where_joints]
    return points3d_rig

def points3d_maintain_by_palm(points3d_rig:np.ndarray, hands3d_rig:np.ndarray, r=Hands.FOCUS)->np.ndarray:
    assert points3d_rig.shape[1] == 3, f"expected shape to be [n,3], but got {points3d_rig.shape}"
    palm = get_palm(hands3d_rig)
    where_focus = ((points3d_rig - palm)**2).sum(-1) < r ** 2
    points3d_rig = points3d_rig[where_focus]
    return points3d_rig


