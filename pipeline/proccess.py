"""
    Frame -> Pointcloud/RGB/Depth

"""
from typing import Union,Optional
import open3d as o3d
from ..io import Frame,HumanFrame
from .experimental import pointcloud_paint_by_joints,\
                            pointcloud_paint_by_palm,\
                            points3d_to_spheres,\
                            pointcloud_cluster_maintain_by_plam
from .functional import points4d_to_pointcloud,\
                        points3d_to_pointcloud,\
                        get_hands3d,\
                        depth_to_points4d,\
                        points4d_map_rgb,\
                        points4d_to_3d,\
                        points3d_to_4d,\
                        points3d_remove_by_hands,\
                        points3d_maintain_by_palm
from .filter import HandLinearRegressionFilter,HandStaticFilter,PointCloudClusterFilter

class PointCloudPaint:
    def __call__(self,frame:Frame)->o3d.geometry.PointCloud:
        points4d = depth_to_points4d(frame.depth_frame,frame.rig_frame)
        points4d,colors = points4d_map_rgb(points4d, frame.rgb_frame, frame.rig_frame)
        pcd      = points4d_to_pointcloud(points4d, colors)
        hands3d  = get_hands3d(frame.human_frame, frame.rig_frame)
        if hands3d is not None:
            sphs= points3d_to_spheres(hands3d, radius=0.005)
            pcd = pointcloud_paint_by_joints(pcd, hands3d)
            pcd = pointcloud_paint_by_palm(pcd, hands3d, r=0.5)
            return [pcd, sphs]
        else:
            return pcd

class PointCloudHandFilterPaint:
    def __init__(self,window_size=16):
        self.lefthand_filter = HandStaticFilter(window_size=window_size)
        self.righthand_filter = HandStaticFilter(window_size=window_size)
    def __call__(self, frame:Frame)->Optional[o3d.geometry.PointCloud]:
        if self.lefthand_filter(frame.human_frame.lefthand) and self.righthand_filter(frame.human_frame.righthand):
            points4d = depth_to_points4d(frame.depth_frame,frame.rig_frame)
            points4d,colors = points4d_map_rgb(points4d, frame.rgb_frame, frame.rig_frame)
            pcd      = points4d_to_pointcloud(points4d, colors)
            hands3d  = get_hands3d(frame.human_frame, frame.rig_frame)
            sphs= points3d_to_spheres(hands3d, radius=0.005)
            pcd = pointcloud_paint_by_joints(pcd, hands3d)
            pcd = pointcloud_paint_by_palm(pcd, hands3d, r=0.5)
            return [pcd,sphs]
        else:
            return None
class PointCloudRemove:
    def __call__(self, frame:Frame)->o3d.geometry.PointCloud:
        points4d = depth_to_points4d(frame.depth_frame,frame.rig_frame)
        hands3d  = get_hands3d(frame.human_frame, frame.rig_frame)
        points3d = points4d_to_3d(points4d)
        if hands3d is not None:
            points3d = points3d_remove_by_hands(points3d, hands3d)
            points3d = points3d_maintain_by_palm(points3d, hands3d)
        points4d = points3d_to_4d(points3d)
        points4d,colors = points4d_map_rgb(points4d, frame.rgb_frame, frame.rig_frame)
        pcd      = points4d_to_pointcloud(points4d, colors)
        return pcd

class PointCloudHandFilterRemove:
    def __init__(self, pcd_std_scale:float=1.0, window_size:int=16, use_pointcloud_filter:bool=True):
        self.lefthand_filter    = HandStaticFilter(window_size=window_size)
        self.righthand_filter   = HandStaticFilter(window_size=window_size)
        if use_pointcloud_filter:
            self.pointcloud_filter  = PointCloudClusterFilter(std_scale=pcd_std_scale, window_size=window_size)
        else:
            self.pointcloud_filter  = None
    def __call__(self, frame:Frame)->Optional[o3d.geometry.PointCloud]:
        if self.lefthand_filter(frame.human_frame.lefthand) and self.righthand_filter(frame.human_frame.righthand):
            points4d = depth_to_points4d(frame.depth_frame,frame.rig_frame)
            hands3d  = get_hands3d(frame.human_frame, frame.rig_frame)
            points3d = points4d_to_3d(points4d)
            points3d = points3d_maintain_by_palm(points3d, hands3d)
            points3d = points3d_remove_by_hands(points3d, hands3d)
            points4d = points3d_to_4d(points3d)
            points4d,colors = points4d_map_rgb(points4d, frame.rgb_frame, frame.rig_frame)
            pcd      = points4d_to_pointcloud(points4d, colors)
            if self.pointcloud_filter is not None:
                pcd      = self.pointcloud_filter(pcd, hands3d)
            return pcd
        else:
            return None