from typing import Union,Optional
from ..io import Frame,DepthFrame,RGBFrame
from scipy.spatial import ConvexHull
import open3d as o3d
import mediapipe as mp
import cv2
import numpy as np


class PointCloudNaiveRegister:
    def __init__(self, voxelsize=0.001):
        self.Ts = []
        self.O  = np.eye(4)
        self.voxelsize = voxelsize
        self.pcd = None
    def __call__(self, pcd:Optional[o3d.geometry.PointCloud])->Optional[o3d.geometry.PointCloud]:
        if pcd is None:
            return None
        if self.pcd is None:
            self.pcd = pcd.voxel_down_sample(self.voxelsize)
        else:
            src_pcd = self.pcd
            dst_pcd = pcd.voxel_down_sample(self.voxelsize)
            T =  o3d.pipelines.registration.registration_generalized_icp(
                src_pcd,
                dst_pcd,
                0.1,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
                    o3d.pipelines.registration.TukeyLoss(0.1).k
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(1e-7,1e-7,300)
            ).transformation
            # T = o3d.pipelines.registration.registration_colored_icp(
            #     src_pcd, dst_pcd, 0.1, np.eye(4),
            #     o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            #     o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
            #                                                     relative_rmse=1e-6,
            #                                                     max_iteration=50)).transformation
            self.Ts.append(T)
            self.O = self.O @ T 
            self.pcd = self.pcd.transform(T) + dst_pcd
            self.pcd = self.pcd.voxel_down_sample(self.voxelsize)
            # _, ind   = self.pcd.remove_statistical_outlier(nb_neighbors=8,std_ratio=1.0)
            # self.pcd = self.pcd.select_by_index(ind)
        return self.pcd
    def finalize(self)->o3d.geometry.PointCloud:
        return self.pcd
