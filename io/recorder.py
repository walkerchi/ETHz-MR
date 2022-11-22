__all__ = [
    "DepthVideoRecorder",
    "RGBVideoRecorder",
    "PointCloudPLYRecorder",
    "PointCloudVideoRecorder",
    "PointCloudRecorder"
]
from typing import Union,List,Tuple,Optional
import open3d as o3d
import cv2
import numpy as np
import os

Geometry = Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]
Geometries = Union[Geometry, List[Geometry],Tuple[Geometry,...]]


class RGBVideoRecorder:
    def __init__(self, 
                path:str    = "rgb_recorder.mp4", 
                width:int   = 1080,
                height:int  = 960,
                fps:int     = 24
                ):
        assert path.endswith(".mp4")
        self.path   = path
        self.width  = width 
        self.height = height
        self.video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.width, self.height))
    def __call__(self, rgb:np.ndarray)->None:
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.width,self.height))
        self.video.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)) 
    def __del__(self):
        if hasattr(self, "video"):
            self.video.release()
    

class DepthVideoRecorder:
    def __init__(self, 
                path:str    = "rgb_recorder.mp4", 
                width:int   = 1080,
                height:int  = 960,
                fps:int     = 24
                ):
        assert path.endswith(".mp4")
        self.path   = path
        self.width  = width 
        self.height = height
        self.video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.width, self.height))
    def __call__(self, gray:np.ndarray)->None:
        gray = cv2.resize(gray, (self.width,self.height))
        self.video.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)) 
    def __del__(self):
        if hasattr(self, "video"):
            self.video.release()
    

class GeometryPLYRecorder:
    def __init__(self,
                path:str="./.ply_recorder"):
        self.counter = 0
        self.path    = path 
        if not os.path.exists(path):
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))
            os.mkdir(path)
    def save_single_geometry(self, geo:Geometry)->None:
        if isinstance(geo, o3d.geometry.PointCloud):
            o3d.io.write_point_cloud(os.path.join(self.path,f"pointcloud-{self.counter}.ply"), geo)
        elif isinstance(geo, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(os.path.join(self.path,f"trianglemesh-{self.counter}.ply"), geo)
        else:
            raise Exception(f"Expected Type to be o3d.geometry.PointCloud or o3d.geometry.TriangleMesh not {type(geo)}")
    def __call__(self, geos:Optional[Geometries])->None:
        if geos is None:
            return 
        if isinstance(geos, (list, tuple)):
            for geo in geos:
                self.save_single_geometry(geo)
        else:
            self.save_single_geometry(geos)
        self.counter += 1

class GeometryVideoRecorder:
    def __init__(self, 
                path:str     = "obj_recoder.mp4",
                width:int    = 1024,
                height:int   = 960,
                fps:int      = 24,
                pointsize:float=0.8,
                background:np.ndarray=np.array([0.2,0.2,0.2])
                ):
        assert path.endswith(".mp4")
        self.path      = path 
        self.pointsize = pointsize
        self.width     = width 
        self.height    = height
        self.video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.width, self.height))
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width,height=self.height,visible=False,)
        render_opt = self.vis.get_render_option()
        render_opt.point_size = self.pointsize
        render_opt.show_coordinate_frame = True
        render_opt.background_color = background
    def __call__(self, geos:Optional[Geometries])->None:
        if geos is None:
            return 
        if isinstance(geos, (list, tuple)):
            for geo  in geos:
                self.vis.add_geometry(geo)
        else:
            self.vis.add_geometry(geos)
        self.vis.poll_events()
        self.vis.update_renderer()
        image = self.vis.capture_screen_float_buffer(do_render=True)
        self.vis.clear_geometries()
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)
        self.video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    def __del__(self):
        if hasattr(self, "vis"):
            self.vis.destroy_window()
        if hasattr(self, "video"):
            self.video.release()

class GeometryRecorder:
    def __init__(self,
                mp4_path= "obj_render.mp4",
                ply_path= "./.obj_render",
                width   = 1080,
                height  = 960,
                fps     = 24,
                pointsize=0.8,
                background=np.array([0.2,0.2,0.2])
                ):
        self.mp4_path = mp4_path
        self.ply_path = ply_path
        if self.mp4_path is not None:
            self.mp4_recorder = GeometryVideoRecorder(self.mp4_path, width, height, fps, pointsize, background)
        if self.ply_path is not None:
            self.ply_recorder = GeometryPLYRecorder(self.ply_path)
    def __call__(self, geos:Optional[Geometries])->None:
        if geos is None:
            return 
        if self.mp4_path is not None:
            self.mp4_recorder(geos)
        if self.ply_path is not None:
            self.ply_recorder(geos)