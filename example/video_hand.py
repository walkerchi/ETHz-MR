import numpy as np
from tqdm import tqdm
from ..io.dataset import HumanDataset
from ..io.frame_loader import RGBBasedFrameLoader
from ..io.recorder import GeometryVideoRecorder
from ..pipeline.experimental import points3d_to_spheres,\
                                    pointcloud_paint_by_joints,\
                                    pointcloud_paint_by_palm
from ..pipeline.functional import points4d_to_3d,\
                                    depth_to_points4d,\
                                    points4d_map_rgb,\
                                    points4d_to_pointcloud,\
                                    get_hands3d



class HandFilter:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.window = []
        self.alpha  = 0.05 
    def __call__(self, hand):
        self.window.append(np.linalg.norm(hand[:,:3]) if hand is not None else 0)
        if len(self.window) > self.capacity:
            self.window.pop(0)
        array = np.array(self.window)
        num_none = (array == 0).sum()
        std      = array[array!=0].std()
        if std > self.alpha or num_none >= 1:
            return False
        return True
if __name__ == '__main__':
    dataset = RGBBasedFrameLoader('.')
    recorder = GeometryVideoRecorder("script/.video/hand-only.mp4")

    lefthand_filter = HandFilter()
    righthand_filter = HandFilter()
    for frame in tqdm(dataset):
        if lefthand_filter(frame.human_frame.lefthand) and righthand_filter(frame.human_frame.righthand):
            points4d = depth_to_points4d(frame.depth_frame,frame.rig_frame)
            points4d,colors = points4d_map_rgb(points4d, frame.rgb_frame, frame.rig_frame)
            pcd      = points4d_to_pointcloud(points4d, colors)
            hands3d  = get_hands3d(frame.human_frame, frame.rig_frame)
            sphs= points3d_to_spheres(hands3d, radius=0.005)
            pcd = pointcloud_paint_by_joints(pcd, hands3d)
            pcd = pointcloud_paint_by_palm(pcd, hands3d, r=0.5)
            recorder([sphs,pcd])
            
      