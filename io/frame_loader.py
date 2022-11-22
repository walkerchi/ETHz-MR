__all__ = [
    "RGBBasedFrameLoader"
]

from . import Frame
from .dataset import RGBDataset,\
                    DepthDataset,\
                    HumanDataset,\
                    RigDataset

class RGBBasedFrameLoader:
    def __init__(self, base:str="../.."):
        self.rgb_dataset   = RGBDataset(base)
        self.depth_dataset = DepthDataset(base)
        self.human_dataset = HumanDataset(base)
        self.rig_dataset   = RigDataset(base)
    def __len__(self):
        return len(self.rgb_dataset)
    def __getitem__(self, index:int)->Frame:
        rgb_frame   = self.rgb_dataset[index]
        depth_frame = self.depth_dataset.timestamp(rgb_frame.timestamp)
        human_frame = self.human_dataset.timestamp(rgb_frame.timestamp)
        rig_frame   = self.rig_dataset.timestamp(rgb_frame.timestamp)
        return Frame(rgb_frame  = rgb_frame,
                    depth_frame = depth_frame,
                    human_frame = human_frame,
                    rig_frame   = rig_frame
                    )
