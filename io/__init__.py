__all__ = [
    "Frame",
    "RGBFrame",
    "DepthFrame",
    "HumanFrame",
    "RigFrame",
]
from typing import Union
from dataclasses import dataclass,asdict
import numpy as np
@dataclass
class RGBFrame:
    rgb:np.ndarray      # var (rgb)     [428,760,3]    
    rgb2world:np.ndarray# var           [4,4]
    intrinsic:np.ndarray# const         [3,3]
    timestamp:np.uint64 # var       
@dataclass
class DepthFrame:       
    depth:np.ndarray    # var (cam)     [512, 512]
    lut:np.ndarray      # const         [512*512,3]
    timestamp:np.uint64
@dataclass
class HumanFrame:
    head:np.ndarray                 # var   [4,4] 
    lefthand:Union[np.ndarray,None] # var   (world)[26,4]
    righthand:Union[np.ndarray,None]# var   (world)[26,4]
    gaze:Union[np.ndarray,None]     # var   (world)[26,4]
    timestamp:np.uint64
@dataclass
class RigFrame:
    rig2cam:np.ndarray  # const     [4,4]
    rig2world:np.ndarray# var       [4,4]
    timestamp:np.uint64
@dataclass
class Frame:
    rgb_frame:RGBFrame
    depth_frame:DepthFrame
    human_frame:HumanFrame
    rig_frame:RigFrame
   
   