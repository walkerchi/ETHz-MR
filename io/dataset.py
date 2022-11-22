"""
    RGB
        principle_point, (rgb[rgb],rgb2world,focal_length) 
    Depth
        lut, (depth[cam])
    Human
        (head[world], lefthand[world], righthand[world], gaze[world])
    Rig
        rig2cam,(rig2world) 
"""

__all__ = [
    "RGBDataset",
    "DepthDataset",
    "HumanDataset",
    "RigDataset"
]
from typing import Tuple
import os
import cv2
import numpy as np
from glob import glob 
from PIL import Image
import re
from . import RGBFrame,\
                DepthFrame,\
                HumanFrame,\
                RigFrame


Tuple8ndarray = Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]


class RGBDataset:
    def __init__(self,base:str="../.."):
        meta_path = os.path.join(base,[i for i in os.listdir(base) if i.endswith("pv.txt")][0])
        self.w, self.h, self.intrinsics, self.rgb2worlds, self.timestamps = self.load_meta(meta_path)
        self.paths = [os.path.join(base, "PV", f"{t}.bytes") for t in self.timestamps]
        self.sort_timeline()
    def sort_timeline(self):
        index = np.argsort(self.timestamp)
        if np.all((index[1:] - index[:-1]) > 0):
            return 
        else:
            self.timestamps = self.timestamps[index]
            self.paths      = self.paths[index]
            self.intrinsics = self.intrinsics[index]
            self.rgb2worlds = self.rgb2worlds[index]
    def load_meta(self,meta_path:str)->Tuple[int,int,np.ndarray,np.ndarray,np.ndarray]:   
        w, h = None, None
        cx, cy = None, None
        intrinsics = []
        rgb2worlds = []
        timestamps  = []
        with open(meta_path) as f:
            lines = f.readlines()
            metas = lines[0].strip().split(",")
            cx, cy, w, h = float(metas[0]), float(metas[1]), int(metas[2]), int(metas[3])
            for frame in lines[1:]:
                items                   = frame.split(',')
                timestamps.append(np.uint64(items[0]))
                fx, fy                  = float(items[1]), float(items[2])
                intrinsics.append(np.array([[fx, 0, w-cx],
                                           [0, fy,   cy],
                                           [0,  0,    1]]))
                rgb2worlds.append(np.array(items[3:20]).astype(np.float32).reshape([4,4]))
        return w, h, \
                np.stack(intrinsics, 0).astype(np.float32), \
                np.stack(rgb2worlds, 0).astype(np.float32), \
                np.array(timestamps,dtype=np.uint64)
    def load_byte(self,byte_path:str)->np.ndarray:
        with open(byte_path,"rb") as f:
            bgra = np.frombuffer(f.read(), dtype=np.uint8)
        bgra = bgra.reshape([self.h, self.w, 4])
        rgb  = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        return rgb
    def __getitem__(self,index:int)->RGBFrame:
        return RGBFrame(rgb=self.load_byte(self.paths[index]), 
                        rgb2world=self.rgb2worlds[index],
                        intrinsic=self.intrinsics[index],
                        timestamp=self.timestamps[index])
    def __len__(self):
        return len(self.timestamps)
    def timestamp(self,t:np.uint64):
        index = np.argmin(abs(self.timestamps - t))
        return self.__getitem__(index)
        
class DepthDataset:
    def __init__(self, base:str="../.."):
        path = [os.path.join(base,i) for i  in os.listdir(base) if i.endswith("_lut.bin")][0]
        self.lut        = self.load_bin(path)
        path = [os.path.join(base,i) for i  in os.listdir(base) if i.startswith("Depth") and os.path.isdir(os.path.join(base,i))][0]
        self.paths      = sorted(glob(f"{path}/*[0-9].pgm"))
        # self.paths      = [os.path.join(path, i) for i in self.paths]
        self.timestamps = np.array([np.uint64(re.findall(f"[0-9]+",i)[0]) for i in self.paths])    
    def __len__(self):
        return len(self.timestamps)
    def load_pgm(self, pgm_path:str):
        return np.array(Image.open(pgm_path), dtype=np.uint16)
    def load_bin(self, bin_path:str):
        with open(bin_path, "rb") as f:
            lut = np.frombuffer(f.read(), dtype="f").reshape([-1, 3]).astype(np.float32)
        return lut
    def __getitem__(self,index:int)->DepthFrame:
        return DepthFrame(
            depth       = self.load_pgm(self.paths[index]),
            lut         = self.lut,
            timestamp   = self.timestamps[index]
        )
    def timestamp(self,t):
        index = np.argmin(abs(self.timestamps - t))
        return self.__getitem__(index)

class HumanDataset:
    def __init__(self,base:str="../.."):
        path = [os.path.join(base,i) for i in os.listdir(base) if i.endswith(".csv")][0]
        self.heads,\
        self.lefthands,\
        self.righthands,\
        self.gazes,\
        self.is_lefthand_avails,\
        self.is_righthand_avails,\
        self.is_gaze_avails,\
        self.timestamps = self.load_csv(path)
    def load_csv(self, csv_path:str)->Tuple8ndarray:
        timestamps = []
        heads      = []
        lefthands  = []
        righthands = []
        gazes      = []
        is_lefthand_avails  = []
        is_righthand_avails = []
        is_gaze_avails      = []
        with open(csv_path) as f:
            for line in f:
                items = line.strip().split(',')
                timestamps.append(
                        np.uint64(items[0])
                    )
                heads.append(
                        np.array(items[1:17]).astype(np.float32).reshape([4,4])
                    )
                if int(items[17]) == 1:
                    lefthands.append(
                        np.array(items[18:434]).astype(np.float32).reshape([26, 4, 4])[:,:,3]
                    )
                    is_lefthand_avails.append(True)
                else:
                    lefthands.append(
                        np.zeros([26,4], dtype=np.float32)
                    )
                    is_lefthand_avails.append(False)
                if int(items[434]) == 1:
                    righthands.append(
                        np.array(items[435:851]).astype(np.float32).reshape([26,4,4])[:,:,3]
                    )
                    is_righthand_avails.append(True)
                else:
                    righthands.append(
                        np.zeros([26,4], dtype=np.float32)
                    )
                    is_righthand_avails.append(False)
                if int(items[851]) == 1:
                    gazes.append(
                        np.array(items[852:861]).astype(np.float32)
                    )
                    is_gaze_avails.append(True)
                else:
                    gazes.append(
                        np.zeros([9], dtype=np.float32)
                    )
                    is_gaze_avails.append(False)
        return np.stack(heads, 0),\
                np.stack(lefthands, 0),\
                np.stack(righthands,0),\
                np.stack(gazes, 0),\
                np.array(is_lefthand_avails),\
                np.array(is_righthand_avails),\
                np.array(is_gaze_avails),\
                np.array(timestamps)
    def sort_timeline(self):
        index = np.argsort(self.timestamp)
        if np.all((index[1:] - index[:-1]) > 0):
            return 
        else:
            self.timestamps = self.timestamps[index]
            self.heads      = self.heads[index]
            self.lefthands  = self.lefthands[index]
            self.righthands = self.righthands[index]
            self.gazes      = self.gazes[index]
            self.is_lefthand_avails = self.is_lefthand_avails[index]
            self.is_righthand_avails= self.is_righthand_avails[index]
            self.is_gaze_avails     = self.is_gaze_avails[index]
    def __len__(self):
        return len(self.timestamps)
    def __getitem__(self,index:int)->HumanFrame:
        return HumanFrame(
            head        = self.heads[index],
            lefthand    = self.lefthands[index] if self.is_lefthand_avails[index] else None,
            righthand   = self.righthands[index] if self.is_righthand_avails[index] else None,
            gaze        = self.gazes[index] if self.is_gaze_avails[index] else None,
            timestamp   = self.timestamps[index]
        ) 
    def timestamp(self, t:np.uint64):
        index = np.argmin(abs(self.timestamps - t))
        return self.__getitem__(index)

class RigDataset:
    def __init__(self,base:str="../.."):
        path = [os.path.join(base,i) for i in os.listdir(base) if i.endswith("_rig2world.txt")][0]
        self.rig2worlds, self.timestamps = self.load_rig2world(path)
        path = [os.path.join(base,i) for i in os.listdir(base) if i.endswith("_extrinsics.txt")][0]
        self.rig2cam = self.load_extrinsics(path)

    def load_extrinsics(self,path:str)->np.ndarray:
        return np.loadtxt(path,delimiter=",").reshape([4,4]).astype(np.float32)
    def load_rig2world(self, path:str)->np.ndarray:
        timestamps = []
        rig2worlds = []
        with open(path) as f:
            for line in f:
                items = line.strip().split(',')
                timestamps.append(
                    np.uint64(items[0])
                )
                rig2worlds.append(
                    np.array(items[1:]).astype(np.float32).reshape([4,4])
                )
        return np.stack(rig2worlds, 0).astype(np.float32),\
                np.array(timestamps, dtype=np.uint64)
    def __len__(self):
        return len(self.timestamps)
    def __getitem__(self, index:int)->RigFrame:
        return RigFrame(
            rig2cam     = self.rig2cam,
            rig2world   = self.rig2worlds[index],
            timestamp   = self.timestamps[index]
        )
    def timestamp(self, t:np.uint64):
        index = np.argmin(abs(self.timestamps - t))
        return self.__getitem__(index)


if __name__ == '__main__':
    print(RGBDataset(".")[10])
    print(DepthDataset(".")[10])
    print(HumanDataset(".")[10])
    print(RigDataset(".")[100])