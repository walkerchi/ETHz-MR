from tqdm import tqdm
import numpy as np
from ..io.frame_loader import RGBBasedFrameLoader
from ..io.recorder import GeometryRecorder,GeometryVideoRecorder
from ..pipeline.proccess import PointCloudHandFilterRemove
from ..pipeline.register import PointCloudNaiveRegister
from ..pipeline.colors import Colors
from argparse import ArgumentParser

if __name__ == "__main__":
    loader = RGBBasedFrameLoader(base=".")
    process = PointCloudHandFilterRemove()
    register = PointCloudNaiveRegister()
    record = GeometryRecorder(
        mp4_path="script/.video/pointcloud-register.mp4",
        ply_path="script/.ply/register",
        background=Colors.WHITE
    )
    for frame in tqdm(loader):
        record(register(process(frame)))