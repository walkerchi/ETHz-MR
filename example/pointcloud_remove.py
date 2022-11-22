from tqdm import tqdm
from ..io.frame_loader import RGBBasedFrameLoader
from ..io.recorder import GeometryRecorder,GeometryVideoRecorder
from ..pipeline.proccess import PointCloudHandFilterRemove
from ..pipeline.register import PointCloudNaiveRegister
from argparse import ArgumentParser

if __name__ == "__main__":
    loader = RGBBasedFrameLoader(base=".")
    process = PointCloudHandFilterRemove()
    record = GeometryRecorder(
        mp4_path="script/.video/pointcloud-remove.mp4",
        ply_path="script/.ply/remove"
    )
    for frame in tqdm(loader):
        record(process(frame))