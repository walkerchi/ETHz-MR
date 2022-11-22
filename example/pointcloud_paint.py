from tqdm import tqdm
from ..io.frame_loader import RGBBasedFrameLoader
from ..io.recorder import GeometryRecorder,GeometryVideoRecorder
from ..pipeline.proccess import PointCloudPaint, PointCloudHandFilterPaint
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filter", action="store_true")
    config = parser.parse_args()
    loader = RGBBasedFrameLoader(base=".")
    if config.filter:
        process = PointCloudHandFilterPaint()
        record = GeometryVideoRecorder(
        "script/.video/pointcloud-hand-filter-paint.mp4"
        )
    else:
        process= PointCloudPaint()
        record = GeometryVideoRecorder(
        "script/.video/pointcloud-paint.mp4"
        )
    
    for frame in tqdm(loader):
        record(process(frame))