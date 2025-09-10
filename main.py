from utils import readVideoFrames, showImages, multimodalSemanticRetrieval
from ultralytics import YOLO
import cv2
from datetime import datetime

# 存储YOLO模型的绝对路径
YOLO_MODEL_PATH = "D:\\ultralytics\\YOLO+CLIP\\yolov8n.pt"
# 存储CLIP模型的绝对路径
CLIP_MODEL_PATH = "D:\\ultralytics\models\\chinese-clip-vit-huge-patch14"
# 需要解析的视频的绝对路径
VIDEO_PATH = "D:\\ultralytics\\YOLO+CLIP\\videos\\video03\\video03-1080p.mp4"
# 目标图像的绝对路径
TARGET_IMAGE_PATH = "D:\\ultralytics\\YOLO+CLIP\\YoloClipOutput\\90-crop.jpg"

def frameExtract(frames: list, frameRate: int):# 每秒读取五分之一帧率大小的帧数
    print(f"视频帧率是{frameRate}fps,每秒钟按照相同间隔抽取{int(frameRate/5)}帧")
    frameExtracted = frames[::int(frameRate/5)]
    print(f"一共抽取了{len(frameExtracted)}帧")
    return frameExtracted


if __name__ == "__main__":
    start_time = datetime.now()
    video_frames, fps = readVideoFrames.read_mp4_opencv(VIDEO_PATH)
    frameExtracted = frameExtract(video_frames, fps)
    # video_frames_selected = readVideoFrames.read_specific_frames(VIDEO_PATH, frame_indices = [100, 200, 300])
    # showImages.display_frames_opencv(video_frames_selected)
    multimodalSemanticRetrieval.search_video_frame( 
        yolo_model_path=YOLO_MODEL_PATH,
        clip_model_path=CLIP_MODEL_PATH,
        video_frames=frameExtracted,
        text="一张紫红色轿车的照片",
        targetImage=TARGET_IMAGE_PATH,
        yolo_conf=0.5,
        verbose=True
    )
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"视频处理完成，耗时: {duration}")
