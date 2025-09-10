import cv2
import numpy as np

def read_mp4_opencv(video_path):
    """
    使用OpenCV读取MP4文件
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) # 获取视频的帧率
    frames = []
    # num = 0
    while True:
        # num += 1
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV默认使用BGR格式，转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if num == 50:
        #     cv2.imshow("function_cvt", frame_rgb)
        #     cv2.waitKey(1)  # 等待1毫秒，让窗口有机会刷新
        #     cv2.imshow("function_original", frame)
        #     cv2.waitKey(1)  # 等待1毫秒，让窗口有机会刷新
        frames.append(frame_rgb)
    cap.release() # 释放内存资源
    print(f"读取到了{len(frames)}帧，视频帧率是{fps}fps")
    return np.array(frames), fps # 这里为什么要转化为np数组，因为后续要使用CLIP模型，CLIP模型的输入是np数组

def read_specific_frames(video_path, frame_indices):
    """
    读取特定帧号的帧，返回的不是np数组

    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame_idx, frame_rgb))
    cap.release()
    return frames
