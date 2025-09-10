import cv2

def display_frames_opencv(selected_frames):
    """
    使用OpenCV窗口展示帧（适合快速查看）
    """
    for frame_idx, frame_data in selected_frames:
        # OpenCV使用BGR格式，所以需要转换，那检测的时候是不是也需要转换一下

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 在图像上添加文本信息
        cv2.putText(frame_rgb, f'Frame: {frame_idx}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) # 2代表的是文本线条宽度
        cv2.putText(frame_rgb, f'Size: {frame_data.shape[1]}x{frame_data.shape[0]}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # 显示图像
        cv2.imshow(f'Frame {frame_idx}', frame_rgb)
        
        # 等待按键，按任意键继续下一张，按'q'退出
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        
        # cv2.destroyAllWindows()
    cv2.destroyAllWindows()