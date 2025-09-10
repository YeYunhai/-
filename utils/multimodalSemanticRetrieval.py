from ultralytics import YOLO
import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import cv2
import numpy as np
from torch.nn.functional import cosine_similarity


# 输入参数——YOLO模型的存储地址、视频帧列表、需要检测的目标图片对应的文本内容
# 输出参数——满足查询条件的帧画面列表、检测到的目标、这一帧在原视频中的时间点、系统对检测到的目标语义相似度打分
# 需要包装成一个类吗？？
# 帧的时间点是不是应该在另一个方法中提供？？？
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 检验模型是否在GPU上运行
def verify_gpu_operation(model_name ,model, device):
    """验证模型是否在GPU上运行"""
    # 检查模型参数所在的设备
    model_device = next(model.parameters()).device
    if 'cuda' in str(model_device):
        print(f"✅ 确认: {model_name}模型正在GPU上运行")
    else:
        print(f"❌ 警告: {model_name}模型在CPU上运行")
# 返回检测到的物体对应的裁剪区域
def get_crop_regions(yolo_result: list, frame: Image):
    # (1) 获取检测到的目标的位置信息和类别信息
    boxes = yolo_result[0].boxes.xyxy.tolist()  # 检测框坐标
    classes = yolo_result[0].boxes.cls.tolist()  # 类别ID
    # 一个物体都没检测到的情况
    if boxes == []:
        print("该帧图像中没有检测到物体")
        return None
    # (2) 提取每个物体区域，并用 CLIP 编码
    image_pil = Image.fromarray(frame)
    crop_regions = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = image_pil.crop((x1, y1, x2, y2))
        crop_regions.append(crop) # 裁剪后的区域
    return crop_regions


# 返回置信度高出某一定值的视频帧
def search_video_frame(yolo_model_path, clip_model_path, video_frames, text, targetImage=None, yolo_conf=0.4, verbose=False):
    target_image = Image.open(targetImage) if targetImage else None
    # (1) 加载模型
    try:
        clip_model = ChineseCLIPModel.from_pretrained(clip_model_path).to(DEVICE)
        clip_processor = ChineseCLIPProcessor.from_pretrained(clip_model_path)
        verify_gpu_operation("CLIP", clip_model, DEVICE) # 检验CLIP是不是部署到了GPU上
    except:
        print("CLIP模型加载失败")
        return None
    try:
        yolo_model = YOLO(yolo_model_path).to(DEVICE)
        verify_gpu_operation("YOLO", yolo_model, DEVICE) # 检验YOLO是不是部署到了GPU上
    except:
        print("YOLO模型加载失败")
        return None
    # (2) text2image，传入文本和待检测图像，返回图文对的得分
    def text2image(text: str, objects: list):
        inputs = clip_processor(
            text=[text], 
            images=objects, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # 相似度矩阵，但是只有一个text
            # 这里找的是最大的，但是目标也有可能是次大的，所以还是需要优化一下
            # 进行softmax的缺点是如果只检测到了较少的目标，那么softmax的结果波动性非常大
            # 所以需要对softmax的处理方式改进一下
            logits_per_image_softmax = logits_per_image.softmax(dim=0)
            # max_confidence = (torch.max(logits_per_image_softmax, dim=0)).values.item()
        return logits_per_image_softmax

    # (3) image2image，传入目标图像和待检测图像，返回图像对的得分
    def image2image(image, objects):
        images = objects + [image]
        inputs = clip_processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs.to(DEVICE))
        # 图像之间不需要归一化
        similarity_scores = []
        target_image_feature = image_features[-1].unsqueeze(0)
        for image_feature in image_features[:-1]:
            similarity_scores.append(cosine_similarity(target_image_feature, image_feature.unsqueeze(0)))
        return similarity_scores

    def objectWeight(totalObjects: int):
        return 1-torch.exp(torch.tensor(-(totalObjects-1), dtype=torch.float32))

    def getWeightedResult(T2Iscores, I2Iscores, total_objects, alpha=0.4): # alpha是T2I的基础权重, 1-T2I的值就是I2I的权重
        weighted_scores = []
        object_weight = objectWeight(total_objects).item()
        for t2i_score, i2i_score in zip(T2Iscores, I2Iscores):
            # weighted_score = alpha * t2i_score + beta * i2i_score
            # weighted_scores.append(weighted_score)
            # 保存超过设置的检索阈值的视频图像帧
            # 检测到1个目标，直接以image2image的得分为准
            t2i_score, i2i_score = t2i_score.item(), i2i_score.item()
            if total_objects == 1:
                weighted_score = i2i_score
            # 检测到1个以上的目标
            elif total_objects <= 5:
                weighted_score = t2i_score * alpha * object_weight + i2i_score * (1 - alpha * object_weight)
            else:
                weighted_score = t2i_score * alpha + i2i_score * (1 - alpha) # 避免冗余计算
            weighted_scores.append(weighted_score)
        return weighted_scores
    
    def save(crop_region, max_score, T2I_score, I2I_score, frame_index, total_objects, result):
        crop_region = crop_regions[index]
        # 命名规则：最终得分-T2T得分-I2I得分-crop.jpeg
        save_path = f"D:\\ultralytics\\YOLO+CLIP\\YOLO+CLIP_output_T2I_I2I\\{frame_index}-{max_score:.4f}-{T2I_score:.4f}-{I2I_score:.4f}-crop.jpg"
        region_np = np.array(crop_region) # region是PIL Image格式，而cv2.imwrite()函数只能保存numpy数组
        annotated_frame = result[0].plot() # 这里会返回BGR
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR) # 转换为RGB
        # 命名规则：帧的索引-检测到的目标个数.jpg
        cv2.imwrite(f"D:\\ultralytics\\YOLO+CLIP\\YOLO+CLIP_output_T2I_I2I\\{frame_index}-{total_objects}.jpg", annotated_frame)        
        cv2.imwrite(save_path, cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR)) # 保存裁剪区域

    for index, frame in enumerate(video_frames):
        frame_index = index * 6 # 每6帧处理一次
        result = yolo_model.predict(
            frame,
            verbose=False,
            conf=yolo_conf,
            # batch=8 # 当前这个参数无效
        ) # 逐帧进行目标检测

        # 获取YOLO检测到的目标个数
        total_objects = len(result[0].boxes)
        crop_regions = get_crop_regions(result, frame) # 获取检测后每一帧中裁剪的目标子图
        # if frame_index == 50:
        #     # 使用PIL展示裁切区域
        #     for region in crop_regions:
        #         region.show()
        if crop_regions == None:
            continue
        T2Iscores = text2image(text, crop_regions)
        I2Iscores = image2image(target_image, crop_regions)
        if len(T2Iscores) == len(crop_regions) and len(I2Iscores) == len(crop_regions):
            weighted_scores = torch.tensor(getWeightedResult(T2Iscores, I2Iscores, total_objects))
            max_score = torch.max(weighted_scores).item()
            index = torch.argmax(weighted_scores).item()
            if (total_objects == 1 and max_score >= 0.84) or (total_objects == 2 and max_score >= 0.81536) or (total_objects >= 3 and max_score >= 0.827):
                save(crop_regions[index], max_score, T2Iscores[index].item(), I2Iscores[index].item(), frame_index, total_objects, result)
                if verbose:
                    # 打印日志
                    print(f"在第{frame_index}帧图像中检测到了目标，系统打分为{max_score:.4f}，其中T2I得分为{T2Iscores[index].item():.4f}，I2I得分为{I2Iscores[index].item():.4f}")
                    print(f"所有目标的分数分别为{weighted_scores}")
        else:
            print("分数计算有误，T2I和I2I的分数个数与检测到的目标个数不匹配")
            return None

