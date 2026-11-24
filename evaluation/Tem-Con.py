import os
import ssl
import urllib.request
import re

# 禁用 SSL 验证
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import clip
import numpy as np
from PIL import Image
import os
import glob

# 强制使用CPU，避免CUDA兼容性问题
device = "cpu"
print(f"使用设备: {device}")

# 加载CLIP模型和预处理函数
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_frame_features(frame_path):
    """
    提取单帧图像的特征。
    
    参数:
        frame_path (str): 图像帧的文件路径。
    
    返回:
        torch.Tensor: 归一化的特征向量。
    """
    try:
        # 加载图像
        image = Image.open(frame_path)
        # 预处理图像
        image = preprocess(image).unsqueeze(0).to(device)
        # 提取特征
        with torch.no_grad():
            features = model.encode_image(image)
        # 归一化特征
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze()
    except Exception as e:
        print(f"处理图像 {frame_path} 时出错: {e}")
        return None

def temporal_consistency(frame_paths):
    """
    计算视频帧之间的时间一致性 (Tem-Con)。
    
    参数:
        frame_paths (list): 包含所有帧路径的列表。
    
    返回:
        float: 平均时间一致性得分。
    """
    # 提取所有帧的特征
    frame_features = []
    valid_paths = []
    
    for path in frame_paths:
        features = extract_frame_features(path)
        if features is not None:
            frame_features.append(features.cpu().numpy())  # 将特征移动到CPU并转换为NumPy数组
            valid_paths.append(path)
    
    if len(frame_features) < 2:
        print("错误: 需要至少2个有效的帧来计算时间一致性")
        return 0.0
    
    frame_features = np.array(frame_features)
    
    # 计算连续帧之间的余弦相似度
    num_frames = len(frame_features)
    consistency_scores = []
    
    for i in range(num_frames - 1):
        cos_similarity = np.dot(frame_features[i], frame_features[i+1])
        consistency_scores.append(cos_similarity)
    
    # 计算平均一致性得分
    avg_consistency = np.mean(consistency_scores)
    return avg_consistency

def extract_number(filename):
    """
    从文件名中提取数字，支持多种格式：
    - 0.png, 1.jpg (纯数字)
    - frame_001.png, frame_1.jpg (frame_前缀)
    - img_001.png, image_1.jpg (其他前缀)
    """
    # 移除文件扩展名
    name_without_ext = os.path.splitext(filename)[0]
    
    # 尝试匹配文件名末尾的数字
    match = re.search(r'(\d+)$', name_without_ext)
    if match:
        return int(match.group(1))
    
    # 如果上述方法失败，尝试匹配文件名中的任何数字
    matches = re.findall(r'\d+', name_without_ext)
    if matches:
        return int(matches[-1])  # 取最后一个数字
    
    # 如果都没有数字，返回0（保持原顺序）
    return 0

def get_frame_paths(folder_path, extensions=("png", "jpg", "jpeg")):
    """
    从文件夹中获取所有帧图像的路径，并按帧号排序。
    支持多种命名格式：0.png, 1.jpg, frame_001.png, frame_1.jpg 等。
    
    参数:
        folder_path (str): 包含帧图像的文件夹路径。
        extensions (tuple): 支持的图像文件扩展名。
    
    返回:
        list: 排序后的帧路径列表。
    """
    frame_paths = []
    for ext in extensions:
        # 匹配所有图像文件，不限制前缀
        pattern = os.path.join(folder_path, f"*.{ext}")
        frame_paths.extend(glob.glob(pattern))
        # 同时匹配大写扩展名
        pattern_upper = os.path.join(folder_path, f"*.{ext.upper()}")
        frame_paths.extend(glob.glob(pattern_upper))
    
    # 去重并排序
    frame_paths = list(set(frame_paths))
    frame_paths.sort(key=lambda x: extract_number(os.path.basename(x)))
    
    return frame_paths

# 示例用法
if __name__ == "__main__":
    folder_path = "data/tokenflow-dancer"
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 {folder_path} 不存在")
        exit(1)
    
    frame_paths = get_frame_paths(folder_path)

    if not frame_paths:
        print("未找到任何帧图像，请检查文件夹路径和文件名格式。")
        print(f"检查文件夹内容: {os.listdir(folder_path)}")
    else:
        print(f"找到 {len(frame_paths)} 帧图像")
        print("前5个文件:", frame_paths[:5])  # 显示前5个文件检查排序是否正确
        
        # 显示处理进度
        print("开始计算时间一致性...")
        tem_con_score = temporal_consistency(frame_paths)
        print(f"{folder_path}: {tem_con_score:.4f}")