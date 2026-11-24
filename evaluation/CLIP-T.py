import torch
import clip
from PIL import Image
import os

# 强制使用CPU
device = "cpu"

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_features(model, preprocess, image, text, device):
    # 处理图像
    image_input = preprocess(image).unsqueeze(0).to(device)
    # 处理文本
    text_input = clip.tokenize([text]).to(device)
    
    # 提取特征
    with torch.no_grad():
        image_feature = model.encode_image(image_input)
        text_feature = model.encode_text(text_input)
    
    return image_feature, text_feature

def calculate_clip_t(image, text, model, preprocess, device):
    image_feature, text_feature = extract_features(model, preprocess, image, text, device)
    
    # 计算相似度
    similarity = torch.cosine_similarity(image_feature, text_feature)
    clip_t_score = similarity.item()
    
    return clip_t_score

def calculate_average_clip_t_score(frames_dir, text, device="cpu"):
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    clip_t_scores = []
    
    # 遍历frames_dir中的所有图片
    for filename in os.listdir(frames_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(frames_dir, filename)
            image = Image.open(image_path)
            
            clip_t_score = calculate_clip_t(image, text, model, preprocess, device)
            clip_t_scores.append(clip_t_score)
    
    if not clip_t_scores:
        return 0
    
    average_score = sum(clip_t_scores) / len(clip_t_scores)
    return average_score

# 使用示例
if __name__ == "__main__":
    frames_directory = "data/tokenflow-dancer"  # 修改为您的帧目录
    text_description = "a young woman in the light of sunset"  # 修改为您的文本描述
    
    average_clip_t_score = calculate_average_clip_t_score(frames_directory, text_description, device="cpu")
    print(f"{frames_directory}--- {average_clip_t_score:.4f}")