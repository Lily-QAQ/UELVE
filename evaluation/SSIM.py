import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(original_frames_dir, edited_frames_dir):
    # 获取原始视频和编辑后视频的帧文件列表
    original_frames = sorted([os.path.join(original_frames_dir, f) for f in os.listdir(original_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    edited_frames = sorted([os.path.join(edited_frames_dir, f) for f in os.listdir(edited_frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # 确保帧数一致
    assert len(original_frames) == len(edited_frames), "原始视频和编辑后视频的帧数不一致"

    total_ssim = 0
    for orig_frame_path, edit_frame_path in zip(original_frames, edited_frames):
        # 读取帧图像
        orig_frame = cv2.imread(orig_frame_path)
        edit_frame = cv2.imread(edit_frame_path)

        # 转换为灰度图像（SSIM 计算需要灰度图像）
        orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        edit_gray = cv2.cvtColor(edit_frame, cv2.COLOR_BGR2GRAY)

        # 计算 SSIM
        similarity, _ = ssim(orig_gray, edit_gray, full=True)
        total_ssim += similarity

    # 计算平均 SSIM
    avg_ssim = total_ssim / len(original_frames)
    return avg_ssim

# 示例用法
original_frames_dir = "data/dancer200f"
edited_frames_dir = "data/tokenflow-dancer"

ssim_value = calculate_ssim(original_frames_dir, edited_frames_dir)
print(f"{edited_frames_dir}---SSIM 评估指标: {ssim_value}")