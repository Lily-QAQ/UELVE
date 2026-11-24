import cv2
import numpy as np
from rembg import remove
from PIL import Image
import json
from tqdm import tqdm
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchmetrics
from torchmetrics.image import PeakSignalNoiseRatio

class VideoBackgroundPSNREvaluator:
    def __init__(self, device='auto'):
        """
        初始化视频背景PSNR评估器
        
        Args:
            device: 设备设置 (auto/cpu/cuda)
        """
        self.device = device
        # 初始化PSNR计算器
        self.psnr_metric = PeakSignalNoiseRatio(data_range=255.0)
        
    def extract_frames(self, video_path, frame_interval=5, max_frames=50):
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            frame_interval: 帧间隔
            max_frames: 最大帧数
            
        Returns:
            frames: 帧列表 (RGB格式)
            fps: 视频帧率
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        frame_count = 0
        
        print(f"视频总帧数: {total_frames}, 帧率: {fps:.2f}")
        
        with tqdm(total=min(total_frames//frame_interval, max_frames), 
                 desc="提取视频帧") as pbar:
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    pbar.update(1)
                
                frame_count += 1
                if frame_count >= total_frames:
                    break
        
        cap.release()
        return frames, fps
    
    def generate_background_mask(self, frame, frame_idx=0, save_masks=False, masks_dir='masks'):
        """
        使用rembg生成背景mask (背景=1, 前景=0)
        
        Args:
            frame: 输入帧 (RGB格式)
            frame_idx: 帧索引，用于保存文件名
            save_masks: 是否保存mask图片
            masks_dir: mask保存目录
            
        Returns:
            mask: 背景mask
        """
        # 将numpy数组转换为PIL Image
        pil_image = Image.fromarray(frame)
        
        # 使用rembg移除背景（返回RGBA图像，alpha通道为mask）
        result = remove(pil_image, post_process_mask=True)
        
        # 保存rembg原始结果（可选）
        if save_masks:
            os.makedirs(masks_dir, exist_ok=True)
            result.save(f'{masks_dir}/frame_{frame_idx:04d}_rembg.png')
        
        # 提取alpha通道并反转（rembg返回的是前景mask）
        if result.mode == 'RGBA':
            alpha = np.array(result.split()[-1])  # alpha通道
            # 反转：背景=1, 前景=0
            background_mask = (alpha == 0).astype(np.float32)
        else:
            # 如果没有alpha通道，创建全背景mask
            h, w = frame.shape[:2]
            background_mask = np.ones((h, w), dtype=np.float32)
        
        # 形态学操作优化mask
        kernel = np.ones((5, 5), np.uint8)
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
        
        # 保存处理后的mask
        if save_masks:
            # 保存二值mask
            mask_uint8 = (background_mask * 255).astype(np.uint8)
            cv2.imwrite(f'{masks_dir}/frame_{frame_idx:04d}_mask_binary.png', mask_uint8)
            
            # 保存可视化mask（红色表示背景，绿色表示前景）
            visualization = self.create_mask_visualization(frame, background_mask)
            cv2.imwrite(f'{masks_dir}/frame_{frame_idx:04d}_mask_visualization.png', 
                       cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            
            # 保存前景和背景分离的图像
            self.save_separated_images(frame, background_mask, frame_idx, masks_dir)
        
        return background_mask
    
    def create_mask_visualization(self, frame, mask):
        """
        创建mask可视化图像
        
        Args:
            frame: 原始帧
            mask: 背景mask
            
        Returns:
            visualization: 可视化图像
        """
        h, w = frame.shape[:2]
        visualization = frame.copy()
        
        # 创建彩色mask：背景用半透明红色，前景用半透明绿色
        red_mask = np.zeros((h, w, 3), dtype=np.uint8)
        red_mask[:, :, 0] = 255  # 红色通道
        green_mask = np.zeros((h, w, 3), dtype=np.uint8)
        green_mask[:, :, 1] = 255  # 绿色通道
        
        # 应用mask
        background_area = (mask == 1)
        foreground_area = (mask == 0)
        
        # 叠加颜色
        alpha = 0.3  # 透明度
        visualization[background_area] = cv2.addWeighted(
            visualization[background_area], 1 - alpha, 
            red_mask[background_area], alpha, 0
        )
        visualization[foreground_area] = cv2.addWeighted(
            visualization[foreground_area], 1 - alpha, 
            green_mask[foreground_area], alpha, 0
        )
        
        return visualization
    
    def save_separated_images(self, frame, mask, frame_idx, masks_dir):
        """
        保存前景和背景分离的图像
        
        Args:
            frame: 原始帧
            mask: 背景mask
            frame_idx: 帧索引
            masks_dir: 保存目录
        """
        # 分离前景和背景
        background = frame * mask[:, :, np.newaxis]
        foreground = frame * (1 - mask[:, :, np.newaxis])
        
        # 保存背景图像
        bg_uint8 = background.astype(np.uint8)
        cv2.imwrite(f'{masks_dir}/frame_{frame_idx:04d}_background.png', 
                   cv2.cvtColor(bg_uint8, cv2.COLOR_RGB2BGR))
        
        # 保存前景图像
        fg_uint8 = foreground.astype(np.uint8)
        cv2.imwrite(f'{masks_dir}/frame_{frame_idx:04d}_foreground.png', 
                   cv2.cvtColor(fg_uint8, cv2.COLOR_RGB2BGR))
    
    def calculate_masked_psnr_skimage(self, original_frame, edited_frame, mask):
        """
        使用skimage计算masked PSNR
        
        Args:
            original_frame: 原始帧
            edited_frame: 编辑后的帧
            mask: 背景mask
            
        Returns:
            psnr_value: PSNR值
        """
        # 调整编辑帧大小以匹配原始帧
        if original_frame.shape != edited_frame.shape:
            edited_frame = cv2.resize(edited_frame, 
                                    (original_frame.shape[1], original_frame.shape[0]))
        
        # 应用mask（只保留背景区域）
        original_background = original_frame * mask[:, :, np.newaxis]
        edited_background = edited_frame * mask[:, :, np.newaxis]
        
        # 检查mask区域是否有效
        if np.sum(mask) == 0:
            return 0  # 没有背景区域
        
        # 检查两幅图像是否完全相同
        if np.array_equal(original_background, edited_background):
            return 100  # 设置一个上限值，而不是inf
        
        try:
            psnr_value = psnr(original_background.astype(np.float32), 
                             edited_background.astype(np.float32), 
                             data_range=255)
            
            # 处理无穷大值
            if np.isinf(psnr_value):
                return 100
            return psnr_value
            
        except ValueError:
            # 如果mask全为0，返回0
            return 0
    
    def calculate_masked_psnr_torchmetrics(self, original_frame, edited_frame, mask):
        """
        使用torchmetrics计算masked PSNR
        
        Args:
            original_frame: 原始帧
            edited_frame: 编辑后的帧
            mask: 背景mask
            
        Returns:
            psnr_value: PSNR值
        """
        import torch
        
        # 转换为torch tensor
        original_tensor = torch.from_numpy(original_frame.astype(np.float32)).permute(2, 0, 1)
        edited_tensor = torch.from_numpy(edited_frame.astype(np.float32)).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask.astype(np.float32))
        
        # 应用mask
        original_masked = original_tensor * mask_tensor
        edited_masked = edited_tensor * mask_tensor
        
        # 检查是否所有值都为0
        if torch.sum(original_masked) == 0 and torch.sum(edited_masked) == 0:
            return 100.0
        
        # 添加batch维度
        original_masked = original_masked.unsqueeze(0)
        edited_masked = edited_masked.unsqueeze(0)
        
        # 计算PSNR
        psnr_value = self.psnr_metric(edited_masked, original_masked)
        
        return psnr_value.item() if not torch.isnan(psnr_value) and not torch.isinf(psnr_value) else 100.0
    
    def evaluate_video_background(self, original_video, edited_video, 
                                frame_interval=5, max_frames=50, 
                                psnr_method='skimage', save_masks=False):
        """
        评估视频背景保持效果
        
        Args:
            original_video: 原始视频路径
            edited_video: 编辑后视频路径
            frame_interval: 帧采样间隔
            max_frames: 最大处理帧数
            psnr_method: PSNR计算方法 ('skimage' 或 'torchmetrics')
            save_masks: 是否保存mask图片
            
        Returns:
            results: 评估结果字典
            masks: 生成的mask列表
        """
        print("正在提取视频帧...")
        
        # 提取帧
        original_frames, original_fps = self.extract_frames(original_video, 
                                                          frame_interval, max_frames)
        edited_frames, edited_fps = self.extract_frames(edited_video, 
                                                      frame_interval, max_frames)
        
        # 确保帧数一致
        min_frames = min(len(original_frames), len(edited_frames))
        original_frames = original_frames[:min_frames]
        edited_frames = edited_frames[:min_frames]
        
        print(f"开始处理 {min_frames} 帧...")
        
        # 创建mask保存目录
        if save_masks:
            masks_dir = 'mask_results'
            os.makedirs(masks_dir, exist_ok=True)
            print(f"Mask图片将保存到: {masks_dir}")
        
        # 计算每帧的masked PSNR
        psnr_values = []
        masks = []
        background_areas = []
        
        for i, (orig_frame, edit_frame) in enumerate(
            tqdm(zip(original_frames, edited_frames), 
                 total=min_frames, desc="计算Masked PSNR")):
            
            # 生成背景mask（并保存）
            mask = self.generate_background_mask(orig_frame, i, save_masks, 
                                               masks_dir if save_masks else 'masks')
            masks.append(mask)
            
            # 计算背景区域比例
            background_ratio = np.mean(mask)
            background_areas.append(background_ratio)
            
            # 计算masked PSNR
            if psnr_method == 'skimage':
                frame_psnr = self.calculate_masked_psnr_skimage(orig_frame, edit_frame, mask)
            else:
                frame_psnr = self.calculate_masked_psnr_torchmetrics(orig_frame, edit_frame, mask)
            
            psnr_values.append(frame_psnr)
        
        # 统计结果
        results = {
            'mean_psnr': float(np.mean(psnr_values)),
            'std_psnr': float(np.std(psnr_values)),
            'min_psnr': float(np.min(psnr_values)),
            'max_psnr': float(np.max(psnr_values)),
            'median_psnr': float(np.median(psnr_values)),
            'frame_psnr': [float(x) for x in psnr_values],
            'background_ratio_mean': float(np.mean(background_areas)),
            'total_frames_processed': min_frames,
            'original_video': original_video,
            'edited_video': edited_video,
            'frame_interval': frame_interval,
            'masks_saved': save_masks
        }
        
        if save_masks:
            results['masks_directory'] = masks_dir
        
        return results, masks
    
    def save_visualization(self, original_frame, edited_frame, mask, frame_idx, save_dir='results'):
        """
        保存可视化结果
        
        Args:
            original_frame: 原始帧
            edited_frame: 编辑后的帧
            mask: 背景mask
            frame_idx: 帧索引
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建对比图像
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始帧
        axes[0, 0].imshow(original_frame)
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        # 编辑后的帧
        axes[0, 1].imshow(edited_frame)
        axes[0, 1].set_title('Edited Frame')
        axes[0, 1].axis('off')
        
        # Mask
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title(f'Background Mask (ratio: {np.mean(mask):.2f})')
        axes[0, 2].axis('off')
        
        # 原始背景
        original_bg = original_frame * mask[:, :, np.newaxis]
        axes[1, 0].imshow(original_bg.astype(np.uint8))
        axes[1, 0].set_title('Original Background')
        axes[1, 0].axis('off')
        
        # 编辑后的背景
        edited_bg = edited_frame * mask[:, :, np.newaxis]
        axes[1, 1].imshow(edited_bg.astype(np.uint8))
        axes[1, 1].set_title('Edited Background')
        axes[1, 1].axis('off')
        
        # 差异图
        diff = np.abs(original_bg.astype(float) - edited_bg.astype(float))
        axes[1, 2].imshow(diff.astype(np.uint8))
        axes[1, 2].set_title('Background Difference')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/frame_{frame_idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    # 在这里直接设置参数，不再使用命令行参数
    params = {
        'original_video': 'data/dancer200f.mp4',
        'edited_video': 'data/tokenflow-dancer.mp4',
        'frame_interval': 1,
        'max_frames': 8,
        'output_file': 'background_psnr_results.json',
        'visualize': False,
        'psnr_method': 'skimage',  # 'skimage' 或 'torchmetrics'
        'save_masks': False
    }
    
    # 初始化评估器
    evaluator = VideoBackgroundPSNREvaluator()
    
    print("开始视频背景保持效果评估...")
    print(f"原始视频: {params['original_video']}")
    print(f"编辑视频: {params['edited_video']}")
    print(f"帧间隔: {params['frame_interval']}, 最大帧数: {params['max_frames']}")
    print(f"PSNR计算方法: {params['psnr_method']}")
    print(f"保存mask图片: {params['save_masks']}")
    
    # 执行评估
    results, masks = evaluator.evaluate_video_background(
        params['original_video'], 
        params['edited_video'], 
        frame_interval=params['frame_interval'],
        max_frames=params['max_frames'],
        psnr_method=params['psnr_method'],
        save_masks=params['save_masks']
    )
    
    # 保存结果
    # with open(params['output_file'], 'w', encoding='utf-8') as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n" + "="*50)
    print("视频背景保持效果评估结果")
    print("="*50)
    print(f"平均背景PSNR: {results['mean_psnr']:.2f} dB")
    print(f"PSNR标准差: {results['std_psnr']:.2f} dB")
    print(f"最小PSNR: {results['min_psnr']:.2f} dB")
    print(f"最大PSNR: {results['max_psnr']:.2f} dB")
    print(f"中位数PSNR: {results['median_psnr']:.2f} dB")
    print(f"平均背景区域比例: {results['background_ratio_mean']:.2%}")
    print(f"处理帧数: {results['total_frames_processed']}")
    if params['save_masks']:
        print(f"Mask图片保存到: {results.get('masks_directory', 'mask_results')}")
    #print(f"详细结果已保存到: {params['output_file']}")
    
    # 可视化结果
    if params['visualize']:
        print("生成可视化结果...")
        original_frames, _ = evaluator.extract_frames(params['original_video'], params['frame_interval'], params['max_frames'])
        edited_frames, _ = evaluator.extract_frames(params['edited_video'], params['frame_interval'], params['max_frames'])
        
        # 只可视化前5帧
        visualize_frames = min(5, len(original_frames))
        for i in tqdm(range(visualize_frames), desc="生成可视化"):
            evaluator.save_visualization(original_frames[i], edited_frames[i], masks[i], i)

if __name__ == "__main__":
    # 如果没有安装matplotlib，跳过导入
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("警告: matplotlib未安装，可视化功能不可用")
    
    main()