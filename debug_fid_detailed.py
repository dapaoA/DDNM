#!/usr/bin/env python3
"""
详细调试FID计算问题
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3
import shutil
from pathlib import Path

def debug_fid_detailed():
    """详细调试FID计算"""
    
    print("详细调试FID计算...")
    print("=" * 60)
    
    # 设置路径
    task_dir = Path("/workspace/DDNM/exp/image_samples/ffhq_inpainting_rand_T1000")
    apy_dir = task_dir / "Apy"
    
    # 创建临时目录
    temp_gen_dir = Path("/tmp/fid_detailed_gen")
    temp_ref_dir = Path("/tmp/fid_detailed_ref")
    
    # 清理并创建临时目录
    if temp_gen_dir.exists():
        shutil.rmtree(temp_gen_dir)
    if temp_ref_dir.exists():
        shutil.rmtree(temp_ref_dir)
    
    temp_gen_dir.mkdir()
    temp_ref_dir.mkdir()
    
    # 复制所有图像
    gen_images = list(task_dir.glob("*.png"))
    gen_images = [img for img in gen_images if img.is_file() and not img.name.startswith('.')]
    gen_images.sort(key=lambda x: int(x.stem.split('_')[0]) if x.stem.split('_')[0] != '-1' else -1)
    
    ref_images = list(apy_dir.glob("orig_*.png"))
    ref_images.sort(key=lambda x: int(x.stem.split('_')[1]))
    
    print(f"复制 {len(gen_images)} 个生成图像")
    print(f"复制 {len(ref_images)} 个参考图像")
    
    # 复制生成图像
    for i, img_path in enumerate(gen_images):
        shutil.copy2(img_path, temp_gen_dir / f"{i:06d}.png")
    
    # 复制参考图像
    for i, img_path in enumerate(ref_images):
        shutil.copy2(img_path, temp_ref_dir / f"{i:06d}.png")
    
    # 检查图像统计信息
    print("\n检查图像统计信息...")
    
    def get_image_stats(image_dir):
        """获取图像目录的统计信息"""
        images = list(image_dir.glob("*.png"))
        if not images:
            return None
        
        # 随机选择10个图像计算统计信息
        import random
        sample_images = random.sample(images, min(10, len(images)))
        
        all_pixels = []
        for img_path in sample_images:
            img = Image.open(img_path)
            img_array = np.array(img)
            all_pixels.extend(img_array.flatten())
        
        all_pixels = np.array(all_pixels)
        return {
            'mean': all_pixels.mean(),
            'std': all_pixels.std(),
            'min': all_pixels.min(),
            'max': all_pixels.max()
        }
    
    gen_stats = get_image_stats(temp_gen_dir)
    ref_stats = get_image_stats(temp_ref_dir)
    
    print(f"生成图像统计: 均值={gen_stats['mean']:.2f}, 标准差={gen_stats['std']:.2f}")
    print(f"参考图像统计: 均值={ref_stats['mean']:.2f}, 标准差={ref_stats['std']:.2f}")
    
    # 检查图像分布差异
    print(f"\n图像分布差异:")
    print(f"  均值差异: {abs(gen_stats['mean'] - ref_stats['mean']):.2f}")
    print(f"  标准差差异: {abs(gen_stats['std'] - ref_stats['std']):.2f}")
    
    # 计算FID
    print(f"\n计算FID...")
    try:
        fid_value = calculate_fid_given_paths(
            [str(temp_gen_dir), str(temp_ref_dir)],
            batch_size=50,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dims=2048
        )
        print(f"FID值: {fid_value:.4f}")
    except Exception as e:
        print(f"FID计算错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 尝试使用不同的FID计算参数
    print(f"\n尝试不同的FID计算参数...")
    
    # 使用更小的batch size
    try:
        fid_value_small = calculate_fid_given_paths(
            [str(temp_gen_dir), str(temp_ref_dir)],
            batch_size=10,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dims=2048
        )
        print(f"FID值 (batch_size=10): {fid_value_small:.4f}")
    except Exception as e:
        print(f"FID计算错误 (batch_size=10): {e}")
    
    # 使用不同的dims
    try:
        fid_value_dims = calculate_fid_given_paths(
            [str(temp_gen_dir), str(temp_ref_dir)],
            batch_size=50,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dims=768
        )
        print(f"FID值 (dims=768): {fid_value_dims:.4f}")
    except Exception as e:
        print(f"FID计算错误 (dims=768): {e}")
    
    # 检查图像数量是否匹配
    print(f"\n检查图像数量:")
    print(f"  生成图像数量: {len(list(temp_gen_dir.glob('*.png')))}")
    print(f"  参考图像数量: {len(list(temp_ref_dir.glob('*.png')))}")
    
    # 清理临时目录
    shutil.rmtree(temp_gen_dir)
    shutil.rmtree(temp_ref_dir)
    
    print("\n详细调试完成!")

if __name__ == "__main__":
    debug_fid_detailed()
