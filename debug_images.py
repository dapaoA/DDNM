#!/usr/bin/env python3
"""
调试图像质量和数据范围
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def analyze_images():
    """分析图像的基本属性"""
    
    # 检查生成的图像
    gen_img_path = Path("/workspace/DDNM/exp/image_samples/ffhq_inpainting_rand_T1000/0_0.png")
    ref_img_path = Path("/workspace/DDNM/exp/datasets/ffhq/00000.png")
    
    print("分析图像属性...")
    print("=" * 50)
    
    # 加载图像
    gen_img = Image.open(gen_img_path).convert('RGB')
    ref_img = Image.open(ref_img_path).convert('RGB')
    
    print(f"生成图像: {gen_img_path}")
    print(f"  - 尺寸: {gen_img.size}")
    print(f"  - 模式: {gen_img.mode}")
    print(f"  - 像素值范围: {min(gen_img.getdata())} - {max(gen_img.getdata())}")
    
    print(f"\n参考图像: {ref_img_path}")
    print(f"  - 尺寸: {ref_img.size}")
    print(f"  - 模式: {ref_img.mode}")
    print(f"  - 像素值范围: {min(ref_img.getdata())} - {max(ref_img.getdata())}")
    
    # 转换为numpy数组
    gen_array = np.array(gen_img)
    ref_array = np.array(ref_img)
    
    print(f"\n生成图像数组:")
    print(f"  - 形状: {gen_array.shape}")
    print(f"  - 数据类型: {gen_array.dtype}")
    print(f"  - 数值范围: {gen_array.min()} - {gen_array.max()}")
    print(f"  - 均值: {gen_array.mean():.2f}")
    print(f"  - 标准差: {gen_array.std():.2f}")
    
    print(f"\n参考图像数组:")
    print(f"  - 形状: {ref_array.shape}")
    print(f"  - 数据类型: {ref_array.dtype}")
    print(f"  - 数值范围: {ref_array.min()} - {ref_array.max()}")
    print(f"  - 均值: {ref_array.mean():.2f}")
    print(f"  - 标准差: {ref_array.std():.2f}")
    
    # 转换为tensor
    transform = transforms.ToTensor()
    gen_tensor = transform(gen_img)
    ref_tensor = transform(ref_img)
    
    print(f"\n转换为tensor后:")
    print(f"  - 生成图像tensor范围: {gen_tensor.min():.4f} - {gen_tensor.max():.4f}")
    print(f"  - 参考图像tensor范围: {ref_tensor.min():.4f} - {ref_tensor.max():.4f}")
    
    # 计算简单的MSE
    mse = torch.mean((gen_tensor - ref_tensor) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    print(f"\n简单PSNR计算:")
    print(f"  - MSE: {mse:.6f}")
    print(f"  - PSNR: {psnr:.3f}")
    
    # 检查图像是否相似
    diff = torch.abs(gen_tensor - ref_tensor)
    print(f"\n差异分析:")
    print(f"  - 平均绝对差异: {diff.mean():.4f}")
    print(f"  - 最大差异: {diff.max():.4f}")
    print(f"  - 差异>0.1的像素比例: {(diff > 0.1).float().mean():.4f}")

if __name__ == "__main__":
    from pathlib import Path
    analyze_images()
