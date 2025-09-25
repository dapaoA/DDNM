#!/usr/bin/env python3
"""
调试FID计算问题
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pytorch_fid.fid_score import calculate_fid_given_paths
import shutil
from pathlib import Path

def debug_fid_calculation():
    """调试FID计算"""
    
    # 设置路径
    task_dir = Path("/workspace/DDNM/exp/image_samples/ffhq_inpainting_rand_T1000")
    apy_dir = task_dir / "Apy"
    
    # 创建临时目录
    temp_gen_dir = Path("/tmp/fid_debug_gen")
    temp_ref_dir = Path("/tmp/fid_debug_ref")
    
    # 清理并创建临时目录
    if temp_gen_dir.exists():
        shutil.rmtree(temp_gen_dir)
    if temp_ref_dir.exists():
        shutil.rmtree(temp_ref_dir)
    
    temp_gen_dir.mkdir()
    temp_ref_dir.mkdir()
    
    print("调试FID计算...")
    print("=" * 50)
    
    # 复制生成的图像
    gen_images = list(task_dir.glob("*.png"))
    gen_images = [img for img in gen_images if img.is_file() and not img.name.startswith('.')]
    
    print(f"找到 {len(gen_images)} 个生成图像")
    
    # 复制前10个图像用于调试
    for i, img_path in enumerate(gen_images[:10]):
        shutil.copy2(img_path, temp_gen_dir / f"{i:06d}.png")
    
    # 复制参考图像
    ref_images = list(apy_dir.glob("orig_*.png"))
    ref_images.sort(key=lambda x: int(x.stem.split('_')[1]))
    
    print(f"找到 {len(ref_images)} 个参考图像")
    
    # 复制前10个图像用于调试
    for i, img_path in enumerate(ref_images[:10]):
        shutil.copy2(img_path, temp_ref_dir / f"{i:06d}.png")
    
    print(f"临时目录:")
    print(f"  生成图像: {temp_gen_dir}")
    print(f"  参考图像: {temp_ref_dir}")
    
    # 检查图像质量
    print("\n检查图像质量...")
    
    # 检查生成图像
    gen_img = Image.open(temp_gen_dir / "000000.png")
    print(f"生成图像: {gen_img.size}, 模式: {gen_img.mode}")
    gen_array = np.array(gen_img)
    print(f"  像素值范围: {gen_array.min()} - {gen_array.max()}")
    print(f"  均值: {gen_array.mean():.2f}, 标准差: {gen_array.std():.2f}")
    
    # 检查参考图像
    ref_img = Image.open(temp_ref_dir / "000000.png")
    print(f"参考图像: {ref_img.size}, 模式: {ref_img.mode}")
    ref_array = np.array(ref_img)
    print(f"  像素值范围: {ref_array.min()} - {ref_array.max()}")
    print(f"  均值: {ref_array.mean():.2f}, 标准差: {ref_array.std():.2f}")
    
    # 计算FID
    print("\n计算FID...")
    try:
        fid_value = calculate_fid_given_paths(
            [str(temp_gen_dir), str(temp_ref_dir)],
            batch_size=10,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dims=2048
        )
        print(f"FID值: {fid_value:.4f}")
    except Exception as e:
        print(f"FID计算错误: {e}")
    
    # 检查图像是否匹配
    print("\n检查图像匹配...")
    
    # 使用正确的匹配关系
    gen_img_path = task_dir / "-1_0.png"  # 第一个生成图像
    ref_img_path = apy_dir / "orig_0.png"  # 对应的参考图像
    
    if gen_img_path.exists() and ref_img_path.exists():
        gen_img = Image.open(gen_img_path)
        ref_img = Image.open(ref_img_path)
        
        print(f"检查匹配的图像对:")
        print(f"  生成图像: {gen_img_path.name}")
        print(f"  参考图像: {ref_img_path.name}")
        
        # 转换为tensor
        transform = transforms.ToTensor()
        gen_tensor = transform(gen_img)
        ref_tensor = transform(ref_img)
        
        # 计算差异
        diff = torch.abs(gen_tensor - ref_tensor)
        print(f"  平均绝对差异: {diff.mean():.4f}")
        print(f"  最大差异: {diff.max():.4f}")
        print(f"  差异>0.1的像素比例: {(diff > 0.1).float().mean():.4f}")
        
        # 计算简单的MSE和PSNR
        mse = torch.mean((gen_tensor - ref_tensor) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        print(f"  MSE: {mse:.6f}")
        print(f"  PSNR: {psnr:.3f}")
    
    # 清理临时目录
    shutil.rmtree(temp_gen_dir)
    shutil.rmtree(temp_ref_dir)
    
    print("\n调试完成!")

if __name__ == "__main__":
    debug_fid_calculation()
