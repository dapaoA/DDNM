#!/usr/bin/env python3
"""
分析FID高值的原因
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from pytorch_fid.fid_score import calculate_fid_given_paths
import shutil
from pathlib import Path

def analyze_fid_issue():
    """分析FID高值的原因"""
    
    print("分析FID高值的原因...")
    print("=" * 60)
    
    # 设置路径
    task_dir = Path("/workspace/DDNM/exp/image_samples/ffhq_inpainting_rand_T1000")
    apy_dir = task_dir / "Apy"
    
    # 创建临时目录
    temp_gen_dir = Path("/tmp/analyze_gen")
    temp_ref_dir = Path("/tmp/analyze_ref")
    
    # 清理并创建临时目录
    if temp_gen_dir.exists():
        shutil.rmtree(temp_gen_dir)
    if temp_ref_dir.exists():
        shutil.rmtree(temp_ref_dir)
    
    temp_gen_dir.mkdir()
    temp_ref_dir.mkdir()
    
    # 获取图像
    gen_images = list(task_dir.glob("*.png"))
    gen_images = [img for img in gen_images if img.is_file() and not img.name.startswith('.')]
    gen_images.sort(key=lambda x: int(x.stem.split('_')[0]) if x.stem.split('_')[0] != '-1' else -1)
    
    ref_images = list(apy_dir.glob("orig_*.png"))
    ref_images.sort(key=lambda x: int(x.stem.split('_')[1]))
    
    # 复制前20个图像对进行分析
    matched_pairs = 0
    for i, gen_img in enumerate(gen_images[:20]):
        stem = gen_img.stem
        if '_' in stem:
            num_part = stem.split('_')[0]
            gen_index = int(num_part)
            ref_index = gen_index + 1
            
            ref_img_path = apy_dir / f"orig_{ref_index}.png"
            if ref_img_path.exists():
                shutil.copy2(gen_img, temp_gen_dir / f"{matched_pairs:06d}.png")
                shutil.copy2(ref_img_path, temp_ref_dir / f"{matched_pairs:06d}.png")
                matched_pairs += 1
    
    print(f"分析 {matched_pairs} 对图像")
    
    # 检查图像统计信息
    print("\n检查图像统计信息...")
    
    def get_image_stats(image_dir):
        """获取图像目录的统计信息"""
        images = list(image_dir.glob("*.png"))
        all_pixels = []
        for img_path in images:
            img = Image.open(img_path)
            img_array = np.array(img)
            all_pixels.extend(img_array.flatten())
        
        all_pixels = np.array(all_pixels)
        return {
            'mean': all_pixels.mean(),
            'std': all_pixels.std(),
            'min': all_pixels.min(),
            'max': all_pixels.max(),
            'count': len(all_pixels)
        }
    
    gen_stats = get_image_stats(temp_gen_dir)
    ref_stats = get_image_stats(temp_ref_dir)
    
    print(f"生成图像统计:")
    print(f"  像素数量: {gen_stats['count']:,}")
    print(f"  均值: {gen_stats['mean']:.2f}")
    print(f"  标准差: {gen_stats['std']:.2f}")
    print(f"  范围: {gen_stats['min']} - {gen_stats['max']}")
    
    print(f"\n参考图像统计:")
    print(f"  像素数量: {ref_stats['count']:,}")
    print(f"  均值: {ref_stats['mean']:.2f}")
    print(f"  标准差: {ref_stats['std']:.2f}")
    print(f"  范围: {ref_stats['min']} - {ref_stats['max']}")
    
    # 计算差异
    print(f"\n图像分布差异:")
    print(f"  均值差异: {abs(gen_stats['mean'] - ref_stats['mean']):.2f}")
    print(f"  标准差差异: {abs(gen_stats['std'] - ref_stats['std']):.2f}")
    
    # 计算FID
    print(f"\n计算FID...")
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
    
    # 检查单个图像对的差异
    print(f"\n检查单个图像对的差异...")
    
    gen_img = Image.open(temp_gen_dir / "000000.png")
    ref_img = Image.open(temp_ref_dir / "000000.png")
    
    # 转换为tensor
    transform = transforms.ToTensor()
    gen_tensor = transform(gen_img)
    ref_tensor = transform(ref_img)
    
    # 计算差异
    diff = torch.abs(gen_tensor - ref_tensor)
    print(f"单个图像对分析:")
    print(f"  平均绝对差异: {diff.mean():.4f}")
    print(f"  最大差异: {diff.max():.4f}")
    print(f"  差异>0.1的像素比例: {(diff > 0.1).float().mean():.4f}")
    
    # 计算PSNR
    mse = torch.mean((gen_tensor - ref_tensor) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    print(f"  PSNR: {psnr:.3f}")
    
    # 分析FID高的可能原因
    print(f"\nFID高的可能原因分析:")
    
    # 1. 检查图像分布
    mean_diff = abs(gen_stats['mean'] - ref_stats['mean'])
    std_diff = abs(gen_stats['std'] - ref_stats['std'])
    
    if mean_diff > 10:
        print(f"  1. 图像均值差异较大 ({mean_diff:.2f}) - 可能影响FID")
    if std_diff > 5:
        print(f"  2. 图像标准差差异较大 ({std_diff:.2f}) - 可能影响FID")
    
    # 2. 检查图像数量
    if matched_pairs < 50:
        print(f"  3. 图像数量较少 ({matched_pairs}) - 可能影响FID稳定性")
    
    # 3. 检查图像质量
    if diff.mean() > 0.05:
        print(f"  4. 图像差异较大 ({diff.mean():.4f}) - 可能影响FID")
    
    # 清理临时目录
    shutil.rmtree(temp_gen_dir)
    shutil.rmtree(temp_ref_dir)
    
    print("\n分析完成!")

if __name__ == "__main__":
    analyze_fid_issue()
