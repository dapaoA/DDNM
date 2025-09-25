#!/usr/bin/env python3
"""
简单的FID修复 - 只使用orig_*.png
"""

import shutil
from pathlib import Path
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch

def fix_fid_calculation():
    """修复FID计算"""
    
    print("修复FID计算 - 只使用orig_*.png作为参考图像")
    print("=" * 60)
    
    # 设置路径
    task_dir = Path("/workspace/DDNM/exp/image_samples/ffhq_inpainting_rand_T1000")
    apy_dir = task_dir / "Apy"
    
    # 创建临时目录
    temp_gen_dir = Path("/tmp/fixed_fid_gen")
    temp_ref_dir = Path("/tmp/fixed_fid_ref")
    
    # 清理并创建临时目录
    if temp_gen_dir.exists():
        shutil.rmtree(temp_gen_dir)
    if temp_ref_dir.exists():
        shutil.rmtree(temp_ref_dir)
    
    temp_gen_dir.mkdir()
    temp_ref_dir.mkdir()
    
    # 获取生成图像
    gen_images = list(task_dir.glob("*.png"))
    gen_images = [img for img in gen_images if img.is_file() and not img.name.startswith('.')]
    gen_images.sort(key=lambda x: int(x.stem.split('_')[0]) if x.stem.split('_')[0] != '-1' else -1)
    
    # 获取参考图像 - 只使用orig_*.png
    ref_images = list(apy_dir.glob("orig_*.png"))
    ref_images.sort(key=lambda x: int(x.stem.split('_')[1]))
    
    print(f"生成图像: {len(gen_images)} 个")
    print(f"参考图像: {len(ref_images)} 个 (只使用orig_*.png)")
    
    # 复制匹配的图像对
    matched_pairs = 0
    for i, gen_img in enumerate(gen_images):
        stem = gen_img.stem
        if '_' in stem:
            num_part = stem.split('_')[0]
            gen_index = int(num_part)
            ref_index = gen_index + 1
            
            # 查找对应的参考图像
            ref_img_path = apy_dir / f"orig_{ref_index}.png"
            if ref_img_path.exists():
                # 复制到临时目录
                shutil.copy2(gen_img, temp_gen_dir / f"{matched_pairs:06d}.png")
                shutil.copy2(ref_img_path, temp_ref_dir / f"{matched_pairs:06d}.png")
                matched_pairs += 1
    
    print(f"成功匹配 {matched_pairs} 对图像")
    
    # 检查图像数量
    gen_count = len(list(temp_gen_dir.glob("*.png")))
    ref_count = len(list(temp_ref_dir.glob("*.png")))
    
    print(f"临时目录图像数量: 生成={gen_count}, 参考={ref_count}")
    
    # 计算FID
    print("\n计算FID...")
    try:
        fid_value = calculate_fid_given_paths(
            [str(temp_gen_dir), str(temp_ref_dir)],
            batch_size=50,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dims=2048
        )
        print(f"修复后的FID值: {fid_value:.4f}")
    except Exception as e:
        print(f"FID计算错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理临时目录
    shutil.rmtree(temp_gen_dir)
    shutil.rmtree(temp_ref_dir)
    
    print("\n修复完成!")

if __name__ == "__main__":
    fix_fid_calculation()
