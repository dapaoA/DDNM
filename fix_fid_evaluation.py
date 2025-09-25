#!/usr/bin/env python3
"""
修复FID计算 - 只使用orig_*.png作为参考图像
"""

import json
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_all_tasks import TaskEvaluator

def create_fixed_fid_evaluator():
    """创建修复FID计算的评估器"""
    evaluator = TaskEvaluator(Path("/workspace/DDNM/exp"))
    
    # 重新定义获取数据集信息的函数
    def fixed_get_dataset_info(task_name):
        """修复的数据集信息获取"""
        if task_name.startswith('ffhq'):
            # 对于inpainting任务，使用Apy子目录中的orig_*.png作为参考
            if 'inpainting' in task_name:
                task_dir = evaluator.image_samples_dir / task_name
                apy_dir = task_dir / 'Apy'
                if apy_dir.exists():
                    return apy_dir, 'ffhq_inpainting_orig'
            # 其他任务使用原始数据集
            dataset_dir = evaluator.datasets_dir / 'ffhq'
            return dataset_dir, 'ffhq'
        elif task_name.startswith('imagenet'):
            # 对于inpainting任务，使用Apy子目录中的orig_*.png作为参考
            if 'inpainting' in task_name:
                task_dir = evaluator.image_samples_dir / task_name
                apy_dir = task_dir / 'Apy'
                if apy_dir.exists():
                    return apy_dir, 'imagenet_inpainting_orig'
            # 其他任务使用原始数据集
            dataset_dir = evaluator.datasets_dir / 'imagenet'
            return dataset_dir, 'imagenet'
        return None, None
    
    # 重新定义图像匹配函数
    def fixed_match_image_pairs(generated_images, reference_images):
        """修复的图像匹配逻辑"""
        matched_pairs = []
        
        # 创建参考图像的索引 - 只使用orig_*.png
        ref_dict = {}
        for ref_img in reference_images:
            stem = ref_img.stem
            # 只处理orig_*.png格式，忽略Apy_*.png
            if stem.startswith('orig_'):
                try:
                    num = int(stem.split('_')[1])
                    ref_dict[num] = ref_img
                except (ValueError, IndexError):
                    continue
            # 处理00000.png格式
            elif stem.isdigit():
                try:
                    num = int(stem)
                    ref_dict[num] = ref_img
                except ValueError:
                    continue
        
        print(f"找到 {len(ref_dict)} 个有效的参考图像 (只计算orig_*.png)")
        
        # 匹配生成的图像
        for gen_img in generated_images:
            stem = gen_img.stem
            try:
                # 处理格式如 "-1_0", "0_0", "1_0" 等
                if '_' in stem:
                    num_part = stem.split('_')[0]
                    gen_index = int(num_part)
                    
                    # 关键修正：-1对应orig_0, 0对应orig_1, 1对应orig_2, ...
                    ref_index = gen_index + 1
                    
                    if ref_index in ref_dict:
                        matched_pairs.append((gen_img, ref_dict[ref_index]))
            except ValueError:
                continue
        
        print(f"成功匹配 {len(matched_pairs)} 对图像")
        return matched_pairs
    
    # 重新定义FID计算函数 - 确保只使用匹配的图像
    def fixed_calculate_fid(self, task_dir, reference_dir):
        """修复的FID计算"""
        if not FID_AVAILABLE:
            return -1.0
        
        try:
            # 创建临时目录用于FID计算
            temp_gen_dir = self.results_dir / f"temp_gen_{task_dir.name}"
            temp_ref_dir = self.results_dir / f"temp_ref_{task_dir.name}"
            
            # 获取匹配的图像对
            generated_images = list(task_dir.glob("*.png"))
            generated_images = [img for img in generated_images if img.is_file() and not img.name.startswith('.')]
            
            reference_images = list(reference_dir.glob("orig_*.png"))  # 只使用orig_*.png
            
            # 使用修复的匹配逻辑
            matched_pairs = fixed_match_image_pairs(generated_images, reference_images)
            
            if len(matched_pairs) == 0:
                print("没有匹配的图像对，跳过FID计算")
                return -1.0
            
            # 准备FID目录
            self._prepare_fid_directories_fixed(task_dir, reference_dir, temp_gen_dir, temp_ref_dir, matched_pairs)
            
            # 计算FID
            fid_value = calculate_fid_given_paths(
                [str(temp_gen_dir), str(temp_ref_dir)],
                batch_size=50,
                device=device,
                dims=2048
            )
            
            # 清理临时目录
            import shutil
            if temp_gen_dir.exists():
                shutil.rmtree(temp_gen_dir)
            if temp_ref_dir.exists():
                shutil.rmtree(temp_ref_dir)
            
            return fid_value
            
        except Exception as e:
            print(f"Error calculating FID for {task_dir.name}: {e}")
            return -1.0
    
    def _prepare_fid_directories_fixed(self, task_dir, reference_dir, temp_gen_dir, temp_ref_dir, matched_pairs):
        """为FID计算准备目录 - 只使用匹配的图像对"""
        import shutil
        
        # 清理并创建临时目录
        if temp_gen_dir.exists():
            shutil.rmtree(temp_gen_dir)
        if temp_ref_dir.exists():
            shutil.rmtree(temp_ref_dir)
        
        temp_gen_dir.mkdir(parents=True)
        temp_ref_dir.mkdir(parents=True)
        
        # 复制匹配的图像对
        for i, (gen_path, ref_path) in enumerate(matched_pairs):
            shutil.copy2(gen_path, temp_gen_dir / f"{i:06d}.png")
            shutil.copy2(ref_path, temp_ref_dir / f"{i:06d}.png")
        
        print(f"FID计算准备: {len(matched_pairs)} 对匹配的图像")
    
    # 替换函数
    evaluator.get_dataset_info = fixed_get_dataset_info
    evaluator._match_image_pairs = fixed_match_image_pairs
    evaluator.calculate_fid = fixed_calculate_fid.__get__(evaluator, TaskEvaluator)
    
    return evaluator

def test_fixed_fid():
    """测试修复的FID计算"""
    evaluator = create_fixed_fid_evaluator()
    
    # 测试一个任务
    task_name = 'ffhq_inpainting_rand_T1000'
    
    print(f"测试修复的FID计算: {task_name}")
    print("=" * 60)
    
    try:
        result = evaluator.evaluate_task(task_name)
        if result:
            print(f"✓ {task_name}:")
            print(f"    PSNR: {result['psnr']:.3f}")
            print(f"    LPIPS: {result['lpips']:.4f}")
            print(f"    FID: {result['fid']:.2f}")
        else:
            print(f"✗ {task_name}: 评估失败")
    except Exception as e:
        print(f"✗ {task_name}: 错误 - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 导入必要的模块
    try:
        import pandas as pd
    except ImportError:
        import datetime
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    return datetime.datetime.now()
    
    import torch
    from pytorch_fid.fid_score import calculate_fid_given_paths
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检查FID可用性
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        FID_AVAILABLE = True
    except ImportError:
        FID_AVAILABLE = False
        print("Warning: pytorch-fid not available.")
    
    test_fixed_fid()
