#!/usr/bin/env python3
"""
修复评估脚本 - 重新评估几个高质量任务
"""

import json
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_all_tasks import TaskEvaluator

def fix_image_matching():
    """修复图像匹配逻辑"""
    evaluator = TaskEvaluator(Path("/workspace/DDNM/exp"))
    
    # 重新定义匹配函数
    def fixed_match_image_pairs(generated_images, reference_images):
        """修复的图像匹配逻辑"""
        matched_pairs = []
        
        # 创建参考图像的索引 - 使用文件名中的数字
        ref_dict = {}
        for ref_img in reference_images:
            stem = ref_img.stem
            # 提取数字部分（去除前导零）
            try:
                num = int(stem)
                ref_dict[num] = ref_img
            except ValueError:
                continue
        
        # 匹配生成的图像
        for gen_img in generated_images:
            stem = gen_img.stem
            try:
                # 处理格式如 "0_0", "10_0", "-1_0" 等
                if '_' in stem:
                    num_part = stem.split('_')[0]
                    if num_part == '-1':
                        # 跳过 -1 索引
                        continue
                    num = int(num_part)
                else:
                    num = int(stem)
                
                if num in ref_dict:
                    matched_pairs.append((gen_img, ref_dict[num]))
            except ValueError:
                continue
        
        return matched_pairs
    
    # 替换匹配函数
    evaluator._match_image_pairs = fixed_match_image_pairs
    
    return evaluator

def re_evaluate_high_quality_tasks():
    """重新评估高质量任务"""
    evaluator = fix_image_matching()
    
    # 选择几个高质量任务重新评估
    high_quality_tasks = [
        'ffhq_inpainting_rand_T1000',
        'ffhq_inpainting_rand_T100', 
        'imagenet_inpainting_rand_T1000',
        'imagenet_inpainting_T1000',
        'ffhq_sr_bicubic_T1000'
    ]
    
    print("重新评估高质量任务...")
    print("=" * 60)
    
    results = {}
    for task_name in high_quality_tasks:
        print(f"\n重新评估: {task_name}")
        print("-" * 40)
        
        try:
            result = evaluator.evaluate_task(task_name)
            if result:
                results[task_name] = result
                print(f"✓ {task_name}: PSNR={result['psnr']:.3f}, LPIPS={result['lpips']:.4f}, FID={result['fid']:.2f}")
            else:
                print(f"✗ {task_name}: 评估失败")
        except Exception as e:
            print(f"✗ {task_name}: 错误 - {e}")
    
    # 保存修复后的结果
    if results:
        results_path = evaluator.results_dir / "fixed_evaluation_results.json"
        
        output_data = {
            'metadata': {
                'total_tasks': len(results),
                'evaluation_time': str(pd.Timestamp.now()),
                'device': str(device),
                'fid_available': True,
                'note': 'Fixed image matching logic'
            },
            'results': results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n修复后的结果保存到: {results_path}")
        
        # 显示修复后的结果
        print("\n" + "=" * 80)
        print("修复后的评估结果")
        print("=" * 80)
        
        for task_name, result in results.items():
            print(f"{task_name}:")
            print(f"  PSNR: {result['psnr']:.3f}")
            print(f"  LPIPS: {result['lpips']:.4f}")
            print(f"  FID: {result['fid']:.2f}")
            print()

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    re_evaluate_high_quality_tasks()
