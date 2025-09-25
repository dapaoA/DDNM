#!/usr/bin/env python3
"""
修正的评估脚本 - 使用正确的参考图像
"""

import json
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_all_tasks import TaskEvaluator

def create_corrected_evaluator():
    """创建修正的评估器"""
    evaluator = TaskEvaluator(Path("/workspace/DDNM/exp"))
    
    # 重新定义获取数据集信息的函数
    def corrected_get_dataset_info(task_name):
        """修正的数据集信息获取"""
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
    def corrected_match_image_pairs(generated_images, reference_images):
        """修正的图像匹配逻辑"""
        matched_pairs = []
        
        # 创建参考图像的索引
        ref_dict = {}
        for ref_img in reference_images:
            stem = ref_img.stem
            # 处理orig_*.png格式
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
        
        # 匹配生成的图像
        for gen_img in generated_images:
            stem = gen_img.stem
            try:
                # 处理格式如 "0_0", "10_0" 等
                if '_' in stem:
                    num_part = stem.split('_')[0]
                    if num_part == '-1':
                        continue
                    num = int(num_part)
                else:
                    num = int(stem)
                
                if num in ref_dict:
                    matched_pairs.append((gen_img, ref_dict[num]))
            except ValueError:
                continue
        
        return matched_pairs
    
    # 替换函数
    evaluator.get_dataset_info = corrected_get_dataset_info
    evaluator._match_image_pairs = corrected_match_image_pairs
    
    return evaluator

def evaluate_inpainting_tasks():
    """重新评估inpainting任务"""
    evaluator = create_corrected_evaluator()
    
    # 选择inpainting任务重新评估
    inpainting_tasks = [
        'ffhq_inpainting_rand_T1000',
        'ffhq_inpainting_rand_T100', 
        'ffhq_inpainting_rand_T20',
        'imagenet_inpainting_rand_T1000',
        'imagenet_inpainting_rand_T100',
        'imagenet_inpainting_rand_T20',
        'imagenet_inpainting_T1000',
        'imagenet_inpainting_T100',
        'imagenet_inpainting_T20'
    ]
    
    print("重新评估inpainting任务（使用正确的参考图像）...")
    print("=" * 70)
    
    results = {}
    for task_name in inpainting_tasks:
        print(f"\n评估: {task_name}")
        print("-" * 50)
        
        try:
            result = evaluator.evaluate_task(task_name)
            if result:
                results[task_name] = result
                print(f"✓ {task_name}:")
                print(f"    PSNR: {result['psnr']:.3f}")
                print(f"    LPIPS: {result['lpips']:.4f}")
                print(f"    FID: {result['fid']:.2f}")
            else:
                print(f"✗ {task_name}: 评估失败")
        except Exception as e:
            print(f"✗ {task_name}: 错误 - {e}")
    
    # 保存修正后的结果
    if results:
        results_path = evaluator.results_dir / "corrected_inpainting_results.json"
        
        output_data = {
            'metadata': {
                'total_tasks': len(results),
                'evaluation_time': str(pd.Timestamp.now()),
                'device': str(device),
                'fid_available': True,
                'note': 'Corrected evaluation using orig_*.png as reference for inpainting tasks'
            },
            'results': results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n修正后的结果保存到: {results_path}")
        
        # 显示修正后的结果汇总
        print("\n" + "=" * 80)
        print("修正后的inpainting任务评估结果")
        print("=" * 80)
        
        # 按数据集分组
        ffhq_results = {k: v for k, v in results.items() if k.startswith('ffhq')}
        imagenet_results = {k: v for k, v in results.items() if k.startswith('imagenet')}
        
        if ffhq_results:
            print("\nFFHQ Inpainting Tasks:")
            print("-" * 40)
            for task_name, result in ffhq_results.items():
                print(f"{task_name}: PSNR={result['psnr']:.3f}, LPIPS={result['lpips']:.4f}, FID={result['fid']:.2f}")
        
        if imagenet_results:
            print("\nImageNet Inpainting Tasks:")
            print("-" * 40)
            for task_name, result in imagenet_results.items():
                print(f"{task_name}: PSNR={result['psnr']:.3f}, LPIPS={result['lpips']:.4f}, FID={result['fid']:.2f}")
        
        # 计算平均值
        if ffhq_results:
            avg_psnr = sum(r['psnr'] for r in ffhq_results.values()) / len(ffhq_results)
            avg_lpips = sum(r['lpips'] for r in ffhq_results.values()) / len(ffhq_results)
            avg_fid = sum(r['fid'] for r in ffhq_results.values()) / len(ffhq_results)
            print(f"\nFFHQ平均: PSNR={avg_psnr:.3f}, LPIPS={avg_lpips:.4f}, FID={avg_fid:.2f}")
        
        if imagenet_results:
            avg_psnr = sum(r['psnr'] for r in imagenet_results.values()) / len(imagenet_results)
            avg_lpips = sum(r['lpips'] for r in imagenet_results.values()) / len(imagenet_results)
            avg_fid = sum(r['fid'] for r in imagenet_results.values()) / len(imagenet_results)
            print(f"ImageNet平均: PSNR={avg_psnr:.3f}, LPIPS={avg_lpips:.4f}, FID={avg_fid:.2f}")

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
    
    evaluate_inpainting_tasks()
