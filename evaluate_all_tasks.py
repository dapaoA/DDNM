#!/usr/bin/env python3
"""
评估脚本：计算所有任务的PSNR、FID、LPIPS指标

使用方法:
    python evaluate_all_tasks.py

功能:
    1. 自动发现所有任务目录
    2. 批量计算PSNR和LPIPS
    3. 计算FID指标
    4. 保存结果到JSON文件
    5. 生成汇总报告
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# 尝试导入FID相关库
try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    from pytorch_fid.inception import InceptionV3
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch-fid not available. FID calculation will be skipped.")
    print("Install with: pip install pytorch-fid")

# 全局设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TaskEvaluator:
    """任务评估器"""
    
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.image_samples_dir = exp_dir / "image_samples"
        self.datasets_dir = exp_dir / "datasets"
        self.results_dir = exp_dir / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 初始化评估指标
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True
        ).to(device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # 结果存储
        self.all_results = {}
        
    def discover_tasks(self) -> List[str]:
        """发现所有任务目录"""
        if not self.image_samples_dir.exists():
            print(f"Error: Image samples directory not found: {self.image_samples_dir}")
            return []
        
        task_dirs = []
        for item in self.image_samples_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                task_dirs.append(item.name)
        
        task_dirs.sort()
        print(f"Found {len(task_dirs)} task directories:")
        for task in task_dirs:
            print(f"  - {task}")
        
        return task_dirs
    
    def get_dataset_info(self, task_name: str) -> Tuple[Optional[Path], Optional[str]]:
        """根据任务名推断对应的数据集"""
        if task_name.startswith('ffhq'):
            dataset_dir = self.datasets_dir / 'ffhq'
            dataset_type = 'ffhq'
        elif task_name.startswith('imagenet'):
            dataset_dir = self.datasets_dir / 'imagenet'
            dataset_type = 'imagenet'
        else:
            return None, None
            
        if dataset_dir.exists():
            return dataset_dir, dataset_type
        return None, None
    
    def load_image(self, image_path: Path) -> torch.Tensor:
        """加载图像为tensor"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(device)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def batch_load_images(self, image_paths: List[Path], batch_size: int = 32) -> List[torch.Tensor]:
        """批量加载图像"""
        images = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                img = self.load_image(path)
                if img is not None:
                    batch_images.append(img)
            
            if batch_images:
                images.extend(batch_images)
        
        return images
    
    def calculate_psnr_lpips_batch(self, generated_images: List[torch.Tensor], 
                                 reference_images: List[torch.Tensor]) -> Tuple[float, float]:
        """批量计算PSNR和LPIPS"""
        if len(generated_images) != len(reference_images):
            print(f"Warning: Mismatch in image counts: {len(generated_images)} vs {len(reference_images)}")
            min_len = min(len(generated_images), len(reference_images))
            generated_images = generated_images[:min_len]
            reference_images = reference_images[:min_len]
        
        psnr_values = []
        lpips_values = []
        
        # 批量处理以提高效率
        batch_size = 16
        for i in range(0, len(generated_images), batch_size):
            batch_gen = generated_images[i:i + batch_size]
            batch_ref = reference_images[i:i + batch_size]
            
            if not batch_gen or not batch_ref:
                continue
                
            # 堆叠为批量张量
            gen_batch = torch.cat(batch_gen, dim=0)
            ref_batch = torch.cat(batch_ref, dim=0)
            
            # 计算PSNR
            batch_psnr = self.psnr_metric(gen_batch, ref_batch)
            psnr_values.append(batch_psnr.item())
            
            # 计算LPIPS
            batch_lpips = self.lpips_metric(gen_batch, ref_batch)
            lpips_values.append(batch_lpips.item())
        
        avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
        avg_lpips = np.mean(lpips_values) if lpips_values else 0.0
        
        return avg_psnr, avg_lpips
    
    def calculate_fid(self, task_dir: Path, reference_dir: Path) -> float:
        """计算FID"""
        if not FID_AVAILABLE:
            return -1.0
        
        try:
            # 创建临时目录用于FID计算
            temp_gen_dir = self.results_dir / f"temp_gen_{task_dir.name}"
            temp_ref_dir = self.results_dir / f"temp_ref_{task_dir.name}"
            
            # 复制图像到临时目录
            self._prepare_fid_directories(task_dir, reference_dir, temp_gen_dir, temp_ref_dir)
            
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
    
    def _prepare_fid_directories(self, task_dir: Path, reference_dir: Path, 
                               temp_gen_dir: Path, temp_ref_dir: Path):
        """为FID计算准备目录"""
        import shutil
        
        # 清理并创建临时目录
        if temp_gen_dir.exists():
            shutil.rmtree(temp_gen_dir)
        if temp_ref_dir.exists():
            shutil.rmtree(temp_ref_dir)
        
        temp_gen_dir.mkdir(parents=True)
        temp_ref_dir.mkdir(parents=True)
        
        # 复制生成的图像
        gen_images = list(task_dir.glob("*.png"))
        for i, img_path in enumerate(gen_images):
            if img_path.is_file():
                shutil.copy2(img_path, temp_gen_dir / f"{i:06d}.png")
        
        # 复制参考图像
        ref_images = list(reference_dir.glob("*.png"))
        for i, img_path in enumerate(ref_images):
            if img_path.is_file():
                shutil.copy2(img_path, temp_ref_dir / f"{i:06d}.png")
    
    def evaluate_task(self, task_name: str) -> Dict[str, Any]:
        """评估单个任务"""
        print(f"\n{'='*60}")
        print(f"Evaluating task: {task_name}")
        print(f"{'='*60}")
        
        task_dir = self.image_samples_dir / task_name
        if not task_dir.exists():
            print(f"Error: Task directory not found: {task_dir}")
            return {}
        
        # 获取数据集信息
        dataset_dir, dataset_type = self.get_dataset_info(task_name)
        if dataset_dir is None:
            print(f"Warning: Could not determine dataset for task {task_name}")
            return {}
        
        print(f"Dataset: {dataset_type} at {dataset_dir}")
        
        # 获取图像文件列表
        generated_images = list(task_dir.glob("*.png"))
        reference_images = list(dataset_dir.glob("*.png"))
        
        # 过滤掉子目录中的图像
        generated_images = [img for img in generated_images if img.is_file()]
        reference_images = [img for img in reference_images if img.is_file()]
        
        print(f"Found {len(generated_images)} generated images")
        print(f"Found {len(reference_images)} reference images")
        
        if len(generated_images) == 0:
            print(f"Error: No generated images found in {task_dir}")
            return {}
        
        if len(reference_images) == 0:
            print(f"Error: No reference images found in {dataset_dir}")
            return {}
        
        # 匹配图像对
        matched_pairs = self._match_image_pairs(generated_images, reference_images)
        print(f"Matched {len(matched_pairs)} image pairs")
        
        if len(matched_pairs) == 0:
            print("Error: No matching image pairs found")
            return {}
        
        # 批量加载图像
        print("Loading images...")
        gen_paths, ref_paths = zip(*matched_pairs)
        generated_tensors = self.batch_load_images(list(gen_paths))
        reference_tensors = self.batch_load_images(list(ref_paths))
        
        print(f"Loaded {len(generated_tensors)} generated images")
        print(f"Loaded {len(reference_tensors)} reference images")
        
        # 计算PSNR和LPIPS
        print("Calculating PSNR and LPIPS...")
        psnr, lpips = self.calculate_psnr_lpips_batch(generated_tensors, reference_tensors)
        
        # 计算FID
        print("Calculating FID...")
        fid = self.calculate_fid(task_dir, dataset_dir)
        
        # 整理结果
        result = {
            'task_name': task_name,
            'dataset_type': dataset_type,
            'num_images': len(matched_pairs),
            'psnr': float(psnr),
            'lpips': float(lpips),
            'fid': float(fid) if fid >= 0 else None,
            'generated_dir': str(task_dir),
            'reference_dir': str(dataset_dir)
        }
        
        print(f"Results:")
        print(f"  PSNR: {psnr:.4f}")
        print(f"  LPIPS: {lpips:.4f}")
        print(f"  FID: {fid:.4f}" if fid >= 0 else "  FID: Failed")
        
        return result
    
    def _match_image_pairs(self, generated_images: List[Path], 
                          reference_images: List[Path]) -> List[Tuple[Path, Path]]:
        """匹配生成的图像和参考图像"""
        matched_pairs = []
        
        # 创建参考图像的索引
        ref_dict = {}
        for ref_img in reference_images:
            # 提取文件名中的数字部分
            stem = ref_img.stem
            try:
                # 尝试提取数字
                if '_' in stem:
                    num_part = stem.split('_')[0]
                else:
                    num_part = stem
                num = int(num_part)
                ref_dict[num] = ref_img
            except ValueError:
                # 如果无法解析数字，使用原始文件名
                ref_dict[stem] = ref_img
        
        # 匹配生成的图像
        for gen_img in generated_images:
            stem = gen_img.stem
            try:
                # 提取数字部分
                if '_' in stem:
                    num_part = stem.split('_')[0]
                else:
                    num_part = stem
                
                # 尝试不同的匹配策略
                if num_part.isdigit():
                    num = int(num_part)
                    if num in ref_dict:
                        matched_pairs.append((gen_img, ref_dict[num]))
                        continue
                
                # 尝试直接匹配文件名
                if stem in ref_dict:
                    matched_pairs.append((gen_img, ref_dict[stem]))
                    continue
                
                # 尝试匹配去除后缀的数字
                base_name = stem.split('_')[0] if '_' in stem else stem
                if base_name.isdigit():
                    num = int(base_name)
                    if num in ref_dict:
                        matched_pairs.append((gen_img, ref_dict[num]))
                        continue
                
            except ValueError:
                continue
        
        return matched_pairs
    
    def save_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json"):
        """保存结果到JSON文件"""
        results_path = self.results_dir / filename
        
        # 添加元数据
        output_data = {
            'metadata': {
                'total_tasks': len(results),
                'evaluation_time': str(pd.Timestamp.now()),
                'device': str(device),
                'fid_available': FID_AVAILABLE
            },
            'results': results
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {results_path}")
    
    def generate_summary(self, results: Dict[str, Any]):
        """生成汇总报告"""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        if not results:
            print("No results to summarize.")
            return
        
        # 按数据集分组
        ffhq_tasks = {}
        imagenet_tasks = {}
        other_tasks = {}
        
        for task_name, result in results.items():
            dataset_type = result.get('dataset_type', 'unknown')
            if dataset_type == 'ffhq':
                ffhq_tasks[task_name] = result
            elif dataset_type == 'imagenet':
                imagenet_tasks[task_name] = result
            else:
                other_tasks[task_name] = result
        
        # 打印汇总表
        print(f"\n{'Task Name':<40} {'Dataset':<10} {'Images':<8} {'PSNR':<8} {'LPIPS':<8} {'FID':<10}")
        print("-" * 90)
        
        for dataset_tasks, dataset_name in [(ffhq_tasks, 'FFHQ'), (imagenet_tasks, 'ImageNet'), (other_tasks, 'Other')]:
            if not dataset_tasks:
                continue
            
            for task_name, result in dataset_tasks.items():
                psnr = result.get('psnr', 0)
                lpips = result.get('lpips', 0)
                fid = result.get('fid')
                num_images = result.get('num_images', 0)
                
                fid_str = f"{fid:.2f}" if fid is not None else "N/A"
                
                print(f"{task_name:<40} {dataset_name:<10} {num_images:<8} {psnr:<8.3f} {lpips:<8.4f} {fid_str:<10}")
        
        # 计算平均值
        print(f"\n{'='*80}")
        print("AVERAGE METRICS")
        print(f"{'='*80}")
        
        for dataset_tasks, dataset_name in [(ffhq_tasks, 'FFHQ'), (imagenet_tasks, 'ImageNet')]:
            if not dataset_tasks:
                continue
            
            psnr_values = [r.get('psnr', 0) for r in dataset_tasks.values()]
            lpips_values = [r.get('lpips', 0) for r in dataset_tasks.values()]
            fid_values = [r.get('fid') for r in dataset_tasks.values() if r.get('fid') is not None]
            
            avg_psnr = np.mean(psnr_values) if psnr_values else 0
            avg_lpips = np.mean(lpips_values) if lpips_values else 0
            avg_fid = np.mean(fid_values) if fid_values else None
            
            print(f"\n{dataset_name} Dataset ({len(dataset_tasks)} tasks):")
            print(f"  Average PSNR: {avg_psnr:.3f}")
            print(f"  Average LPIPS: {avg_lpips:.4f}")
            print(f"  Average FID: {avg_fid:.2f}" if avg_fid is not None else "  Average FID: N/A")
    
    def run_evaluation(self, task_filter: Optional[List[str]] = None):
        """运行完整评估"""
        print("Starting evaluation of all tasks...")
        
        # 发现任务
        all_tasks = self.discover_tasks()
        if not all_tasks:
            print("No tasks found!")
            return
        
        # 应用过滤器
        if task_filter:
            all_tasks = [task for task in all_tasks if any(pattern in task for pattern in task_filter)]
            print(f"Filtered to {len(all_tasks)} tasks matching: {task_filter}")
        
        # 评估每个任务
        for task_name in all_tasks:
            try:
                result = self.evaluate_task(task_name)
                if result:
                    self.all_results[task_name] = result
            except Exception as e:
                print(f"Error evaluating task {task_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存结果
        if self.all_results:
            self.save_results(self.all_results)
            self.generate_summary(self.all_results)
        else:
            print("No successful evaluations!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all DDNM tasks")
    parser.add_argument("--exp_dir", type=str, default="/workspace/DDNM/exp",
                       help="Path to exp directory")
    parser.add_argument("--tasks", nargs="+", default=None,
                       help="Filter tasks by name patterns (e.g., --tasks ffhq imagenet)")
    parser.add_argument("--skip_fid", action="store_true",
                       help="Skip FID calculation")
    
    args = parser.parse_args()
    
    # 检查目录
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Error: exp directory not found: {exp_dir}")
        return 1
    
    # 创建评估器
    evaluator = TaskEvaluator(exp_dir)
    
    # 运行评估
    try:
        evaluator.run_evaluation(args.tasks)
        return 0
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # 添加pandas导入（用于时间戳）
    try:
        import pandas as pd
    except ImportError:
        import datetime
        class pd:
            class Timestamp:
                @staticmethod
                def now():
                    return datetime.datetime.now()
    
    sys.exit(main())
