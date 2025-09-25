#!/usr/bin/env python3
"""
监控评估进度的脚本
"""

import json
import time
from pathlib import Path

def monitor_evaluation():
    results_file = Path("/workspace/DDNM/exp/evaluation_results/evaluation_results.json")
    
    if not results_file.exists():
        print("结果文件不存在，评估可能还未开始...")
        return
    
    # 获取所有任务列表
    image_samples_dir = Path("/workspace/DDNM/exp/image_samples")
    all_tasks = [d.name for d in image_samples_dir.iterdir() if d.is_dir()]
    
    print(f"总任务数: {len(all_tasks)}")
    print("=" * 80)
    
    while True:
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            completed_tasks = data.get('results', {})
            completed_count = len(completed_tasks)
            
            print(f"\r进度: {completed_count}/{len(all_tasks)} ({completed_count/len(all_tasks)*100:.1f}%)", end='', flush=True)
            
            if completed_count == len(all_tasks):
                print("\n评估完成！")
                break
                
            # 显示最近完成的任务
            if completed_tasks:
                latest_task = list(completed_tasks.keys())[-1]
                result = completed_tasks[latest_task]
                print(f"\n最新完成: {latest_task}")
                print(f"  PSNR: {result['psnr']:.3f}, LPIPS: {result['lpips']:.4f}, FID: {result['fid']:.2f}")
            
            time.sleep(10)  # 每10秒检查一次
            
        except Exception as e:
            print(f"\n监控错误: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_evaluation()
