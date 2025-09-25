#!/usr/bin/env python3
"""
测试评估脚本的基本功能
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试所有必要的导入"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ torchvision: {e}")
        return False
    
    try:
        import torchmetrics
        print(f"✓ torchmetrics {torchmetrics.__version__}")
    except ImportError as e:
        print(f"✗ torchmetrics: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ PIL")
    except ImportError as e:
        print(f"✗ PIL: {e}")
        return False
    
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        print(f"✓ pytorch-fid")
    except ImportError as e:
        print(f"✗ pytorch-fid: {e}")
        print("  (FID calculation will be skipped)")
    
    return True

def test_device():
    """测试设备设置"""
    print("\nTesting device...")
    
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {device}")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  Using CPU")
        
        return True
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False

def test_directories():
    """测试目录结构"""
    print("\nTesting directories...")
    
    exp_dir = Path("/workspace/DDNM/exp")
    
    if not exp_dir.exists():
        print(f"✗ exp directory not found: {exp_dir}")
        return False
    
    print(f"✓ exp directory: {exp_dir}")
    
    image_samples_dir = exp_dir / "image_samples"
    if image_samples_dir.exists():
        task_dirs = [d for d in image_samples_dir.iterdir() if d.is_dir()]
        print(f"✓ Found {len(task_dirs)} task directories")
        
        # 显示前几个任务
        for i, task_dir in enumerate(sorted(task_dirs)[:5]):
            image_count = len(list(task_dir.glob("*.png")))
            print(f"  - {task_dir.name}: {image_count} images")
        
        if len(task_dirs) > 5:
            print(f"  ... and {len(task_dirs) - 5} more")
    else:
        print(f"✗ image_samples directory not found: {image_samples_dir}")
        return False
    
    datasets_dir = exp_dir / "datasets"
    if datasets_dir.exists():
        dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
        print(f"✓ Found {len(dataset_dirs)} dataset directories")
        
        for dataset_dir in dataset_dirs:
            image_count = len(list(dataset_dir.glob("*.png")))
            print(f"  - {dataset_dir.name}: {image_count} images")
    else:
        print(f"✗ datasets directory not found: {datasets_dir}")
        return False
    
    return True

def test_evaluation_class():
    """测试评估类的基本功能"""
    print("\nTesting evaluation class...")
    
    try:
        from evaluate_all_tasks import TaskEvaluator
        exp_dir = Path("/workspace/DDNM/exp")
        evaluator = TaskEvaluator(exp_dir)
        print("✓ TaskEvaluator created successfully")
        
        # 测试任务发现
        tasks = evaluator.discover_tasks()
        print(f"✓ Discovered {len(tasks)} tasks")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("DDNM Evaluation Script Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Device Test", test_device),
        ("Directory Test", test_directories),
        ("Evaluation Class Test", test_evaluation_class),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("✓ All tests passed! The evaluation script should work correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
