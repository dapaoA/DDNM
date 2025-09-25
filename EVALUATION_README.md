# DDNM 任务评估脚本

这个脚本用于自动计算所有DDNM任务的PSNR、FID和LPIPS指标。

## 功能特性

- **自动任务发现**: 自动扫描 `exp/image_samples` 目录下的所有任务
- **批量计算**: 高效的批量PSNR和LPIPS计算
- **FID支持**: 集成pytorch-fid进行FID计算
- **智能匹配**: 自动匹配生成的图像和参考图像
- **结果保存**: 将结果保存为JSON格式
- **汇总报告**: 生成详细的汇总报告

## 安装依赖

```bash
pip install -r requirements_evaluation.txt
```

主要依赖包括：
- torch, torchvision, torchmetrics
- pytorch-fid (用于FID计算)
- numpy, Pillow, pandas

## 使用方法

### 方法1: 使用Shell脚本 (推荐)

```bash
# 评估所有任务
./run_evaluation.sh

# 只评估FFHQ任务
./run_evaluation.sh --tasks ffhq

# 只评估ImageNet任务
./run_evaluation.sh --tasks imagenet

# 跳过FID计算 (更快)
./run_evaluation.sh --skip-fid

# 指定exp目录
./run_evaluation.sh --exp-dir /path/to/exp
```

### 方法2: 直接使用Python脚本

```bash
# 评估所有任务
python evaluate_all_tasks.py

# 只评估特定任务
python evaluate_all_tasks.py --tasks ffhq imagenet

# 跳过FID计算
python evaluate_all_tasks.py --skip-fid

# 指定exp目录
python evaluate_all_tasks.py --exp-dir /workspace/DDNM/exp
```

## 输出结果

### 文件结构

```
exp/
├── image_samples/          # 生成的图像
│   ├── ffhq_sr_bicubic_T20/
│   ├── imagenet_deblur_gauss_T100/
│   └── ...
├── datasets/               # 参考数据集
│   ├── ffhq/
│   └── imagenet/
└── evaluation_results/     # 评估结果
    └── evaluation_results.json
```

### JSON结果格式

```json
{
  "metadata": {
    "total_tasks": 42,
    "evaluation_time": "2024-01-15 10:30:00",
    "device": "cuda:0",
    "fid_available": true
  },
  "results": {
    "ffhq_sr_bicubic_T20": {
      "task_name": "ffhq_sr_bicubic_T20",
      "dataset_type": "ffhq",
      "num_images": 100,
      "psnr": 28.456,
      "lpips": 0.1234,
      "fid": 15.67,
      "generated_dir": "/path/to/ffhq_sr_bicubic_T20",
      "reference_dir": "/path/to/ffhq"
    }
  }
}
```

### 控制台输出示例

```
============================================================
Evaluating task: ffhq_sr_bicubic_T20
============================================================
Dataset: ffhq at /workspace/DDNM/exp/datasets/ffhq
Found 100 generated images
Found 100 reference images
Matched 100 image pairs
Loading images...
Loaded 100 generated images
Loaded 100 reference images
Calculating PSNR and LPIPS...
Calculating FID...
Results:
  PSNR: 28.456
  LPIPS: 0.1234
  FID: 15.67

================================================================================
EVALUATION SUMMARY
================================================================================

Task Name                               Dataset   Images   PSNR     LPIPS    FID        
------------------------------------------------------------------------------------------
ffhq_sr_bicubic_T20                    ffhq      100      28.456   0.1234   15.67
imagenet_deblur_gauss_T100             imagenet  100      25.123   0.1456   18.90
...

================================================================================
AVERAGE METRICS
================================================================================

FFHQ Dataset (21 tasks):
  Average PSNR: 26.789
  Average LPIPS: 0.1345
  Average FID: 17.23

ImageNet Dataset (21 tasks):
  Average PSNR: 24.567
  Average LPIPS: 0.1567
  Average FID: 19.45
```

## 支持的任务类型

脚本自动识别以下任务模式：

- **FFHQ任务**: `ffhq_*`
  - 超分辨率: `ffhq_sr_bicubic_*`
  - 去模糊: `ffhq_deblur_gauss_*`
  - 修复: `ffhq_inpainting_*`

- **ImageNet任务**: `imagenet_*`
  - 超分辨率: `imagenet_sr_bicubic_*`
  - 去模糊: `imagenet_deblur_gauss_*`
  - 修复: `imagenet_inpainting_*`
  - 着色: `imagenet_colorization_*`

## 图像匹配策略

脚本使用多种策略匹配生成的图像和参考图像：

1. **数字匹配**: 提取文件名中的数字进行匹配
2. **直接匹配**: 尝试直接匹配文件名
3. **前缀匹配**: 匹配去除后缀后的数字

例如：
- `0_0.png` → `00000.png`
- `1_0.png` → `00001.png`
- `-1_0.png` → 跳过（负数索引）

## 性能优化

- **批量处理**: PSNR和LPIPS使用批量计算
- **GPU加速**: 自动使用CUDA（如果可用）
- **内存管理**: 分批加载图像避免内存溢出
- **缓存机制**: FID计算使用临时目录避免重复计算

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批量大小或使用CPU
   CUDA_VISIBLE_DEVICES="" python evaluate_all_tasks.py
   ```

2. **FID计算失败**
   ```bash
   # 跳过FID计算
   python evaluate_all_tasks.py --skip-fid
   ```

3. **图像匹配失败**
   - 检查文件名格式
   - 确保生成的图像和参考图像数量匹配

4. **依赖缺失**
   ```bash
   pip install -r requirements_evaluation.txt
   ```

### 调试模式

可以通过修改脚本添加更多调试信息：

```python
# 在脚本开头添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 添加新的评估指标

可以在 `TaskEvaluator` 类中添加新的指标计算方法：

```python
def calculate_custom_metric(self, generated_images, reference_images):
    # 实现自定义指标
    pass
```

### 支持新的数据集

修改 `get_dataset_info` 方法以支持新的数据集：

```python
def get_dataset_info(self, task_name: str):
    if task_name.startswith('custom_dataset'):
        dataset_dir = self.datasets_dir / 'custom_dataset'
        return dataset_dir, 'custom_dataset'
    # ... 其他逻辑
```

## 注意事项

1. **存储空间**: FID计算需要额外的临时存储空间
2. **计算时间**: FID计算比较耗时，可以考虑并行计算
3. **内存使用**: 大量图像可能消耗大量内存
4. **图像格式**: 目前只支持PNG格式的图像

## 许可证

本脚本遵循与DDNM项目相同的许可证。
