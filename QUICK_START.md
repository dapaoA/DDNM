# DDNM 评估脚本快速开始指南

## 1. 安装依赖

```bash
# 运行安装脚本
./install_dependencies.sh

# 或者手动安装
pip3 install -r requirements_evaluation.txt
```

## 2. 测试安装

```bash
# 运行测试脚本验证环境
python3 test_evaluation.py
```

## 3. 运行评估

```bash
# 评估所有任务
./run_evaluation.sh

# 或者直接使用Python脚本
python3 evaluate_all_tasks.py
```

## 4. 查看结果

结果保存在 `exp/evaluation_results/evaluation_results.json`

## 常用选项

```bash
# 只评估FFHQ任务
./run_evaluation.sh --tasks ffhq

# 只评估ImageNet任务  
./run_evaluation.sh --tasks imagenet

# 跳过FID计算（更快）
./run_evaluation.sh --skip-fid

# 指定exp目录
./run_evaluation.sh --exp-dir /path/to/exp
```

## 故障排除

如果遇到依赖问题：
```bash
# 重新安装依赖
pip3 install --upgrade -r requirements_evaluation.txt

# 如果PyTorch有问题，重新安装
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio
```

如果CUDA内存不足：
```bash
# 使用CPU运行
CUDA_VISIBLE_DEVICES="" python3 evaluate_all_tasks.py
```

## 输出说明

- **PSNR**: 峰值信噪比，越高越好
- **LPIPS**: 感知损失，越低越好  
- **FID**: Fréchet Inception Distance，越低越好

详细说明请查看 `EVALUATION_README.md`
