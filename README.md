# 3D-Gaussian-Splatting

基于原版 [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 的实现。

## 项目特性

- 🚀 **简化的配置管理**: 使用YAML配置文件，支持多种预设配置
- ⚡ **优化的训练流程**: 集成训练、渲染和评估于一体
- 🔧 **灵活的参数调整**: 支持继续训练、检查点保存等高级功能
- 📊 **完整的评估指标**: 自动计算PSNR、SSIM、LPIPS等指标
- 🐛 **修复的编译问题**: 解决了CUDA编译相关的常见问题

## 环境要求

- CUDA (测试版本: 12.8)
- Python 3.8+
- UV 包管理器

## 安装步骤

1. **克隆项目**（包含子模块）:
```bash
git clone git@github.com:xcz0/3D-Gaussian-Splatting.git --recursive
cd 3D-Gaussian-Splatting
```

2. **应用CUDA编译修复**:
   编辑 `gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`，在文件开头添加：
```cpp
#include <cstdint>
#include <cstddef>
```

3. **安装依赖**:
```bash
uv sync --no-build-isolation
```

## 数据准备

在训练之前，您需要准备符合COLMAP格式的数据集：

1. **数据格式**: 支持COLMAP格式的相机参数和图像
2. **数据结构**: 
   ```
   data/
   ├── images/          # 输入图像
   ├── sparse/          # COLMAP稀疏重建结果
   │   └── 0/
   │       ├── cameras.bin
   │       ├── images.bin
   │       └── points3D.bin
   └── distorted/       # (可选) 未校正的图像
   ```
3. **数据路径**: 在配置文件中设置 `paths.source_path` 指向数据目录


## 使用方法

### 1. 查看可用配置
```bash
uv run main.py --list
```

### 2. 运行训练
```bash
# 快速测试 (1000次迭代，约20-30分钟)
uv run main.py --config config/quick.yaml

# 高质量训练 (30000次迭代，需要几小时)
uv run main.py --config config/high_quality.yaml
```


## 配置文件说明

### quick.yaml - 快速测试
- **用途**: 快速验证数据和环境是否正常
- **迭代次数**: 1000
- **训练时间**: 约20-30分钟
- **特点**: 低分辨率，快速收敛参数

### high_quality.yaml - 高质量训练
- **用途**: 高质量模型训练
- **迭代次数**: 30000
- **训练时间**: 几小时到一天
- **特点**: 高分辨率，完整训练参数


## 配置文件结构

```yaml
# 基本信息
name: "配置名称"
description: "配置描述"

# 训练参数
training:
  iterations: 10000
  eval: true
  densify_grad_threshold: 0.0003
  # ... 其他训练参数

# 路径配置
paths:
  source_path: "../data"
  model_path: "../output/model_name"

# 后处理
post_processing:
  enable_render: true
  enable_metrics: true

# 特殊设置 (可选)
setup:
  install_accelerated_rasterizer: true
  rasterizer_branch: "3dgs_accel"
```

## 输出结果

训练完成后，结果将保存在指定的模型路径下：

```
output/model_name/
├── cfg_args          # 训练配置参数
├── chkpnt*.pth      # 检查点文件
├── point_cloud/     # 点云数据
│   └── iteration_*/
├── test/            # 测试集渲染结果
│   ├── ours_*/
│   └── gt/
├── train/           # 训练集渲染结果
└── results.json     # 评估指标 (PSNR, SSIM, LPIPS)
```

## 结果评估

训练完成后，系统会自动计算以下指标：
- **PSNR**: 峰值信噪比，数值越高越好
- **SSIM**: 结构相似性指数，数值越高越好  
- **LPIPS**: 学习感知图像块相似度，数值越低越好

## 自定义配置

您可以创建自己的配置文件：

1. **复制现有配置**:
   ```bash
   cp config/quick.yaml config/my_config.yaml
   ```

2. **修改配置参数**:
   ```yaml
   name: "my_custom_training"
   description: "我的自定义训练配置"
   
   paths:
     source_path: "path/to/your/data"
     model_path: "output/my_model"
   
   training:
     iterations: 15000
     resolution: 2
     # ... 其他参数
   ```

3. **运行自定义训练**:
   ```bash
   uv run main.py --config config/my_config.yaml
   ```

## 支持的训练参数

### 核心参数
- `iterations`: 训练迭代次数
- `eval`: 是否启用评估
- `resolution`: 分辨率缩放因子
- `data_device`: 数据设备 (cpu/cuda)
- `quiet`: 静默模式

### 密化参数
- `densify_grad_threshold`: 密化梯度阈值
- `densification_interval`: 密化间隔
- `densify_until_iter`: 密化截止迭代
- `opacity_reset_interval`: 透明度重置间隔

### 学习率参数
- `position_lr_init`: 位置学习率初始值
- `position_lr_final`: 位置学习率最终值
- `scaling_lr`: 缩放学习率

### 评估和保存
- `test_iterations`: 测试迭代列表
- `save_iterations`: 保存迭代列表
- `checkpoint_iterations`: 检查点保存迭代列表

### 损失函数
- `lambda_dssim`: DSSIM损失权重
- `optimizer_type`: 优化器类型

## 监控训练进度

训练过程中，您可以：

1. **查看终端输出**: 了解训练进度和损失变化
2. **使用TensorBoard**: 查看详细的训练曲线和指标
   ```bash
   tensorboard --logdir output/model_name
   ```
3. **检查中间结果**: 查看保存的中间模型文件和渲染结果

## 常见问题 (FAQ)

### Q: 编译时出现 "namespace std has no member uintptr_t" 错误
**A**: 这是因为缺少必要的头文件包含。请按照上述编译修复步骤，在 `rasterizer_impl.h` 文件开头添加：
```cpp
#include <cstdint>
#include <cstddef>
```

### Q: 找不到ninja构建器
**A**: 这是正常现象。系统会自动回退到distutils构建器，虽然速度较慢但不影响结果。如需加速编译，可以安装ninja：
```bash
pip install ninja
```

### Q: 安装成功但导入失败
**A**: 请确保：
1. 已正确应用CUDA编译修复
2. 使用了 `--no-build-isolation` 标志进行安装
3. 激活了正确的虚拟环境
4. CUDA环境变量设置正确

### Q: 训练过程中出现内存不足
**A**: 可以尝试：
1. 降低 `resolution` 参数值
2. 设置 `data_device: "cpu"` 将数据存储在CPU内存中
3. 减少批处理大小或使用更小的数据集

### Q: 如何继续中断的训练
**A**: 在配置文件中设置：
```yaml
training:
  resume_training: true
  resume_from_iteration: 1000  # 从第1000次迭代继续
```

## 许可证

本项目基于原版3D Gaussian Splatting项目，请遵循相应的许可证条款。

## 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

## 致谢

感谢原版 [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 项目的作者们提供的优秀实现。