# PT to ONNX Converter

将 PyTorch (.pt) 模型转换为 ONNX 格式的 Docker 工具。

## 功能特性

- 支持普通 PyTorch 模型 (.pt 文件)
- 支持 HuggingFace Transformers 模型
- 支持 state_dict 格式的权重文件
- 自动检测模型类型
- 支持 CPU 和 GPU 转换
- 支持自定义输入形状和 ONNX opset 版本

## 快速开始

### 1. 构建 Docker 镜像

```bash
docker build -t pt2onnx .
```

### 2. 运行转换

#### 转换普通 PyTorch 模型

```bash
docker run -v $(pwd)/models:/input -v $(pwd)/output:/output pt2onnx \
  --input /input/model.pt \
  --output /output/model.onnx
```

#### 转换 HuggingFace 模型

```bash
docker run -v $(pwd)/my-llm:/input -v $(pwd)/output:/output pt2onnx \
  --input /input \
  --output /output \
  --mode huggingface \
  --task text-generation
```

#### 使用 GPU 加速

```bash
docker run --gpus all -v $(pwd)/models:/input -v $(pwd)/output:/output pt2onnx \
  --input /input/model.pt \
  --output /output/model.onnx \
  --device cuda
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input, -i` | 输入模型路径 | 必填 |
| `--output, -o` | 输出路径 | 必填 |
| `--mode, -m` | 转换模式: auto, pytorch, huggingface, state_dict | auto |
| `--input-shape` | 输入张量形状 (逗号分隔) | 1,3,224,224 |
| `--opset` | ONNX opset 版本 | 14 |
| `--task` | HuggingFace 任务类型 | text-generation |
| `--model-class` | 模型类名 (state_dict 模式) | - |
| `--device` | 运行设备: cpu, cuda | cpu |
| `--verbose, -v` | 显示详细信息 | false |

## 支持的 HuggingFace 任务类型

- `text-generation` / `causal-lm`: 文本生成模型 (GPT, LLaMA 等)
- `text-classification` / `sequence-classification`: 文本分类模型
- `token-classification`: 命名实体识别等
- `question-answering`: 问答模型
- `feature-extraction`: 特征提取

## 示例

### 1. 转换 ResNet 模型

```bash
# 假设你有一个 resnet50.pt 文件
docker run -v $(pwd):/data pt2onnx \
  --input /data/resnet50.pt \
  --output /data/resnet50.onnx \
  --input-shape 1,3,224,224
```

### 2. 转换 LLM 模型 (如 Qwen, LLaMA)

```bash
# 模型目录包含 config.json, model.safetensors 等文件
docker run -v $(pwd)/Qwen-7B:/input -v $(pwd)/output:/output pt2onnx \
  --input /input \
  --output /output \
  --mode huggingface \
  --task text-generation
```

### 3. 转换 state_dict 权重

```bash
docker run -v $(pwd):/data pt2onnx \
  --input /data/weights.pt \
  --output /data/model.onnx \
  --mode state_dict \
  --model-class torchvision.resnet50
```

## 目录结构

```
pt2onnx/
├── Dockerfile          # Docker 镜像定义
├── requirements.txt    # Python 依赖
├── convert.py          # 转换脚本
├── entrypoint.sh       # 入口脚本
└── README.md           # 本文件
```

## 注意事项

1. **大型 LLM 模型**: 转换大型语言模型需要足够的内存，建议至少 16GB RAM
2. **GPU 支持**: 使用 `--gpus all` 需要安装 NVIDIA Container Toolkit
3. **动态形状**: 默认支持动态 batch size，如需其他动态维度请修改脚本
4. **模型兼容性**: 并非所有 PyTorch 操作都能转换为 ONNX，某些自定义操作可能不支持

## 故障排除

### 内存不足

增加 Docker 内存限制：

```bash
docker run -m 32g --memory-swap 32g -v ... pt2onnx ...
```

### CUDA 错误

确保已安装 NVIDIA Container Toolkit：

```bash
# Ubuntu
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 模型加载失败

检查模型文件是否完整，或尝试指定 `--mode` 参数。

## License

MIT
