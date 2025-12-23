# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PT2ONNX is a containerized tool for converting PyTorch models to ONNX format. It supports standard PyTorch models, HuggingFace Transformers, and state_dict files.

## Build & Run Commands

```bash
# Build Docker image
docker build -t pt2onnx .

# Run conversion (CPU)
docker run -v /path/to/models:/input -v /path/to/output:/output pt2onnx -i /input/model.pt -o /output/model.onnx

# Run conversion (GPU)
docker run --gpus all -v /path/to/models:/input -v /path/to/output:/output pt2onnx -i /input/model.pt -o /output/model.onnx --device cuda

# Large LLM conversion (with memory limits)
docker run -m 32g --memory-swap 32g -v /path/to/models:/input -v /path/to/output:/output pt2onnx -i /input/llm -o /output/ --mode huggingface --task text-generation

# Direct Python execution (for development)
python convert.py -i /path/to/model.pt -o /path/to/output.onnx --verbose
```

## Architecture

```
Docker Container (Python 3.10)
├── entrypoint.sh          # CLI wrapper, forwards args to convert.py
└── convert.py             # Main conversion logic
    ├── convert_pytorch_model()      # Standard .pt files → torch.onnx.export()
    ├── convert_huggingface_model()  # HF models → optimum.onnxruntime
    └── convert_state_dict_model()   # State dicts → torchvision model instantiation
```

**Data Flow:** Input model → Auto-detect mode → Load model → Export to ONNX → Validate with onnx.checker → Output

## Key CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input, -i` | required | Model file or directory |
| `--output, -o` | required | Output .onnx file or directory |
| `--mode, -m` | auto | Conversion mode: auto/pytorch/huggingface/state_dict |
| `--input-shape` | 1,3,224,224 | Tensor dimensions for PyTorch models |
| `--opset` | 14 | ONNX opset version |
| `--task` | text-generation | HF task: text-generation, text-classification, token-classification, question-answering, feature-extraction |
| `--model-class` | - | Required for state_dict mode (torchvision class name) |
| `--device` | cpu | Execution device: cpu/cuda |
| `--verbose, -v` | false | Enable detailed logging |

## Conversion Modes

1. **pytorch** - Complete PyTorch `.pt` files with model architecture
2. **huggingface** - HF Transformers from hub or local directory
3. **state_dict** - Weight-only files requiring `--model-class` specification
4. **auto** - Automatic detection (default, recommended)

## Code Conventions

- Logging uses prefixes: `[INFO]`, `[ERROR]`, `[SUCCESS]`
- Bilingual documentation (Chinese/English) in README.md
- ONNX validation required after every export via `onnx.checker.check_model()`
- Dynamic axes enabled by default for batch_size flexibility

## Limitations

- Not all PyTorch operations convert to ONNX (custom ops may fail)
- Large LLM conversion requires 16GB+ RAM
- State_dict mode only supports torchvision models
- GPU mode requires NVIDIA Container Toolkit
- PyTorch 2.6+ requires `weights_only=False` for loading complete models
