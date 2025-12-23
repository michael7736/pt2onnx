#!/bin/bash
set -e

echo "========================================"
echo "  PT to ONNX Converter"
echo "  PyTorch Model to ONNX Format"
echo "========================================"
echo ""

# 如果没有参数，显示帮助
if [ $# -eq 0 ] || [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "用法:"
    echo "  docker run -v /path/to/models:/input -v /path/to/output:/output pt2onnx [选项]"
    echo ""
    echo "选项:"
    echo "  --input, -i      输入模型路径 (容器内路径，如 /input/model.pt)"
    echo "  --output, -o     输出路径 (容器内路径，如 /output/model.onnx)"
    echo "  --mode, -m       转换模式: auto, pytorch, huggingface, state_dict (默认: auto)"
    echo "  --input-shape    输入张量形状 (默认: 1,3,224,224)"
    echo "  --opset          ONNX opset 版本 (默认: 14)"
    echo "  --task           HuggingFace 任务类型 (默认: text-generation)"
    echo "  --model-class    模型类名 (用于 state_dict 模式)"
    echo "  --device         运行设备: cpu, cuda (默认: cpu)"
    echo ""
    echo "示例:"
    echo "  # 转换普通 PyTorch 模型"
    echo "  docker run -v \$(pwd)/models:/input -v \$(pwd)/output:/output pt2onnx \\"
    echo "    --input /input/model.pt --output /output/model.onnx"
    echo ""
    echo "  # 转换 HuggingFace 模型"
    echo "  docker run -v \$(pwd)/my-llm:/input -v \$(pwd)/output:/output pt2onnx \\"
    echo "    --input /input --output /output --mode huggingface --task text-generation"
    echo ""
    echo "  # 使用 GPU"
    echo "  docker run --gpus all -v \$(pwd)/models:/input -v \$(pwd)/output:/output pt2onnx \\"
    echo "    --input /input/model.pt --output /output/model.onnx --device cuda"
    echo ""
    exit 0
fi

# 执行转换
exec python /app/convert.py "$@"
