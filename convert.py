#!/usr/bin/env python3
"""
PT to ONNX Converter
支持将 PyTorch (.pt) 模型转换为 ONNX 格式
支持普通 PyTorch 模型和 HuggingFace Transformers 模型
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import onnx


def convert_pytorch_model(
    model_path: str,
    output_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 14,
    dynamic_axes: dict = None,
    device: str = "cpu"
):
    """
    转换普通 PyTorch 模型到 ONNX

    Args:
        model_path: .pt 文件路径
        output_path: 输出 .onnx 文件路径
        input_shape: 输入张量形状
        opset_version: ONNX opset 版本
        dynamic_axes: 动态轴配置
        device: 设备 (cpu/cuda)
    """
    print(f"[INFO] 加载 PyTorch 模型: {model_path}")

    # 尝试加载模型
    try:
        # 首先尝试作为完整模型加载
        model = torch.load(model_path, map_location=device, weights_only=False)

        # 如果加载的是 state_dict，需要用户提供模型定义
        if isinstance(model, dict):
            if 'model' in model:
                model = model['model']
            elif 'state_dict' in model:
                print("[ERROR] 加载的是 state_dict，需要提供模型架构定义")
                print("[INFO] 请使用 --model-class 参数指定模型类")
                sys.exit(1)
            else:
                print("[ERROR] 无法识别的模型格式")
                sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {e}")
        sys.exit(1)

    model.eval()
    model.to(device)

    # 创建示例输入
    dummy_input = torch.randn(*input_shape).to(device)

    # 设置动态轴
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    print(f"[INFO] 导出 ONNX 模型到: {output_path}")
    print(f"[INFO] 输入形状: {input_shape}")
    print(f"[INFO] ONNX opset 版本: {opset_version}")

    # 导出到 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    # 验证导出的模型
    print("[INFO] 验证 ONNX 模型...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[SUCCESS] ONNX 模型验证通过!")

    return output_path


def convert_huggingface_model(
    model_path: str,
    output_path: str,
    task: str = "text-generation",
    opset_version: int = 14,
    device: str = "cpu"
):
    """
    转换 HuggingFace Transformers 模型到 ONNX
    使用 optimum 库进行转换

    Args:
        model_path: HuggingFace 模型路径或名称
        output_path: 输出目录
        task: 任务类型 (text-generation, text-classification 等)
        opset_version: ONNX opset 版本
        device: 设备
    """
    try:
        from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSequenceClassification
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    except ImportError:
        print("[ERROR] 请安装 optimum: pip install optimum[onnxruntime]")
        sys.exit(1)

    print(f"[INFO] 加载 HuggingFace 模型: {model_path}")
    print(f"[INFO] 任务类型: {task}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if task in ["text-generation", "causal-lm"]:
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # 导出到 ONNX
            print(f"[INFO] 导出模型到: {output_dir}")
            ort_model = ORTModelForCausalLM.from_pretrained(
                model_path,
                export=True,
                trust_remote_code=True
            )
            ort_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        elif task in ["text-classification", "sequence-classification"]:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            ort_model = ORTModelForSequenceClassification.from_pretrained(
                model_path,
                export=True,
                trust_remote_code=True
            )
            ort_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            # 使用通用导出方法
            from optimum.exporters.onnx import main_export
            main_export(
                model_path,
                output=output_dir,
                task=task,
                opset=opset_version,
                device=device
            )

        print(f"[SUCCESS] 模型已导出到: {output_dir}")

    except Exception as e:
        print(f"[ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    return str(output_dir)


def convert_state_dict_model(
    model_path: str,
    output_path: str,
    model_class: str,
    input_shape: tuple = (1, 3, 224, 224),
    opset_version: int = 14,
    device: str = "cpu"
):
    """
    转换只包含 state_dict 的 .pt 文件
    需要提供模型类定义
    """
    print(f"[INFO] 加载 state_dict: {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # 动态导入模型类
    # 这里需要用户提供模型定义文件
    print(f"[INFO] 使用模型类: {model_class}")

    # 示例：如果是 torchvision 模型
    if model_class.startswith("torchvision."):
        import torchvision.models as models
        model_name = model_class.split(".")[-1]
        model = getattr(models, model_name)()
    else:
        print(f"[ERROR] 不支持的模型类: {model_class}")
        print("[INFO] 支持的格式: torchvision.resnet50, torchvision.vgg16 等")
        sys.exit(1)

    model.load_state_dict(state_dict)
    model.eval()

    # 保存为完整模型后再转换
    temp_path = "/tmp/temp_model.pt"
    torch.save(model, temp_path)

    return convert_pytorch_model(
        temp_path,
        output_path,
        input_shape=input_shape,
        opset_version=opset_version,
        device=device
    )


def main():
    parser = argparse.ArgumentParser(
        description="将 PyTorch (.pt) 模型转换为 ONNX 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换普通 PyTorch 模型
  python convert.py --input model.pt --output model.onnx --input-shape 1,3,224,224

  # 转换 HuggingFace 模型
  python convert.py --input ./my-llm --output ./onnx-model --mode huggingface --task text-generation

  # 转换带 state_dict 的模型
  python convert.py --input weights.pt --output model.onnx --mode state_dict --model-class torchvision.resnet50
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入模型路径 (.pt 文件或 HuggingFace 模型目录)"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出路径 (.onnx 文件或目录)"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "pytorch", "huggingface", "state_dict"],
        default="auto",
        help="转换模式 (默认: auto)"
    )

    parser.add_argument(
        "--input-shape",
        type=str,
        default="1,3,224,224",
        help="输入张量形状，逗号分隔 (默认: 1,3,224,224)"
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset 版本 (默认: 14)"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="text-generation",
        help="HuggingFace 任务类型 (默认: text-generation)"
    )

    parser.add_argument(
        "--model-class",
        type=str,
        help="模型类名 (用于 state_dict 模式)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="运行设备 (默认: cpu)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细信息"
    )

    args = parser.parse_args()

    # 解析输入形状
    input_shape = tuple(map(int, args.input_shape.split(",")))

    # 自动检测模式
    mode = args.mode
    if mode == "auto":
        input_path = Path(args.input)
        if input_path.is_dir():
            # 检查是否是 HuggingFace 模型目录
            if (input_path / "config.json").exists():
                mode = "huggingface"
                print("[INFO] 检测到 HuggingFace 模型目录")
            else:
                print("[ERROR] 目录中未找到 config.json，请指定 --mode")
                sys.exit(1)
        elif input_path.suffix == ".pt":
            mode = "pytorch"
            print("[INFO] 检测到 PyTorch 模型文件")
        else:
            print(f"[ERROR] 不支持的文件类型: {input_path.suffix}")
            sys.exit(1)

    # 执行转换
    print("=" * 50)
    print("PT to ONNX Converter")
    print("=" * 50)

    if mode == "pytorch":
        convert_pytorch_model(
            args.input,
            args.output,
            input_shape=input_shape,
            opset_version=args.opset,
            device=args.device
        )
    elif mode == "huggingface":
        convert_huggingface_model(
            args.input,
            args.output,
            task=args.task,
            opset_version=args.opset,
            device=args.device
        )
    elif mode == "state_dict":
        if not args.model_class:
            print("[ERROR] state_dict 模式需要 --model-class 参数")
            sys.exit(1)
        convert_state_dict_model(
            args.input,
            args.output,
            model_class=args.model_class,
            input_shape=input_shape,
            opset_version=args.opset,
            device=args.device
        )

    print("=" * 50)
    print("[DONE] 转换完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
