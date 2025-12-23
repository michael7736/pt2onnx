# PT to ONNX Converter Docker Image
# 支持将 PyTorch 模型转换为 ONNX 格式

FROM python:3.10-slim

LABEL maintainer="pt2onnx"
LABEL description="Convert PyTorch (.pt) models to ONNX format"
LABEL version="1.0"

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制转换脚本
COPY convert.py .
COPY entrypoint.sh .

# 设置脚本权限
RUN chmod +x convert.py entrypoint.sh

# 创建输入输出目录
RUN mkdir -p /input /output

# 设置入口点
ENTRYPOINT ["/app/entrypoint.sh"]

# 默认显示帮助信息
CMD ["--help"]
