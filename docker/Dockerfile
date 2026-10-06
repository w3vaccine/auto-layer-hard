FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        "pillow>=10.0.0" \
        "opencv-python-headless>=4.8.0" \
        "numpy>=1.24.0" \
        "google-genai>=1.0.0" \
        "python-dotenv>=1.0.0" \
        "ultralytics>=8.3.0" \
    && python -c "import torch; assert '+cpu' in torch.__version__, torch.__version__"

RUN curl -fsSL -o /app/sam2_b.pt \
        https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt \
    && test "$(stat -c%s /app/sam2_b.pt)" -gt 100000000

ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TORCHDYNAMO_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    SAM_MODEL=sam2_b.pt \
    PRELOAD_SAM=1

COPY app.py ./
COPY src ./src
COPY web ./web
COPY fixtures ./fixtures

EXPOSE 8787
CMD ["python", "app.py"]
