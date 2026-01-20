FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (smaller, faster build)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

# Install ESM-C from source without dependency resolution backtracking
RUN pip install --no-cache-dir "git+https://github.com/evolutionaryscale/esm.git" --no-deps

CMD ["/bin/bash"]
