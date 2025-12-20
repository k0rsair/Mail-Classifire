FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /python

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# pip
RUN python3 -m pip install --upgrade pip

# код
COPY ./python .

# зависимости
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]