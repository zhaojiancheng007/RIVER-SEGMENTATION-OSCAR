FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/envs/sam3/bin:${CONDA_DIR}/bin:${PATH}
ENV PYTHON_BIN=python
ENV PYTHONPATH=/workspace/river_seg_server_deploy:/workspace/sam3

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    bzip2 \
    git \
    wget \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda clean -afy

WORKDIR /workspace/river_seg_server_deploy

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy \
    && rm /tmp/environment.yml

RUN git clone https://github.com/facebookresearch/sam3.git /workspace/sam3 \
    && cd /workspace/sam3 \
    && git checkout 757bbb0206a0b68bee81b17d7eb4877177025b2f \
    && pip uninstall -y sam3 \
    && pip install --no-deps -e /workspace/sam3

COPY . .

CMD ["bash", "run_server_seg.sh"]
