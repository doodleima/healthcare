## start
FROM ubuntu:21.04
ARG PYTHON_VERSION=3.6
ARG DEBIAN_FRONTEND=noninteractive

## install custom libraries
RUN apt-get update && apt-get upgrade -y && \
         apt-get install -y --no-install-recommends \
         wget && \
         rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install libgtk2.0-dev -y

## install miniconda and needed python packages(for image conversion)
RUN wget -qO ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION flask tqdm && \
    /opt/conda/bin/conda install -y -c conda-forge flask-restx numpy pandas pydicom && \
    /opt/conda/bin/conda install -y -c conda-forge gdcm && \
    /opt/conda/bin/conda install -y --channel https://conda.anaconda.org/menpo opencv3 && \
    /opt/conda/bin/conda clean -ya

ENV PYTHONUNBUFFERED=0 \
    PATH=/opt/conda/bin:$PATH

## ADD Soruce code & prefix(dir)
COPY ./src /src
WORKDIR "/src"

ENTRYPOINT ["python3.6", "api.py"]
# CMD ["--help"]