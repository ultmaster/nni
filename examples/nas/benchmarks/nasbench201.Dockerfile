FROM ufoym/deepo:pytorch-cpu

RUN apt update && apt install -y wget && \
    pip uninstall -y enum34 && \
    pip install --no-cache-dir gdown tqdm peewee

ADD . /nni
RUN mkdir -p /tmp && cp /nni/examples/nas/benchmarks/nasbench201.sh /tmp && \
    cd /nni && echo "y" | bash install.sh

WORKDIR /tmp
