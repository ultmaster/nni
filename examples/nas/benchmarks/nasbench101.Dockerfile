FROM tensorflow/tensorflow:1.15.2-py3

RUN apt update && apt install -y wget git && \
    pip install --no-cache-dir tqdm peewee

ADD . /nni
RUN mkdir -p /tmp && cp /nni/examples/nas/benchmarks/nasbench101.sh /tmp && \
    cd /nni && echo "y" | source install.sh

RUN cd / && git clone https://github.com/google-research/nasbench && \
    cd nasbench && pip install -e . && cd ..

WORKDIR /tmp
