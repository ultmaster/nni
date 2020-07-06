FROM python:3.7

RUN apt update && apt install -y wget zip && \
    pip install --no-cache-dir tqdm peewee

ADD . /nni
RUN mkdir -p /tmp && cp /nni/examples/nas/benchmarks/nasbench201.sh /tmp && \
    cd /nni && echo "y" | bash install.sh

WORKDIR /tmp
