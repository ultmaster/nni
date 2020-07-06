set -e

echo "Downloading NAS-Bench-101..."
wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord

echo "Generating database..."
rm -f /outputs/nasbench101.db /outputs/nasbench101.db-journal
NASBENCHMARK_DIR=/outputs python -m nni.nas.benchmarks.nasbench101.db_gen nasbench_full.tfrecord
