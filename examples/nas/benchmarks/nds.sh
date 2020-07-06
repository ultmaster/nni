set -e

echo "Downloading NDS..."
wget https://dl.fbaipublicfiles.com/nds/data.zip -O data.zip
unzip data.zip

echo "Generating database..."
rm -f /outputs/nds.db /outputs/nds.db-journal
NASBENCHMARK_DIR=/outputs python -m nni.nas.benchmarks.nds.db_gen nds_data
