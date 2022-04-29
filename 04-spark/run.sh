DIR="${HOME}/spark-output"
rm -rf "${DIR}"
mkdir "${DIR}"

ln -s /opt/data/seznam.csv $DIR/seznam.csv
cp -r $HOME/submit_box/* $DIR/

sbatch -A nprg042s $DIR/spark-slurm.sh /home/_teaching/para/04-spark/spark/ eno1 $DIR/

