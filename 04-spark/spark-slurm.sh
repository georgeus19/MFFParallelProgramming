#!/bin/bash
## #SBATCH --time=0:30:00 -N 6 -c 16 -p mpi-homo-short --mem 101G
#SBATCH --time=0:30:00 -p mpi-homo-short --mem 1G

# Run an example non-interactive Spark computation. Requires three arguments:
#
#   1. Image directory
#   2. High-speed network interface name
#   3. R/W directory
#   4. Application
#
# Example:
#
#   $ sbatch spark-slurm.sh /mnt/home/_teaching/para/spark eno1 ~ /mnt/1/spark/myapp.py 
#
# Spark configuration will be generated in ~/slurm-$SLURM_JOB_ID.spark; any
# configuration already there will be clobbered.

set -e

if [[ -z $SLURM_JOB_ID ]]; then
    echo "not running under Slurm" 1>&2
    exit 1
fi

img=$1
dev=$2
rwdir=$3
app=$4
conf=${HOME}/slurm-${SLURM_JOB_ID}.spark

# What IP address to use for master?
if [[ -z $dev ]]; then
    echo "no high-speed network device specified"
    exit 1
fi
master_ip=$(  ip -o -f inet addr show dev "$dev" \
            | sed -r 's/^.+inet ([0-9.]+).+/\1/')
master_url=spark://${master_ip}:7077
if [[ -n $master_ip ]]; then
    echo "Spark master IP: ${master_ip}"
else
    echo "no IP address for ${dev} found"
    exit 1
fi

tmpdir="$HOME/spark/log"
tmpdir_cont="/mnt/2"
rwdir_cont="/mnt/1"
conf_cont="/mnt/0"

# Make Spark configuration
mkdir "$conf"
chmod 700 "$conf"
cat <<EOF > "${conf}/spark-env.sh"
SPARK_LOCAL_DIRS=$tmpdir_cont
SPARK_LOG_DIR=$tmpdir_cont/log
SPARK_WORKER_DIR=$tmpdir_cont
SPARK_LOCAL_IP=127.0.0.1
SPARK_MASTER_HOST=${master_ip}
EOF
mysecret=$(cat /dev/urandom | tr -dc '0-9a-f' | head -c 48)
cat <<EOF > "${conf}/spark-defaults.sh"
spark.authenticate true
spark.authenticate.secret $mysecret
EOF
chmod 600 "${conf}/spark-defaults.sh"

# UTF-8 encoding for Python
export PYTHONIOENCODING=utf8

# Start the Spark master
ch-run -b "$conf:${conf_cont}" -b "$rwdir:${rwdir_cont}" -b "$tmpdir:${tmpdir_cont}" "$img" -- /opt/spark/sbin/start-master.sh
sleep 10
tail -7 $tmpdir/log/*master*.out
grep -Fq 'New state: ALIVE' $tmpdir/log/*master*.out

# Start the Spark workers
srun --mem 1G sh -c "  ch-run -b '${conf}:${conf_cont}' -b '${rwdir}:${rwdir_cont}' -b '${tmpdir}:${tmpdir_cont}' '${img}' -- \
                      /opt/spark/sbin/start-slave.sh ${master_url} \
            && sleep infinity" &
sleep 10
grep -F worker $tmpdir/log/*master*.out
tail -3 $tmpdir/log/*worker*.out

ch-run -b "$conf:${conf_cont}" -b "$rwdir:${rwdir_cont}" -b "$tmpdir:${tmpdir_cont}" "$img" -- \
       /opt/spark/bin/spark-submit --master "$master_url" \
       --executor-memory=24G \
       --executor-cores=4 \
       "$app"
# Let Slurm kill the workers and master
