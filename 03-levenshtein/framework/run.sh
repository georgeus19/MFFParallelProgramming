srun -p mpi-homo-short -A nprg042s make
echo "Running"

echo "--1"
srun -p mpi-homo-short -A nprg042s -c 32 ./levenshtein ../data/01-32k.A ../data/01-32k.B
echo "--2"
srun -p mpi-homo-short -A nprg042s -c 32 ./levenshtein ../data/02-64k.A ../data/02-64k.B
echo "--3"
srun -p mpi-homo-short -A nprg042s -c 32 ./levenshtein ../data/03-128k.A ../data/03-128k.B
echo "--4"
srun -p mpi-homo-short -A nprg042s -c 32 ./levenshtein ../data/04-64k.A ../data/04-128k.B
echo "--5"
srun -p mpi-homo-short -A nprg042s -c 32 ./levenshtein ../data/05-128k.A ../data/05-64k.B