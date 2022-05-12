echo " RUN: srun -p gpu-short -A nprg042s --gres=gpu:1 ./potential ../data/v16k-e64k.gbf"
srun -p gpu-short -A nprg042s --gres=gpu:1 ./potential ../data/v16k-e64k.gbf

echo "RUN: srun -p gpu-short -A nprg042s --gres=gpu:1 ./potential ../data/v32k-e256k.gbf"
srun -p gpu-short -A nprg042s --gres=gpu:1 ./potential ../data/v32k-e256k.gbf

echo "RUN: srun -p gpu-short -A nprg042s --gres=gpu:1 ./potential ../data/v64k-e1024k.gbf"
srun -p gpu-short -A nprg042s --gres=gpu:1 ./potential ../data/v64k-e1024k.gbf