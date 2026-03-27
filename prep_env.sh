salloc -N 1 --gpus=1 --cpus-per-gpu=4 --mem=40gb --time=08:00:00 --qos=dw87
module load cuda/12.8
mamba activate bayes_cu128
export HF_HUB_OFFLINE=1
cd repositories/bayes_design_multi_constraint/
