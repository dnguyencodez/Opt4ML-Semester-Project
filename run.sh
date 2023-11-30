export PYTHONPATH="$PYTHONPATH:./bimodal_exps"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

data_path=./datasets
ann_path=./clip_train
train_image_root=cc3m_subset_100k/
data=cc3m
train_file=${data}_train_subset.json
gamma=0.8
epochs=30

echo "Starting clip knn training"

CUDA_VISIBLE_DEVICES=0 python ./bimodal_exps/clip.py \
    --data_path ${data_path} \
    --ann_path ${ann_path} \
    --train_file ${train_file} \
    --train_image_root ${train_image_root} \
    --output_dir output/clip_knn_${data}_g${gamma}_e${epochs}_2 \
    --init_model \
    --use_amp \
    --ita_type clip_knn \
    --tau_init 0.01 \
    --learnable_temp \
    --sogclr_gamma ${gamma} \
    --eta_init 0.03 --sched cosine \
    --no-distributed \
    --epochs ${epochs}