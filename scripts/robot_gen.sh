#! /bin/bash

# -------------------------------
# Robot Point Cloud Generation
# -------------------------------

cate="robot"
dims="512-512-512"
latent_dims="256-256"
num_blocks=1
latent_num_blocks=1
zdim=128
batch_size=16
lr=2e-3
epochs=1000
ds=robot
log_name="gen/${ds}-cate${cate}"
data_dir="./../dataset_cr/pointcloud_15k/"

python train_cr.py \
    --log_name ${log_name} \
    --lr ${lr} \
    --dataset_type ${ds} \
    --data_dir ${data_dir} \
    --cates ${cate} \
    --dims ${dims} \
    --latent_dims ${latent_dims} \
    --num_blocks ${num_blocks} \
    --latent_num_blocks ${latent_num_blocks} \
    --batch_size ${batch_size} \
    --zdim ${zdim} \
    --epochs ${epochs} \
    --use_adjoint True \
    --rtol 1e-3 --atol 1e-3 \
    --save_freq 10 --viz_freq 1 --log_freq 1 --val_freq 1 \
    --use_latent_flow \
    --no_validation

echo "Robot dataset training finished!"
exit 0
