#! /bin/bash
data_dir="./../dataset_cr/pointcloud_15k/"
python demo_cr.py \
    --cates airplane \
    --data_dir ${data_dir} \
    --resume_checkpoint checkpoints/gen/robot-caterobot/checkpoint-latest.pt \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_latent_flow \
    --num_sample_shapes 20 \
    --num_sample_points 1024