#! /bin/bash
data_dir="ShapeNetCore.v2.PC15k"
python demo.py \
    --cates airplane \
    --data_dir ${data_dir} \
    --resume_checkpoint checkpoints/gen/shapenet15k-cateairplane/checkpoint-latest.pt \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_latent_flow \
    --num_sample_shapes 20 \
    --num_sample_points 4096

