python -m torch.distributed.launch --nproc_per_node=1 --use_env NextQA.py \
    --config ./configs/nextqa.yaml \
    --output_dir output/nextqa \
    --checkpoint ckpt/ALBEF.pth
