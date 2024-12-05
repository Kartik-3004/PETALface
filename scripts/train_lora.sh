### CosFace | TinyFace ###
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29190 train.py \
    --network swin_256new_iqa \
    --head partial_fc \
    --output /mnt/store/knaraya4/PETALface/<model_save_folder> \
    --margin_list 1.0,0.0,0.4 \
    --batch-size 8 \
    --optimizer adamw \
    --weight_decay 0.1 \
    --rec /data/knaraya4/data/<folder_to_dataset_rec_file> \
    --num_classes 2570 \
    --num_image 7804 \
    --num_epoch 50 \
    --lr 0.0005 \
    --fp16 \
    --warmup_epoch 2 \
    --image_size 120 \
    --use_lora \
    --lora_rank 8 \
    --seed 19 \
    --load_pretrained /mnt/store/knaraya4/PETALface/<path_to_pretrained_model>

###
# For CosFace, --margin_list 1.0,0.0,0.4; For ArcFace, --margin_list 1.0,0.5,0.0
# For TinyFace,
#   --num_classes 2570
#   --num_image 7804
#   --num_epoch 50
#   --warmup_epoch 2
# For BRIAR,
#   --num_classes 778
#   --num_image 301000
#   --num_epoch 10
#   --warmup_epoch 1
###