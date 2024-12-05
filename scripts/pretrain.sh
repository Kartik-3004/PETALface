### ArcFace | WebFace 4M ###
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29110 train.py \
    --network swin_256new \
    --head partial_fc \
    --output /cis/home/knaraya4/PETALface/<model_save_folder> \
    --margin_list 1.0,0.5,0.0 \
    --batch-size 128 \
    --optimizer adamw \
    --weight_decay 0.05 \
    --rec /cis/home/knaraya4/data/<webface4m_dataset_rec_file> \
    --num_classes 205990 \
    --num_image 4235242 \
    --num_epoch 26 \
    --lr 0.001 \
    --fp16 \
    --warmup_epoch 1

### ArcFace | WebFace 12M ###
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29120 train.py \
    --network swin_256new \
    --head partial_fc \
    --output /cis/home/knaraya4/PETALface/<model_save_folder> \
    --margin_list 1.0,0.5,0.0 \
    --batch-size 256 \
    --optimizer adamw \
    --weight_decay 0.05 \
    --rec /cis/home/knaraya4/data/webface12m_dataset_rec_file \
    --num_classes 617970 \
    --num_image 12720066 \
    --num_epoch 20 \
    --lr 0.001 \
    --fp16 \
    --warmup_epoch 1
 
# For CosFace, --margin_list 1.0,0.0,0.4; For ArcFace, --margin_list 1.0,0.5,0.0