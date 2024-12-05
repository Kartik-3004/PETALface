### CNNIQA ###
CUDA_VISIBLE_DEVICES=0 python validation_hq/validate_hq_iqa.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/<hq_dataset_folder> \
    --model_type swin_256new_iqa  \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora \
    --iqa cnniqa \
    --threshold threshold

### BRISQUE ###
CUDA_VISIBLE_DEVICES=0 python validation_hq/validate_hq_iqa.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/<hq_dataset_folder> \
    --model_type swin_256new_iqa  \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora \
    --iqa brisque \
    --threshold 0.5

### LoRA ###
CUDA_VISIBLE_DEVICES=0 python validation_hq/validate_hq.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/<hq_dataset_folder> \
    --model_type swin_256new  \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora

### PreTrained ###
CUDA_VISIBLE_DEVICES=0 python validation_hq/validate_hq.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/<hq_dataset_folder> \
    --model_type swin_256new  \
    --image_size 120
