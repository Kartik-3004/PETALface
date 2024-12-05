### BRISQUE ###
CUDA_VISIBLE_DEVICES=0 python validation_lq/validate_tinyface_iqa.py \
    --data_root /data/knaraya4/data \
    --batch_size 512 \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --model_type swin_256new_iqa \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora \
    --iqa brisque \
    --threshold <threshold>

### CNNIQA ###
CUDA_VISIBLE_DEVICES=0 python validation_lq/validate_tinyface_iqa.py \
    --data_root /data/knaraya4/data \
    --batch_size 512 \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --model_type swin_256new_iqa \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora \
    --iqa cnniqa \
    --threshold <threshold>

### LoRA ###
CUDA_VISIBLE_DEVICES=0 python validation_lq/validate_tinyface.py \
    --data_root /data/knaraya4/data \
    --batch_size 512 \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --model_type swin_256new \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora 

### PreTrained ###
CUDA_VISIBLE_DEVICES=0 python validation_lq/validate_tinyface.py \
    --data_root /data/knaraya4/data \
    --batch_size 512 \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --model_type swin_256new \
    --image_size 120