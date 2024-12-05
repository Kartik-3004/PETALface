### BRISQUE ###
CUDA_VISIBLE_DEVICES=0 python validation_lq/validate_ijbs_iqa.py \
    --data_root /mnt/store/knaraya4/data/<ijbs_dataset_folder> \
    --batch_size 2048 \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --model_type swin_256new_iqa \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora \
    --iqa brisque \
    --threshold <threshold>


### CNNIQA ###
CUDA_VISIBLE_DEVICES=0 python validation_lq/validate_ijbs_iqa.py \
    --data_root /mnt/store/knaraya4/data/<ijbs_dataset_folder> \
    --batch_size 2048 \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --model_type swin_256new_iqa \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora \
    --iqa cnniqa \
    --threshold <threshold>

### LoRA ###
CUDA_VISIBLE_DEVICES=0 python validation_lq/validate_ijbs.py \
    --data_root /mnt/store/knaraya4/data/<ijbs_dataset_folder> \
    --batch_size 2048 \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --model_type swin_256new \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora

### PreTrained ###
CUDA_VISIBLE_DEVICES=0 python validation_lq/validate_ijbs.py \
    --data_root /mnt/store/knaraya4/data/<ijbs_dataset_folder> \
    --batch_size 2048 \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --model_type swin_256new \
    --image_size 120