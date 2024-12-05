### IJBC BRISQUE ###
CUDA_VISIBLE_DEVICES=0 python validation_ijb/eval_ijb_iqa.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/ijb/<ijbc_dataset_folder> \
    --batch-size 1024 \
    --model_type swin_256new_iqa \
    --target IJBC \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora \
    --iqa brisque \
    --threshold <threshold>

### IJBB BRISQUE ###
CUDA_VISIBLE_DEVICES=0 python validation_ijb/eval_ijb_iqa.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/<ijbb_dataset_folder> \
    --batch-size 1024 \
    --model_type swin_256new_iqa \
    --target IJBB \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora \
    --iqa brisque \
    --threshold <threshold>

### IJBC CNNIQA ###
CUDA_VISIBLE_DEVICES=0 python validation_ijb/eval_ijb_iqa.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/ijb/<ijbc_dataset_folder> \
    --batch-size 1024 \
    --model_type swin_256new_iqa \
    --target IJBC \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora \
    --iqa cnniqa \
    --threshold <threshold>

### IJBB CNNIQA ###
CUDA_VISIBLE_DEVICES=0 python validation_ijb/eval_ijb_iqa.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/<ijbb_dataset_folder> \
    --batch-size 1024 \
    --model_type swin_256new_iqa \
    --target IJBB \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora \
    --iqa cnniqa \
    --threshold <threshold>


### IJBC LoRA ###
CUDA_VISIBLE_DEVICES=0 python validation_ijb/eval_ijb.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/ijb/<ijbc_dataset_folder> \
    --batch-size 1024 \
    --model_type swin_256new \
    --target IJBC \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora 

### IJBB LoRA ###
CUDA_VISIBLE_DEVICES=0 python validation_ijb/eval_ijb.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/<ijbb_dataset_folder> \
    --batch-size 1024 \
    --model_type swin_256new \
    --target IJBB \
    --image_size 120 \
    --lora_rank 8 \
    --use_lora 

### IJBC PreTrained ###
CUDA_VISIBLE_DEVICES=0 python validation_ijb/eval_ijb.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/ijb/<ijbc_dataset_folder> \
    --batch-size 1024 \
    --model_type swin_256new \
    --target IJBC \
    --image_size 120

### IJBB PreTrained ###
CUDA_VISIBLE_DEVICES=0 python validation_ijb/eval_ijb.py \
    --model_load_path /mnt/store/knaraya4/PETALface/<folder_name>/model.pt \
    --data_root /data/knaraya4/data/<ijbb_dataset_folder> \
    --batch-size 1024 \
    --model_type swin_256new \
    --target IJBB \
    --image_size 120