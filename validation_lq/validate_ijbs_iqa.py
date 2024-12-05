import torch
import numpy as np
from tqdm import tqdm
import data_utils
import argparse
import pandas as pd
import evaluate_helper
import sys, os
sys.path.insert(0, os.getcwd())
from backbones import get_model
import pyiqa

def str2bool(v):
    return v.lower() in ("true", "t", "1")


def load_pretrained_model(model, model_name, gpus, lora_rank, lora_scale, use_lora):
    # load model and pretrained statedict
    ckpt_path = model_name
    model = get_model(model, dropout=0.0, fp16=False, num_features=512, r=lora_rank, scale=lora_scale, use_lora=use_lora)

    statedict = torch.load(ckpt_path, map_location=torch.device('cuda:' + str(gpus[0])))
    model.load_state_dict(statedict)
    model = torch.nn.DataParallel(model, device_ids=gpus)
    model.eval()
    return model


def get_save_path(model_load_path):
    directory, _ = os.path.split(model_load_path)
    results_save_path = os.path.join(directory, 'results')
    
    return results_save_path

def generate_alpha(img, iqa, thresh):
    device = img.device
    BS, C, H, W = img.shape
    alpha = torch.zeros((BS, 1), dtype=torch.float32, device=device)

    score = iqa(img)
    threshold = thresh
    for i in range(BS):
        if score[i] == threshold:
            alpha[i] = 0.5
        elif score[i] < threshold:
            alpha[i] = 0.5 - (threshold - score[i])
        else:
            alpha[i] = 0.5 + (score[i] - threshold)
    return alpha

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--data_root", type=str, default='/mnt/store/knaraya4/data/ijbs_aligned_180')
    parser.add_argument('--model_type', type=str, default='r50')
    parser.add_argument('--model_load_path', type=str, default='ijcb_cosface')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--image_size', type=int, default=120)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_scale', type=int, default=1)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='List of GPU ids')
    parser.add_argument('--fuse_match_method', type=str, default='pre_norm_vector_add_cos',
                        choices=('pre_norm_vector_add_cos'))
    parser.add_argument('--save_features', type=bool, default=True)
    parser.add_argument('--csv_path', type=str, default='/mnt/store/knaraya4/data/IJBS/image_paths_180.csv')
    parser.add_argument('--iqa', type=str, default='brisque')
    parser.add_argument('--threshold', type=float, default=0.5) 
    args = parser.parse_args()

    # load model
    model_load_path = args.model_load_path
    model = load_pretrained_model(args.model_type, model_load_path, args.gpus, args.lora_rank, args.lora_scale, args.use_lora)
    model.to('cuda:{}'.format(args.gpus[0]))

    # make result save root
    save_root = get_save_path(model_load_path)
    os.makedirs(save_root, exist_ok=True)
    image_path_df = pd.read_csv(args.csv_path, index_col=0)
    all_image_paths = image_path_df['path'].apply(lambda x: os.path.join(args.data_root, x)).tolist()

    num_partition = 100
    dataset_split = np.array_split(all_image_paths, num_partition)

    print('total {} images'.format(len(all_image_paths)))
    all_features = []
    device = "cuda:" + str(gpu)
    if iqa == "brisque":
        iqa = pyiqa.create_metric('brisque').cuda()
    elif iqa == "cnniqa":
        iqa = pyiqa.create_metric('cnniqa').cuda()
    threshold = args.threshold
    for partition_idx in tqdm(range(num_partition)):

        image_paths = list(dataset_split[partition_idx])
        dataloader = data_utils.prepare_imagelist_dataloader(image_paths, batch_size=args.batch_size, image_size=args.image_size, num_workers=8)

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()

        features = []
        norms = []
        prev_max_idx = 0
        with torch.no_grad():
            for iter_idx, (img, idx) in enumerate(dataloader):
                assert idx.max().item() > prev_max_idx
                prev_max_idx = idx.max().item()  # order shifting by dataloader checking
                if iter_idx % 100 == 0:
                    print(f"{iter_idx} / {len(dataloader)} done")
                img = img.to('cuda:{}'.format(args.gpus[0]), non_blocking=True)
                alpha = generate_alpha(img, iqa, threshold)
                feature = model(img, alpha)

                if isinstance(feature, tuple) and len(feature) == 2:
                    feature, norm = feature
                    features.append(feature.cpu().numpy())
                    norms.append(norm.cpu().numpy())
                else:
                    norm = torch.norm(feature, 2, 1, True)
                    features.append(feature.cpu().numpy())
                    norms.append(norm.cpu().numpy())

        features = np.concatenate(features, axis=0)
        if args.save_features:
            save_path = os.path.join(save_root, 'feature_extracted/ijbs_pred_{}_{}.npy'.format(args.model_type, partition_idx))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, features)

        if len(norms) > 0:
            norms = np.concatenate(norms, axis=0)
            if args.save_features:
                save_path = os.path.join(save_root, 'feature_extracted/ijbs_pred_{}_norm_{}.npy'.format(args.model_type, partition_idx))
                np.save(save_path, norms)

        if args.fuse_match_method == 'pre_norm_vector_add_cos':
            features = features * norms
        all_features.append(features)
    all_features = np.concatenate(all_features, axis=0)

    # prepare savedir
    os.makedirs(os.path.join(save_root, 'eval_result'), exist_ok=True)
    # evaluate
    evaluate_helper.run_eval_with_features(save_root=save_root,
                                    features=all_features,
                                    image_paths=all_image_paths,
                                    get_retrievals=True,
                                    fuse_match_method=args.fuse_match_method,
                                    ijbs_proto_path=None)