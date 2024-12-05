import datetime
import os
import pickle

import mxnet as mx
import numpy as np
import sklearn
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import sys
sys.path.insert(0, os.getcwd())
from backbones import get_model
from utils.utils_logging import init_logging_test

def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]

class BinDataset(Dataset):
    def __init__(self, bins, image_size):
        self.bins = bins
        self.image_size = image_size

    def __len__(self):
        return len(self.bins)

    def __getitem__(self, idx):
        _bin = self.bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != self.image_size[0]:
            img = mx.image.resize_short(img, self.image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        img = torch.from_numpy(img.asnumpy())
        return img

def load_bin(path, image_size, batch_size, num_workers=4):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3

    dataset = BinDataset(bins, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    return dataloader, issame_list

@torch.no_grad()
def test(dataloader, issame_list, backbone, batch_size, device='cuda', nfolds=10):
    print('Testing verification...')
    embeddings_list = []
    time_consumed = 0.0

    for flip in [0, 1]:
        embeddings = []
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            if flip == 1:
                batch = torch.flip(batch, [3])
            img = ((batch / 255) - 0.5) / 0.5
            time0 = datetime.datetime.now()
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            embeddings.append(_embeddings)
        embeddings = np.concatenate(embeddings, axis=0)
        embeddings_list.append(embeddings)

    _xnorm = np.mean([np.linalg.norm(embed) for embed in np.concatenate(embeddings_list, axis=0)])

    embeddings = sklearn.preprocessing.normalize(embeddings_list[0])
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings + sklearn.preprocessing.normalize(embeddings_list[1])
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print('Infer time:', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list

class ValidateHQ:
    def __init__(self, val_targets, rec_prefix, im_size, batch_size=32, num_workers=4):
        self.ver_list = []
        self.ver_name_list = []
        self.issame_list = []
        self.image_size = (im_size, im_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.init_dataset(val_targets, rec_prefix)

    def ver_test(self, backbone, log_root):
        for i, data_set in enumerate(self.ver_list):
            acc1, std1, acc2, std2, xnorm, embeddings_list = test(data_set, self.issame_list[i], backbone, self.batch_size, device='cuda', nfolds=10)
            log_root.info(f'[{self.ver_name_list[i]}] Accuracy: {acc2:.5f}±{std2:.5f}')

    def init_dataset(self, val_targets, data_dir):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                dataloader, issame_list = load_bin(path, self.image_size, self.batch_size, self.num_workers)
                self.ver_list.append(dataloader)
                self.issame_list.append(issame_list)
                self.ver_name_list.append(name)

    def __call__(self, backbone, log_root):
        backbone.eval()
        self.ver_test(backbone, log_root)

def load_pretrained_model(model, model_name, gpu, lora_rank, lora_scale, use_lora):
    ckpt_path = model_name
    model = get_model(model, dropout=0.0, fp16=False, num_features=512, r=lora_rank, scale=lora_scale, use_lora=use_lora)
    statedict = torch.load(ckpt_path, map_location=torch.device('cuda:' + str(gpu)))
    model.load_state_dict(statedict)
    model.eval()
    return model

def get_save_path(model_load_path):
    directory, _ = os.path.split(model_load_path)
    results_save_path = os.path.join(directory, 'results')
    return results_save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validation High Quality Dataset')
    parser.add_argument('--data_root', default='/mnt/store/knaraya4/data/HQ_val')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--val_targets', default='lfw,cfp_fp,cplfw,agedb_30,calfw,cfp_ff')
    parser.add_argument('--model_load_path', type=str)
    parser.add_argument('--model_type', type=str, default='r50')
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_scale', type=int, default=1)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--image_size', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    args.val_targets = [str(x) for x in args.val_targets.split(',')]

    model_load_path = args.model_load_path
    print("Model Load Path", model_load_path)
    model = load_pretrained_model(args.model_type, model_load_path, args.gpu, args.lora_rank, args.lora_scale, args.use_lora)
    model.to(f'cuda:{args.gpu}')

    save_path = get_save_path(model_load_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('Save Path:', save_path)

    log_root = init_logging_test(0, save_path)
    log_root.info("---" * 15)
    for arg, value in vars(args).items():
        log_root.info(f"{arg}: {value}")
    log_root.info("--" * 15)

    verification = ValidateHQ(val_targets=args.val_targets, rec_prefix=args.data_root, im_size=args.image_size, batch_size=args.batch_size, num_workers=args.num_workers)
    verification(model, log_root)
