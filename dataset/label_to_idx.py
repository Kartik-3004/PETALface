import mxnet as mx
import numpy as np
import pickle
import os

def create_label_to_indices_mapping(rec_file, idx_file, output_file):
    print("Creating label to indices mapping...")
    imgrec = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'r')
    label_to_indices = {}

    for i in range(len(imgrec.keys)):
        idx = imgrec.keys[i]
        header, _ = mx.recordio.unpack(imgrec.read_idx(idx))
        label = int(header.label) if isinstance(header.label, float) else int(header.label[0])
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)

    print("Saving mapping to", output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(label_to_indices, f)

    print("Done.")

# Specify your RecordIO files and output .pkl file
rec_file = '/mnt/store/knaraya4/data/WebFace4M/train.rec'
idx_file = '/mnt/store/knaraya4/data/WebFace4M/train.idx'
output_file = '/mnt/store/knaraya4/data/WebFace4M/train.pkl'

# Make sure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

create_label_to_indices_mapping(rec_file, idx_file, output_file)
