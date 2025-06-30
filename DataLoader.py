from torch.utils.data import Dataset
import torch
import os

original_to_index = {
    36: 0,
    15: 1,
    2: 2,
    1: 3,
    25: 4,
    4: 5,
    3: 6,
    17: 7,
    31: 8,
    6: 9,
    37: 10,
    27: 11,
    14: 12,
    13: 13,
    33: 14,
    32: 15,
    30: 16,
    23: 17,
    24: 18,
    35: 19,
    34: 20,
    29: 21,
    16: 22,
    5: 23,
    12: 24,
    7: 25,
    21: 26,
    22: 27,
    19: 28
}

index_to_class = {
    0: "Entry",
    1: "2.5 Soms.Pike",
    2: "Back",
    3: "Forward",
    4: "3.5 Soms.Tuck",
    5: "Inward",
    6: "Reverse",
    7: "3.5 Soms.Pike",
    8: "1.5 Twists",
    9: "Arm.Back",
    10: "0.5 Som.Pike",
    11: "4.5 Soms.Tuck",
    12: "2 Soms.Pike",
    13: "1.5 Soms.Pike",
    14: "2.5 Twists",
    15: "2 Twists",
    16: "1 Twist",
    17: "2.5 Soms.Tuck",
    18: "3 Soms.Tuck",
    19: "3.5 Twists",
    20: "3 Twists",
    21: "0.5 Twist",
    22: "3 Soms.Pike",
    23: "Arm.Forward",
    24: "1 Som.Pike",
    25: "Arm.Reverse",
    26: "1.5 Soms.Tuck",
    27: "2 Soms.Tuck",
    28: "4.5 Soms.Pike"
}
# print(f"Original to Index Mapping: {index_to_class[4]}")

import pickle

class MotionDataset(Dataset):
    def __init__(self, pt_folder, file_list=None, annotation_folder='/Users/mrinalraj/Downloads/WebDownload/Preprocess/FullAnnotated'):
        self.pt_files = []

        for x, y in file_list:
            file_prefix = f"{x}_{y}"
            video_file = os.path.join(annotation_folder, f"queries_{file_prefix}.mp4")
            # print(f"Processing {file_prefix}: Video file {video_file}")
            if not os.path.exists(video_file):
                print(f" Skipping {file_prefix}: Video not found")
                continue

            # Find all .pt files that match the prefix (e.g., FINA_xyz_seg_*.pt)
            matching_pt_files = [
                f for f in os.listdir(pt_folder)
                if f.startswith(file_prefix + "_seg_") and f.endswith("_tracking.pt")
            ]

            if not matching_pt_files:
                print(f" Skipping {file_prefix}: No matching .pt files found")
                continue

            for pt_file in matching_pt_files:
                full_path = os.path.join(pt_folder, pt_file)
                if os.path.exists(full_path):
                    self.pt_files.append(full_path)

        if not self.pt_files:
            raise ValueError(" No valid .pt files found matching the provided list.")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx])
        seg_len, num_points = data['shape_info']
        pred_tracks = data['pred_tracks'].reshape(seg_len, num_points, 2)
        pred_vis = data['pred_visibility'].reshape(seg_len, num_points, 1)
        pred_tracks = pred_tracks / 512
        input_tensor = torch.cat([pred_tracks, pred_vis], dim=-1)
        label = original_to_index[int(data['label'])]
        return input_tensor, torch.tensor(label).long(), self.pt_files[idx]

def motion_collate_fn(batch):
    """
    Pads the sequences to the max T in the batch.
    """
    from torch.nn.utils.rnn import pad_sequence

    features, labels, pt_paths = zip(*batch)  # list of [T_i, 1000, 3]
    T_max = max(f.shape[0] for f in features)

    padded_features = []
    for f in features:
        pad_len = T_max - f.shape[0]
        if pad_len > 0:
            pad = torch.zeros((pad_len, f.shape[1], f.shape[2]))
            f = torch.cat([f, pad], dim=0)
        padded_features.append(f)

    features_tensor = torch.stack(padded_features)  # [B, T_max, 1000, 3]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return features_tensor, labels_tensor