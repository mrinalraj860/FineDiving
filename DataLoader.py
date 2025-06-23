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

class MotionDataset(Dataset):
    def __init__(self, pt_folder):
        self.pt_files = [os.path.join(pt_folder, f) for f in os.listdir(pt_folder) if f.endswith('.pt')]

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx])
        seg_len, num_points = data['shape_info']
        
        pred_tracks = data['pred_tracks'].reshape(seg_len, num_points, 2)      # [T, 1000, 2]
        pred_vis = data['pred_visibility'].reshape(seg_len, num_points, 1)     # [T, 1000, 1]

        # Normalize x, y to [0, 1]
        pred_tracks = pred_tracks / 512  # Assuming max width-height is 512

        input_tensor = torch.cat([pred_tracks, pred_vis], dim=-1)  # [T, 1000, 3]
        # print(f"Loaded {self.pt_files[idx]}: shape {input_tensor.shape}")
        label = int(data['label'])
        # print(f"Original Label: {label}")
        label = original_to_index[label]
        label = torch.tensor(label).long()
        # print(f"Label: {label} ({index_to_class[label.item()]})")
        return input_tensor, label, self.pt_files[idx]  # return path for visualization
    


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