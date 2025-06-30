# import torch, os
# from torch.utils.data import Dataset
# from collections import defaultdict

# import torch, os
# from torch.utils.data import Dataset
# from collections import defaultdict

# class DiveDifficultyDataset(Dataset):
#     """
#     Returns
#         tracks     :  [T_total, N, 2]   (float32, normalized coords)
#         visibility :  [T_total, N]      (float32)
#         difficulty :  scalar (float32)
#         video_id   :  str
#     """
#     def __init__(self, pt_dir: str, all_difficulties: dict, max_points=1000):
#         self.pt_dir = pt_dir
#         self.groups = defaultdict(list)
#         self.diff = {}

#         # Step 1: Group files by video ID (everything before _seg_)
#         for fname in os.listdir(pt_dir):
#             if fname.endswith(".pt") and "_seg_" in fname:
#                 video_id = fname.rsplit("_seg_", 1)[0]
#                 full_path = os.path.join(pt_dir, fname)
#                 self.groups[video_id].append(full_path)

#         # Step 2: Assign difficulty to each video using the first matching .pt file
#         for video_id, paths in self.groups.items():
#             # Find a matching key in all_difficulties
#             matching_keys = [k for k in all_difficulties if k.startswith(video_id)]
#             if matching_keys:
#                 # self.diff[video_id] = float(all_difficulties[matching_keys[0]])
#                 val = all_difficulties[matching_keys[0]]

#                 # Convert '[3.4]' or [3.4] to float
#                 if isinstance(val, str) and val.startswith("["):
#                     val = eval(val)[0]  # Or use ast.literal_eval for safety
#                 elif isinstance(val, (list, tuple)):
#                     val = val[0]

#                 self.diff[video_id] = float(val)

#         # Step 3: Filter out any videos without difficulty labels
#         self.video_ids = [v for v in sorted(self.groups) if v in self.diff]

#         print(f"Loaded {len(self.video_ids)} video samples with valid difficulty labels.")
#         if not self.video_ids:
#             raise RuntimeError("No videos matched difficulty map!")

#     def __len__(self):
#         return len(self.video_ids)

#     def _load_and_concat(self, pt_paths):
#         tracks_list, vis_list = [], []
#         for p in sorted(pt_paths):
#             d = torch.load(p, map_location="cpu")
#             T, N = d["shape_info"]

#             tr = d["pred_tracks"].reshape(T, N, 2) / 512.0
#             vi = d["pred_visibility"].reshape(T, N)

#             tracks_list.append(tr)
#             vis_list.append(vi)

#         return torch.cat(tracks_list, 0), torch.cat(vis_list, 0)

#     def __getitem__(self, idx):
#         video_id = self.video_ids[idx]
#         pt_paths = self.groups[video_id]
#         tracks, vis = self._load_and_concat(pt_paths)

#         difficulty = torch.tensor(self.diff[video_id], dtype=torch.float32)
#         return tracks, vis, difficulty, video_id

# def pad_collate(batch):
#     """
#     Pads along time axis so every sample in the batch has same T.
#     Returns:
#         tracks : [B, T_max, N, 2]
#         vis    : [B, T_max, N]
#         diff   : [B]  (float)
#         lengths: [B]  (original T)
#         vids   : list[str]
#     """
#     tracks, vis, diff, vids = zip(*batch)
#     lengths = [t.shape[0] for t in tracks]
#     T_max   = max(lengths)
#     N       = tracks[0].shape[1]

#     def _pad(tensor, fill=0.0):
#         pad_T = T_max - tensor.shape[0]
#         if pad_T == 0: return tensor
#         pad_shape = (pad_T, *tensor.shape[1:])
#         return torch.cat([tensor, torch.full(pad_shape, fill, dtype=tensor.dtype)], 0)

#     tracks_pad = torch.stack([_pad(t) for t in tracks])   # [B, T_max, N, 2]
#     vis_pad    = torch.stack([_pad(v) for v in vis])      # [B, T_max, N]

#     return tracks_pad, vis_pad, torch.stack(diff), torch.tensor(lengths), list(vids)


import torch, os, ast
from torch.utils.data import Dataset
from collections import defaultdict

class DiveDifficultyDataset(Dataset):
    """
    Dataset for dive difficulty regression.

    Returns:
        tracks     : [T_total, N, 2]   (float32)
        visibility : [T_total, N]      (float32)
        difficulty : scalar (float32)
        video_id   : str
    """
    def __init__(self, pt_dir: str, all_difficulties: dict, max_points=1000):
        self.pt_dir = pt_dir
        self.groups = defaultdict(list)
        self.diff = {}

        # Step 1: Group .pt files by video ID prefix
        for fname in os.listdir(pt_dir):
            if fname.endswith(".pt") and "_seg_" in fname:
                video_id = fname.rsplit("_seg_", 1)[0]
                full_path = os.path.join(pt_dir, fname)
                self.groups[video_id].append(full_path)

        # Step 2: Assign difficulty from all_difficulties
        for video_id, paths in self.groups.items():
            matching_keys = [k for k in all_difficulties if k.startswith(video_id)]
            if not matching_keys:
                continue

            val = all_difficulties[matching_keys[0]]

            # Safely parse '[3.4]' or [3.4] → float
            try:
                if isinstance(val, str) and val.startswith("["):
                    val = ast.literal_eval(val)[0]
                elif isinstance(val, (list, tuple)):
                    val = val[0]
                val = float(val)
                self.diff[video_id] = val
            except Exception as e:
                print(f"⚠️ Failed to parse difficulty for {video_id}: {val} ({e})")

        # Step 3: Final filtered list of video IDs
        self.video_ids = [v for v in sorted(self.groups) if v in self.diff]

        print(f"✅ Loaded {len(self.video_ids)} video samples with valid difficulty labels.")
        if not self.video_ids:
            raise RuntimeError("❌ No videos matched difficulty map!")

    def __len__(self):
        return len(self.video_ids)

    def _load_and_concat(self, pt_paths):
        tracks_list, vis_list = [], []
        for p in sorted(pt_paths):
            d = torch.load(p, map_location="cpu")
            T, N = d["shape_info"]
            tr = d["pred_tracks"].reshape(T, N, 2) / 512.0
            vi = d["pred_visibility"].reshape(T, N)
            tracks_list.append(tr)
            vis_list.append(vi)
        return torch.cat(tracks_list, dim=0), torch.cat(vis_list, dim=0)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        pt_paths = self.groups[video_id]
        tracks, vis = self._load_and_concat(pt_paths)
        difficulty = torch.tensor(self.diff[video_id], dtype=torch.float32)
        return tracks, vis, difficulty, video_id
    


def pad_collate(batch):
    """
    Pads each sample in the batch along the time axis to max T.
    Returns:
        tracks  : [B, T_max, N, 2]
        vis     : [B, T_max, N]
        diff    : [B]
        lengths : [B] (original time lengths)
        vids    : list of str
    """
    tracks, vis, diff, vids = zip(*batch)
    lengths = [t.shape[0] for t in tracks]
    T_max = max(lengths)
    N = tracks[0].shape[1]

    def _pad(tensor, fill=0.0):
        pad_T = T_max - tensor.shape[0]
        if pad_T == 0:
            return tensor
        pad_shape = (pad_T, *tensor.shape[1:])
        return torch.cat([tensor, torch.full(pad_shape, fill, dtype=tensor.dtype)], dim=0)

    tracks_pad = torch.stack([_pad(t) for t in tracks])  # [B, T_max, N, 2]
    vis_pad    = torch.stack([_pad(v) for v in vis])     # [B, T_max, N]
    diff_tensor = torch.tensor(diff, dtype=torch.float32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32)

    return tracks_pad, vis_pad, diff_tensor, lengths_tensor, list(vids)