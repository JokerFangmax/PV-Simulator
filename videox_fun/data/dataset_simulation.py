# Simulation dataset for PV-Simulator.
# Loads physics trajectory data for Stage 1 (sim-only) and Stage 2 (joint) training.
#
# Key design: bs=1 always. T_raw, n_objects, and N_i (points per object) all vary
# per sample. Use multi-GPU DDP + gradient accumulation for effective batch size.
#
# Data format: each sample is an npz/hdf5 file containing:
#   - x_s_raw: (T_raw, N, 9)  — point states (pos3 + vel3 + ang_vel3)
#   - c_force_raw: (T_raw, N, 6) — force vector (3) + contact point (3)
#   - c_floor: float — floor height
#   - c_mat: (n_objects, 2) — (friction, restitution) per object
#   - c_mass: (n_objects,) — mass per object
#   - c_static: (n_objects,) — static flag per object (0 or 1)
#   - c_init: (n_objects, 9) — initial state (pos3 + vel3 + ang_vel3) per object
#   - point_obj_idx: (N,) — maps each point to its object index
#   - text: str — text description (for Stage 2)
#   - video_path: str — path to rendered video (for Stage 2)

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    """Dataset for physics simulation trajectory data.

    Each sample is loaded as bs=1 with variable shapes. No padding is applied.
    Different samples may have different T_raw, n_objects, and N_i.

    Args:
        ann_path: Path to JSON annotation file. Each entry should have:
            - 'file_path': path to npz data file
            - 'text': text description (optional, for Stage 2)
            - 'video_path': path to video file (optional, for Stage 2)
        data_root: Root directory for data file paths.
        load_video: Whether to load video frames (for Stage 2).
        temporal_compression_ratio: Temporal compression ratio (default 4).
    """

    def __init__(
        self,
        ann_path: str,
        data_root: str = None,
        load_video: bool = False,
        temporal_compression_ratio: int = 4,
    ):
        print(f"Loading simulation annotations from {ann_path} ...")
        self.dataset = json.load(open(ann_path, 'r'))
        self.length = len(self.dataset)
        print(f"Simulation data scale: {self.length}")

        self.data_root = data_root
        self.load_video = load_video
        self.temporal_compression_ratio = temporal_compression_ratio

    def __len__(self):
        return self.length

    def _get_data_path(self, file_path):
        if self.data_root is not None:
            return os.path.join(self.data_root, file_path)
        return file_path

    def _load_npz(self, idx):
        """Load a simulation data sample from npz file."""
        entry = self.dataset[idx]
        data_path = self._get_data_path(entry['file_path'])
        data = np.load(data_path, allow_pickle=True)
        return data, entry

    def __getitem__(self, idx):
        while True:
            try:
                data, entry = self._load_npz(idx)

                # Point states: (T_raw, N, 9)
                x_s_raw = torch.from_numpy(data['x_s_raw'].astype(np.float32))
                T_raw, N, _ = x_s_raw.shape

                # Validate T_raw is compatible with 4x temporal compression
                # T_raw should be 4*(T-1)+1 for some integer T
                assert (T_raw - 1) % (self.temporal_compression_ratio) == 0, \
                    f"T_raw={T_raw} is not compatible with temporal_compression_ratio={self.temporal_compression_ratio}"
                T = (T_raw - 1) // self.temporal_compression_ratio + 1

                # Force: (T_raw, N, 6)
                if 'c_force_raw' in data:
                    c_force_raw = torch.from_numpy(data['c_force_raw'].astype(np.float32))
                else:
                    c_force_raw = torch.zeros(T_raw, N, 6)

                # Floor height: scalar
                c_floor = torch.tensor(float(data['c_floor']))

                # Per-object properties
                c_mat = torch.from_numpy(data['c_mat'].astype(np.float32))       # (n_objects, 2)
                c_mass = torch.from_numpy(data['c_mass'].astype(np.float32))     # (n_objects,)
                c_static = torch.from_numpy(data['c_static'].astype(np.int64))   # (n_objects,)
                n_objects = c_mat.shape[0]

                # Object IDs: 0..n_objects-1
                c_id = torch.arange(n_objects, dtype=torch.long)

                # Initial state: (n_objects, 9) — pos3 + vel3 + ang_vel3
                c_init_state = torch.from_numpy(data['c_init'].astype(np.float32))  # (n_objects, 9)
                # Append condition mask: 1 for the first frame (conditioned), 0 otherwise
                # Shape becomes (n_objects, 10)
                c_init_mask = torch.ones(n_objects, 1)
                c_init = torch.cat([c_init_state, c_init_mask], dim=-1)

                # Point-to-object mapping: (N,)
                point_obj_idx = torch.from_numpy(data['point_obj_idx'].astype(np.int64))

                sample = {
                    'x_s_raw': x_s_raw,                    # (T_raw, N, 9)
                    'c_force_raw': c_force_raw,             # (T_raw, N, 6)
                    'c_floor': c_floor,                     # scalar
                    'c_id': c_id,                           # (n_objects,)
                    'c_mat': c_mat,                         # (n_objects, 2)
                    'c_mass': c_mass,                       # (n_objects,)
                    'c_static': c_static,                   # (n_objects,)
                    'c_init': c_init,                       # (n_objects, 10)
                    'point_obj_idx': point_obj_idx,         # (N,)
                    'T': T,                                 # int
                    'T_raw': T_raw,                         # int
                    'N': N,                                 # int
                    'n_objects': n_objects,                  # int
                }

                # Optional fields for Stage 2
                if 'text' in entry:
                    sample['text'] = entry['text']
                if self.load_video and 'video_path' in entry:
                    sample['video_path'] = self._get_data_path(entry['video_path'])

                return sample

            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                idx = random.randint(0, self.length - 1)


def sim_collate_fn(batch):
    """Collate function for SimulationDataset.

    Since bs=1, this simply adds a batch dimension to each tensor.
    Does NOT pad variable-length dimensions.
    """
    assert len(batch) == 1, "SimulationDataset requires batch_size=1"
    sample = batch[0]
    result = {}
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            result[key] = val.unsqueeze(0)  # add batch dim
        else:
            result[key] = val
    return result
