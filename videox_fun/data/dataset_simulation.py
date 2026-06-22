# Simulation dataset for PV-Simulator.
# Loads physics trajectory data for Stage 1 (sim-only) and Stage 2 (joint) training.
#
# Data format (SimulationDataset, npz-based):
#   - x_s_raw: (T_raw, N, 6)  — point states (pos3 + vel3)
#   - c_force_raw: (T_raw, N, 6) — force vector (3) + contact point (3)
#   - c_floor: float — floor height
#   - c_mat: (n_objects, 2) — (friction, restitution) per object
#   - c_mass: (n_objects,) — mass per object
#   - c_static: (n_objects,) — static flag per object (0 or 1)
#   - c_init: (n_objects, 7) — initial state (pos3 + vel3 + mask1) per object
#   - point_obj_idx: (N,) — maps each point to its object index
#   - text: str — text description (for Stage 2)
#   - video_path: str — path to rendered video (for Stage 2)

import json
import io
import os
import pickle
import random
import tarfile

import numpy as np
import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    """Dataset for physics simulation trajectory data loaded from npz files.

    Each sample may have different T_raw, n_objects, and N_i (points per object).
    When batching with batch_size > 1, all samples in a batch must have identical
    shapes — the DataLoader will raise an error otherwise.

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

                # Point states: (T_raw, N, 6) — pos3 + vel3
                x_s_raw = torch.from_numpy(data['x_s_raw'].astype(np.float32))
                T_raw, N, _ = x_s_raw.shape

                # Validate T_raw is compatible with temporal compression: T_raw = 4k+1
                assert (T_raw - 1) % self.temporal_compression_ratio == 0, \
                    f"T_raw={T_raw} is not compatible with temporal_compression_ratio={self.temporal_compression_ratio}"

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

                # Initial state: (n_objects, 7) — pos3 + vel3 + mask1
                c_init_state = torch.from_numpy(data['c_init'].astype(np.float32))  # (n_objects, 6)
                c_init_mask = torch.ones(n_objects, 1)
                c_init = torch.cat([c_init_state, c_init_mask], dim=-1)              # (n_objects, 7)

                # Point-to-object mapping: (N,)
                point_obj_idx = torch.from_numpy(data['point_obj_idx'].astype(np.int64))

                sample = {
                    'x_s_raw': x_s_raw,                    # (T_raw, N, 6)
                    'c_force_raw': c_force_raw,             # (T_raw, N, 6)
                    'c_floor': c_floor,                     # scalar
                    'c_id': c_id,                           # (n_objects,)
                    'c_mat': c_mat,                         # (n_objects, 2)
                    'c_mass': c_mass,                       # (n_objects,)
                    'c_static': c_static,                   # (n_objects,)
                    'c_init': c_init,                       # (n_objects, 7)
                    'point_obj_idx': point_obj_idx,         # (N,)
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


class MoviSimulationDataset(Dataset):
    """Dataset for the MOVI-AB style simulation data (directory-based format).

    Each sample is a subdirectory containing:
      - point_cloud_states.pkl: point states (T_raw_avail, N, 6), instance point ranges
      - metadata.json: per-instance friction, restitution, mass, positions, velocities

    Frames are clipped to the largest valid T_raw = 4k+1 ≤ T_raw_available.

    Each sample may have different n_objects and N (total points). When batching with
    batch_size > 1, all samples in a batch must have identical shapes — the DataLoader
    will raise an error otherwise.

    Args:
        data_root: Root directory containing subdirectories (one per sample).
        max_objects: Skip samples with more than this many objects.
        temporal_compression_ratio: Temporal compression ratio (default 4).
    """

    def __init__(
        self,
        data_root: str,
        max_objects: int = 16,
        temporal_compression_ratio: int = 4,
    ):
        self.data_root = data_root
        self.max_objects = max_objects
        self.temporal_compression_ratio = temporal_compression_ratio
        self._shard_tars = {}
        self._shard_members = {}
        # DataLoader workers are forked lazily.  If the main process has already
        # loaded a sample (for example, fixed visualization samples), workers
        # inherit these tar handles.  The inherited descriptors share a file
        # offset across processes, so concurrent reads can make an inner payload
        # look like an invalid tar archive.  Track the owning PID and reopen in
        # every worker instead of reusing inherited handles.
        self._shard_handle_pid = os.getpid()

        webdataset_root = self._resolve_webdataset_root(data_root)
        if webdataset_root is not None:
            self.backend = "webdataset"
            self.webdataset_root = webdataset_root
            manifest_path = os.path.join(webdataset_root, "manifest.jsonl")
            self.samples = []
            with open(manifest_path, "r") as f:
                for line in f:
                    record = json.loads(line)
                    self.samples.append({
                        "sample_id": record["sample_id"],
                        "shard_path": os.path.join(webdataset_root, "shards", record["shard_path"]),
                    })
            print(
                f"MoviSimulationDataset: found {len(self.samples)} sharded samples "
                f"in {webdataset_root}"
            )
        else:
            self.backend = "directory"
            # Discover all subdirectories
            all_dirs = sorted([
                d for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            ])

            # Filter: must have both required files and not too many objects
            self.samples = []
            for d in all_dirs:
                sample_dir = os.path.join(data_root, d)
                pkl_path = os.path.join(sample_dir, 'point_cloud_states.pkl')
                meta_path = os.path.join(sample_dir, 'metadata.json')
                if not os.path.exists(pkl_path) or not os.path.exists(meta_path):
                    continue
                self.samples.append(sample_dir)

            print(f"MoviSimulationDataset: found {len(self.samples)} samples in {data_root}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _resolve_webdataset_root(data_root: str):
        candidates = [
            data_root,
            os.path.join(data_root, "webdataset"),
        ]
        for candidate in candidates:
            if (
                os.path.isfile(os.path.join(candidate, "dataset_manifest.json"))
                and os.path.isfile(os.path.join(candidate, "manifest.jsonl"))
                and os.path.isdir(os.path.join(candidate, "shards"))
            ):
                return candidate
        return None

    def _build_sample_from_movi_parts(
        self, point_states, instances, meta_instances, sample_name, physics=None,
    ):
        """Convert MOVI point-cloud + metadata parts into the Stage 1 sample dict."""
        physics = {} if physics is None else physics
        point_states = physics.get('x_s_raw', point_states)
        point_states = point_states.astype(np.float32)  # (T_avail, N, 6)
        T_avail = point_states.shape[0]

        # Clip to largest valid T_raw = 4k+1
        r = self.temporal_compression_ratio
        k = (T_avail - 1) // r
        T_raw = k * r + 1
        assert T_raw >= r + 1, f"Too few frames in {sample_name}: T_avail={T_avail}"
        point_states = point_states[:T_raw]  # (T_raw, N, 6)
        T_raw, N, _ = point_states.shape

        n_objects = len(instances)
        if n_objects > self.max_objects:
            raise ValueError(f"Too many objects ({n_objects}) in {sample_name}")

        # point_obj_idx: (N,) — use generator-side mapping when available.
        point_obj_idx = physics.get('point_obj_idx')
        if point_obj_idx is None:
            point_obj_idx = np.zeros(N, dtype=np.int64)
            for obj_i, inst in enumerate(instances):
                start, end = inst['point_range']
                point_obj_idx[start:end] = obj_i
        else:
            point_obj_idx = np.asarray(point_obj_idx, dtype=np.int64)
            if point_obj_idx.shape != (N,):
                raise ValueError(
                    f"Invalid point_obj_idx shape {point_obj_idx.shape} in {sample_name}; "
                    f"expected {(N,)}"
                )

        assert len(meta_instances) == n_objects, \
            f"Instance count mismatch: pkl={n_objects}, meta={len(meta_instances)}"

        c_mat = np.zeros((n_objects, 2), dtype=np.float32)    # (friction, restitution)
        c_mass = np.zeros(n_objects, dtype=np.float32)
        c_init_state = np.zeros((n_objects, 6), dtype=np.float32)  # pos3 + vel3

        for obj_i, mi in enumerate(meta_instances):
            c_mat[obj_i, 0] = mi.get('friction', 0.5)
            c_mat[obj_i, 1] = mi.get('restitution', 0.5)
            c_mass[obj_i] = mi.get('mass', 1.0)
            pos0 = mi['positions'][0]   # [x, y, z]
            vel0 = mi['velocities'][0]  # [vx, vy, vz]
            c_init_state[obj_i, :3] = pos0
            c_init_state[obj_i, 3:6] = vel0

        # c_init: pos3 + vel3 + mask1 = (n_objects, 7)
        c_init_mask = np.ones((n_objects, 1), dtype=np.float32)
        c_init = np.concatenate([c_init_state, c_init_mask], axis=-1)

        c_force_raw = physics.get('c_force_raw')
        if c_force_raw is None:
            c_force_raw = np.zeros((T_raw, N, 6), dtype=np.float32)
        else:
            c_force_raw = np.asarray(c_force_raw, dtype=np.float32)[:T_raw]
            if c_force_raw.shape != (T_raw, N, 6):
                raise ValueError(
                    f"Invalid c_force_raw shape {c_force_raw.shape} in {sample_name}; "
                    f"expected {(T_raw, N, 6)}"
                )

        if 'c_mat' in physics:
            c_mat = np.asarray(physics['c_mat'], dtype=np.float32)
        if 'c_mass' in physics:
            c_mass = np.asarray(physics['c_mass'], dtype=np.float32)
        if 'c_static' in physics:
            c_static = np.asarray(physics['c_static'], dtype=np.int64)
        else:
            c_static = np.zeros(n_objects, dtype=np.int64)
        if 'c_init' in physics:
            c_init = np.asarray(physics['c_init'], dtype=np.float32)

        if c_mat.shape != (n_objects, 2) or c_mass.shape != (n_objects,):
            raise ValueError(f"Invalid object physics shapes in {sample_name}")
        if c_static.shape != (n_objects,) or c_init.shape != (n_objects, 7):
            raise ValueError(f"Invalid static/init physics shapes in {sample_name}")
        c_floor = float(np.asarray(physics.get('c_floor', 0.0)))

        return {
            'x_s_raw': torch.from_numpy(point_states),             # (T_raw, N, 6)
            'c_force_raw': torch.from_numpy(c_force_raw),           # (T_raw, N, 6)
            'c_floor': torch.tensor(c_floor),                       # scalar
            'c_id': torch.arange(n_objects, dtype=torch.long),     # (n_objects,)
            'c_mat': torch.from_numpy(c_mat),                       # (n_objects, 2)
            'c_mass': torch.from_numpy(c_mass),                     # (n_objects,)
            'c_static': torch.from_numpy(c_static),                 # (n_objects,)
            'c_init': torch.from_numpy(c_init),                     # (n_objects, 7)
            'point_obj_idx': torch.from_numpy(point_obj_idx),       # (N,)
            'T_raw': T_raw,                                          # int
            'N': N,                                                  # int
            'n_objects': n_objects,                                  # int
        }

    def _get_shard_handle(self, shard_path):
        current_pid = os.getpid()
        if current_pid != self._shard_handle_pid:
            for inherited_tf in self._shard_tars.values():
                inherited_tf.close()
            self._shard_tars = {}
            self._shard_members = {}
            self._shard_handle_pid = current_pid

        tf = self._shard_tars.get(shard_path)
        if tf is None:
            tf = tarfile.open(shard_path, "r:")
            self._shard_tars[shard_path] = tf
            self._shard_members[shard_path] = {member.name: member for member in tf.getmembers()}
        return tf, self._shard_members[shard_path]

    def _load_webdataset_sample(self, idx):
        sample_info = self.samples[idx]
        sample_id = sample_info["sample_id"]
        shard_path = sample_info["shard_path"]

        tf, members = self._get_shard_handle(shard_path)
        payload_name = f"{sample_id}.payload.tar"
        payload_member = members[payload_name]
        with tf.extractfile(payload_member) as f:
            payload_bytes = f.read()

        with tarfile.open(fileobj=io.BytesIO(payload_bytes), mode="r:") as inner_tf:
            with inner_tf.extractfile("point_cloud_states.pkl") as f:
                pkl = pickle.load(f)
            with inner_tf.extractfile("metadata.json") as f:
                meta = json.load(f)
            physics = None
            try:
                physics_member = inner_tf.getmember("physics.npz")
            except KeyError:
                physics_member = None
            if physics_member is not None:
                with inner_tf.extractfile(physics_member) as f:
                    with np.load(io.BytesIO(f.read())) as physics_npz:
                        physics = {key: physics_npz[key] for key in physics_npz.files}

        return self._build_sample_from_movi_parts(
            point_states=pkl["point_states"],
            instances=pkl["instances"],
            meta_instances=meta["instances"],
            sample_name=f"{os.path.basename(shard_path)}:{sample_id}",
            physics=physics,
        )

    def __getitem__(self, idx):
        while True:
            try:
                if self.backend == "webdataset":
                    return self._load_webdataset_sample(idx)

                sample_dir = self.samples[idx]
                pkl_path = os.path.join(sample_dir, 'point_cloud_states.pkl')
                meta_path = os.path.join(sample_dir, 'metadata.json')

                with open(pkl_path, 'rb') as f:
                    pkl = pickle.load(f)
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                physics_path = os.path.join(sample_dir, 'physics.npz')
                physics = None
                if os.path.exists(physics_path):
                    with np.load(physics_path) as physics_npz:
                        physics = {key: physics_npz[key] for key in physics_npz.files}

                return self._build_sample_from_movi_parts(
                    point_states=pkl["point_states"],
                    instances=pkl["instances"],
                    meta_instances=meta["instances"],
                    sample_name=sample_dir,
                    physics=physics,
                )

            except Exception as e:
                print(f"Error loading sample {idx} ({self.samples[idx]}): {e}")
                idx = random.randint(0, len(self.samples) - 1)


def sim_collate_fn(batch):
    """Collate function for simulation datasets.

    Works for any batch size when all samples have identical tensor shapes.
    Stacks tensors along a new batch dimension; passes through scalar metadata.
    """
    result = {}
    sample = batch[0]
    for key in sample:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], str):
            result[key] = vals
        else:
            # int/scalar metadata — keep as list; caller can index [0] if needed
            result[key] = vals
    return result


def sim_collate_fn_padded(batch, max_T_raw: int, max_objects: int, max_points_per_object: int):
    """Padded collate function enabling batch_size > 1 with variable-length samples.

    Pads (or truncates) all samples to fixed shapes:
      - T_raw  → max_T_raw  (largest valid 4k+1 ≤ max_T_raw, then zero-pad to max_T_raw)
      - n_obj  → max_objects (truncate or pad with zeros)
      - N_i    → max_points_per_object per object (truncate or zero-pad)
      → max_N  = max_objects × max_points_per_object (total points)

    Each object occupies a fixed block of max_points_per_object slots so that the
    gather-based SimConditionEmbedder continues to work without modification.

    Returns additional keys:
      point_mask  (B, max_N)       — True for valid (non-padded) points
      obj_mask    (B, max_objects) — True for valid (non-padded) objects
      T_raw       (B,)             — actual T_raw per sample (int tensor)
      N           (B,)             — actual total valid points per sample (int tensor)
      n_objects   (B,)             — actual object count per sample (int tensor)

    Args:
        batch: List of sample dicts from SimulationDataset or MoviSimulationDataset.
        max_T_raw: Maximum raw frame count (must satisfy 4k+1 for some k ≥ 1).
        max_objects: Maximum number of objects per scene.
        max_points_per_object: Maximum surface points per object.
    """
    max_N = max_objects * max_points_per_object
    r = 4  # temporal compression ratio

    processed = []
    for sample in batch:
        x_s_raw = sample['x_s_raw']           # (T_raw, N, 6)
        c_force_raw = sample['c_force_raw']    # (T_raw, N, 6)
        point_obj_idx_orig = sample['point_obj_idx']  # (N,) int

        T_orig = x_s_raw.shape[0]
        N_orig = x_s_raw.shape[1]
        n_obj_orig = sample['n_objects']

        # --- Clip T_raw to largest valid 4k+1 ≤ max_T_raw ---
        k_max = (max_T_raw - 1) // r
        k_cur = (T_orig - 1) // r
        k_clip = min(k_max, k_cur)
        T_clip = k_clip * r + 1

        x_s_raw = x_s_raw[:T_clip]             # (T_clip, N_orig, 6)
        c_force_raw = c_force_raw[:T_clip]

        # --- Clip n_objects ---
        n_obj_clip = min(n_obj_orig, max_objects)

        # --- Per-object: gather, truncate, pad points to max_points_per_object ---
        # Outputs: new_x_s_raw (T_clip, max_N, 6), new_c_force (T_clip, max_N, 6)
        # new_point_obj_idx (max_N,), point_mask (max_N,)
        new_x_s_raw = torch.zeros(T_clip, max_N, 6, dtype=x_s_raw.dtype)
        new_c_force = torch.zeros(T_clip, max_N, 6, dtype=c_force_raw.dtype)
        new_point_obj_idx = torch.zeros(max_N, dtype=torch.long)
        point_mask = torch.zeros(max_N, dtype=torch.bool)

        total_valid_N = 0
        for obj_i in range(n_obj_clip):
            slot_start = obj_i * max_points_per_object
            slot_end = slot_start + max_points_per_object

            # Find indices of this object's points in the original flat array
            obj_pt_indices = (point_obj_idx_orig == obj_i).nonzero(as_tuple=True)[0]
            n_pts = obj_pt_indices.shape[0]
            n_keep = min(n_pts, max_points_per_object)

            if n_keep > 0:
                keep_idx = obj_pt_indices[:n_keep]
                new_x_s_raw[:, slot_start:slot_start + n_keep, :] = x_s_raw[:, keep_idx, :]
                new_c_force[:, slot_start:slot_start + n_keep, :] = c_force_raw[:, keep_idx, :]
                point_mask[slot_start:slot_start + n_keep] = True
                total_valid_N += n_keep

            # point_obj_idx for the entire block (including padding) points to obj_i
            new_point_obj_idx[slot_start:slot_end] = obj_i

        # --- Pad T_raw dim to max_T_raw ---
        if T_clip < max_T_raw:
            pad_T = max_T_raw - T_clip
            x_pad = torch.zeros(pad_T, max_N, 6, dtype=x_s_raw.dtype)
            new_x_s_raw = torch.cat([new_x_s_raw, x_pad], dim=0)
            new_c_force = torch.cat([new_c_force, x_pad], dim=0)

        # --- Clip / pad per-object tensors ---
        c_id_orig = sample['c_id']       # (n_obj_orig,)
        c_mat_orig = sample['c_mat']     # (n_obj_orig, 2)
        c_mass_orig = sample['c_mass']   # (n_obj_orig,)
        c_static_orig = sample['c_static']  # (n_obj_orig,)
        c_init_orig = sample['c_init']   # (n_obj_orig, 7)

        c_id = torch.zeros(max_objects, dtype=c_id_orig.dtype)
        c_mat = torch.zeros(max_objects, 2, dtype=c_mat_orig.dtype)
        c_mass = torch.zeros(max_objects, dtype=c_mass_orig.dtype)
        c_static = torch.zeros(max_objects, dtype=c_static_orig.dtype)
        c_init = torch.zeros(max_objects, 7, dtype=c_init_orig.dtype)

        c_id[:n_obj_clip] = c_id_orig[:n_obj_clip]
        c_mat[:n_obj_clip] = c_mat_orig[:n_obj_clip]
        c_mass[:n_obj_clip] = c_mass_orig[:n_obj_clip]
        c_static[:n_obj_clip] = c_static_orig[:n_obj_clip]
        c_init[:n_obj_clip] = c_init_orig[:n_obj_clip]

        obj_mask = torch.zeros(max_objects, dtype=torch.bool)
        obj_mask[:n_obj_clip] = True

        p = {
            'x_s_raw': new_x_s_raw,          # (max_T_raw, max_N, 6)
            'c_force_raw': new_c_force,        # (max_T_raw, max_N, 6)
            'c_floor': sample['c_floor'],      # scalar
            'c_id': c_id,                       # (max_objects,)
            'c_mat': c_mat,                     # (max_objects, 2)
            'c_mass': c_mass,                   # (max_objects,)
            'c_static': c_static,               # (max_objects,)
            'c_init': c_init,                   # (max_objects, 7)
            'point_obj_idx': new_point_obj_idx, # (max_N,)
            'point_mask': point_mask,           # (max_N,) bool
            'obj_mask': obj_mask,               # (max_objects,) bool
            'T_raw': T_clip,
            'N': total_valid_N,
            'n_objects': n_obj_clip,
        }
        if 'text' in sample:
            p['text'] = sample['text']
        processed.append(p)

    # --- Stack all samples ---
    result = {}
    keys_tensor = ['x_s_raw', 'c_force_raw', 'c_floor', 'c_id', 'c_mat', 'c_mass',
                   'c_static', 'c_init', 'point_obj_idx', 'point_mask', 'obj_mask']
    for key in keys_tensor:
        result[key] = torch.stack([p[key] for p in processed], dim=0)

    # Scalar metadata as int tensors (not lists) for convenient use in training
    result['T_raw'] = torch.tensor([p['T_raw'] for p in processed], dtype=torch.long)
    result['N'] = torch.tensor([p['N'] for p in processed], dtype=torch.long)
    result['n_objects'] = torch.tensor([p['n_objects'] for p in processed], dtype=torch.long)

    if 'text' in processed[0]:
        result['text'] = [p['text'] for p in processed]

    return result
