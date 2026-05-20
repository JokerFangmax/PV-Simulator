"""Visualization of point cloud motion for PV-Simulator.

Provides a reusable function and a CLI script to render point cloud trajectories
as animated GIF or MP4. Supports multiple views side-by-side in a single animation,
with optional keypoint tracking and GT-vs-pred overlay for failure analysis.

Supported views:
  birdseye  — top-down orthographic (X vs Y), square, equal scale
  side      — side orthographic (X vs Z, Z=height), square, equal scale
  front     — front orthographic (Y vs Z), square, equal scale
  iso       — true 3D matplotlib axes at 45° elevation / 45° azimuth

All views share the same world-space bounding box so 1 unit = same size everywhere.

Usage (CLI):
    python scripts/pv_simulator/visualize.py \
        --data_dir datasets/movi_ab_10k/00000 \
        --output /tmp/out.mp4 \
        --views birdseye side iso

    python scripts/pv_simulator/visualize.py \
        --point_states_npy /path/to/states.npy \
        --point_obj_idx_npy /path/to/obj_idx.npy \
        --output /tmp/out.mp4

Programmatic usage:
    from scripts.pv_simulator.visualize import visualize_point_cloud_motion
    visualize_point_cloud_motion(point_states, point_obj_idx, "out.mp4",
                                 views=["birdseye", "side", "iso"])
"""

import argparse
import os
import pickle
import sys

import numpy as np

current_file_path = os.path.abspath(__file__)
for _root in [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]:
    if _root not in sys.path:
        sys.path.insert(0, _root)

DEFAULT_COLORS = [
    "#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#8d99ae",
    "#ef476f", "#118ab2", "#06d6a0", "#ffd166", "#8338ec",
    "#fb5607", "#3a86ff", "#e9c46a", "#264653", "#a8dadc",
    "#e76f51",
]

_VIEW_TITLES = {
    'birdseye': 'Top (X–Y)',
    'side':     'Side (X–Z)',
    'front':    'Front (Y–Z)',
    'iso':      '3D (45°)',
}

# axis indices and labels for orthographic views
_ORTHO_AXES = {
    'birdseye': (0, 1, 'X', 'Y'),
    'side':     (0, 2, 'X', 'Z'),
    'front':    (1, 2, 'Y', 'Z'),
}


def _hex_to_rgba(hex_color: str, alpha: float = 1.0):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return (r, g, b, alpha)


def _world_bounds(all_pos: np.ndarray, pad: float = 0.3):
    """Compute a uniform half-range and per-axis midpoints from (T, N, 3) positions.

    Returns (mid_x, mid_y, mid_z, half_range) so that
    [mid_? - half_range, mid_? + half_range] encloses all data with padding,
    and all axes share the same scale.
    """
    x_min, x_max = all_pos[:, :, 0].min(), all_pos[:, :, 0].max()
    y_min, y_max = all_pos[:, :, 1].min(), all_pos[:, :, 1].max()
    z_min, z_max = all_pos[:, :, 2].min(), all_pos[:, :, 2].max()

    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2
    mid_z = (z_min + z_max) / 2

    half_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2 + pad
    return mid_x, mid_y, mid_z, half_range


def _normalize_keypoint_indices(keypoint_indices, n_points: int):
    if keypoint_indices is None:
        return []
    arr = np.asarray(keypoint_indices, dtype=np.int64).reshape(-1)
    arr = [int(i) for i in arr.tolist() if 0 <= int(i) < n_points]
    return arr


def _sample_keypoints_by_strategy(
    point_states: np.ndarray,
    point_obj_idx: np.ndarray,
    num_keypoints: int,
):
    """Pick representative keypoints from frame0 geometry.

    Strategy: for each object choose farthest-from-centroid points first, then
    round-robin across objects until num_keypoints are filled.
    """
    if num_keypoints <= 0:
        return []

    frame0 = point_states[0, :, :3]
    chosen = []
    per_obj_ranked = []
    for obj_id in np.unique(point_obj_idx):
        obj_idx = np.where(point_obj_idx == obj_id)[0]
        if len(obj_idx) == 0:
            continue
        obj_pos = frame0[obj_idx]
        centroid = obj_pos.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(obj_pos - centroid, axis=-1)
        ranked = obj_idx[np.argsort(-dist)].tolist()
        per_obj_ranked.append(ranked)

    cursor = 0
    while len(chosen) < num_keypoints and any(len(r) > 0 for r in per_obj_ranked):
        ranked = per_obj_ranked[cursor % max(len(per_obj_ranked), 1)]
        if ranked:
            chosen.append(ranked.pop(0))
        cursor += 1

    return chosen[:num_keypoints]


def _draw_keypoints_ortho(
    ax,
    pts,
    keypoint_indices,
    keypoint_labels,
    ax0_i,
    ax1_i,
    marker,
    edgecolor,
    text_color,
    past_traj=None,
    linestyle='-',
):
    for ki, kp in enumerate(keypoint_indices):
        x = pts[kp, ax0_i]
        y = pts[kp, ax1_i]
        ax.scatter([x], [y], s=64, marker=marker, facecolors='none',
                   edgecolors=edgecolor, linewidths=1.8, zorder=6)
        label = keypoint_labels[ki]
        ax.text(x, y, label, color=text_color, fontsize=7, zorder=7)
        if past_traj is not None and past_traj.shape[0] > 1:
            ax.plot(
                past_traj[:, kp, ax0_i], past_traj[:, kp, ax1_i],
                color=edgecolor, linewidth=1.0, linestyle=linestyle, alpha=0.9, zorder=5,
            )


def _draw_keypoints_3d(
    ax,
    pts,
    keypoint_indices,
    keypoint_labels,
    marker,
    edgecolor,
    text_color,
    past_traj=None,
    linestyle='-',
):
    for ki, kp in enumerate(keypoint_indices):
        x, y, z = pts[kp, 0], pts[kp, 1], pts[kp, 2]
        ax.scatter([x], [y], [z], s=72, marker=marker, facecolors='none',
                   edgecolors=edgecolor, linewidths=1.8, depthshade=False)
        ax.text(x, y, z, keypoint_labels[ki], color=text_color, fontsize=7)
        if past_traj is not None and past_traj.shape[0] > 1:
            ax.plot(
                past_traj[:, kp, 0], past_traj[:, kp, 1], past_traj[:, kp, 2],
                color=edgecolor, linewidth=1.0, linestyle=linestyle, alpha=0.9,
            )


def visualize_point_cloud_motion(
    point_states,
    point_obj_idx,
    output_path: str,
    fps: int = 10,
    views=None,
    colors=None,
    max_points_per_object: int = None,
    show_velocity: bool = False,
    velocity_scale: float = 0.1,
    iso_elev: int = 25,
    iso_azim: int = 45,
    figsize_per_panel: tuple = (5, 5),
    dpi: int = 100,
    keypoint_indices=None,
    keypoint_labels=None,
    reference_point_states=None,
    keypoint_history: bool = True,
    auto_select_keypoints: int = 0,
):
    """Render point cloud motion as an animated GIF or MP4.

    Args:
        point_states: (T, N, 6) array or tensor — pos(3)+vel(3) per point per frame.
        point_obj_idx: (N,) int array or tensor — object index for each point.
        output_path: Output file path (.gif or .mp4).
        fps: Frames per second.
        views: List of view names. Each can be 'birdseye', 'side', 'front', or 'iso'.
               Defaults to ["birdseye", "side", "iso"].
        colors: Optional list of hex color strings, one per object.
        max_points_per_object: Subsample each object's points to at most this count.
        show_velocity: Draw velocity arrows (orthographic views only).
        velocity_scale: Arrow length multiplier.
        figsize_per_panel: Figure size (w, h) in inches per panel.
        dpi: Output resolution.
        keypoint_indices: Optional iterable of point indices to highlight.
        keypoint_labels: Optional labels for each highlighted keypoint.
        reference_point_states: Optional second trajectory (T, N, 6), typically GT,
            used only for keypoint overlay comparison against point_states.
        keypoint_history: Draw each keypoint's past trajectory up to frame t.
        auto_select_keypoints: If >0 and keypoint_indices is empty, choose that many
            representative keypoints automatically from frame0.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
    import imageio

    if views is None:
        views = ["birdseye", "side", "iso"]
    if isinstance(views, str):
        views = [views]
    n_panels = len(views)

    # Convert to numpy
    if hasattr(point_states, 'numpy'):
        point_states = point_states.detach().cpu().numpy()
    if hasattr(point_obj_idx, 'numpy'):
        point_obj_idx = point_obj_idx.detach().cpu().numpy()
    point_states = np.asarray(point_states, dtype=np.float32)
    point_obj_idx = np.asarray(point_obj_idx, dtype=np.int64)
    if reference_point_states is not None:
        if hasattr(reference_point_states, 'numpy'):
            reference_point_states = reference_point_states.detach().cpu().numpy()
        reference_point_states = np.asarray(reference_point_states, dtype=np.float32)
        assert reference_point_states.shape == point_states.shape, (
            f"reference_point_states shape {reference_point_states.shape} "
            f"must match point_states shape {point_states.shape}"
        )

    T, N, _ = point_states.shape
    obj_ids = np.unique(point_obj_idx)
    n_objects = len(obj_ids)

    keypoint_indices = _normalize_keypoint_indices(keypoint_indices, N)
    if not keypoint_indices and auto_select_keypoints > 0:
        keypoint_indices = _sample_keypoints_by_strategy(
            point_states=reference_point_states if reference_point_states is not None else point_states,
            point_obj_idx=point_obj_idx,
            num_keypoints=auto_select_keypoints,
        )
    if keypoint_labels is None:
        keypoint_labels = [f"K{i}" for i in range(len(keypoint_indices))]
    else:
        keypoint_labels = list(keypoint_labels)
        if len(keypoint_labels) < len(keypoint_indices):
            keypoint_labels += [f"K{i}" for i in range(len(keypoint_labels), len(keypoint_indices))]

    if colors is None:
        colors = DEFAULT_COLORS
    obj_colors = [_hex_to_rgba(colors[i % len(colors)]) for i in range(n_objects)]

    all_pos = point_states[:, :, :3]   # (T, N, 3)

    # Shared world bounds — same scale for all views
    mid_x, mid_y, mid_z, half_range = _world_bounds(all_pos)
    lim = {
        'x': (mid_x - half_range, mid_x + half_range),
        'y': (mid_y - half_range, mid_y + half_range),
        'z': (mid_z - half_range, mid_z + half_range),
    }

    # Per-orthographic-view limits using the common half_range so scales match
    ortho_lims = {
        'birdseye': (lim['x'], lim['y']),
        'side':     (lim['x'], lim['z']),
        'front':    (lim['y'], lim['z']),
    }

    # Per-object point masks
    obj_masks = []
    for oid in obj_ids:
        mask = np.where(point_obj_idx == oid)[0]
        if max_points_per_object is not None and len(mask) > max_points_per_object:
            mask = np.random.choice(mask, max_points_per_object, replace=False)
        obj_masks.append(mask)

    figw = figsize_per_panel[0] * n_panels
    figh = figsize_per_panel[1]

    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(figw, figh), dpi=dpi)
        fig.patch.set_facecolor('#1a1a2e')

        for panel_i, v in enumerate(views):
            is_3d = (v == 'iso')
            subplot_kw = dict(projection='3d') if is_3d else {}
            ax = fig.add_subplot(1, n_panels, panel_i + 1, **subplot_kw)

            # Common styling
            ax.set_facecolor('#1a1a2e')
            ax.set_title(f"{_VIEW_TITLES.get(v, v)}  [{t}/{T-1}]",
                         color='white', fontsize=9, pad=4)

            if is_3d:
                # ---- 3D view ----
                ax.set_xlim(*lim['x'])
                ax.set_ylim(*lim['y'])
                ax.set_zlim(*lim['z'])
                ax.view_init(elev=iso_elev, azim=iso_azim)

                ax.set_xlabel('X', color='#aaaaaa', fontsize=8, labelpad=2)
                ax.set_ylabel('Y', color='#aaaaaa', fontsize=8, labelpad=2)
                ax.set_zlabel('Z', color='#aaaaaa', fontsize=8, labelpad=2)

                # Dark pane backgrounds
                ax.xaxis.pane.fill = True
                ax.yaxis.pane.fill = True
                ax.zaxis.pane.fill = True
                ax.xaxis.pane.set_facecolor('#1a1a2e')
                ax.yaxis.pane.set_facecolor('#1a1a2e')
                ax.zaxis.pane.set_facecolor('#1a1a2e')
                ax.xaxis.pane.set_edgecolor('#333344')
                ax.yaxis.pane.set_edgecolor('#333344')
                ax.zaxis.pane.set_edgecolor('#333344')

                ax.tick_params(colors='#888888', labelsize=6)
                ax.xaxis.line.set_color('#555566')
                ax.yaxis.line.set_color('#555566')
                ax.zaxis.line.set_color('#555566')

                for oi, (oid, mask) in enumerate(zip(obj_ids, obj_masks)):
                    pts = point_states[t, mask]   # (n_pts, 6)
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                               s=4, color=obj_colors[oi], alpha=0.85,
                               linewidths=0, depthshade=True)

                if keypoint_indices:
                    past = point_states[:t + 1, :, :3] if keypoint_history else None
                    _draw_keypoints_3d(
                        ax, point_states[t, :, :3], keypoint_indices, keypoint_labels,
                        marker='o', edgecolor='#ffffff', text_color='#ffffff',
                        past_traj=past, linestyle='-',
                    )
                    if reference_point_states is not None:
                        ref_past = reference_point_states[:t + 1, :, :3] if keypoint_history else None
                        _draw_keypoints_3d(
                            ax, reference_point_states[t, :, :3], keypoint_indices, keypoint_labels,
                            marker='x', edgecolor='#ffd166', text_color='#ffd166',
                            past_traj=ref_past, linestyle='--',
                        )

            else:
                # ---- Orthographic view ----
                ax0_i, ax1_i, xlabel, ylabel = _ORTHO_AXES[v]
                xlim, ylim = ortho_lims[v]

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel(xlabel, color='#888888', fontsize=8)
                ax.set_ylabel(ylabel, color='#888888', fontsize=8)
                ax.tick_params(colors='#888888', labelsize=6)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#333344')

                for oi, (oid, mask) in enumerate(zip(obj_ids, obj_masks)):
                    pts = point_states[t, mask]   # (n_pts, 6)
                    x = pts[:, ax0_i]
                    y = pts[:, ax1_i]
                    c = obj_colors[oi]
                    ax.scatter(x, y, s=5, color=c, alpha=0.85, linewidths=0)

                    if show_velocity:
                        vx = pts[:, 3 + ax0_i]
                        vy = pts[:, 3 + ax1_i]
                        ax.quiver(x, y, vx * velocity_scale, vy * velocity_scale,
                                  color=c, alpha=0.6, width=0.003,
                                  scale=1, scale_units='xy')

                if keypoint_indices:
                    past = point_states[:t + 1, :, :3] if keypoint_history else None
                    _draw_keypoints_ortho(
                        ax, point_states[t, :, :3], keypoint_indices, keypoint_labels,
                        ax0_i, ax1_i, marker='o', edgecolor='#ffffff',
                        text_color='#ffffff', past_traj=past, linestyle='-',
                    )
                    if reference_point_states is not None:
                        ref_past = reference_point_states[:t + 1, :, :3] if keypoint_history else None
                        _draw_keypoints_ortho(
                            ax, reference_point_states[t, :, :3], keypoint_indices, keypoint_labels,
                            ax0_i, ax1_i, marker='x', edgecolor='#ffd166',
                            text_color='#ffd166', past_traj=ref_past, linestyle='--',
                        )

                if keypoint_indices and panel_i == 0:
                    legend_lines = ["white circle/solid = current trajectory",
                                    "yellow x/dashed = reference trajectory"] if reference_point_states is not None else \
                                   ["white circle/solid = tracked keypoints"]
                    ax.text(
                        0.02, 0.98, "\n".join(legend_lines),
                        transform=ax.transAxes, va='top', ha='left',
                        fontsize=7, color='#dddddd',
                        bbox=dict(facecolor='#111122', edgecolor='#333344', alpha=0.75, pad=4),
                    )

        plt.tight_layout(pad=0.5)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    # Write output
    ext = os.path.splitext(output_path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if ext == '.gif':
        imageio.mimsave(output_path, frames, duration=1.0 / fps, loop=0)
    elif ext == '.mp4':
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                    quality=8, macro_block_size=1)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    else:
        raise ValueError(f"Unsupported output format: {ext!r}. Use .gif or .mp4")

    print(f"Saved {T}-frame animation ({n_panels} view(s)) to {output_path}")


# ---------------------------------------------------------------------------
# MOVI loader
# ---------------------------------------------------------------------------

def load_from_movi_dir(data_dir: str):
    """Load point_states and point_obj_idx from a MOVI-AB sample directory."""
    pkl_path = os.path.join(data_dir, 'point_cloud_states.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing {pkl_path}")

    with open(pkl_path, 'rb') as f:
        pkl = pickle.load(f)

    point_states = pkl['point_states'].astype(np.float32)
    instances = pkl['instances']
    N = point_states.shape[1]
    point_obj_idx = np.zeros(N, dtype=np.int64)
    for i, inst in enumerate(instances):
        start, end = inst['point_range']
        point_obj_idx[start:end] = i

    return point_states, point_obj_idx


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize point cloud motion")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_dir", type=str,
                       help="MOVI-AB sample directory containing point_cloud_states.pkl")
    group.add_argument("--point_states_npy", type=str,
                       help="Path to .npy file with point states (T, N, 6)")
    parser.add_argument("--reference_point_states_npy", type=str, default=None,
                        help="Optional reference trajectory (T, N, 6), e.g. GT when "
                             "visualizing a predicted trajectory with keypoint comparison.")
    parser.add_argument("--point_obj_idx_npy", type=str, default=None,
                        help="Path to .npy file with point-to-object mapping (N,). "
                             "If omitted, all points are treated as one object.")
    parser.add_argument("--output", type=str, default="/tmp/point_cloud_motion.mp4")
    parser.add_argument("--views", type=str, nargs='+',
                        default=["birdseye", "side", "iso"],
                        choices=["birdseye", "side", "front", "iso"],
                        help="Views to render side by side. Default: birdseye side iso")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_points_per_object", type=int, default=None)
    parser.add_argument("--show_velocity", action="store_true")
    parser.add_argument("--velocity_scale", type=float, default=0.1)
    parser.add_argument("--iso_elev", type=int, default=25,
                        help="Elevation angle (degrees) for the 3D iso view. Default: 25")
    parser.add_argument("--iso_azim", type=int, default=45,
                        help="Azimuth angle (degrees) for the 3D iso view. Default: 45")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--keypoint_indices", type=int, nargs='*', default=None,
                        help="Specific point indices to highlight and track.")
    parser.add_argument("--auto_select_keypoints", type=int, default=0,
                        help="Automatically choose this many representative keypoints if "
                             "--keypoint_indices is not provided.")
    parser.add_argument("--no_keypoint_history", action="store_true",
                        help="Disable drawing each keypoint's past trajectory.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_dir is not None:
        point_states, point_obj_idx = load_from_movi_dir(args.data_dir)
    else:
        point_states = np.load(args.point_states_npy)
        if args.point_obj_idx_npy is not None:
            point_obj_idx = np.load(args.point_obj_idx_npy)
        else:
            point_obj_idx = np.zeros(point_states.shape[1], dtype=np.int64)
    reference_point_states = None
    if args.reference_point_states_npy is not None:
        reference_point_states = np.load(args.reference_point_states_npy)

    visualize_point_cloud_motion(
        point_states=point_states,
        point_obj_idx=point_obj_idx,
        output_path=args.output,
        fps=args.fps,
        views=args.views,
        max_points_per_object=args.max_points_per_object,
        show_velocity=args.show_velocity,
        velocity_scale=args.velocity_scale,
        iso_elev=args.iso_elev,
        iso_azim=args.iso_azim,
        dpi=args.dpi,
        keypoint_indices=args.keypoint_indices,
        reference_point_states=reference_point_states,
        keypoint_history=not args.no_keypoint_history,
        auto_select_keypoints=args.auto_select_keypoints,
    )


if __name__ == "__main__":
    main()
