"""Visualization of point cloud motion for PV-Simulator.

Provides a reusable function and a CLI script to render point cloud trajectories
as animated GIF or MP4. Supports multiple views side-by-side in a single animation.

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

    T, N, _ = point_states.shape
    obj_ids = np.unique(point_obj_idx)
    n_objects = len(obj_ids)

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
    )


if __name__ == "__main__":
    main()
