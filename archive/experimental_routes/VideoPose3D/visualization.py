# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np

import io
import imageio

import cv2

H36M17_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),  # Hip->RLeg
    (0, 4),
    (4, 5),
    (5, 6),  # Hip->LLeg
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),  # Spine->Head
    (8, 11),
    (11, 12),
    (12, 13),  # Thorax->LArm
    (8, 14),
    (14, 15),
    (15, 16),  # Thorax->RArm
]


def read_video(filename, skip=0, limit=-1):

    while True:
        try:
            cv2_cap = cv2.VideoCapture(filename)
            if not cv2_cap.isOpened():
                raise IOError(f"Cannot open video: {filename}")
            fps = cv2_cap.get(cv2.CAP_PROP_FPS)
            break
        except Exception as e:
            print(f"Error opening video {filename}: {e}. Retrying...")

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cv2_cap.read()
        if not ret:
            break
        if frame_idx >= skip:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
        if limit != -1 and len(frames) >= limit:
            break
    cv2_cap.release()
    return frames, fps


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation(
    keypoints,  # 2d kpt (N, J, 2)
    keypoints_metadata,
    poses,
    skeleton,
    fps,
    bitrate,
    azim,
    output,
    viewport,
    limit=-1,
    downsample=1,
    size=6,
    input_video_path=None,
    input_video_skip=0,
):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title("Input")

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection="3d")
        ax.view_init(elev=15.0, azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect("equal")
        except NotImplementedError:
            ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros(
            (keypoints.shape[0], viewport[1], viewport[0]), dtype="uint8"
        )
    else:
        # Load video using ffmpeg
        all_frames, fps = read_video(input_video_path, skip=input_video_skip)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

        keypoints = keypoints[input_video_skip:]  # todo remove
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype("uint8")
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d(
                [
                    -radius / 2 + trajectories[n][i, 0],
                    radius / 2 + trajectories[n][i, 0],
                ]
            )
            ax.set_ylim3d(
                [
                    -radius / 2 + trajectories[n][i, 1],
                    radius / 2 + trajectories[n][i, 1],
                ]
            )

        # Update 2D poses
        joints_right_2d = keypoints_metadata["keypoints_symmetry"][1]
        colors_2d = np.full(keypoints.shape[1], "black")
        colors_2d[joints_right_2d] = "red"
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect="equal")

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if (
                    len(parents) == keypoints.shape[1]
                    and keypoints_metadata["layout_name"] != "coco"
                ):
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(
                        ax_in.plot(
                            [keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]],
                            color="pink",
                        )
                    )

                col = "red" if j in skeleton.joints_right() else "black"
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(
                        ax.plot(
                            [pos[j, 0], pos[j_parent, 0]],
                            [pos[j, 1], pos[j_parent, 1]],
                            [pos[j, 2], pos[j_parent, 2]],
                            zdir="z",
                            c=col,
                        )
                    )

            points = ax_in.scatter(
                *keypoints[i].T, 10, color=colors_2d, edgecolors="white", zorder=10
            )

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if (
                    len(parents) == keypoints.shape[1]
                    and keypoints_metadata["layout_name"] != "coco"
                ):
                    lines[j - 1][0].set_data(
                        [keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                        [keypoints[i, j, 1], keypoints[i, j_parent, 1]],
                    )

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata(
                        np.array([pos[j, 0], pos[j_parent, 0]])
                    )
                    lines_3d[n][j - 1][0].set_ydata(
                        np.array([pos[j, 1], pos[j_parent, 1]])
                    )
                    lines_3d[n][j - 1][0].set_3d_properties(
                        np.array([pos[j, 2], pos[j_parent, 2]]), zdir="z"
                    )

            points.set_offsets(keypoints[i])

        print("{}/{}      ".format(i, limit), end="\r")

    fig.tight_layout()

    anim = FuncAnimation(
        fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False
    )
    if output.endswith(".mp4"):
        Writer = writers["ffmpeg"]
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith(".gif"):
        anim.save(output, dpi=80, writer="imagemagick")
    else:
        raise ValueError("Unsupported output format (only .mp4 and .gif are supported)")
    plt.close()


def set_equal_aspect_3d(ax, X, Y, Z):
    xs, ys, zs = np.array(X), np.array(Y), np.array(Z)
    max_range = np.array(
        [xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]
    ).max()
    mid_x = (xs.max() + xs.min()) / 2
    mid_y = (ys.max() + ys.min()) / 2
    mid_z = (zs.max() + zs.min()) / 2
    r = max_range / 2 * 1.05
    ax.set_xlim(mid_x - r, mid_x + r)
    ax.set_ylim(mid_y - r, mid_y + r)
    ax.set_zlim(mid_z - r, mid_z + r)


def plot_coco3d_frame(points, swap_yz=True, show_labels=False, elev=20, azim=0):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    x = points[:, 0]
    y = points[:, 2] if swap_yz else points[:, 1]
    z = points[:, 1] if swap_yz else points[:, 2]

    ax.scatter(x, y, z, s=20, color="k")
    for a, b in H36M17_EDGES:
        ax.plot([x[a], x[b]], [y[a], y[b]], [z[a], z[b]], c="blue", lw=2)
    if show_labels:
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            ax.text(xi, yi, zi, f"{i}", color="red", fontsize=7)

    ax.set_xlabel("X")
    ax.set_ylabel("Z" if swap_yz else "Y")
    ax.set_zlabel("Y" if swap_yz else "Z")
    ax.view_init(elev=elev, azim=azim)
    set_equal_aspect_3d(ax, x, y, z)
    plt.tight_layout()
    return fig


def save_coco3d_gif_multi_view(
    all_fused, gif_prefix: Path = "fused_pose", swap_yz=True, fps=10
):

    gif_prefix.mkdir(parents=True, exist_ok=True)

    views = [
        dict(name="front", elev=0, azim=0),
        dict(name="left", elev=90, azim=-90),
        dict(name="top", elev=0, azim=90),
        dict(name="default", elev=20, azim=45),
    ]
    for view in views:
        frames = []
        for i, f in enumerate(all_fused):
            fig = plot_coco3d_frame(
                f,
                swap_yz=swap_yz,
                show_labels=False,
                elev=view["elev"],
                azim=view["azim"],
            )
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.v2.imread(buf))
            buf.close()
        gif_path = gif_prefix / f"{view['name']}.gif"
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"✅ Saved {gif_path}")
