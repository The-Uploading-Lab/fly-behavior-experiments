# Portability adaptation of the frozen R3 silhouette estimator; numerical equations unchanged.
"""Read frozen poses with the R2 silhouette estimator; no dynamics are rerun."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import mujoco as mj
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
ARMS = ('control', 'ibuprofen', 'deprivation_b_half')
ARM = {'control': 'control', 'ibuprofen': 'ibuprofen', 'calorie_restriction': 'deprivation_b_half'}
FPS, START = 240, 72
CAMERA = {'projection': 'orthographic', 'lookat_mm': [2., 1., 0.],
          'distance': 16., 'azimuth_deg': 90., 'elevation_deg': -90., 'width_px': 480, 'height_px': 480}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(xy, lengths, intervals):
    assert len(xy) == intervals+1 and len(lengths) == len(xy)
    sm = median_filter(xy, size=(5, 1))
    length = float(np.median(lengths))
    paths = {}
    for stride in (3, 6, 12):
        ix = np.unique(np.r_[np.arange(0, len(sm), stride), len(sm)-1])
        paths[stride] = float(np.linalg.norm(np.diff(sm[ix], axis=0), axis=1).sum())
    duration = intervals/FPS
    net = float(np.linalg.norm(sm[-1]-sm[0]))
    return {'duration_s': duration, 'frame_intervals': intervals,
            'silhouette_length_px': length, 'path_px': paths[6], 'net_px': net,
            'speed_px_s': paths[6]/duration,
            'speed_silh_s': paths[6]/duration/length,
            'path_silh': paths[6]/length, 'net_silh': net/length,
            'straightness': net/paths[6] if paths[6] else None,
            'sensitivity_min_silh_s': min(paths.values())/duration/length,
            'sensitivity_max_silh_s': max(paths.values())/duration/length,
            'stride_speeds_silh_s': {str(k): v/duration/length for k, v in paths.items()}}


def project(folder, intervals):
    arm = folder.name.removesuffix("-walk")
    manifest = json.loads((folder/'manifest.json').read_text())
    for filename, expected in manifest['files'].items():
        if sha(folder/filename) != expected:
            raise ValueError(f'changed source: {folder/filename}')
    readout = json.loads((folder/'readout.json').read_text())
    if not readout['successor_retention_pass']:
        return {'arm': arm, 'status': 'CENSORED_STANDING_FAILURE', 'windows': {}}
    out = folder/'projection'
    out.mkdir(parents=True, exist_ok=True)
    poses = np.load(folder/'poses.npz', allow_pickle=False)
    model = mj.MjModel.from_binary_path(str(folder/'body.mjb'))
    data = mj.MjData(model)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.orthographic = True
    camera.lookat[:] = CAMERA['lookat_mm']
    camera.distance = CAMERA['distance']
    camera.azimuth, camera.elevation = CAMERA['azimuth_deg'], CAMERA['elevation_deg']
    fly_ids = [i for i in range(model.ngeom)
               if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) or '').startswith('fb/')]
    maximum = max(intervals)
    rows = []
    with mj.Renderer(model, height=480, width=480) as renderer:
        for ix in range(START, START+maximum+1):
            data.qpos[:] = poses['qpos'][ix]
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            if ix == START:
                Image.fromarray(renderer.render()).save(out/'first-frame.png')
            renderer.enable_segmentation_rendering()
            seg = renderer.render()
            mask = (seg[:, :, 1] == int(mj.mjtObj.mjOBJ_GEOM)) & np.isin(seg[:, :, 0], fly_ids)
            renderer.disable_segmentation_rendering()
            if ix == START:
                Image.fromarray((mask*255).astype(np.uint8)).save(out/'first-mask.png')
            yy, xx = np.where(mask)
            if len(xx) < 100 or xx.min() == 0 or xx.max() == 479 or yy.min() == 0 or yy.max() == 479:
                raise ValueError(f'empty or clipped silhouette at frame {ix}')
            points = np.column_stack([xx, yy])
            eig = np.linalg.eigvalsh(np.cov(points.T))
            rows.append([ix, float(poses['time_s'][ix]), float(xx.mean()), float(yy.mean()),
                         float(4*np.sqrt(eig[-1])), int(len(xx))])
    rows = np.asarray(rows)
    np.savez_compressed(out/'silhouette-track.npz', track=rows)
    windows = {str(n): summarize(rows[:n+1, 2:4], rows[:n+1, 4], n) for n in sorted(set(intervals))}
    for n, value in windows.items():
        value.update(start_nominal_s=START/FPS, end_nominal_s=(START+int(n))/FPS,
                     start_physics_s=float(rows[0, 1]), end_physics_s=float(rows[int(n), 1]))
    return {'arm': arm, 'status': 'COMPUTED', 'windows': windows,
            'source_manifest_sha256': sha(folder/'manifest.json'), 'camera': CAMERA,
            'track_columns': ['pose_index', 'physics_time_s', 'centroid_x_px', 'centroid_y_px', 'silhouette_length_px', 'area_px'],
            'track_sha256': sha(out/'silhouette-track.npz'),
            'nominal_time_max_error_ms': float(np.abs(rows[:, 1]-rows[:, 0]/FPS).max()*1000)}

