"""Optional movies replay saved poses; no neural or body dynamics are rerun."""
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from portable_integrity import sha


def render(folder, output):
    output = Path(output)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Changed capture: {folder / name}")
    result = json.loads((folder / "integrated-receipt.json").read_text())
    readout = json.loads((folder / "readout.json").read_text())
    poses = np.load(folder / "poses.npz", allow_pickle=False)
    model = mj.MjModel.from_binary_path(str(folder / "body.mjb"))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    data = mj.MjData(model)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 11., 135., -24.
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol, day = manifest["protocol"], manifest["age_days"]
    speed = 1. if protocol == "walk" else .125
    end = min(float(poses["time_s"][-1]), manifest["duration_ms"] / 1000.)
    clock = np.arange(0, end, speed / 30.)
    title = (f"Day {day}: synaptic efficacy {2 ** (-day / 45):.6f}" if day is not None else
             {"control": "Control", "ibuprofen": "Ibuprofen adaptation model",
              "deprivation_b_half": "Calorie-restriction circuit hypothesis"}[manifest["condition"]])
    behavior = "Walking" if protocol == "walk" else "Antennal input" if protocol == "puff" else "No-input control"
    font = ImageFont.load_default(size=30)
    small = ImageFont.load_default(size=20)
    # Fonts are Pillow's bundled font, so rendering needs no OS font path.
    with mj.Renderer(model, height=720, width=1280) as renderer, imageio.get_writer(
            output, fps=30, codec="libx264", quality=8, macro_block_size=16,
            ffmpeg_log_level="error") as writer:
        for i, t in enumerate(clock):
            ix = min(np.searchsorted(poses["time_s"], t), len(poses["time_s"]) - 1)
            data.qpos[:] = poses["qpos"][ix]
            mj.mj_forward(model, data)
            camera.lookat[:] = poses["thorax_xyz"][ix]
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1280, 98), fill=(11, 16, 26))
            draw.text((26, 15), title, font=font, fill=(210, 234, 250))
            draw.text((26, 57), f"{behavior} | simulated time {t:.3f} s | playback {speed:g}x | seed 0", font=small, fill=(182, 197, 212))
            if protocol == "puff" and .6 <= t <= .63:
                draw.rounded_rectangle((985, 25, 1255, 70), radius=8, fill=(35, 154, 187))
                draw.text((1000, 34), "30 ms sensory pulse", font=small, fill="white")
            draw.rectangle((0, 666, 1280, 720), fill=(11, 16, 26))
            if protocol == "walk":
                label = "Both-body standing gate: PASS" if readout["successor_retention_pass"] else "Standing gate: FAIL | walking statistics censored"
            else:
                latency = readout["takeoff_latency_ms"]
                event = f"support loss {latency:.1f} ms after input" if latency is not None else "no support loss after input"
                label = f"{event} | flight/return gate: {'PASS' if readout['successor_retention_pass'] else 'FAIL'}"
            draw.text((26, 682), label, font=small, fill=(195, 213, 226))
            writer.append_data(np.asarray(frame))
            if i == min(20, len(clock) - 1):
                frame.save(output.with_suffix(".png"))
    output.with_suffix(".json").write_text(json.dumps({"capture_manifest_sha256": sha(folder / "manifest.json"),
        "video_sha256": sha(output), "frames": len(clock), "fps": 30, "playback_speed": speed,
        "rendering": "Nearest saved native pose; following camera; no dynamics rerun",
        "limitations": result["limitations"]}, indent=2) + "\n")
    print(f"Video: {output}", flush=True)
