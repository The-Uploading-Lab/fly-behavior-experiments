"""The first behavioural parameter search: what makes it WALK?

WHY NOW AND NOT BEFORE. Until today a search would have tuned a rig that could
not walk regardless: no antagonists engaged, no load signal, no physiological
regime. All three exist now, the legs are rhythmic (load autocorr 0.53-0.64)
but uncoordinated (tripod score n.s. against the shift null), and displacement
is measurable in real physics. AGENTS.md's own plan says this is stage 4.

WHAT IS SEARCHED -- the sensorimotor interface only. The BRAIN stays frozen at
the verified physiological operating point (balance 1.10, alpha 0.10, w_syn
0.65): searching brain parameters would abandon the regime four measurements
pinned. The searched parameters are the unmeasured couplings:

    camp_gain    20-300 Hz   load -> campaniform rate
    F_ref        10-80       load normalisation
    claw/hook    +/-1 each   reflex polarities (BANC does not name tuning)
    drive_scale  1e-3..8e-3  muscle gain
    tau_joint    8-30 ms     joint filter
    adapt_b      0-1.2       adaptation strength
    cmd          DNg100/DNp09/MDN at 20-120 Hz   the descending command

OBJECTIVE: net planar displacement (mm) over 4 s of closed loop. Random search
first (40 evals), then hill-climb from the best (20 evals).

⚠️ THE NULL PROMISE, pre-registered per AGENTS.md stage 3-6: any real-vs-
shuffled claim about a parameter set found here REQUIRES training the shuffled
connectomes with the SAME search budget first. Until that runs, a walking
parameter set is a capability demo, not evidence about the connectome.
"""
import hashlib, json, os, time
import numpy as np, pandas as pd
import body as B
import cns
from compiled_body_positions import CompiledBodyPositionSelector
from nobody import Readout
from load_loop import (LEG_KEYS, camp_by_leg, camp_extra_by_leg,
                       leg_of_geom, contact_loads,
                       seg_of_geom, contact_loads_by_segment, SEG_TAGS)

SPACE = {
    "camp_gain": (20.0, 300.0), "F_ref": (10.0, 80.0),
    "drive_scale": (0.001, 0.008), "tau_joint": (8.0, 30.0),
    "adapt_b": (0.0, 1.2), "cmd_hz": (20.0, 120.0),
}
CATS = {"claw": (1, -1), "hook": (1, -1),
        "cmd": ("DNg100", "DNp09", "MDN")}

# The nervous system's own parameters. SPACE above is the muscle, sensory and
# command model: of its six continuous entries only adapt_b is intrinsic to a
# neuron, and neighbour() proposes only SPACE and CATS members, so no search
# that has ever run could vary a synaptic parameter. SPACE is left untouched so
# no existing search changes behaviour; a runner that wants the nervous system
# in scope uses SPACE | SPACE_NETWORK. Bounds are the ones the validators
# enforce or the ones the ledger rows declare.
SPACE_NETWORK = {
    "delay": (0.5, 12.0),          # measured law: 7.09 ms period per ms
    "tau_w": (20.0, 400.0),        # adaptation time constant
    "bal_target": (0.0, 1.6),      # per-neuron E/I homeostatic scaling
    "w_syn_scale": (0.5, 2.0),     # lane D measured a x2 gain effect 16:44
    "graded_gain": (0.0002, 0.0020),   # 2 to 20 spike-equivalents/s
    "graded_v50": (-55.0, -42.0),      # release curve midpoint, mV
    "graded_slope": (1.8, 9.0),        # mV per e-fold; 2.5 was a constant
    "std_U": (0.0, 0.6),           # short-term depression, cited, never used
}
CATS_NETWORK = {
    "signs": ("measured", None),
    "delay_mode": ("length", "length_bv", "by_type", "length_by_type", None),
    "threshold_mode": (
        "volume_rank", "mcns_active_proxy", "by_type", "by_exact_type",
        None),
    "v_rest_mode": ("by_exact_type", None),
    "graded": ("19a006_13ba", "hyp", None),
}


# v3.2 (re-freeze event 2026-08-31): additive per-evaluation extras, read
# by the battery; cleared and rewritten by each evaluate() call.
LAST_EXTRAS = {}


_OPTOMOTOR_DNA02 = {
    "left": "720575941510475536",
    "right": "720575941456897005",
}
_OPTOMOTOR_HS_TYPES = ("HSS", "HSN", "HSE")
_OPTOMOTOR_T4T5_TYPES = ("T4a", "T4b", "T5a", "T5b")
_DIRECT_DNA02_MODE = "mano2023_dna02_bridge_v0"
_HS_DNA02_RELAY_MODE = "mano2023_hs_dna02_relay_v0"
# C-M86 (2026-09-04): sight with no decoder. Every carrier delivery site in
# this file sits inside `if visual_bridge is not None`, and the bridge is only
# built when `vision_cmd` names a relay, so removing the decoder removes the
# vision -- measured, arms bit-identical on seeds 59346/59347. This mode runs
# the relay's setup path unchanged, so the carrier tape reaches T4/T5, and
# writes nothing to DNa02: the connectome's own edges are the only route from
# eye to cord. Absent = untouched = bit-identical. Ported from lane C by the
# judge 2026-09-05 for the behavioural vision grade.
_DELIVER_ONLY_MODE = "carrier_deliver_only_v1"
_LABEL_VISUAL_INPUT = "label_schedule"
_RENDERED_VISUAL_INPUT = "rendered_panorama_reichardt_v1"
_FLYVIS_HS_TAPE_VISUAL_INPUT = "flyvis_t4t5_opponent_hs_tape_v1"
_FLYVIS_BANC_OPPONENT_T4T5_VISUAL_INPUT = (
    "flyvis_banc_opponent_t4t5_rate_tape_v1")
# C-M16: the carrier front end. The tape hands T4a and T4b their opposed
# rates ready-made, so the model receives direction rather than producing
# it. This source drives the COLUMNAR CARRIERS instead -- Mi1, Mi9, Mi4, C3
# on the ON side and Tm9, Tm2, Tm1, Tm4 on the OFF side, none of which is
# direction-selective -- and leaves T4/T5 to whatever the connectome does
# with them. Per-cell rates are computed here from the committed frame file
# rather than carried on the theta, because 2,400 ticks by 12,713 cells is
# about 244 MB and a theta is not a place for it.
_RENDERED_CARRIER_VISUAL_INPUT = "rendered_carrier_frame_v1"
_CARRIER_ON_TYPES = ("Mi1", "Mi9", "Mi4", "C3")
_CARRIER_OFF_TYPES = ("Tm9", "Tm2", "Tm1", "Tm4")
_CONSTANT_IMAGE_MOTION_PROFILE = "constant"
_SINGLE_REVERSAL_IMAGE_MOTION_PROFILE = "single_reversal_v1"
_MOTION_STOP_REVERSAL_PROFILE = "motion_stop_reversal_v1"
# Lane B's +1 s shift of the same movie, ported to ask whether a mechanism
# follows the image or the clock. Same three phases, every breakpoint later.
_MOTION_STOP_REVERSAL_SHIFT1S_PROFILE = "motion_stop_reversal_shift1s_v1"
_MOTION_STOP_PROFILES = (
    _MOTION_STOP_REVERSAL_PROFILE,
    _MOTION_STOP_REVERSAL_SHIFT1S_PROFILE,
)
_HS_CLASS_RELAY_CODES = ("hs_centroid_v1", "hs_graded_opponent")
# C-V72: BANC v888 types 39.5% of left optic-lobe intrinsic rows against
# 81.9% on the right while the central brain and cord are symmetric to
# within 0.3 points, so selecting T4/T5 drive rows by `cell_type` alone
# drives the right population 2.84x harder than the left at the same
# stimulus. `fafb_alignment_cell_type` names a T4/T5 subtype on 1,494 of
# the untyped left optic-lobe rows and disagrees with `cell_type` on 0 of
# the 8,866 rows where both are present, so the second source fills the
# gaps without overwriting anything. Default is the original column.
_T4T5_LABEL_SOURCES = {
    "cell_type": (283, 290, 810, 801, 219, 347, 816, 812),
    "cell_type_or_fafb_alignment": (633, 673, 822, 820, 594, 674, 825, 831),
}
_T4T5_ALIGNMENT_COLUMN = "fafb_alignment_cell_type"


def _is_motion_stop_profile(motion_profile):
    """True for the movie that moves, stops and reverses, shifted or not."""
    return motion_profile in _MOTION_STOP_PROFILES


def _motion_stop_first_seconds(motion_profile):
    """Seconds of first-direction motion before the image stops."""
    if motion_profile == _MOTION_STOP_REVERSAL_SHIFT1S_PROFILE:
        return 2.75
    return 1.75


def _motion_stop_reverse_seconds(motion_profile):
    """Seconds from stimulus onset at which the image reverses."""
    if motion_profile == _MOTION_STOP_REVERSAL_SHIFT1S_PROFILE:
        return 4.25
    return 3.25


def _motion_stop_static_ms(motion_profile):
    """The static-image interval in run time, or None off this family."""
    if not _is_motion_stop_profile(motion_profile):
        return None
    return (500.0 + 1000.0 * _motion_stop_first_seconds(motion_profile),
            500.0 + 1000.0 * _motion_stop_reverse_seconds(motion_profile))
_HS_CENTROID_COUNTS_V1 = np.asarray([
    [758.0, 1736.0, 1374.0, 0.0, 0.0, 0.0],
    [448.0, 1724.0, 1311.0, 737.0, 749.0, 735.0],
])
_HS_CENTROIDS_V1 = (
    _HS_CENTROID_COUNTS_V1
    / _HS_CENTROID_COUNTS_V1.sum(axis=1, keepdims=True)
)
_MCNS_ACTIVE_STRUCTURE_PATH = os.path.join(
    os.path.dirname(__file__),
    "lanes/E/2026-09-01-mcns-active-neural-structure-r40.feather")
_MCNS_ACTIVE_STRUCTURE_SHA256 = (
    "d50d8a6da99e4d0950a8658822bdeb94d76f5e04707126eb9d069d7eef6ae7af")


def _optomotor_active_side(t_ms, direction, contrast):
    """Return 0=left, 1=right, or None outside the visual stimulus."""
    if not 500.0 <= float(t_ms) < 5500.0:
        return None
    side = 0 if direction == 1 else 1
    if contrast == 1.0 and float(t_ms) >= 1500.0:
        side = 1 - side
    return side


def _map_optomotor_direction(direction, side_map):
    """Map stimulus direction to the prototype's commanded body side."""
    if side_map is None:
        return direction
    if side_map != "swapped":
        raise ValueError("vision_side_map must be absent or 'swapped'")
    return -direction


def _optomotor_dna02_rates(t_ms, direction, contrast, rate_hz):
    """Return [left, right] DNa02 drive for the visual BUILD adapter.

    This is an explicit command-boundary hypothesis, not a model of the
    omitted T4/T5-to-DNa02 dynamics. The timing and sign transform follow
    Mano et al. 2023: a five-second stimulus starts at 500 ms; low contrast
    remains syn-directional, while high contrast reverses after one second.
    """
    rates = np.zeros(2, dtype=float)
    side = _optomotor_active_side(t_ms, direction, contrast)
    if side is None:
        return rates
    rates[side] = rate_hz
    return rates


def _optomotor_hs_rates(t_ms, direction, contrast, rate_hz, sides,
                        baseline_hz=0.0):
    """Drive six HS cells at baseline, raising the stimulus-side three."""
    sides = np.asarray(sides, dtype=np.int8)
    rates = np.full(len(sides), baseline_hz, dtype=float)
    side = _optomotor_active_side(t_ms, direction, contrast)
    if side is not None:
        rates[sides == side] = float(rate_hz)
    return rates


def _body_yaw(qpos):
    """MuJoCo root quaternion (w,x,y,z) to wrapped world yaw, radians."""
    qw, qx, qy, qz = (float(qpos[index]) for index in range(3, 7))
    return float(np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))


def _body_forward_speed(qpos, qvel):
    """Causal positive body-axis speed from MuJoCo root state."""
    velocity = np.asarray(qvel, dtype=float)
    if velocity.ndim != 1 or len(velocity) < 2 or not np.isfinite(
            velocity[:2]).all():
        raise ValueError("body velocity must expose finite world x/y rates")
    yaw = _body_yaw(qpos)
    forward = (np.cos(yaw) * velocity[0]
               + np.sin(yaw) * velocity[1])
    return max(0.0, float(forward))


def _hs_forward_speed_phase_strength(forward_speed, running_max,
                                     motion_sensed):
    """Causal scale-free HS steering strength and updated speed maximum."""
    if (type(forward_speed) not in (int, float)
            or type(running_max) not in (int, float)
            or not np.isfinite(forward_speed)
            or not np.isfinite(running_max)
            or float(forward_speed) < 0.0 or float(running_max) < 0.0
            or type(motion_sensed) is not bool):
        raise ValueError("HS forward-speed state is invalid")
    updated_max = max(float(running_max), float(forward_speed))
    if not motion_sensed:
        return updated_max, 0.0
    ratio = (0.0 if updated_max <= 0.0
             else float(forward_speed) / updated_max)
    return updated_max, 1.0 + ratio


def _rendered_panorama_motion_cycles(t_ms, direction, temporal_hz,
                                     motion_profile, body_yaw_rad,
                                     body_reafference):
    """Return the exact world-minus-body phase used by the pixel renderer."""
    active = 500.0 <= float(t_ms) < 5500.0
    elapsed_s = ((float(t_ms) - 500.0) / 1000.0 if active else 0.0)
    motion_cycles = float(direction) * float(temporal_hz) * elapsed_s
    if (active
            and motion_profile == _SINGLE_REVERSAL_IMAGE_MOTION_PROFILE
            and elapsed_s > 2.5):
        half_cycles = float(temporal_hz) * 2.5
        motion_cycles = float(direction) * (
            half_cycles - float(temporal_hz) * (elapsed_s - 2.5))
    if active and _is_motion_stop_profile(motion_profile):
        first_seconds = _motion_stop_first_seconds(motion_profile)
        reverse_seconds = _motion_stop_reverse_seconds(motion_profile)
        if elapsed_s <= first_seconds:
            motion_cycles = (
                float(direction) * float(temporal_hz) * elapsed_s)
        elif elapsed_s <= reverse_seconds:
            motion_cycles = (
                float(direction) * float(temporal_hz) * first_seconds)
        else:
            motion_cycles = float(direction) * (
                float(temporal_hz) * first_seconds
                - float(temporal_hz) * (elapsed_s - reverse_seconds))
    if body_reafference:
        wrapped_yaw = float(np.arctan2(
            np.sin(float(body_yaw_rad)), np.cos(float(body_yaw_rad))))
        motion_cycles -= 6.0 * wrapped_yaw / (2.0 * np.pi)
    return motion_cycles


def _rendered_panorama_frame(t_ms, direction, contrast, temporal_hz,
                             motion_profile=_CONSTANT_IMAGE_MOTION_PROFILE,
                             body_yaw_rad=0.0, body_reafference=False):
    """Render the frozen 72x24 vertical-bar panorama for C-V19/C-V20."""
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("rendered panorama direction must be -1 or +1")
    if (type(contrast) not in (int, float) or not np.isfinite(contrast)
            or float(contrast) != 0.25):
        raise ValueError("rendered panorama contrast must be exactly 0.25")
    if (type(temporal_hz) not in (int, float)
            or not np.isfinite(temporal_hz)
            or not 0.0 <= float(temporal_hz) <= 20.0):
        raise ValueError(
            "rendered panorama temporal frequency must be in [0, 20] Hz")
    if motion_profile not in (
            (_CONSTANT_IMAGE_MOTION_PROFILE,
             _SINGLE_REVERSAL_IMAGE_MOTION_PROFILE) + _MOTION_STOP_PROFILES):
        raise ValueError("unknown rendered panorama motion profile")
    if ((motion_profile == _SINGLE_REVERSAL_IMAGE_MOTION_PROFILE
         or _is_motion_stop_profile(motion_profile))
            and float(temporal_hz) <= 0.0):
        raise ValueError("dynamic rendered profiles require positive frequency")
    if type(body_reafference) is not bool:
        raise ValueError("rendered panorama body reafference must be boolean")
    if (type(body_yaw_rad) not in (int, float)
            or not np.isfinite(body_yaw_rad)):
        raise ValueError("rendered panorama body yaw must be finite")
    motion_cycles = _rendered_panorama_motion_cycles(
        t_ms, direction, temporal_hz, motion_profile, body_yaw_rad,
        body_reafference)
    x = np.arange(72, dtype=float)
    phase = 2.0 * np.pi * (
        6.0 * x / 72.0
        - motion_cycles)
    row = 0.5 * (1.0 + float(contrast) * np.sin(phase))
    return np.repeat(row[None, :], 24, axis=0)


def _rendered_panorama_phase_delta_cycles(previous, current):
    """Recover signed six-cycle panorama motion from two pixel frames."""
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if (previous.shape != (24, 72) or current.shape != (24, 72)
            or not np.isfinite(previous).all()
            or not np.isfinite(current).all()):
        raise ValueError("panorama phase frames must be finite 24x72 arrays")
    x = np.arange(72, dtype=float)
    basis = np.exp(-2j * np.pi * 6.0 * x / 72.0)
    previous_coefficient = np.sum(
        (previous[0] - np.mean(previous[0])) * basis)
    current_coefficient = np.sum(
        (current[0] - np.mean(current[0])) * basis)
    if (abs(previous_coefficient) <= 1e-12
            or abs(current_coefficient) <= 1e-12):
        raise ValueError("panorama six-cycle Fourier component is absent")
    phase_delta = float(np.angle(
        current_coefficient * np.conjugate(previous_coefficient)))
    return -phase_delta / (2.0 * np.pi)


def _new_rendered_motion_state():
    """Create evaluation-local state for the rendered motion detector."""
    return {
        "previous_frame": None,
        "frame_digest": hashlib.sha256(),
        "rendered_frames": 0,
        "stimulus_frames": 0,
        "stimulus_frames_by_epoch": np.zeros(2, dtype=np.int64),
        "detector_positive_ticks": 0,
        "detector_negative_ticks": 0,
        "detector_zero_ticks": 0,
        "selected_ticks": np.zeros(2, dtype=np.int64),
        "selected_ticks_by_epoch_side": np.zeros((2, 2), dtype=np.int64),
        "frames_by_segment": np.zeros(3, dtype=np.int64),
        "selected_ticks_by_segment_side": np.zeros(
            (3, 2), dtype=np.int64),
        "zero_ticks_by_segment": np.zeros(3, dtype=np.int64),
        "score_sum": 0.0,
        "score_abs_max": 0.0,
        "last_score": 0.0,
        "body_yaw_samples": 0,
        "body_yaw_min_rad": None,
        "body_yaw_max_rad": None,
        "body_phase_contribution_abs_max_cycles": 0.0,
        "previous_body_yaw_rad": None,
        "last_body_yaw_rate_deg_s": 0.0,
        "retinal_delta_abs_max_cycles": 0.0,
        "body_delta_abs_max_cycles": 0.0,
        "corrected_delta_abs_max_cycles": 0.0,
        "raw_retinal_selected_ticks": np.zeros(2, dtype=np.int64),
        "compensation_changed_ticks": 0,
        "previous_motion_cycles": None,
        "previous_external_motion_cycles": None,
        "retinal_slip_nonzero_ticks": 0,
        "detector_retinal_slip_match_ticks": 0,
        "detector_retinal_slip_mismatch_ticks": 0,
        "retinal_slip_opposes_external_by_epoch": np.zeros(
            2, dtype=np.int64),
    }


def _new_flyvis_side_tape_state(bridge):
    """Create evaluation-local accounting for a frozen FlyVis side tape."""
    tape = np.asarray(bridge["flyvis_side_tape"], dtype=np.int8)
    return {
        "previous_frame": None,
        "frame_digest": hashlib.sha256(),
        "frame_sha256": bridge["flyvis_movie_sha256"],
        "side_tape_sha256": hashlib.sha256(
            np.asarray(tape, dtype="<i1").tobytes(order="C")).hexdigest(),
        "source_model": _FLYVIS_HS_TAPE_VISUAL_INPUT,
        "rendered_frames": 0,
        "stimulus_frames": 0,
        "stimulus_frames_by_epoch": np.zeros(2, dtype=np.int64),
        "detector_positive_ticks": 0,
        "detector_negative_ticks": 0,
        "detector_zero_ticks": 0,
        "selected_ticks": np.zeros(2, dtype=np.int64),
        "selected_ticks_by_epoch_side": np.zeros((2, 2), dtype=np.int64),
        "score_sum": 0.0,
        "score_abs_max": 0.0,
        "last_score": 0.0,
        # Open-loop tape: reafference-only diagnostics are honest zero/nulls.
        "body_yaw_samples": 0,
        "body_yaw_min_rad": None,
        "body_yaw_max_rad": None,
        "body_phase_contribution_abs_max_cycles": 0.0,
        "retinal_slip_nonzero_ticks": 0,
        "detector_retinal_slip_match_ticks": 0,
        "detector_retinal_slip_mismatch_ticks": 0,
        "retinal_slip_opposes_external_by_epoch": np.zeros(
            2, dtype=np.int64),
    }


def _new_flyvis_banc_t4t5_rate_tape_state(bridge):
    """Create accounting for frozen FlyVis rates into BANC T4/T5 rows."""
    tape = np.asarray(bridge["flyvis_t4t5_rate_tape"], dtype="<f8")
    n_groups = int(tape.shape[1])
    observed_sha256 = hashlib.sha256(
        tape.tobytes(order="C")).hexdigest()
    if observed_sha256 != bridge["flyvis_t4t5_rate_tape_sha256"]:
        raise ValueError("FlyVis T4/T5 rate tape SHA-256 mismatch")
    return {
        "source_model": bridge["input_source"],
        "rate_tape_sha256": observed_sha256,
        "movie_sha256": bridge["flyvis_movie_sha256"],
        "ticks": 0,
        "tape_rows_consumed": 0,
        "last_tape_index": None,
        "stimulus_ticks": 0,
        "nonzero_ticks_inside_window": 0,
        "nonzero_ticks_outside_window": 0,
        "positive_ticks_by_group": np.zeros(n_groups, dtype=np.int64),
        "rate_sum_hz_by_group": np.zeros(n_groups, dtype=float),
        "rate_sum_hz_by_group_inside_window": np.zeros(
            n_groups, dtype=float),
        "rate_sum_hz_by_group_outside_window": np.zeros(
            n_groups, dtype=float),
        "max_rate_hz_by_group": np.zeros(n_groups, dtype=float),
        "selected_ticks_by_side": np.zeros(2, dtype=np.int64),
        "selected_ticks_by_epoch_side": np.zeros((2, 2), dtype=np.int64),
        "zero_or_tied_ticks": 0,
    }


def _rendered_motion_hs_rates(
        t_ms, bridge, state, body_yaw_rad=0.0, tick_ms=2.5):
    """Render pixels, run the two-frame correlator and drive one HS side."""
    if (type(tick_ms) not in (int, float) or not np.isfinite(tick_ms)
            or float(tick_ms) <= 0.0):
        raise ValueError("rendered motion tick_ms must be finite and positive")
    state.setdefault("previous_body_yaw_rad", None)
    state.setdefault("last_body_yaw_rate_deg_s", 0.0)
    state.setdefault("retinal_delta_abs_max_cycles", 0.0)
    state.setdefault("body_delta_abs_max_cycles", 0.0)
    state.setdefault("corrected_delta_abs_max_cycles", 0.0)
    state.setdefault(
        "raw_retinal_selected_ticks", np.zeros(2, dtype=np.int64))
    state.setdefault("compensation_changed_ticks", 0)
    state.setdefault("previous_motion_cycles", None)
    state.setdefault("previous_external_motion_cycles", None)
    state.setdefault("retinal_slip_nonzero_ticks", 0)
    state.setdefault("detector_retinal_slip_match_ticks", 0)
    state.setdefault("detector_retinal_slip_mismatch_ticks", 0)
    state.setdefault(
        "retinal_slip_opposes_external_by_epoch",
        np.zeros(2, dtype=np.int64))
    sides = np.asarray(bridge["input_sides"], dtype=np.int8)
    if not np.array_equal(sides, np.asarray([0, 0, 0, 1, 1, 1])):
        raise ValueError("rendered motion requires three HS cells per side")
    motion_profile = bridge.get(
        "image_motion_profile", _CONSTANT_IMAGE_MOTION_PROFILE)
    reafferent = bridge.get("image_body_reafference", False)
    motion_cycles = _rendered_panorama_motion_cycles(
        t_ms, bridge["stimulus_direction"], bridge["image_motion_hz"],
        motion_profile, body_yaw_rad, reafferent)
    external_motion_cycles = _rendered_panorama_motion_cycles(
        t_ms, bridge["stimulus_direction"], bridge["image_motion_hz"],
        motion_profile, 0.0, False)
    frame = _rendered_panorama_frame(
        t_ms, bridge["stimulus_direction"], bridge["contrast"],
        bridge["image_motion_hz"], motion_profile,
        body_yaw_rad=body_yaw_rad,
        body_reafference=reafferent)
    if frame.shape != (24, 72) or not np.isfinite(frame).all():
        raise ValueError("rendered panorama frame is invalid")
    state["frame_digest"].update(
        np.asarray(frame, dtype="<f4").tobytes(order="C"))
    state["rendered_frames"] += 1
    if reafferent:
        wrapped_yaw = float(np.arctan2(
            np.sin(float(body_yaw_rad)), np.cos(float(body_yaw_rad))))
        state["body_yaw_samples"] += 1
        state["body_yaw_min_rad"] = (
            wrapped_yaw if state["body_yaw_min_rad"] is None
            else min(state["body_yaw_min_rad"], wrapped_yaw))
        state["body_yaw_max_rad"] = (
            wrapped_yaw if state["body_yaw_max_rad"] is None
            else max(state["body_yaw_max_rad"], wrapped_yaw))
        state["body_phase_contribution_abs_max_cycles"] = max(
            state["body_phase_contribution_abs_max_cycles"],
            abs(6.0 * wrapped_yaw / (2.0 * np.pi)))
    in_window = 500.0 <= float(t_ms) < 5500.0
    segment = None
    if in_window and _is_motion_stop_profile(motion_profile):
        static_start, static_end = _motion_stop_static_ms(motion_profile)
        segment = (0 if float(t_ms) < static_start else
                   1 if float(t_ms) < static_end else 2)
    epoch = int(
        (motion_profile == _SINGLE_REVERSAL_IMAGE_MOTION_PROFILE
         and float(t_ms) > 3000.0)
        or (_is_motion_stop_profile(motion_profile)
            and segment == 2))
    if in_window:
        state["stimulus_frames"] += 1
        state["stimulus_frames_by_epoch"][epoch] += 1
        if segment is not None:
            state["frames_by_segment"][segment] += 1
    previous = state["previous_frame"]
    previous_body_yaw = state["previous_body_yaw_rad"]
    score = 0.0
    retinal_delta_cycles = 0.0
    body_delta_cycles = 0.0
    corrected_delta_cycles = 0.0
    body_yaw_rate_deg_s = 0.0
    if previous is not None:
        score = float(np.mean(
            previous[:, :-1] * frame[:, 1:]
            - previous[:, 1:] * frame[:, :-1]))
        retinal_delta_cycles = _rendered_panorama_phase_delta_cycles(
            previous, frame)
        if reafferent and previous_body_yaw is not None:
            body_delta_rad = float(np.arctan2(
                np.sin(wrapped_yaw - float(previous_body_yaw)),
                np.cos(wrapped_yaw - float(previous_body_yaw))))
            body_delta_cycles = 6.0 * body_delta_rad / (2.0 * np.pi)
            body_yaw_rate_deg_s = float(
                np.rad2deg(body_delta_rad) / (float(tick_ms) / 1000.0))
        corrected_delta_cycles = retinal_delta_cycles + body_delta_cycles
    state["previous_frame"] = frame
    state["previous_body_yaw_rad"] = float(np.arctan2(
        np.sin(float(body_yaw_rad)), np.cos(float(body_yaw_rad))))
    state["last_body_yaw_rate_deg_s"] = body_yaw_rate_deg_s
    state["last_score"] = score
    state["score_sum"] += score
    state["score_abs_max"] = max(state["score_abs_max"], abs(score))
    state["retinal_delta_abs_max_cycles"] = max(
        state["retinal_delta_abs_max_cycles"], abs(retinal_delta_cycles))
    state["body_delta_abs_max_cycles"] = max(
        state["body_delta_abs_max_cycles"], abs(body_delta_cycles))
    state["corrected_delta_abs_max_cycles"] = max(
        state["corrected_delta_abs_max_cycles"],
        abs(corrected_delta_cycles))
    rates = np.zeros(6, dtype=float)
    active_side = None
    raw_side = None
    motion_enabled = not (
        _is_motion_stop_profile(motion_profile) and segment == 1)
    if (in_window and motion_enabled
            and float(bridge["image_motion_hz"]) > 0.0
            and abs(score) > 1e-12):
        raw_side = 0 if score > 0.0 else 1
        state["raw_retinal_selected_ticks"][raw_side] += 1
    decision_value = (
        corrected_delta_cycles
        if bridge.get("image_motion_cancellation", False) else score)
    if (in_window and motion_enabled
            and float(bridge["image_motion_hz"]) > 0.0
            and abs(decision_value) > 1e-12):
        active_side = 0 if decision_value > 0.0 else 1
        if bridge.get("hs_signed_drive", False):
            rates[sides == active_side] = 0.5 * float(bridge["rate_hz"])
            rates[sides != active_side] = -0.5 * float(bridge["rate_hz"])
        else:
            rates[sides == active_side] = float(bridge["rate_hz"])
        state["selected_ticks"][active_side] += 1
        state["selected_ticks_by_epoch_side"][epoch, active_side] += 1
        if segment is not None:
            state["selected_ticks_by_segment_side"][
                segment, active_side] += 1
        if decision_value > 0.0:
            state["detector_positive_ticks"] += 1
        else:
            state["detector_negative_ticks"] += 1
    else:
        state["detector_zero_ticks"] += 1
        if segment is not None:
            state["zero_ticks_by_segment"][segment] += 1
    if (in_window and bridge.get("image_motion_cancellation", False)
            and raw_side != active_side):
        state["compensation_changed_ticks"] += 1
    previous_cycles = state["previous_motion_cycles"]
    previous_external = state["previous_external_motion_cycles"]
    if in_window and previous_cycles is not None:
        slip_delta = float(motion_cycles - previous_cycles)
        if abs(slip_delta) > 1e-12:
            slip_side = 0 if slip_delta > 0.0 else 1
            state["retinal_slip_nonzero_ticks"] += 1
            if active_side == slip_side:
                state["detector_retinal_slip_match_ticks"] += 1
            else:
                state["detector_retinal_slip_mismatch_ticks"] += 1
            external_delta = float(
                external_motion_cycles - previous_external)
            if (abs(external_delta) > 1e-12
                    and slip_side != (0 if external_delta > 0.0 else 1)):
                state["retinal_slip_opposes_external_by_epoch"][epoch] += 1
    state["previous_motion_cycles"] = motion_cycles
    state["previous_external_motion_cycles"] = external_motion_cycles
    return rates, active_side


def _flyvis_side_tape_hs_rates(t_ms, bridge, state):
    """Drive HS from a frozen, causal FlyVis T4/T5 opponent side tape."""
    sides = np.asarray(bridge["input_sides"], dtype=np.int8)
    hss_only = bridge.get("flyvis_hss_only", False)
    expected_sides = np.asarray(
        [0, 1] if hss_only else [0, 0, 0, 1, 1, 1], dtype=np.int8)
    if not np.array_equal(sides, expected_sides):
        expected = "one HSS cell" if hss_only else "three HS cells"
        raise ValueError(f"FlyVis tape requires {expected} per side")
    tick_ms = float(bridge["flyvis_side_tape_tick_ms"])
    if np.isclose(float(t_ms), 0.0, rtol=0.0, atol=1e-12):
        state["rendered_frames"] += 1
        state["detector_zero_ticks"] += 1
        state["last_score"] = 0.0
        return np.zeros(len(sides), dtype=float), None
    tick_number = int(round(float(t_ms) / tick_ms))
    if (tick_number < 1
            or not np.isclose(tick_number * tick_ms, float(t_ms),
                              rtol=0.0, atol=1e-9)):
        raise ValueError("FlyVis tape time is not aligned to its frozen tick")
    tape = bridge["flyvis_side_tape"]
    index = tick_number - 1
    if index >= len(tape):
        raise ValueError("FlyVis side tape ended before the body run")
    side = int(tape[index])
    in_window = 500.0 <= float(t_ms) < 5500.0
    if not in_window and side != -1:
        raise ValueError("FlyVis side tape must be silent outside stimulus")
    epoch = int(float(t_ms) > 3000.0)
    state["rendered_frames"] += 1
    if in_window:
        state["stimulus_frames"] += 1
        state["stimulus_frames_by_epoch"][epoch] += 1
    score = 1.0 if side == 0 else (-1.0 if side == 1 else 0.0)
    state["last_score"] = score
    state["score_sum"] += score
    state["score_abs_max"] = max(state["score_abs_max"], abs(score))
    rates = np.zeros(len(sides), dtype=float)
    active_side = None
    if in_window and side in (0, 1):
        active_side = side
        rates[sides == side] = float(bridge["rate_hz"])
        state["selected_ticks"][side] += 1
        state["selected_ticks_by_epoch_side"][epoch, side] += 1
        state["detector_positive_ticks" if side == 0
              else "detector_negative_ticks"] += 1
    else:
        state["detector_zero_ticks"] += 1
    return rates, active_side


def _flyvis_banc_t4t5_rate_tape_rates(t_ms, bridge, state):
    """Return one causal rate for each frozen T4/T5 type/side group."""
    n_groups = len(bridge["flyvis_t4t5_groups"])
    tick_ms = float(bridge["flyvis_t4t5_rate_tape_tick_ms"])
    if np.isclose(float(t_ms), 0.0, rtol=0.0, atol=1e-12):
        state["ticks"] += 1
        state["zero_or_tied_ticks"] += 1
        return np.zeros(n_groups, dtype=float), None
    tick_number = int(round(float(t_ms) / tick_ms))
    tolerance = max(
        1e-7,
        32.0 * np.finfo(float).eps
        * max(1.0, abs(float(t_ms)), abs(tick_number * tick_ms)))
    if (tick_number < 1
            or not np.isclose(tick_number * tick_ms, float(t_ms),
                              rtol=0.0, atol=tolerance)):
        raise ValueError(
            "FlyVis T4/T5 rate tape time is not aligned to its frozen tick")
    tape = np.asarray(bridge["flyvis_t4t5_rate_tape"], dtype=float)
    index = tick_number - 1
    if index >= len(tape):
        raise ValueError("FlyVis T4/T5 rate tape ended before the body run")
    rates = np.asarray(tape[index], dtype=float)
    if (rates.shape != (n_groups,) or not np.isfinite(rates).all()
            or np.any(rates < 0.0)
            or np.any(rates > float(bridge["rate_hz"]) + 1e-12)):
        raise ValueError("FlyVis T4/T5 rate tape row is invalid")
    in_window = 500.0 <= float(t_ms) < 5500.0
    if n_groups != 8:
        raise ValueError("opponent FlyVis BANC tape requires eight groups")
    # Preferred a on one eye is paired with null-direction b on the other.
    # This score is diagnostic only; live evidence-graded HS release controls
    # the selected body's phase path.
    side_totals = np.asarray([
        rates[0:2].sum() + rates[6:8].sum(),
        rates[2:4].sum() + rates[4:6].sum(),
    ])
    active_side = None
    if in_window and side_totals[0] != side_totals[1]:
        active_side = int(np.argmax(side_totals))
        if side_totals[active_side] <= 0.0:
            active_side = None
    state["ticks"] += 1
    state["tape_rows_consumed"] += 1
    state["last_tape_index"] = int(index)
    if in_window:
        state["stimulus_ticks"] += 1
        state["nonzero_ticks_inside_window"] += int(np.any(rates != 0.0))
        state["rate_sum_hz_by_group_inside_window"] += rates
    else:
        state["nonzero_ticks_outside_window"] += int(np.any(rates != 0.0))
        state["rate_sum_hz_by_group_outside_window"] += rates
    state["positive_ticks_by_group"] += (rates > 0.0).astype(np.int64)
    state["rate_sum_hz_by_group"] += rates
    state["max_rate_hz_by_group"] = np.maximum(
        state["max_rate_hz_by_group"], rates)
    if active_side is None:
        state["zero_or_tied_ticks"] += 1
    else:
        epoch = int(float(t_ms) > 3000.0)
        state["selected_ticks_by_side"][active_side] += 1
        state["selected_ticks_by_epoch_side"][epoch, active_side] += 1
    return rates, active_side


def _update_visual_relay(ema, counts, tick_ms, tau_ms, swap):
    """Update two causal DNa02 count states and return the selected gate side."""
    ema = np.asarray(ema, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if ema.shape != (2,) or counts.shape != (2,):
        raise ValueError("visual relay state and counts must each have shape (2,)")
    if not np.isfinite(ema).all() or not np.isfinite(counts).all():
        raise ValueError("visual relay state and counts must be finite")
    if type(swap) is not bool:
        raise ValueError("visual relay swap must be bool")
    state = ema * np.exp(-float(tick_ms) / float(tau_ms)) + counts
    if state[0] == state[1]:
        return state, None
    side = int(np.argmax(state))
    return state, (1 - side if swap else side)


def _update_visual_delta_relay(ema, counts, baseline, tick_ms, tau_ms, swap):
    """Update DNa02 state from count innovation over a per-cell baseline."""
    baseline = np.asarray(baseline, dtype=float)
    if baseline.shape != (2,) or not np.isfinite(baseline).all():
        raise ValueError("visual relay baseline must be finite with shape (2,)")
    state, _ = _update_visual_relay(
        ema, np.asarray(counts, dtype=float) - baseline,
        tick_ms, tau_ms, swap=False)
    if state[0] == state[1] or float(np.max(state)) <= 0.0:
        return state, None
    side = int(np.argmax(state))
    return state, (1 - side if swap else side)


def _update_visual_hs_opponent_relay(ema, counts, sides, tick_ms, tau_ms):
    """Decode equal-weight HS side counts into the contralateral motor side."""
    ema = np.asarray(ema, dtype=float)
    counts = np.asarray(counts, dtype=float)
    sides = np.asarray(sides, dtype=np.int8)
    if ema.shape != (2,) or counts.shape != (6,) or sides.shape != (6,):
        raise ValueError(
            "HS opponent state, counts and sides must have shapes (2,), "
            "(6,) and (6,)")
    if not np.isfinite(ema).all() or not np.isfinite(counts).all():
        raise ValueError("HS opponent state and counts must be finite")
    if not np.array_equal(sides, np.asarray([0, 0, 0, 1, 1, 1])):
        raise ValueError("HS opponent sides must be three left then three right")
    side_counts = np.asarray([
        float(counts[sides == side].sum()) for side in (0, 1)
    ])
    state = ema * np.exp(-float(tick_ms) / float(tau_ms)) + side_counts
    if state[0] == state[1]:
        return state, None
    # Fixed opponent architecture: the more active HS side suppresses the
    # contralateral return-phase neural drive. This is not a fitted per-cell
    # map; all three identified HS cells on a side carry equal weight.
    return state, 1 - int(np.argmax(state))


def _update_visual_hs_opponent_prestim_relay(
        ema, counts, baseline, sides, tick_ms, tau_ms):
    """Compare the two HS sides' VISUALLY DRIVEN activity, not their total.

    Measured on the champion (C-V77/C-V77N, seed 59106): the three left HS
    cells fire 401 spikes with the stimulus and 401 without it, while the
    three right fire 137 and 0. A comparison of absolute counts is
    therefore mostly a comparison of vision-independent activity, and it
    decoded one side on 65% of gated ticks. Subtracting each cell's own
    pre-stimulus rate leaves the part the stimulus put there. The residual
    is signed and is not clipped: a cell the stimulus suppresses carries
    direction as much as one it excites.
    """
    ema = np.asarray(ema, dtype=float)
    counts = np.asarray(counts, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    sides = np.asarray(sides, dtype=np.int8)
    if (ema.shape != (2,) or counts.shape != (6,)
            or baseline.shape != (6,) or sides.shape != (6,)):
        raise ValueError(
            "prestim HS opponent state, counts, baseline and sides must "
            "have shapes (2,), (6,), (6,) and (6,)")
    if (not np.isfinite(ema).all() or not np.isfinite(counts).all()
            or not np.isfinite(baseline).all()):
        raise ValueError(
            "prestim HS opponent state, counts and baseline must be finite")
    if not np.array_equal(sides, np.asarray([0, 0, 0, 1, 1, 1])):
        raise ValueError(
            "prestim HS opponent sides must be three left then three right")
    residual = counts - baseline
    side_residual = np.asarray([
        float(residual[sides == side].sum()) for side in (0, 1)
    ])
    state = ema * np.exp(-float(tick_ms) / float(tau_ms)) + side_residual
    if state[0] == state[1]:
        return state, None
    return state, 1 - int(np.argmax(state))


def _update_visual_hs_centroid_relay(ema, counts, tick_ms, tau_ms):
    """Classify causal six-HS activity by frozen C-V12 zero-arm centroids."""
    ema = np.asarray(ema, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if ema.shape != (6,) or counts.shape != (6,):
        raise ValueError("HS centroid state and counts must each have shape (6,)")
    if not np.isfinite(ema).all() or not np.isfinite(counts).all():
        raise ValueError("HS centroid state and counts must be finite")
    state = ema * np.exp(-float(tick_ms) / float(tau_ms)) + counts
    total = float(state.sum())
    if total <= 0.0:
        return state, None
    normalized = state / total
    distances = np.linalg.norm(_HS_CENTROIDS_V1 - normalized, axis=1)
    if distances[0] == distances[1]:
        return state, None
    return state, int(np.argmin(distances))


def _update_visual_hs_graded_opponent(signal, sides):
    """Decode continuous HS release without spike or magnitude thresholds."""
    signal = np.asarray(signal, dtype=float)
    sides = np.asarray(sides, dtype=np.int8)
    if signal.shape != (6,) or sides.shape != (6,):
        raise ValueError(
            "graded HS signal and sides must have shapes (6,) and (6,)")
    if not np.isfinite(signal).all():
        raise ValueError("graded HS release signal must be finite")
    if not np.array_equal(sides, np.asarray([0, 0, 0, 1, 1, 1])):
        raise ValueError("graded HS sides must be three left then three right")
    side_release = np.asarray([
        float(signal[sides == side].mean()) for side in (0, 1)
    ])
    if side_release[0] == side_release[1]:
        return signal.copy(), None
    return signal.copy(), int(np.argmax(side_release))


def _visual_side_magnitude(ema):
    """How unequal the two HS sides are, in [0, 1].

    The opponent relays reduce their state to `argmax`, so a side that
    leads by one spike and a side that leads by a thousand deliver the same
    command. This is the discarded quantity: the normalised difference of
    the same state the argmax reads, so it needs no new measurement and no
    new decode. Zero when the sides are equal or silent, one when all the
    activity is on one side.
    """
    ema = np.asarray(ema, dtype=float)
    if ema.shape != (2,) or not np.isfinite(ema).all():
        raise ValueError("HS side magnitude needs a finite state of shape (2,)")
    total = float(ema.sum())
    if total <= 0.0:
        return 0.0
    return float(min(1.0, abs(float(ema[0]) - float(ema[1])) / total))


def _vision_gated_cmd_rates(cmd_hz, cmd_side, gain, side, swap=False,
                            magnitude=1.0):
    """Per-row command rate, boosted on the side the HS readout names.

    `side` is the relay's decoded contralateral motor side for this tick,
    or None when nothing decoded; with None every row keeps `cmd_hz`, so a
    run with no visual decode is identical to the ungated one.

    `magnitude` scales the gain, and is 1.0 unless the caller passes the
    relay's side imbalance, which makes the command carry how unequal the
    sides are rather than only which one leads.
    """
    base = np.full(len(cmd_side), float(cmd_hz), dtype=float)
    if side is None:
        return base
    if int(side) not in (0, 1):
        raise ValueError("vision-gated command side must be 0, 1 or None")
    m = float(magnitude)
    if not np.isfinite(m) or not 0.0 <= m <= 1.0:
        raise ValueError("vision-gated command magnitude must be in [0, 1]")
    g = float(gain) * m
    # Which side the readout's decode should boost is NOT established: the
    # relay returns the contralateral index by a convention inherited from
    # the DNa02 adapter, and no measurement ties it to a turn direction on
    # this body. It is a parameter for the same reason claw_polarity is.
    target = (1 - int(side)) if bool(swap) else int(side)
    scale = np.where(cmd_side == target, 1.0 + g, 1.0 - g)
    return base * scale


def _carrier_drive_tape(frame, rows, u, is_on, ticks, tick_ms, spec):
    """Per-cell carrier rates for every tick, from a drifting grating.

    Each carrier samples the grating at its own retinotopic u. ON cells take
    the rectified positive luminance change and OFF cells the rectified
    negative, which is contrast polarity and carries no direction: any
    direction selectivity downstream is the connectome's.
    """
    cycles = float(spec["cycles"])
    temporal_hz = float(spec["temporal_hz"])
    direction = float(spec["direction"])
    peak_hz = float(spec["peak_hz"])
    lo_ms, hi_ms = spec["window_ms"]
    tape = np.zeros((ticks, len(rows)), dtype=np.float64)
    previous = None
    for t in range(ticks):
        t_ms = t * tick_ms
        phase = direction * temporal_hz * (t_ms / 1000.0)
        current = 0.5 + 0.5 * np.sin(
            2.0 * np.pi * (cycles * u - phase))
        if previous is not None and lo_ms <= t_ms < hi_ms:
            delta = current - previous
            tape[t] = np.where(is_on, np.clip(delta, 0.0, None),
                               np.clip(-delta, 0.0, None))
        previous = current
    peak = float(tape.max())
    if peak > 0.0:
        tape *= peak_hz / peak
    return tape


def _integer_spike_feedback(signal, label):
    """Recover exact spike counts from a mixed spike/release callback."""
    signal = np.asarray(signal, dtype=float)
    rounded = np.rint(signal)
    if (not np.isfinite(signal).all()
            or not np.array_equal(signal, rounded)):
        raise ValueError(f"{label} feedback must contain integer spike counts")
    return rounded.astype(np.int64)


def _resolve_visual_teacher_steering_legs(th):
    """Resolve the single categorical teacher-correction leg scope."""
    scope = th.get("vision_teacher_steering_legs", "all")
    if scope not in ("all", "front_middle"):
        raise ValueError(
            "vision_teacher_steering_legs must be 'all' or 'front_middle'")
    return scope


def _resolve_visual_teacher_leg_phase_asymmetry(th):
    """Resolve an optional [front, middle, hind] phase-speed allocation."""
    value = th.get("vision_teacher_leg_phase_asymmetry")
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(item) not in (int, float)
                   or not np.isfinite(item)
                   or not 0.0 <= float(item) < 1.0
                   for item in value)):
        raise ValueError(
            "vision_teacher_leg_phase_asymmetry must be three finite "
            "values in [0, 1)")
    return np.asarray(value, dtype=float)


def _resolve_visual_phase_from_detector(
        th, input_source, relay_code, phase_control_active):
    """Validate the default-off visual-class phase source."""
    value = th.get("vision_phase_from_detector", False)
    if type(value) is not bool:
        raise ValueError("vision_phase_from_detector must be Boolean")
    if value and (
            input_source not in (
                _RENDERED_VISUAL_INPUT, _FLYVIS_HS_TAPE_VISUAL_INPUT)
            or relay_code not in _HS_CLASS_RELAY_CODES
            or not phase_control_active):
        raise ValueError(
            "vision_phase_from_detector requires a rendered or frozen-FlyVis "
            "hs_centroid_v1 side-phase controller")
    return value


def _resolve_visual_phase_neural_motion_control(
        th, input_source, relay_code, phase_control_active,
        phase_from_detector, motion_profile, body_reafference,
        motion_cancellation):
    """Authorize live HS phase alongside the frozen motion-stop brake."""
    value = th.get("vision_phase_neural_motion_control", False)
    if type(value) is not bool:
        raise ValueError(
            "vision_phase_neural_motion_control must be Boolean")
    rendered_source = bool(
        input_source == _RENDERED_VISUAL_INPUT
        and body_reafference and motion_cancellation)
    actual_t4t5_source = bool(
        input_source == _FLYVIS_BANC_OPPONENT_T4T5_VISUAL_INPUT
        and not body_reafference and not motion_cancellation)
    if value and not (
            (rendered_source or actual_t4t5_source)
            and relay_code in _HS_CLASS_RELAY_CODES
            and phase_control_active
            and not phase_from_detector
            and _is_motion_stop_profile(motion_profile)):
        raise ValueError(
            "vision_phase_neural_motion_control requires either "
            "body-reafferent cancelled motion-stop pixels or the frozen "
            "actual-BANC T4/T5 source, live HS phase control, and no "
            "detector phase override")
    return value


def _resolve_visual_phase_brake_on_no_motion(
        th, input_source, motion_profile, body_reafference,
        motion_cancellation, phase_from_detector, phase_reset_on_no_motion):
    """Validate the default-off body-yaw brake for the static movie segment."""
    value = th.get("vision_phase_brake_on_no_motion", False)
    if type(value) is not bool:
        raise ValueError("vision_phase_brake_on_no_motion must be Boolean")
    if value and not (
            input_source == _RENDERED_VISUAL_INPUT
            and _is_motion_stop_profile(motion_profile)
            and body_reafference and motion_cancellation
            and phase_from_detector and phase_reset_on_no_motion):
        raise ValueError(
            "vision_phase_brake_on_no_motion requires the body-reafferent "
            "cancelled motion-stop detector with no-motion phase reset")
    return value


def _resolve_visual_hs_forward_speed_phase_gain(
        th, relay_code, phase_control_active, body_reafference):
    """Validate speed-dependent HS steering from the fly's own movement."""
    value = th.get("vision_hs_forward_speed_phase_gain", False)
    if type(value) is not bool:
        raise ValueError(
            "vision_hs_forward_speed_phase_gain must be Boolean")
    if value and not (
            relay_code == "hs_graded_opponent"
            and phase_control_active and body_reafference):
        raise ValueError(
            "HS forward-speed phase gain requires graded HS phase control "
            "and body reafference")
    return value


def _resolve_visual_phase_brake_filter_body_rate(th, brake_enabled):
    """Authorize optional sensory filtering only on the causal yaw brake."""
    value = th.get("vision_phase_brake_filter_body_rate", False)
    if type(value) is not bool:
        raise ValueError(
            "vision_phase_brake_filter_body_rate must be Boolean")
    if value and not brake_enabled:
        raise ValueError(
            "vision_phase_brake_filter_body_rate requires the body-yaw brake")
    return value


def _resolve_visual_phase_brake_proportional_body_rate(
        th, brake_enabled, filter_enabled):
    """Authorize a continuous, self-normalized sensed-yaw brake."""
    value = th.get("vision_phase_brake_proportional_body_rate", False)
    if type(value) is not bool:
        raise ValueError(
            "vision_phase_brake_proportional_body_rate must be Boolean")
    if value and not (brake_enabled and filter_enabled):
        raise ValueError(
            "proportional body-rate brake requires the filtered yaw brake")
    return value


def _resolve_visual_phase_brake_memoryless_body_rate(
        th, brake_enabled, filter_enabled, proportional_enabled):
    """Authorize the non-integrating form of the continuous yaw brake."""
    value = th.get("vision_phase_brake_memoryless_body_rate", False)
    if type(value) is not bool:
        raise ValueError(
            "vision_phase_brake_memoryless_body_rate must be Boolean")
    if value and not (brake_enabled and filter_enabled):
        raise ValueError(
            "memoryless body-rate brake requires the filtered yaw brake")
    if value and proportional_enabled:
        raise ValueError(
            "accumulating and memoryless proportional brakes are exclusive")
    return value


def _visual_phase_brake_proportional_class_strength(
        body_yaw_rate_deg_s, previous_envelope_deg_s):
    """Return opposing class and strength normalized by the running envelope."""
    values = (body_yaw_rate_deg_s, previous_envelope_deg_s)
    if (any(type(value) not in (int, float) or not np.isfinite(value)
            for value in values) or previous_envelope_deg_s < 0.0):
        raise ValueError("proportional brake inputs must be finite")
    magnitude = abs(float(body_yaw_rate_deg_s))
    envelope = max(float(previous_envelope_deg_s), magnitude)
    if magnitude == 0.0:
        return None, 0.0, envelope
    decoded_class = 1 if body_yaw_rate_deg_s > 0.0 else 0
    return decoded_class, magnitude / envelope, envelope


def _visual_phase_brake_filtered_rate(previous, measured, tick_ms, tau_ms):
    """One exact first-order sensory integration step for body yaw rate."""
    values = (previous, measured, tick_ms, tau_ms)
    if (any(type(value) not in (int, float) or not np.isfinite(value)
            for value in values) or tick_ms <= 0.0 or tau_ms <= 0.0):
        raise ValueError("brake rate filter inputs must be finite and positive")
    alpha = float(np.exp(-float(tick_ms) / float(tau_ms)))
    return alpha * float(previous) + (1.0 - alpha) * float(measured)


def _dna02_phase_amplitude_update(previous, counts, tick_ms, tau_ms,
                                  max_rate_hz):
    """Filter the two DNa02 counts and return a bounded phase multiplier.

    Direction remains the live HS centroid.  DNa02 supplies only descending
    response magnitude.  Its normalization is the two cells' own summed
    refractory-limited capacity, so this adds no fixed firing threshold or
    universal gain.  A silent DNa02 pair is exactly multiplier 1 (the selected
    model); saturation can at most double the existing phase command.
    """
    previous = np.asarray(previous, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if (previous.shape != (2,) or counts.shape != (2,)
            or not np.isfinite(previous).all()
            or not np.isfinite(counts).all() or np.any(counts < 0.0)
            or type(tick_ms) not in (int, float) or not tick_ms > 0.0
            or type(tau_ms) not in (int, float) or not tau_ms > 0.0
            or type(max_rate_hz) not in (int, float)
            or not np.isfinite(max_rate_hz) or not max_rate_hz > 0.0):
        raise ValueError("DNa02 phase amplitude inputs are invalid")
    alpha = 1.0 - np.exp(-float(tick_ms) / float(tau_ms))
    state = previous + alpha * (counts - previous)
    rate_hz = float(state.sum()) * 1000.0 / float(tick_ms)
    multiplier = 1.0 + float(np.clip(rate_hz / float(max_rate_hz), 0.0, 1.0))
    return state, multiplier, rate_hz


def _dna02_routed_phase_amplitude(state, tick_ms, max_rate_hz_by_side,
                                  decoded_class):
    """Read amplitude from the DNa02 on the HS-decoded anatomical side."""
    state = np.asarray(state, dtype=float)
    capacities = np.asarray(max_rate_hz_by_side, dtype=float)
    if (state.shape != (2,) or capacities.shape != (2,)
            or not np.isfinite(state).all() or np.any(state < 0.0)
            or not np.isfinite(capacities).all() or np.any(capacities <= 0.0)
            or type(tick_ms) not in (int, float) or not tick_ms > 0.0
            or (decoded_class is not None
                and (type(decoded_class) not in (int, np.int64)
                     or int(decoded_class) not in (0, 1)))):
        raise ValueError("routed DNa02 phase amplitude inputs are invalid")
    if decoded_class is None:
        return 1.0, 0.0
    side = int(decoded_class)
    rate_hz = float(state[side]) * 1000.0 / float(tick_ms)
    multiplier = 1.0 + float(np.clip(
        rate_hz / capacities[side], 0.0, 1.0))
    return multiplier, rate_hz


def _visual_phase_brake_class(t_ms, body_yaw_rate_deg_s, enabled,
                              static_ms=(2250.0, 3750.0), deadband=5.0):
    """Return the opposing phase class inside the static segment.

    The interval is the movie's own, so a shifted movie brakes when its own
    image stops rather than at a fixed clock time.
    """
    if type(enabled) is not bool:
        raise ValueError("visual phase brake enabled flag must be Boolean")
    if (type(body_yaw_rate_deg_s) not in (int, float)
            or not np.isfinite(body_yaw_rate_deg_s)):
        raise ValueError("visual phase brake yaw rate must be finite")
    static_start, static_end = static_ms
    if not enabled or not static_start <= float(t_ms) < static_end:
        return None
    # Lane D measured that the brake's law transfers to a shifted movie and
    # its authority does not: after an extra second of turning it left the
    # static window at 11.647 deg/s against a 5 deg/s band. A narrower
    # deadband is the authority knob, not a retune of the trigger, because it
    # engages the brake on body rates it currently ignores. 5.0 is the frozen
    # value and keeps every existing number.
    if (type(deadband) not in (int, float) or not np.isfinite(deadband)
            or not 0.0 < float(deadband) <= 20.0):
        raise ValueError("visual phase brake deadband must be in (0, 20]")
    if float(body_yaw_rate_deg_s) > float(deadband):
        return 1
    if float(body_yaw_rate_deg_s) < -float(deadband):
        return 0
    return None


def _visual_phase_brake_opposes(decoded_class, body_yaw_rate_deg_s,
                                 deadband):
    """Whether a brake class opposes a rate beyond its configured deadband."""
    return bool(
        (decoded_class == 1 and body_yaw_rate_deg_s > deadband)
        or (decoded_class == 0 and body_yaw_rate_deg_s < -deadband))


def _resolve_visual_image_body_motion(th, input_source):
    """Resolve default-off reafferent pixels and angular cancellation."""
    reafference = th.get("vision_image_body_reafference", False)
    cancellation = th.get("vision_image_motion_cancellation", False)
    if type(reafference) is not bool:
        raise ValueError("vision_image_body_reafference must be boolean")
    if type(cancellation) is not bool:
        raise ValueError("vision_image_motion_cancellation must be boolean")
    if ((reafference or cancellation)
            and input_source != _RENDERED_VISUAL_INPUT):
        raise ValueError(
            "visual body motion requires rendered panorama input")
    if cancellation and not reafference:
        raise ValueError(
            "visual motion cancellation requires body reafference")
    return reafference, cancellation


def _resolve_visual_named_rows(meta, net, cell_types, super_class):
    """Resolve one unguessed visual row per type and side.

    The former DNa02 resolver hard-coded two BANC IDs, even though both BANC
    and MCNS carry exact type, side and class annotations.  Type/side identity
    is the cross-connectome invariant; an optional ``label_guessed`` column is
    a fail-closed provenance guard for adapted male datasets.
    """
    rows = []
    sides = []
    for side_index, side in enumerate(("left", "right")):
        for cell_type in cell_types:
            selected_rows = net.select(
                cell_type=cell_type, side=side, super_class=super_class)
            if len(selected_rows) != 1:
                raise ValueError(
                    f"visual {cell_type} {side} count is "
                    f"{len(selected_rows)}, expected 1")
            row = int(selected_rows[0])
            selected = net.meta.iloc[row]
            if row >= len(meta):
                raise ValueError("visual metadata row is outside caller table")
            caller = meta.iloc[row]
            identifier = str(selected.get("banc_888_id"))
            if (not identifier or identifier.lower() in ("nan", "none")
                    or str(caller.get("banc_888_id")) != identifier
                    or str(selected.get("cell_type")) != cell_type
                    or str(selected.get("side")).lower() != side
                    or str(selected.get("super_class")) != super_class):
                raise ValueError(
                    f"visual {cell_type} {side} metadata identity mismatch")
            if ("label_guessed" in selected.index
                    and bool(selected.get("label_guessed"))):
                raise ValueError(
                    f"visual {cell_type} {side} label is guessed")
            rows.append(row)
            sides.append(side_index)
    if len(np.unique(rows)) != len(rows):
        raise ValueError("visual named rows are not unique")
    return (np.asarray(rows, dtype=np.int64),
            np.asarray(sides, dtype=np.int8))


def _resolve_optomotor_bridge(th, meta, net):
    """Validate and resolve an optional visual-command adapter."""
    mode = th.get("vision_cmd")
    if mode is None:
        return None
    if mode not in (_DIRECT_DNA02_MODE, _HS_DNA02_RELAY_MODE,
                    _DELIVER_ONLY_MODE):
        raise ValueError("unknown th['vision_cmd']")
    deliver_only = mode == _DELIVER_ONLY_MODE
    if deliver_only:
        # identical resolution to the relay, including the carrier tape; only
        # the write to DNa02 is withheld, at the single gate below.
        mode = _HS_DNA02_RELAY_MODE
    direction = th.get("vision_dir")
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("vision_dir must be the integer -1 or +1")
    side_map = th.get("vision_side_map")
    mapped_direction = _map_optomotor_direction(direction, side_map)
    input_source = th.get("vision_input_source", _LABEL_VISUAL_INPUT)
    if input_source not in (
            _LABEL_VISUAL_INPUT, _RENDERED_VISUAL_INPUT,
            _FLYVIS_HS_TAPE_VISUAL_INPUT,
            _FLYVIS_BANC_OPPONENT_T4T5_VISUAL_INPUT,
            _RENDERED_CARRIER_VISUAL_INPUT):
        raise ValueError(
            "vision_input_source must be label, rendered panorama, FlyVis "
            "or the rendered carrier frame")
    image_motion_hz = th.get("vision_image_motion_hz", 0.0)
    if (type(image_motion_hz) not in (int, float)
            or not np.isfinite(image_motion_hz)
            or not 0.0 <= float(image_motion_hz) <= 20.0):
        raise ValueError("vision_image_motion_hz must be in [0, 20]")
    if (input_source == _LABEL_VISUAL_INPUT
            and float(image_motion_hz) != 0.0):
        raise ValueError(
            "vision_image_motion_hz requires rendered panorama input")
    image_motion_profile = th.get(
        "vision_image_motion_profile", _CONSTANT_IMAGE_MOTION_PROFILE)
    image_body_reafference, image_motion_cancellation = (
        _resolve_visual_image_body_motion(th, input_source))
    if image_motion_profile not in (
            (_CONSTANT_IMAGE_MOTION_PROFILE,
             _SINGLE_REVERSAL_IMAGE_MOTION_PROFILE) + _MOTION_STOP_PROFILES):
        raise ValueError("unknown vision_image_motion_profile")
    if (input_source == _LABEL_VISUAL_INPUT
            and image_motion_profile != _CONSTANT_IMAGE_MOTION_PROFILE):
        raise ValueError(
            "vision_image_motion_profile requires rendered panorama input")
    if ((image_motion_profile == _SINGLE_REVERSAL_IMAGE_MOTION_PROFILE
         or _is_motion_stop_profile(image_motion_profile))
            and float(image_motion_hz) <= 0.0):
        raise ValueError(
            "dynamic image profiles require positive motion")
    contrast = th.get("vision_contrast")
    if type(contrast) not in (int, float) or float(contrast) not in (0.25, 1.0):
        raise ValueError("vision_contrast must be exactly 0.25 or 1.0")
    rate_hz = th.get("vision_hz")
    if (type(rate_hz) not in (int, float) or not np.isfinite(rate_hz)
            or not 0.0 <= float(rate_hz) <= 1000.0):
        raise ValueError("vision_hz must be finite and in [0, 1000]")
    return_scale = th.get("vision_return_scale")
    if (return_scale is not None
            and (type(return_scale) not in (int, float)
                 or not np.isfinite(return_scale)
                 or not 0.0 <= float(return_scale) <= 1.0)):
        raise ValueError("vision_return_scale must be absent or in [0, 1]")
    # LIFTED 2026-09-03 by lane C, which wrote the blanket rejection on
    # 2026-08-31 to stop a second command overlapping the visual input rows:
    # in the `absolute` and `prestim_delta` relays the visual input rows ARE
    # the DNa02 rows, so a cmd2 naming them would drive the same cells twice
    # and the relay would read its own injection. The T4/T5 tape source puts
    # the visual input on optic-lobe rows instead, which cannot collide with
    # a descending command, so the protection is an overlap check rather
    # than a ban. The overlap itself is verified against the resolved rows
    # further down, where both sets exist; this is the cheap early refusal
    # for the relays that share their rows with the command by construction.
    _cmd2_requested = bool(
        th.get("cmd2") or th.get("cmd2_banc_ids") is not None)
    if _cmd2_requested and mode != _HS_DNA02_RELAY_MODE:
        raise ValueError(
            "cmd2 with a visual relay requires the HS-DNa02 relay mode")
    if (_cmd2_requested
            and input_source != _FLYVIS_BANC_OPPONENT_T4T5_VISUAL_INPUT):
        raise ValueError(
            "cmd2 with a visual relay requires the T4/T5 input source, "
            "whose input rows are optic-lobe intrinsic and so cannot be a "
            "descending command")
    relay_tau_ms = th.get("vision_relay_tau_ms", 100.0)
    if (type(relay_tau_ms) not in (int, float)
            or not np.isfinite(relay_tau_ms) or float(relay_tau_ms) <= 0.0):
        raise ValueError("vision_relay_tau_ms must be finite and positive")
    hs_baseline_hz = th.get("vision_hs_baseline_hz", 0.0)
    if (type(hs_baseline_hz) not in (int, float)
            or not np.isfinite(hs_baseline_hz)
            or not 0.0 <= float(hs_baseline_hz) <= float(rate_hz)):
        raise ValueError(
            "vision_hs_baseline_hz must be finite and in [0, vision_hz]")
    if mode != _HS_DNA02_RELAY_MODE and float(hs_baseline_hz) != 0.0:
        raise ValueError("vision_hs_baseline_hz requires the HS-DNa02 relay")
    if input_source in (
            _RENDERED_VISUAL_INPUT, _FLYVIS_HS_TAPE_VISUAL_INPUT,
            _FLYVIS_BANC_OPPONENT_T4T5_VISUAL_INPUT):
        if mode != _HS_DNA02_RELAY_MODE:
            raise ValueError("rendered panorama input requires the HS relay")
        if side_map is not None:
            raise ValueError("rendered panorama input forbids vision_side_map")
        if float(contrast) != 0.25:
            raise ValueError("rendered panorama input requires contrast 0.25")
        if float(hs_baseline_hz) != 0.0:
            raise ValueError("rendered panorama input requires zero HS baseline")
    flyvis_side_tape = None
    flyvis_movie_sha256 = None
    flyvis_side_tape_tick_ms = None
    flyvis_t4t5_rate_tape = None
    flyvis_t4t5_rate_tape_sha256 = None
    flyvis_t4t5_rate_tape_tick_ms = None
    flyvis_t4t5_groups = None
    flyvis_t4t5_label_source = "cell_type"
    flyvis_hss_only = th.get("vision_flyvis_hss_only", False)
    if not isinstance(flyvis_hss_only, bool):
        raise ValueError("vision_flyvis_hss_only must be Boolean")
    if (flyvis_hss_only
            and input_source != _FLYVIS_HS_TAPE_VISUAL_INPUT):
        raise ValueError(
            "vision_flyvis_hss_only requires the FlyVis HS side tape")
    carrier_frame = None
    carrier_spec = None
    carrier_u = None
    carrier_on = None
    flyvis_t4t5_stimulus_from_drive = False
    if input_source == _FLYVIS_HS_TAPE_VISUAL_INPUT:
        if any(key in th for key in (
                "vision_flyvis_t4t5_rate_tape",
                "vision_flyvis_t4t5_rate_tape_sha256",
                "vision_flyvis_t4t5_rate_tape_tick_ms",
                "vision_flyvis_t4t5_groups",
                "vision_flyvis_t4t5_label_source",
                "vision_tape_stimulus_from_drive")):
            raise ValueError(
                "FlyVis HS side-tape input rejects T4/T5 tape fields")
        supplied = th.get("vision_flyvis_side_tape")
        if not isinstance(supplied, list):
            raise ValueError("FlyVis input requires a list side tape")
        flyvis_side_tape = np.asarray(supplied)
        if (flyvis_side_tape.ndim != 1 or len(flyvis_side_tape) == 0
                or not np.isin(flyvis_side_tape, (-1, 0, 1)).all()
                or any(type(value) is not int for value in supplied)):
            raise ValueError(
                "FlyVis side tape must be a nonempty integer -1/0/1 vector")
        flyvis_side_tape = flyvis_side_tape.astype(np.int8)
        flyvis_movie_sha256 = th.get("vision_flyvis_movie_sha256")
        if (type(flyvis_movie_sha256) is not str
                or len(flyvis_movie_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in flyvis_movie_sha256)):
            raise ValueError("FlyVis movie SHA-256 must be lowercase hex")
        flyvis_side_tape_tick_ms = th.get("tick_ms", 5.0)
        if (type(flyvis_side_tape_tick_ms) not in (int, float)
                or not np.isfinite(flyvis_side_tape_tick_ms)
                or float(flyvis_side_tape_tick_ms) <= 0.0):
            raise ValueError("FlyVis side tape tick must be finite and positive")
    elif input_source == _FLYVIS_BANC_OPPONENT_T4T5_VISUAL_INPUT:
        if "vision_flyvis_side_tape" in th:
            raise ValueError(
                "FlyVis T4/T5 input rejects the retired HS side tape")
        expected_groups = [
            {"cell_type": "T4a", "side": "left"},
            {"cell_type": "T5a", "side": "left"},
            {"cell_type": "T4a", "side": "right"},
            {"cell_type": "T5a", "side": "right"},
            {"cell_type": "T4b", "side": "left"},
            {"cell_type": "T5b", "side": "left"},
            {"cell_type": "T4b", "side": "right"},
            {"cell_type": "T5b", "side": "right"},
        ]
        supplied = th.get("vision_flyvis_t4t5_rate_tape")
        if not isinstance(supplied, list):
            raise ValueError("FlyVis T4/T5 input requires a list rate tape")
        flyvis_t4t5_rate_tape = np.asarray(supplied)
        if (flyvis_t4t5_rate_tape.ndim != 2
                or flyvis_t4t5_rate_tape.shape[1] != len(expected_groups)
                or len(flyvis_t4t5_rate_tape) == 0
                or not np.isfinite(flyvis_t4t5_rate_tape).all()
                or np.any(flyvis_t4t5_rate_tape < 0.0)
                or np.any(
                    flyvis_t4t5_rate_tape > float(rate_hz) + 1e-12)):
            raise ValueError(
                "opponent FlyVis T4/T5 rate tape must be finite N x 8 in "
                "[0, vision_hz]")
        flyvis_t4t5_rate_tape = flyvis_t4t5_rate_tape.astype(float)
        flyvis_t4t5_rate_tape_sha256 = th.get(
            "vision_flyvis_t4t5_rate_tape_sha256")
        if (type(flyvis_t4t5_rate_tape_sha256) is not str
                or len(flyvis_t4t5_rate_tape_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in flyvis_t4t5_rate_tape_sha256)):
            raise ValueError(
                "FlyVis T4/T5 tape SHA-256 must be lowercase hex")
        observed_tape_sha256 = hashlib.sha256(
            np.asarray(flyvis_t4t5_rate_tape, dtype="<f8")
            .tobytes(order="C")).hexdigest()
        if observed_tape_sha256 != flyvis_t4t5_rate_tape_sha256:
            raise ValueError("FlyVis T4/T5 rate tape SHA-256 mismatch")
        flyvis_movie_sha256 = th.get("vision_flyvis_movie_sha256")
        if (type(flyvis_movie_sha256) is not str
                or len(flyvis_movie_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in flyvis_movie_sha256)):
            raise ValueError("FlyVis movie SHA-256 must be lowercase hex")
        if th.get("vision_flyvis_t4t5_groups") != expected_groups:
            raise ValueError(
                "FlyVis T4/T5 groups do not match the exact source order")
        flyvis_t4t5_groups = expected_groups
        flyvis_t4t5_label_source = th.get(
            "vision_flyvis_t4t5_label_source", "cell_type")
        # C-V80: the tape path derives "the stimulus is present" from the
        # class decode, so a relay that reports a SIDE and no class - the
        # only kind that can carry direction - leaves in_stimulus False on
        # every tick and its drive gate never applies (C-V76R, gate ticks
        # 0/0 with the HS side EMA at 5.73 against 0.063). Reading it from
        # the tape's own nonzero drive instead says what the words mean.
        # Default False is byte-identical.
        flyvis_t4t5_stimulus_from_drive = th.get(
            "vision_tape_stimulus_from_drive", False)
        if not isinstance(flyvis_t4t5_stimulus_from_drive, bool):
            raise ValueError(
                "vision_tape_stimulus_from_drive must be Boolean")
        if flyvis_t4t5_label_source not in _T4T5_LABEL_SOURCES:
            raise ValueError(
                "vision_flyvis_t4t5_label_source must be 'cell_type' or "
                "'cell_type_or_fafb_alignment'")
        flyvis_t4t5_rate_tape_tick_ms = th.get(
            "vision_flyvis_t4t5_rate_tape_tick_ms")
        if (type(flyvis_t4t5_rate_tape_tick_ms) not in (int, float)
                or not np.isfinite(flyvis_t4t5_rate_tape_tick_ms)
                or not np.isclose(
                    float(flyvis_t4t5_rate_tape_tick_ms), 2.5,
                    rtol=0.0, atol=1e-12)
                or not np.isclose(
                    float(th.get("tick_ms", 5.0)),
                    float(flyvis_t4t5_rate_tape_tick_ms),
                    rtol=0.0, atol=1e-12)):
            raise ValueError(
                "FlyVis T4/T5 v1 tape and body callback ticks must both "
                "equal the frozen 2.5 ms artifact cadence")
    elif input_source == _RENDERED_CARRIER_VISUAL_INPUT:
        frame_rel = th.get("vision_carrier_frame")
        if not isinstance(frame_rel, str) or not frame_rel:
            raise ValueError("carrier input requires vision_carrier_frame")
        frame_path = os.path.join(os.path.dirname(__file__), frame_rel)
        with open(frame_path, "rb") as handle:
            frame_bytes = handle.read()
        observed = hashlib.sha256(frame_bytes).hexdigest()
        declared = th.get("vision_carrier_frame_sha256")
        if (type(declared) is not str or len(declared) != 64
                or observed != declared):
            raise ValueError("carrier frame SHA-256 mismatch")
        carrier_frame = json.loads(frame_bytes.decode())
        spec = {}
        for key, lo, hi in (("peak_hz", 0.0, 1000.0),
                            ("cycles", 0.5, 64.0),
                            ("temporal_hz", 0.0, 50.0)):
            value = th.get("vision_carrier_" + key)
            if (type(value) not in (int, float) or not np.isfinite(value)
                    or not lo <= float(value) <= hi):
                raise ValueError(
                    f"vision_carrier_{key} must be finite in [{lo}, {hi}]")
            spec[key] = float(value)
        carrier_direction = th.get("vision_carrier_direction")
        if carrier_direction not in (1, -1):
            raise ValueError("vision_carrier_direction must be 1 or -1")
        spec["direction"] = float(carrier_direction)
        window = th.get("vision_carrier_window_ms", [500.0, 5500.0])
        if (not isinstance(window, list) or len(window) != 2
                or not all(type(x) in (int, float) and np.isfinite(x)
                           for x in window)
                or not 0.0 <= window[0] < window[1]):
            raise ValueError("vision_carrier_window_ms must be [lo, hi]")
        spec["window_ms"] = [float(window[0]), float(window[1])]
        carrier_spec = spec
    elif ("vision_flyvis_side_tape" in th
          or "vision_flyvis_t4t5_rate_tape" in th
          or "vision_flyvis_t4t5_rate_tape_sha256" in th
          or "vision_flyvis_t4t5_rate_tape_tick_ms" in th
          or "vision_flyvis_t4t5_groups" in th
          or "vision_flyvis_t4t5_label_source" in th
          or "vision_tape_stimulus_from_drive" in th
          or "vision_flyvis_movie_sha256" in th):
        raise ValueError("FlyVis tape fields require the FlyVis input source")
    relay_code = th.get("vision_relay_code", "absolute")
    if relay_code not in (
            "absolute", "prestim_delta", "hs_opponent", "hs_centroid_v1",
            "hs_graded_opponent"):
        raise ValueError(
            "vision_relay_code must be 'absolute', 'prestim_delta' or "
            "one of 'hs_opponent'/'hs_centroid_v1'/'hs_graded_opponent'")
    if mode != _HS_DNA02_RELAY_MODE and relay_code != "absolute":
        raise ValueError("vision_relay_code requires the HS-DNa02 relay")
    if relay_code in (
            "hs_opponent", "hs_centroid_v1", "hs_graded_opponent"
            ) and side_map is not None:
        raise ValueError(
            "HS activity decoders have fixed maps and forbid vision_side_map")
    hs_signed_drive = th.get("vision_hs_signed_drive", False)
    if type(hs_signed_drive) is not bool:
        raise ValueError("vision_hs_signed_drive must be Boolean")
    if hs_signed_drive and not (
            mode == _HS_DNA02_RELAY_MODE
            and relay_code == "hs_graded_opponent"
            and input_source == _RENDERED_VISUAL_INPUT):
        raise ValueError(
            "vision_hs_signed_drive requires the rendered graded HS relay")
    teacher_steering_gain = th.get("vision_teacher_steering_gain", 0.0)
    teacher_steering_legs = _resolve_visual_teacher_steering_legs(th)
    teacher_phase_asymmetry = th.get(
        "vision_teacher_phase_asymmetry", 0.0)
    teacher_leg_phase_asymmetry = (
        _resolve_visual_teacher_leg_phase_asymmetry(th))
    if (type(teacher_steering_gain) not in (int, float)
            or not np.isfinite(teacher_steering_gain)
            or not 0.0 <= float(teacher_steering_gain) <= 1.0):
        raise ValueError(
            "vision_teacher_steering_gain must be finite and in [0, 1]")
    if (float(teacher_steering_gain) != 0.0
            and (mode != _HS_DNA02_RELAY_MODE
                 or relay_code not in _HS_CLASS_RELAY_CODES)):
        raise ValueError(
            "vision_teacher_steering_gain requires hs_centroid_v1")
    if (teacher_steering_legs != "all"
            and (mode != _HS_DNA02_RELAY_MODE
                 or relay_code not in _HS_CLASS_RELAY_CODES)):
        raise ValueError(
            "vision_teacher_steering_legs requires hs_centroid_v1")
    if float(teacher_steering_gain) != 0.0 and return_scale is not None:
        raise ValueError(
            "teacher steering and neural return scaling are mutually exclusive")
    if (type(teacher_phase_asymmetry) not in (int, float)
            or not np.isfinite(teacher_phase_asymmetry)
            or not 0.0 <= float(teacher_phase_asymmetry) < 1.0):
        raise ValueError(
            "vision_teacher_phase_asymmetry must be finite in [0, 1)")
    if (float(teacher_phase_asymmetry) != 0.0
            and teacher_leg_phase_asymmetry is not None):
        raise ValueError(
            "global and per-leg visual phase asymmetry are mutually exclusive")
    phase_control_active = bool(
        float(teacher_phase_asymmetry) != 0.0
        or (teacher_leg_phase_asymmetry is not None
            and np.any(teacher_leg_phase_asymmetry != 0.0)))
    phase_from_detector = _resolve_visual_phase_from_detector(
        th, input_source, relay_code, phase_control_active)
    phase_neural_motion_control = (
        _resolve_visual_phase_neural_motion_control(
            th, input_source, relay_code, phase_control_active,
            phase_from_detector, image_motion_profile,
            image_body_reafference, image_motion_cancellation))
    phase_dna02_amplitude = th.get(
        "vision_phase_dna02_amplitude", False)
    if type(phase_dna02_amplitude) is not bool:
        raise ValueError("vision_phase_dna02_amplitude must be Boolean")
    phase_dna02_routed_amplitude = th.get(
        "vision_phase_dna02_routed_amplitude", False)
    if type(phase_dna02_routed_amplitude) is not bool:
        raise ValueError(
            "vision_phase_dna02_routed_amplitude must be Boolean")
    if phase_dna02_amplitude and phase_dna02_routed_amplitude:
        raise ValueError("summed and routed DNa02 amplitude are exclusive")
    phase_dna02_amplitude_active = bool(
        phase_dna02_amplitude or phase_dna02_routed_amplitude)
    dna02_step_return_gate = th.get(
        "vision_dna02_measured_step_return_gate", False)
    if type(dna02_step_return_gate) is not bool:
        raise ValueError(
            "vision_dna02_measured_step_return_gate must be Boolean")
    if dna02_step_return_gate and not phase_dna02_routed_amplitude:
        raise ValueError(
            "DNa02 measured-step return gate requires routed amplitude")
    if phase_dna02_amplitude_active and not (
            mode == _HS_DNA02_RELAY_MODE
            and relay_code in _HS_CLASS_RELAY_CODES
            and phase_neural_motion_control
            and phase_control_active
            and float(rate_hz) > 0.0):
        raise ValueError(
            "vision_phase_dna02_amplitude requires positive-rate live-HS "
            "centroid phase control")
    phase_reset_on_no_motion = th.get(
        "vision_phase_reset_on_no_motion", False)
    if type(phase_reset_on_no_motion) is not bool:
        raise ValueError("vision_phase_reset_on_no_motion must be Boolean")
    if (phase_reset_on_no_motion
            and (not (phase_from_detector or phase_neural_motion_control)
                 or not _is_motion_stop_profile(image_motion_profile))):
        raise ValueError(
            "vision_phase_reset_on_no_motion requires detector or neural "
            "motion phase and "
            "motion_stop_reversal_v1")
    if (phase_reset_on_no_motion
            and input_source == _FLYVIS_BANC_OPPONENT_T4T5_VISUAL_INPUT):
        raise ValueError(
            "actual T4/T5 input forbids a parallel tape-to-body no-motion "
            "reset; live evidence-graded HS must own body phase alone")
    phase_brake_deadband_deg_s = th.get(
        "vision_phase_brake_deadband_deg_s", 5.0)
    if (type(phase_brake_deadband_deg_s) not in (int, float)
            or not np.isfinite(phase_brake_deadband_deg_s)
            or not 0.0 < float(phase_brake_deadband_deg_s) <= 20.0):
        raise ValueError(
            "vision_phase_brake_deadband_deg_s must be in (0, 20]")
    phase_brake_deadband_deg_s = float(phase_brake_deadband_deg_s)
    # Default-off proportional braking: the correction scales with how far
    # the measured body rate exceeds the deadband, capped. Absent = the
    # fixed-size correction the brake has always applied.
    phase_brake_proportional_cap = th.get(
        "vision_phase_brake_proportional_cap")
    if phase_brake_proportional_cap is not None and (
            type(phase_brake_proportional_cap) not in (int, float)
            or not np.isfinite(phase_brake_proportional_cap)
            or not 1.0 <= float(phase_brake_proportional_cap) <= 10.0):
        raise ValueError(
            "vision_phase_brake_proportional_cap must be in [1, 10]")
    if phase_brake_proportional_cap is not None:
        phase_brake_proportional_cap = float(phase_brake_proportional_cap)
    phase_brake_on_no_motion = _resolve_visual_phase_brake_on_no_motion(
        th, input_source, image_motion_profile, image_body_reafference,
        image_motion_cancellation,
        phase_from_detector or phase_neural_motion_control,
        phase_reset_on_no_motion)
    hs_forward_speed_phase_gain = (
        _resolve_visual_hs_forward_speed_phase_gain(
            th, relay_code, phase_control_active,
            image_body_reafference))
    phase_brake_filter_body_rate = (
        _resolve_visual_phase_brake_filter_body_rate(
            th, phase_brake_on_no_motion))
    phase_brake_proportional_body_rate = (
        _resolve_visual_phase_brake_proportional_body_rate(
            th, phase_brake_on_no_motion,
            phase_brake_filter_body_rate))
    phase_brake_memoryless_body_rate = (
        _resolve_visual_phase_brake_memoryless_body_rate(
            th, phase_brake_on_no_motion,
            phase_brake_filter_body_rate,
            phase_brake_proportional_body_rate))
    phase_brake_continuous_body_rate = bool(
        phase_brake_proportional_body_rate
        or phase_brake_memoryless_body_rate)
    if (phase_control_active
            and (mode != _HS_DNA02_RELAY_MODE
                 or relay_code not in _HS_CLASS_RELAY_CODES)):
        raise ValueError(
            "visual teacher phase control requires hs_centroid_v1")
    if (phase_control_active
            and (float(teacher_steering_gain) != 0.0
                 or return_scale is not None)):
        raise ValueError(
            "visual side-phase control is mutually exclusive with drive gates")

    dna_rows, dna_sides = _resolve_visual_named_rows(
        meta, net, ("DNa02",), "descending")
    dna02_max_rate_hz = None
    dna02_max_rate_hz_by_side = None
    if phase_dna02_amplitude_active:
        if dna_sides.tolist() != [0, 1]:
            raise ValueError("DNa02 rows must resolve in left/right order")
        refractory = np.broadcast_to(
            np.asarray(net.p.t_refrac, dtype=float), (net.N,))
        selected_refractory = refractory[dna_rows]
        if (not np.isfinite(selected_refractory).all()
                or np.any(selected_refractory <= 0.0)):
            raise ValueError("DNa02 refractory values must be positive")
        dna02_max_rate_hz_by_side = 1000.0 / selected_refractory
        dna02_max_rate_hz = float(np.sum(dna02_max_rate_hz_by_side))
    input_rows = dna_rows
    input_sides = np.asarray([0, 1], dtype=np.int8)
    input_group_indices = None
    hs_rows = dna_rows[:0]
    hs_sides = np.asarray([], dtype=np.int8)
    if mode == _HS_DNA02_RELAY_MODE:
        hs_rows, hs_sides = _resolve_visual_named_rows(
            meta, net, _OPTOMOTOR_HS_TYPES, "visual_projection")
        input_rows, input_sides = hs_rows, hs_sides
        if flyvis_hss_only:
            input_rows, input_sides = _resolve_visual_named_rows(
                meta, net, ("HSS",), "visual_projection")
        if input_source == _RENDERED_CARRIER_VISUAL_INPUT:
            cells = carrier_frame["cells"]
            id_to_row = {str(b): r for r, b in enumerate(net.ids)}
            rows, us, ons = [], [], []
            for banc_id, rec in cells.items():
                row = id_to_row.get(str(banc_id))
                if row is None:
                    continue
                rows.append(row)
                us.append(float(rec["u"]))
                ons.append(rec["pathway"] == "ON")
            if len(rows) < 1000:
                raise ValueError(
                    "carrier frame resolved too few rows in this core")
            order = np.argsort(np.asarray(rows, dtype=np.int64))
            input_rows = np.asarray(rows, dtype=np.int64)[order]
            carrier_u = np.asarray(us, dtype=float)[order]
            carrier_on = np.asarray(ons, dtype=bool)[order]
            if len(np.unique(input_rows)) != len(input_rows):
                raise ValueError("carrier frame named a row twice")
            if (np.intersect1d(input_rows, hs_rows).size
                    or np.intersect1d(input_rows, dna_rows).size):
                raise ValueError(
                    "carrier rows overlap the HS or DNa02 populations")
            graded = getattr(net, "graded", None)
            if (graded is not None
                    and np.asarray(graded, dtype=bool)[input_rows].any()):
                raise ValueError(
                    "carrier drive requires zero graded input rows")
            side_of = meta["side"].astype(str).to_numpy()[input_rows]
            input_sides = np.where(side_of == "left", 0, 1).astype(np.int8)
            input_group_indices = (
                (~carrier_on).astype(np.int8) * 2 + input_sides).astype(np.int8)
        elif input_source == _FLYVIS_BANC_OPPONENT_T4T5_VISUAL_INPUT:
            row_blocks = []
            side_blocks = []
            group_blocks = []
            expected_counts = list(
                _T4T5_LABEL_SOURCES[flyvis_t4t5_label_source])
            merged_label = None
            if flyvis_t4t5_label_source != "cell_type":
                primary = meta["cell_type"].astype("string")
                fallback = meta[_T4T5_ALIGNMENT_COLUMN].astype("string")
                both = primary.notna() & fallback.notna()
                contradiction = (
                    primary[both].isin(_OPTOMOTOR_T4T5_TYPES)
                    & (primary[both] != fallback[both]).fillna(False))
                if bool(np.asarray(contradiction, dtype=bool).any()):
                    raise ValueError(
                        "FlyVis BANC T4/T5 alignment labels contradict "
                        "cell_type on a T4/T5 row")
                merged_label = primary.where(primary.notna(), fallback)
            for group_index, (group, expected_count) in enumerate(zip(
                    flyvis_t4t5_groups, expected_counts)):
                if merged_label is None:
                    selected_rows = net.select(
                        cell_type=group["cell_type"], side=group["side"],
                        super_class="optic_lobe_intrinsic")
                else:
                    selected_rows = np.flatnonzero(
                        (merged_label == group["cell_type"])
                        .fillna(False).to_numpy(dtype=bool)
                        & (meta["side"] == group["side"])
                        .to_numpy(dtype=bool)
                        & (meta["super_class"] == "optic_lobe_intrinsic")
                        .to_numpy(dtype=bool))
                if len(selected_rows) != expected_count:
                    raise ValueError(
                        "FlyVis BANC T4/T5 group row-count drift")
                selected_rows = np.asarray(selected_rows, dtype=np.int64)
                identifiers = meta.iloc[selected_rows]["banc_888_id"].astype(str)
                if identifiers.isin(("", "nan", "None")).any():
                    raise ValueError(
                        "FlyVis BANC T4/T5 group has missing identities")
                row_blocks.append(selected_rows)
                side_index = 0 if group["side"] == "left" else 1
                side_blocks.append(np.full(
                    len(selected_rows), side_index, dtype=np.int8))
                group_blocks.append(np.full(
                    len(selected_rows), group_index, dtype=np.int8))
            input_rows = np.concatenate(row_blocks)
            input_sides = np.concatenate(side_blocks)
            input_group_indices = np.concatenate(group_blocks)
            if (len(np.unique(input_rows)) != len(input_rows)
                    or np.intersect1d(input_rows, hs_rows).size
                    or np.intersect1d(input_rows, dna_rows).size):
                raise ValueError(
                    "FlyVis BANC T4/T5 rows overlap each other or HS/DNa02")
            graded = getattr(net, "graded", None)
            if (graded is None
                    or np.asarray(graded, dtype=bool).shape != (net.N,)
                    or np.asarray(graded, dtype=bool)[input_rows].any()):
                raise ValueError(
                    "FlyVis BANC T4/T5 drive requires zero graded input "
                    "rows; the cross-backend contract is Poisson-spiking "
                    "T4/T5 into graded HS")
    hs_opponent_prestim_baseline = th.get(
        "vision_hs_opponent_prestim_baseline", False)
    if not isinstance(hs_opponent_prestim_baseline, bool):
        raise ValueError(
            "vision_hs_opponent_prestim_baseline must be Boolean")
    if hs_opponent_prestim_baseline and relay_code != "hs_opponent":
        raise ValueError(
            "vision_hs_opponent_prestim_baseline requires hs_opponent")
    if relay_code == "hs_graded_opponent":
        graded = getattr(net, "graded", None)
        if (graded is None
                or np.asarray(graded, dtype=bool).shape != (net.N,)
                or not np.asarray(graded, dtype=bool)[hs_rows].all()):
            raise ValueError(
                "hs_graded_opponent requires all resolved HS rows graded")
    if hs_signed_drive:
        if float(rate_hz) <= 0.0:
            raise ValueError(
                "vision_hs_signed_drive requires positive vision_hz")
        if float(hs_baseline_hz) != 0.0:
            raise ValueError(
                "vision_hs_signed_drive requires zero HS baseline")
    return {
        "mode": mode,
        "deliver_only": deliver_only,
        "input_source": input_source,
        "image_motion_hz": float(image_motion_hz),
        "image_motion_profile": image_motion_profile,
        "image_body_reafference": image_body_reafference,
        "image_motion_cancellation": image_motion_cancellation,
        "flyvis_side_tape": flyvis_side_tape,
        "flyvis_side_tape_tick_ms": (
            None if flyvis_side_tape_tick_ms is None
            else float(flyvis_side_tape_tick_ms)),
        "flyvis_movie_sha256": flyvis_movie_sha256,
        "flyvis_t4t5_rate_tape": flyvis_t4t5_rate_tape,
        "flyvis_t4t5_rate_tape_sha256": flyvis_t4t5_rate_tape_sha256,
        "flyvis_t4t5_rate_tape_tick_ms": (
            None if flyvis_t4t5_rate_tape_tick_ms is None
            else float(flyvis_t4t5_rate_tape_tick_ms)),
        "flyvis_t4t5_groups": flyvis_t4t5_groups,
        "flyvis_t4t5_label_source": flyvis_t4t5_label_source,
        "flyvis_hss_only": flyvis_hss_only,
        "carrier_spec": carrier_spec,
        "carrier_u": carrier_u,
        "carrier_on": carrier_on,
        "flyvis_t4t5_stimulus_from_drive": flyvis_t4t5_stimulus_from_drive,
        "hs_opponent_prestim_baseline": hs_opponent_prestim_baseline,
        "rows": dna_rows,
        "input_rows": input_rows,
        "input_sides": input_sides,
        "input_group_indices": input_group_indices,
        "hs_rows": hs_rows,
        "hs_sides": hs_sides,
        "direction": mapped_direction,
        "stimulus_direction": direction,
        "side_map": side_map,
        "contrast": float(contrast),
        "rate_hz": float(rate_hz),
        "return_scale": (None if return_scale is None
                         else float(return_scale)),
        "relay_tau_ms": float(relay_tau_ms),
        "hs_baseline_hz": float(hs_baseline_hz),
        "hs_signed_drive": hs_signed_drive,
        "relay_code": relay_code,
        "teacher_steering_gain": float(teacher_steering_gain),
        "teacher_steering_legs": teacher_steering_legs,
        "teacher_phase_asymmetry": float(teacher_phase_asymmetry),
        "teacher_leg_phase_asymmetry": (
            None if teacher_leg_phase_asymmetry is None
            else teacher_leg_phase_asymmetry.astype(float).tolist()),
        "phase_from_detector": phase_from_detector,
        "phase_neural_motion_control": phase_neural_motion_control,
        "phase_dna02_amplitude": phase_dna02_amplitude_active,
        "phase_dna02_amplitude_mode": (
            "routed" if phase_dna02_routed_amplitude
            else "summed" if phase_dna02_amplitude else "off"),
        "dna02_max_rate_hz": dna02_max_rate_hz,
        "dna02_max_rate_hz_by_side": (
            None if dna02_max_rate_hz_by_side is None
            else dna02_max_rate_hz_by_side.astype(float).tolist()),
        "dna02_measured_step_return_gate": dna02_step_return_gate,
        "phase_reset_on_no_motion": phase_reset_on_no_motion,
        "phase_brake_on_no_motion": phase_brake_on_no_motion,
        "hs_forward_speed_phase_gain": hs_forward_speed_phase_gain,
        "phase_brake_deadband_deg_s": phase_brake_deadband_deg_s,
        "phase_brake_proportional_cap": phase_brake_proportional_cap,
        "phase_brake_filter_body_rate": phase_brake_filter_body_rate,
        "phase_brake_proportional_body_rate": (
            phase_brake_continuous_body_rate),
        "phase_brake_memoryless_body_rate": (
            phase_brake_memoryless_body_rate),
    }


def _apply_optomotor_return_gate(t_ms, bridge, dofs_by_side, prev_loads,
                                 *channels, active_side=None,
                                 scale_override=None):
    """Scale commanded-side unloaded-leg drive; return affected leg names."""
    if dofs_by_side is None:
        return ()
    if active_side is None:
        active_side = _optomotor_active_side(
            t_ms, bridge["direction"], bridge["contrast"])
    if active_side is None:
        return ()
    scale = (bridge["return_scale"] if scale_override is None
             else float(scale_override))
    if scale is None:
        return ()
    affected = []
    for leg, dofs in dofs_by_side[active_side].items():
        # Previous-tick contact is the causal phase signal. The first tick has
        # no observation and is deliberately left unchanged.
        if leg in prev_loads and float(prev_loads[leg]) <= 1.0:
            for channel in channels:
                channel[dofs] *= scale
            affected.append(leg)
    return tuple(affected)


def _resolve_steering_trim(th):
    """Return the bounded default-off bilateral drive trim."""
    value = th.get("steering_trim")
    if value is None:
        return 0.0
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not -0.5 <= float(value) <= 0.5):
        raise ValueError("steering_trim must be finite and in [-0.5, 0.5]")
    return float(value)


def _resolve_neural_motor_scale(th):
    """Return one global final neural motor-readout coefficient.

    Scope is one scalar shared by every mapped motor-neuron row, both muscle
    channels and all 42 DOFs. The absent/default value is exactly 1.0.
    """
    value = th.get("neural_motor_scale", 1.0)
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not 0.0 <= float(value) <= 1.0):
        raise ValueError("neural_motor_scale must be finite and in [0, 1]")
    return float(value)


def _resolve_neural_motor_right_class_dof_substitution(th):
    """Resolve the default-off drive-budget-preserving MN substitution."""
    value = th.get("neural_motor_right_class_dof_substitution", False)
    if type(value) is not bool:
        raise ValueError(
            "neural_motor_right_class_dof_substitution must be Boolean")
    detector_zero = th.get(
        "neural_motor_detector_zero_dof_substitution", False)
    if type(detector_zero) is not bool:
        raise ValueError(
            "neural_motor_detector_zero_dof_substitution must be Boolean")
    if detector_zero and not value:
        raise ValueError(
            "detector-zero substitution requires right-class substitution")
    if not value:
        return None
    return {
        "detector_zero": detector_zero,
        "ticks_class0_class1_none": np.zeros(3, dtype=np.int64),
        "eligible_right_ticks": 0,
        "eligible_detector_zero_ticks": 0,
        "substituted_dof_ticks": 0,
        "fallback_dof_ticks": 0,
        "zero_budget_dof_ticks": 0,
        "offered_neural_abs_drive": 0.0,
        "teacher_abs_drive": 0.0,
        "substituted_teacher_abs_drive": 0.0,
        "final_abs_drive": 0.0,
        "max_dof_budget_abs_error": 0.0,
        "sum_dof_budget_abs_error": 0.0,
    }


def _apply_neural_motor_right_class_dof_substitution(
        raw_flex, raw_ext, step_flex, step_ext, eligible, config,
        eligible_source="right"):
    """Replace a right-class teacher command with normalized MN composition."""
    if config is None:
        return raw_flex + step_flex, raw_ext + step_ext
    step_flex = np.asarray(step_flex, dtype=float)
    step_ext = np.asarray(step_ext, dtype=float)
    if not eligible:
        return step_flex.copy(), step_ext.copy()
    if eligible_source not in ("right", "detector_zero"):
        raise ValueError("unknown neural DOF substitution source")
    config[f"eligible_{eligible_source}_ticks"] += 1
    raw_flex = np.asarray(raw_flex, dtype=float)
    raw_ext = np.asarray(raw_ext, dtype=float)
    if not (raw_flex.shape == raw_ext.shape == step_flex.shape
            == step_ext.shape):
        raise ValueError("neural DOF substitution channels must share shape")
    raw_abs = np.abs(raw_flex) + np.abs(raw_ext)
    step_abs = np.abs(step_flex) + np.abs(step_ext)
    offered = raw_abs > 0.0
    budgeted = step_abs > 0.0
    substituted = offered & budgeted
    fallback = ~offered & budgeted
    zero_budget = ~budgeted
    final_flex = step_flex.copy()
    final_ext = step_ext.copy()
    final_flex[zero_budget] = 0.0
    final_ext[zero_budget] = 0.0
    factor = np.zeros_like(step_abs)
    factor[substituted] = step_abs[substituted] / raw_abs[substituted]
    final_flex[substituted] = raw_flex[substituted] * factor[substituted]
    final_ext[substituted] = raw_ext[substituted] * factor[substituted]
    final_abs = np.abs(final_flex) + np.abs(final_ext)
    error = np.abs(final_abs - step_abs)
    config["substituted_dof_ticks"] += int(substituted.sum())
    config["fallback_dof_ticks"] += int(fallback.sum())
    config["zero_budget_dof_ticks"] += int(zero_budget.sum())
    config["offered_neural_abs_drive"] += float(raw_abs.sum())
    config["teacher_abs_drive"] += float(step_abs.sum())
    config["substituted_teacher_abs_drive"] += float(
        step_abs[substituted].sum())
    config["final_abs_drive"] += float(final_abs.sum())
    config["max_dof_budget_abs_error"] = max(
        float(config["max_dof_budget_abs_error"]), float(error.max()))
    config["sum_dof_budget_abs_error"] += float(error.sum())
    return final_flex, final_ext


def _visual_detector_zero_in_stimulus(
        time_ms, rendered_motion_state, selected_side):
    """Return sensed detector silence inside the existing visual envelope."""
    return bool(
        rendered_motion_state is not None
        and 500.0 <= float(time_ms) < 5500.0
        and selected_side is None)


def _apply_side_drive_trim(dofs_by_side, trim, *channels):
    """Apply one static left/right gain contrast to pre-muscle drive."""
    if dofs_by_side is None or trim == 0.0:
        return
    for side, factor in ((0, 1.0 + trim), (1, 1.0 - trim)):
        for dofs in dofs_by_side[side].values():
            for channel in channels:
                channel[dofs] *= factor


def _resolve_body_drive_leg_scale(th):
    """Resolve optional final motor-drive gains in motormap leg order."""
    values = th.get("body_drive_leg_gains")
    if values is None:
        return None
    if (not isinstance(values, list) or len(values) != 6
            or any(type(item) not in (int, float)
                   or not np.isfinite(item) or float(item) < 0.0
                   for item in values)):
        raise ValueError(
            "body_drive_leg_gains must be six finite nonnegative numbers, "
            "ordered lf lm lh rf rm rh")
    # DOF_NAMES is seven contiguous joints per leg in exactly LEGS order.
    return np.repeat(np.asarray(values, dtype=float), 7)


def _resolve_adhesion_leg_threshold_scales(th):
    """Resolve optional per-leg adhesion thresholds in motormap leg order.

    The legacy flex gate uses one absolute threshold for all six legs.  This
    default-off seam preserves that path exactly while allowing the threshold
    to be a searched per-leg body parameter when the motor pools occupy
    different activation ranges.
    """
    values = th.get("adh_leg_threshold_scales")
    if values is None:
        return None
    if (not isinstance(values, list) or len(values) != 6
            or any(type(item) not in (int, float)
                   or not np.isfinite(item) or float(item) <= 0.0
                   for item in values)):
        raise ValueError(
            "adh_leg_threshold_scales must be six finite positive numbers, "
            "ordered lf lm lh rf rm rh")
    return np.asarray(values, dtype=float)


def _resolve_motor_balance_adhesion(th, use_split):
    """Validate the default-off antagonist-balance adhesion mode."""
    if th.get("adh") != "motor_balance":
        return False
    if not use_split:
        raise ValueError(
            "th['adh']='motor_balance' requires the split muscle path")
    forbidden = [key for key in ("adh_thresh", "adh_leg_threshold_scales")
                 if key in th]
    if forbidden:
        raise ValueError(
            "motor_balance adhesion forbids " + ", ".join(forbidden))
    return True


def _resolve_adhesion_knee_legs(th):
    """Per-leg knee-tied adhesion release, default off (judge-ported from
    lane A's CD1-PZ seam, 2026-09-03).

    CD1-PZ (lanes/A/2026-09-03-1525-cd1pz-result.md): under the flex gate
    the hind legs' release is decided by the ThC_roll and TiTa_pitch flexor
    channels (55-65% and 19-29% of the gate mean) and by the levator at
    under 1%, so the hind foot's grip cycle carries no relation to its knee,
    which is the joint that moves the hind tip fore-aft (0.84-0.98 mm/rad
    against 0.36-0.44 for the CTr). For the legs listed here the gate reads
    the FTi_pitch a_ext channel alone: the channel that raises the FTi
    coordinate, which closes the knee on this body on all six legs
    (dknee/dFTi = -1.00, lanes/A/cd1pz_kin.py), so the foot releases while
    the knee is driven closed (swing) and grips while it is driven open
    (stance). Same threshold as the leg's flex gate. Absent = bit-identical.
    """
    values = th.get("adh_knee_legs")
    if values is None:
        return None
    legs = ["lf", "lm", "lh", "rf", "rm", "rh"]
    if (not isinstance(values, list) or not values
            or any(type(item) is not str or item not in legs
                   for item in values)
            or len(set(values)) != len(values)):
        raise ValueError(
            "adh_knee_legs must be a non-empty list of distinct legs "
            "among lf lm lh rf rm rh")
    return np.asarray([leg in values for leg in legs], dtype=bool)


def _motor_balance_adhesion_onoff(a_flex, a_ext, leg_dofs):
    """Attach legs whose mean extensor-minus-flexor activation is nonnegative."""
    if (len(leg_dofs) != 6
            or any(np.asarray(dofs).size != 7 for dofs in leg_dofs)):
        raise ValueError("motor_balance adhesion needs six 7-DOF leg groups")
    flex = np.asarray(a_flex, dtype=float)
    ext = np.asarray(a_ext, dtype=float)
    if flex.shape != ext.shape:
        raise ValueError("motor_balance flex/ext activations must match")
    return np.asarray([
        np.mean(ext[dofs] - flex[dofs]) >= 0.0 for dofs in leg_dofs
    ], dtype=bool)


def _resolve_body_joint_bias(th):
    """Resolve optional per-DOF passive-equilibrium offsets in radians."""
    import motormap as MM2

    values = th.get("body_joint_rest_offsets_deg")
    if values is None:
        return None
    if (not isinstance(values, dict) or not values
            or any(name not in MM2.DOF_INDEX for name in values)
            or any(type(value) not in (int, float)
                   or not np.isfinite(value) or abs(float(value)) > 20.0
                   for value in values.values())):
        raise ValueError(
            "body_joint_rest_offsets_deg must map known DOFs to finite "
            "offsets within +/-20 degrees")
    result = np.zeros(len(MM2.DOF_NAMES), dtype=float)
    for name, value in values.items():
        result[MM2.DOF_INDEX[name]] = np.radians(float(value))
    return result


def _resolve_body_joint_tau_act(th):
    """Resolve optional per-DOF muscle activation time constants."""
    import motormap as MM2

    values = th.get("body_joint_tau_act_ms")
    if values is None:
        return None
    if (not isinstance(values, dict) or not values
            or any(name not in MM2.DOF_INDEX for name in values)
            or any(type(value) not in (int, float)
                   or not np.isfinite(value)
                   or not 1.0 <= float(value) <= 200.0
                   for value in values.values())):
        raise ValueError(
            "body_joint_tau_act_ms must map known DOFs to finite values "
            "within [1, 200] ms")
    result = np.full(len(MM2.DOF_NAMES), B.TAU_ACT_MS, dtype=float)
    for name, value in values.items():
        result[MM2.DOF_INDEX[name]] = float(value)
    return result


def _resolve_tau_deact(th):
    """Resolve the optional relaxation constant of the two-channel
    activation (design 686abf38's build, ported for design c593eb5c), a
    scalar in ms within [1, 1000]. Absent => None => the symmetric filter,
    bit-identical."""
    value = th.get("tau_deact_ms")
    if value is None:
        return None
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not 1.0 <= float(value) <= 1000.0):
        raise ValueError("tau_deact_ms must be finite within [1, 1000] ms")
    return float(value)


def _resolve_body_joint_drive_scale(th):
    """Resolve optional per-DOF muscle loop gain (the joint-equilibrium
    denominator). Absent => None => Body builds a uniform vector at the scalar
    th['drive_scale'], bit-identical to the scalar path. A LOWER value on a DOF
    makes a given activation swing that joint further, which is the lever for a
    leg whose motor drive reaches the muscle weakly (right hind)."""
    import motormap as MM2

    values = th.get("body_joint_drive_scale")
    if values is None:
        return None
    base = float(th.get("drive_scale", B.Body().drive_scale))
    if (not isinstance(values, dict) or not values
            or any(name not in MM2.DOF_INDEX for name in values)
            or any(type(value) not in (int, float)
                   or not np.isfinite(value)
                   or not 1e-5 <= float(value) <= 2000.0
                   for value in values.values())):
        raise ValueError(
            "body_joint_drive_scale must map known DOFs to finite values "
            "within [1e-5, 2000]")
    result = np.full(len(MM2.DOF_NAMES), base, dtype=float)
    for name, value in values.items():
        result[MM2.DOF_INDEX[name]] = float(value)
    return result


def _resolve_body_joint_drive_offset(th):
    """Resolve the optional per-DOF activation offset o_d added to a before the
    tanh, theta_eq = mid + half tanh((a + o_d) / s_d) (design fcb48834 Build 2,
    the posture-matched control Xo). Absent, or every value zero, => None =>
    the Body's original expression, bit-identical (probe P2). Bound
    |o_d| <= 100 drive_scale, in activation units."""
    import motormap as MM2

    values = th.get("body_joint_drive_offset")
    if values is None:
        return None
    base = float(th.get("drive_scale", B.Body().drive_scale))
    if (not isinstance(values, dict) or not values
            or any(name not in MM2.DOF_INDEX for name in values)
            or any(type(value) not in (int, float)
                   or not np.isfinite(value)
                   or abs(float(value)) > 100.0 * base
                   for value in values.values())):
        raise ValueError(
            "body_joint_drive_offset must map known DOFs to finite values "
            "with |o_d| <= 100 drive_scale")
    if all(float(value) == 0.0 for value in values.values()):
        return None
    result = np.zeros(len(MM2.DOF_NAMES), dtype=float)
    for name, value in values.items():
        result[MM2.DOF_INDEX[name]] = float(value)
    return result


def _resolve_musc_class_gain(th):
    """Resolve design fcb48834 Build 7's per-class multiplier {fast, mid, slow}
    on the finalised motor map's rows (the cap's constant-gain twin T,
    z = g_class y). Absent, or every value 1.0, => None, bit-identical
    (probe P6)."""
    values = th.get("musc_class_gain")
    if values is None:
        return None
    if (not isinstance(values, dict) or set(values) != {"fast", "mid", "slow"}
            or any(type(v) not in (int, float) or not np.isfinite(v)
                   or not 0.0 < float(v) <= 10.0 for v in values.values())):
        raise ValueError("musc_class_gain must map fast, mid and slow to "
                         "finite values in (0, 10]")
    if all(float(v) == 1.0 for v in values.values()):
        return None
    return {k: float(v) for k, v in values.items()}


def _resolve_body_joint_f0(th):
    """Resolve optional per-DOF joint resonance. Absent => None => Body takes
    the scalar th['f0_hz'], bit-identical to the scalar path.

    The dial exists because the scalar cannot express the defect it is aimed at.
    C-N335 measured that the cord commands the full joint range on every graded
    DOF (ASK 1.000) and the joint's own second-order dynamics deliver 0.66 to
    0.86 of it; C-N336 then raised the scalar through its own measured anchor
    band and delivery rose on all six graded DOFs, front FTi range with it,
    while standing fell 11, 5, 2, 0 of twelve across 20.55, 26.0, 31.4 and 40.0
    Hz. The cost is paid by joints that were tracking their command already,
    which a scalar cannot exempt. Bounds [1, 200] Hz are the resolver's
    sanity rails and not the anchor: a run states its own band and asserts it.
    Lane C, 2026-09-11."""
    import motormap as MM2

    values = th.get("body_joint_f0")
    if values is None:
        return None
    base = th.get("f0_hz")
    if base is None:
        raise ValueError(
            "body_joint_f0 needs a scalar th['f0_hz'] as the base for every "
            "DOF it does not name; without one the first-order joint and the "
            "second-order joint would be mixed inside one body")
    if (not isinstance(values, dict) or not values
            or any(name not in MM2.DOF_INDEX for name in values)
            or any(type(value) not in (int, float)
                   or not np.isfinite(value)
                   or not 1.0 <= float(value) <= 200.0
                   for value in values.values())):
        raise ValueError(
            "body_joint_f0 must map known DOFs to finite values within [1, 200]")
    result = np.full(len(MM2.DOF_NAMES), float(base), dtype=float)
    for name, value in values.items():
        result[MM2.DOF_INDEX[name]] = float(value)
    return result


def _motor_sign_vector(mode, thc_yaw_moment_arm_orientation=None,
                       thc_roll_moment_arm_orientation=None):
    """Resolve measured signs and opt-in tied ThC orientation probes.

    The six absolute ThC-yaw motor-map signs are not among the 24 signs with
    geometry-derived evidence. ``"alternate"`` is therefore a binary BUILD
    hypothesis about that unresolved coordinate convention, not a measured
    biological correction. Absent remains the historical path exactly.

    ThC-roll (lane C, C-N212, 2026-09-09) has the same standing: its six
    absolute signs are not among the measured 24 either, and C-N209 measured
    it as the joint class with the largest lever from joint angle to foot
    azimuth and the only significant contribution OPPOSING the other three
    significant classes, which is why it gets the same shape of default-off
    probe. Both probes are independent and either may be named alone.
    """
    import motormap as MM2

    vector = None
    if mode is not None:
        if type(mode) is not str or mode != "measured":
            raise ValueError("th['signs'] must be absent, None, or 'measured'")
        signs = MM2.measured_sign_flip()
        if set(signs) != set(MM2.DOF_NAMES):
            raise ValueError(
                "measured motor signs do not cover the 42 DOFs exactly")
        vector = np.asarray(
            [signs[name] for name in MM2.DOF_NAMES], dtype=float)

    if thc_yaw_moment_arm_orientation is not None:
        if (type(thc_yaw_moment_arm_orientation) is not str
                or thc_yaw_moment_arm_orientation != "alternate"):
            raise ValueError(
                "th['thc_yaw_moment_arm_orientation'] must be absent, None, "
                "or 'alternate'")
        if vector is None:
            vector = np.ones(len(MM2.DOF_NAMES), dtype=float)
        else:
            vector = vector.copy()
        yaw_columns = np.asarray([
            index for index, name in enumerate(MM2.DOF_NAMES)
            if name.endswith("_ThC_yaw")
        ], dtype=int)
        if len(yaw_columns) != 6:
            raise ValueError(
                "alternate ThC-yaw orientation must resolve exactly six DOFs")
        vector[yaw_columns] *= -1.0

    if thc_roll_moment_arm_orientation is not None:
        if (type(thc_roll_moment_arm_orientation) is not str
                or thc_roll_moment_arm_orientation != "alternate"):
            raise ValueError(
                "th['thc_roll_moment_arm_orientation'] must be absent, None, "
                "or 'alternate'")
        if vector is None:
            vector = np.ones(len(MM2.DOF_NAMES), dtype=float)
        else:
            vector = vector.copy()
        roll_columns = np.asarray([
            index for index, name in enumerate(MM2.DOF_NAMES)
            if name.endswith("_ThC_roll")
        ], dtype=int)
        if len(roll_columns) != 6:
            raise ValueError(
                "alternate ThC-roll orientation must resolve exactly six DOFs")
        vector[roll_columns] *= -1.0

    if vector is not None and (
            vector.shape != (len(MM2.DOF_NAMES),)
            or not np.isfinite(vector).all()
            or not np.isin(vector, (-1.0, 1.0)).all()):
        raise ValueError("motor signs must be finite -1/+1 values")
    return vector


# CD2-AB (lane A, 2026-09-05): motor neurons whose BANC muscle label disagrees
# with both male reconstructions, moved onto the muscle the males name. The
# middle legs carry 6 femur-reductor labels per side in BANC against 3 in MANC
# and 3 in MaleCNS, and 1 (left) and 0 (right) pleural remotor-and-abductor
# labels against 3 and 3; these six BANC femur-reductor cells are the ones
# BANC's own `manc_cell_type` column matches to the MANC remotor/abductor type
# (lanes/A/2026-09-05-cd2ab-thc-annotation-count.json). Kept only under the
# male-transfer rule (AGENTS.md, 2026-09-04): labelled TRANSFERRED, reversed if
# the fly does not improve.
_MN_TRANSFER_SETS = {
    "t2_remotor_manc6": {"to_muscle": "pleural_remotor_and_abductor_muscle", "ids": [
        720575941513362252, 720575941478307380, 720575941585484766, 720575941505580066,   # right middle
        720575941471300448, 720575941470371851]},                                         # left middle
    # C-N350 (lane C, 2026-09-12): the sternal adductor is the map's only
    # primary ThC_roll flexor and BANC annotates one on five legs and none on
    # the left middle, so that leg's roll joint has no flexor (column 99.98%
    # extension against the right middle's 69.5/30.5, the largest left-right
    # asymmetry of the 21 pairs by 8.9x). The male carries one on all six legs
    # (T2 1/1), which makes the absence an annotation gap on BANC rather than
    # dimorphism. The cell is identified by BANC's own arrival bundle, not by a
    # fit: MesoLN_dorsal_5 holds five cells on each side, four sternal
    # posterior rotators on both, and a fifth labelled to opposite muscles of
    # one axis, left 720575941553503460 remotor/abductor against right
    # 720575941644474312 adductor, the only two abduction-axis cells in T2.
    # The measured size proxy agrees: each is the smallest of its own bundle,
    # 1556 and 1419 input synapses against the rotators' 3896 to 6667.
    "lm_adductor_banc1": {"to_muscle": "sternal_adductor_muscle", "ids": [
        720575941553503460]},                                                             # left middle
}


def _apply_mn_transfer(mapping, mn_ids, spec):
    """th["mn_transfer"]: absent = bit-identical. {"set": <name>} rewrites the
    named cells' rows in a copy of the mapping to the target muscle's entries,
    exactly as motormap.build_map writes them (live and hypothesised), and
    rebuilds the raw matrix from that copy. The cell set is unchanged, so the
    row order is the readout's; asserted.

    {"sets": [<name>, ...]} applies several in the order given, which one set
    cannot express because a set carries one target muscle (C-N350, lane C,
    2026-09-12: the champion's own t2_remotor_manc6 closes the right middle's
    missing remotor, and the left middle's missing adductor is a different
    muscle). {"sets": [x]} is identical to {"set": x} and a single set is
    unchanged by this. The sets must be disjoint: two sets naming one cell
    would make the result depend on an ordering nobody stated.

    Design 171d6f1a B1 (study cfd71a09): an element of "sets" may also be an
    inline set {"to_muscle": <a motormap.MUSCLE_TO_DOF name>, "ids": [<banc
    id>, ...]}, checked as a registered set is (distinct ids, sets disjoint,
    every id a mapped leg motor neuron) and applied by the same loop, so an
    inline copy of a registered set gives the same mapping (probe P1). An
    unknown muscle is refused, and so is a cell whose target is the muscle it
    already carries in the mapping given, BANC's label.
    """
    import motormap as MM2
    if not isinstance(spec, dict) or ("set" in spec) == ("sets" in spec):
        raise ValueError("th['mn_transfer'] must be absent or carry exactly one of "
                         f"{{'set': <name>}} or {{'sets': [<name> | {{'to_muscle': <muscle>, 'ids': [...]}}, ...]}}, "
                         f"names among {sorted(_MN_TRANSFER_SETS)}")
    names = [spec["set"]] if "set" in spec else list(spec["sets"])
    registered = [n for n in names if not isinstance(n, dict)]
    if (not names or any(n not in _MN_TRANSFER_SETS for n in registered)
            or len(set(registered)) != len(registered) or ("set" in spec and not registered)):
        raise ValueError(f"th['mn_transfer'] names must be distinct and among {sorted(_MN_TRANSFER_SETS)}")
    sets = []
    for n in names:
        if not isinstance(n, dict):
            ts = _MN_TRANSFER_SETS[n]
            sets.append((ts["to_muscle"], {str(i) for i in ts["ids"]}))
            continue
        if set(n) != {"to_muscle", "ids"}:
            raise ValueError("mn_transfer: an inline set is {'to_muscle': <muscle>, 'ids': [<banc id>, ...]}")
        if n["to_muscle"] not in MM2.MUSCLE_TO_DOF:
            raise ValueError(f"mn_transfer: unknown muscle {n['to_muscle']!r}, not among {sorted(MM2.MUSCLE_TO_DOF)}")
        ids = [str(i) for i in n["ids"]]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("mn_transfer: an inline set's ids must be distinct and not empty")
        sets.append((n["to_muscle"], set(ids)))
    seen, moved = set(), 0
    mapping = mapping.copy()
    import pandas as pd
    for to_muscle, ids in sets:
        targets = MM2.MUSCLE_TO_DOF[to_muscle]   # BANC ids are strings in the mapping
        if ids & seen:
            raise ValueError(f"mn_transfer: sets must be disjoint; {sorted(ids & seen)} appears twice")
        seen |= ids
        hit = mapping["banc_888_id"].astype(str).isin(ids)
        if mapping.loc[hit, "banc_888_id"].nunique() != len(ids) or not mapping.loc[hit, "dof"].notna().all():
            raise ValueError("mn_transfer: every cell in the set must be a mapped leg motor neuron")
        same = sorted(mapping.loc[hit & (mapping["muscle"].astype(str) == to_muscle), "banc_888_id"].astype(str).unique())
        if same:
            raise ValueError(f"mn_transfer: {same} already carry {to_muscle}")
        new = []
        for cid, grp in mapping[hit].groupby("banc_888_id", sort=False):
            base = grp.iloc[0].to_dict()
            for joint_axis, sign, hypo in targets:
                new.append({**base, "muscle": to_muscle, "dof": f"{base['leg']}_{joint_axis}", "joint_axis": joint_axis,
                            "sign": float(sign), "gain": 0.0 if hypo else 1.0, "hypothesised": hypo, "status": "mapped", "transferred": True})
        mapping = pd.concat([mapping[~hit], pd.DataFrame(new)], ignore_index=True)
        moved += len(ids)
    M_raw, ids2 = MM2.to_matrix(mapping)
    if list(ids2) != list(mn_ids):
        raise ValueError("mn_transfer changed the motor-neuron row set")
    return mapping, M_raw, moved


def _mn_transfer_receipt(before, after):
    """LAST_EXTRAS["mn_transfer"] (design 171d6f1a B1): each cell
    _apply_mn_transfer moved, with its leg, its muscle in the mapping it was
    given (BANC's label) and its target, sorted by id, and the sha256 of that
    list's canonical JSON."""
    was = dict(zip(before["banc_888_id"].astype(str), before["muscle"].astype(str)))
    moved = after[after["transferred"].eq(True)].drop_duplicates("banc_888_id")
    cells = sorted(({"id": str(r.banc_888_id), "leg": str(r.leg), "banc_muscle": was[str(r.banc_888_id)],
                     "target": str(r.muscle)} for r in moved.itertuples(index=False)), key=lambda c: c["id"])
    blob = json.dumps(cells, sort_keys=True, separators=(",", ":")).encode()
    return {"cells": cells, "sha256": hashlib.sha256(blob).hexdigest()}


# CD2-AA (lane A, 2026-09-05): each thorax-coxa muscle on the hinge whose
# motion is its named action, signed by the body's own forward kinematics
# (lanes/A/2026-09-05-cd1zy-geom-jacobian-wb1m.json, `femur_base_mm_per_deg`
# and the tip Jacobian, same numbers at the all/swing/stance postures). On
# every leg: ThC_yaw's axis is body x, so it swings the coxa's distal end
# outward (+yaw = abduction) and moves nothing fore-aft; +ThC_pitch swings
# the coxa's distal end backward (remotion, 0.003-0.0055 mm/deg); ThC_roll
# turns the coxa about its own long axis (distal end fixed, the coxal
# rotators' action) and +roll moves the tip backward (0.007-0.018 mm/deg).
# The shipped map has the rotators on yaw, the abductor/adductor on roll and
# the promotor at +pitch. Keys: (muscle, hinge the shipped map drives) ->
# (group, hinge of the named action, FINAL sign in the body's convention).
_THC_MAP = {
    ("sternal_anterior_rotator_muscle", "ThC_yaw"): ("rotators", "ThC_roll", -1.0),
    ("sternal_posterior_rotator_muscle", "ThC_yaw"): ("rotators", "ThC_roll", 1.0),
    ("pleural_remotor_and_abductor_muscle", "ThC_roll"): ("abductors", "ThC_yaw", 1.0),
    ("sternal_adductor_muscle", "ThC_roll"): ("abductors", "ThC_yaw", -1.0),
    ("tergopleural_promotor_muscle", "ThC_pitch"): ("promotors", "ThC_pitch", -1.0),
    ("pleural_remotor_and_abductor_muscle", "ThC_pitch"): ("promotors", "ThC_pitch", 1.0),
}


def _apply_thc_map(M_raw, mapping, mn_ids, spec, sign_vector):
    """th["thc_map"]: absent = bit-identical. A dict with optional "groups"
    (subset of rotators/abductors/promotors, default all three) and "legs"
    (default all six). Moves the LIVE row of each muscle in scope from the
    hinge the shipped map drives to the hinge in _THC_MAP with the final sign;
    hypothesised entries and every later gain are untouched. Applied to the
    unsigned raw map, so it requires the sign vector to be +1 on the ThC
    columns in scope (true under signs "measured")."""
    import motormap as MM2
    if not isinstance(spec, dict):
        raise ValueError("th['thc_map'] must be absent or a dict")
    known = {g for g, _, _ in _THC_MAP.values()}
    groups = set(spec.get("groups", sorted(known)))
    legs = list(spec.get("legs", MM2.LEGS))
    if not groups or not groups <= known:
        raise ValueError(f"thc_map groups must be a non-empty subset of {sorted(known)}")
    if not legs or not set(legs) <= set(MM2.LEGS) or len(set(legs)) != len(legs):
        raise ValueError("thc_map legs must be distinct entries of motormap.LEGS")
    if sign_vector is not None:
        for l_ in legs:
            for ax in ("ThC_yaw", "ThC_pitch", "ThC_roll"):
                if sign_vector[MM2.DOF_INDEX[f"{l_}_{ax}"]] != 1.0:
                    raise ValueError("thc_map signs are final: the sign vector must be +1 on the ThC columns in scope")
    row_of = {i: k for k, i in enumerate(mn_ids)}
    live = mapping[(~mapping["hypothesised"]) & mapping["dof"].notna() & mapping["leg"].isin(legs)]
    M_raw = M_raw.copy()
    moved = 0
    for r_ in live.itertuples(index=False):
        tgt = _THC_MAP.get((r_.muscle, r_.joint_axis))
        if tgt is None or tgt[0] not in groups or r_.banc_888_id not in row_of:
            continue
        i = row_of[r_.banc_888_id]
        j_old = MM2.DOF_INDEX[r_.dof]
        old = r_.sign * r_.gain * MM2.SIGN_FLIP[r_.dof]
        if abs(M_raw[i, j_old] - old) > 1e-9:
            raise ValueError(f"thc_map: raw entry for {r_.banc_888_id} {r_.dof} is not the mapping row's")
        M_raw[i, j_old] -= old
        new_dof = f"{r_.leg}_{tgt[1]}"
        M_raw[i, MM2.DOF_INDEX[new_dof]] += tgt[2] * r_.gain * MM2.SIGN_FLIP[new_dof]
        moved += 1
    if moved == 0:
        raise ValueError("thc_map moved no live entries")
    return M_raw, moved                               # entries moved; a remotor-and-abductor row carries two

def _dof_index_for_receipt():
    """motormap's DOF name -> column index, for the channel_pool_map receipt."""
    import motormap as _MM
    return dict(_MM.DOF_INDEX)


def _apply_motor_sign_vector(matrix, vector):
    """Apply signs to the fully assembled map; absent mode returns it unchanged."""
    if vector is None:
        return matrix
    if matrix.ndim != 2 or matrix.shape[1] != len(vector):
        raise ValueError("motor sign vector does not match the motor map")
    return matrix * vector[None, :]


def _motor_sequence_config(th, use_split):
    """Build the opt-in tripod motor-sequence scaffold.

    The zero/absent path returns before importing the motor map or allocating
    sequence arrays, preserving the historical evaluator exactly. Positive
    gain is a BUILD hypothesis: it supplements, rather than replaces, neural
    flexor/extensor drive with proximal-to-distal timing.
    """
    gain = float(th.get("motor_sequence_gain", 0.0))
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("motor_sequence_gain must be finite and nonnegative")
    if gain == 0.0:
        return None
    if not use_split:
        raise ValueError("motor_sequence_gain requires the split muscle path")
    if "motor_sequence_hz" not in th or "motor_sequence_lag_ms" not in th:
        raise ValueError(
            "positive motor_sequence_gain requires motor_sequence_hz and "
            "motor_sequence_lag_ms")
    hz = float(th["motor_sequence_hz"])
    lag_ms = float(th["motor_sequence_lag_ms"])
    drive_scale = float(th["drive_scale"])
    if not np.isfinite(hz) or hz <= 0.0:
        raise ValueError("motor_sequence_hz must be finite and positive")
    if not np.isfinite(lag_ms) or lag_ms < 0.0:
        raise ValueError("motor_sequence_lag_ms must be finite and nonnegative")
    if not np.isfinite(drive_scale) or drive_scale <= 0.0:
        raise ValueError("drive_scale must be finite and positive")

    import motormap as MM2

    tripod_a = {"lf", "rm", "lh"}
    stage = {"ThC": 0.0, "CTr": 1.0, "FTi": 2.0, "TiTa": 3.0}
    phase = np.empty(len(MM2.DOF_NAMES), dtype=float)
    for column, name in enumerate(MM2.DOF_NAMES):
        leg, joint_axis = name.split("_", 1)
        joint = joint_axis.split("_", 1)[0]
        if leg not in set(MM2.LEGS) or joint not in stage:
            raise ValueError(f"unexpected motor-map DOF for sequence scaffold: {name}")
        tripod_phase = 0.0 if leg in tripod_a else np.pi
        delay_phase = 2.0 * np.pi * hz * stage[joint] * lag_ms / 1000.0
        phase[column] = tripod_phase - delay_phase
    return {
        "amplitude": gain * drive_scale,
        "omega_per_ms": 2.0 * np.pi * hz / 1000.0,
        "phase": phase,
    }


def _motor_sequence_drive(time_ms, config):
    """Return nonnegative (flexor, extensor) additions for one tick."""
    wave = np.sin(config["omega_per_ms"] * float(time_ms) + config["phase"])
    amplitude = float(config["amplitude"])
    return amplitude * np.maximum(-wave, 0.0), amplitude * np.maximum(wave, 0.0)


def _warp_phase_to_swing_frac(phase, swing_end, target_frac):
    """Spend target_frac of the cycle in [0, swing_end]. Identity at the native fraction."""
    two_pi = 2.0 * np.pi
    p = np.mod(np.asarray(phase, dtype=float), two_pi)
    u = p / two_pi
    out = np.empty_like(p, dtype=float)
    swing = u < target_frac
    out[swing] = (u[swing] / target_frac) * swing_end
    stance = ~swing
    denom = 1.0 - target_frac
    out[stance] = swing_end + (
        (u[stance] - target_frac) / denom) * (two_pi - swing_end)
    return out


def _resolve_measured_step_front_rom_floor(th):
    """Resolve the opt-in measured front-leg ROM floor without coercion."""
    value = th.get("measured_step_front_rom_floor", False)
    if type(value) is not bool:
        raise ValueError("measured_step_front_rom_floor must be bool")
    return value


def _resolve_measured_step_front_fti_rom_floor(th):
    """Resolve the opt-in bilateral front-FTi-only ROM floor."""
    value = th.get("measured_step_front_fti_rom_floor", False)
    if type(value) is not bool:
        raise ValueError("measured_step_front_fti_rom_floor must be bool")
    if value and _resolve_measured_step_front_rom_floor(th):
        raise ValueError("front ROM floor modes are mutually exclusive")
    return value


def _resolve_measured_step_front_rom_floor_motion_gate(th):
    """Resolve sensory motion-gating of the measured front-ROM tape."""
    value = th.get("measured_step_front_rom_floor_motion_gate", False)
    if type(value) is not bool:
        raise ValueError(
            "measured_step_front_rom_floor_motion_gate must be bool")
    if value and not _resolve_measured_step_front_rom_floor(th):
        raise ValueError(
            "front ROM motion gate requires the bilateral FTi/CTr floor")
    return value


def _apply_measured_step_front_rom_floor(
        theta, bod, enabled, joints=("FTi_pitch", "CTr_pitch")):
    """Expand front FTi/CTr tapes to Haustein's two-s.d. lower ROM bounds."""
    if not enabled:
        return None
    import motormap as MM2

    target_deg = {"FTi_pitch": 81.2, "CTr_pitch": 72.2}
    joints = tuple(joints)
    if not joints or any(joint not in target_deg for joint in joints):
        raise ValueError("front ROM floor joints are invalid")
    names = [f"{leg}_{joint}" for leg in ("lf", "rf") for joint in joints]
    indices = [MM2.DOF_INDEX[name] for name in names]
    unchanged = [index for index in range(theta.shape[1])
                 if index not in indices]
    before_unchanged = hashlib.sha256(
        np.ascontiguousarray(theta[:, unchanged]).tobytes()).hexdigest()
    per_dof = {}
    for name, index in zip(names, indices):
        values = theta[:, index]
        old_span = float(np.ptp(values))
        target = float(np.radians(target_deg[name[3:]]))
        if not np.isfinite(old_span) or old_span <= 0.0:
            raise ValueError(f"measured-step {name} has unusable ROM")
        midpoint = 0.5 * (float(np.min(values)) + float(np.max(values)))
        theta[:, index] = midpoint + (values - midpoint) * target / old_span
        low_clearance = float(np.min(theta[:, index]) - bod.lo[index])
        high_clearance = float(bod.hi[index] - np.max(theta[:, index]))
        if min(low_clearance, high_clearance) < -1e-12:
            raise ValueError(f"measured-step {name} ROM floor exceeds stops")
        per_dof[name] = {
            "old_span_deg": float(np.degrees(old_span)),
            "new_span_deg": float(np.degrees(np.ptp(theta[:, index]))),
            "target_span_deg": target_deg[name[3:]],
            "low_limit_clearance_deg": float(np.degrees(low_clearance)),
            "high_limit_clearance_deg": float(np.degrees(high_clearance)),
        }
    after_unchanged = hashlib.sha256(
        np.ascontiguousarray(theta[:, unchanged]).tobytes()).hexdigest()
    if after_unchanged != before_unchanged:
        raise RuntimeError("front ROM floor changed an unregistered trajectory")
    return {
        "mode": ("front_fti_ctr_measured_two_sd_floor"
                 if set(joints) == set(target_deg)
                 else "front_fti_measured_two_sd_floor"),
        "changed_dof_names": names,
        "unchanged_dof_count": len(unchanged),
        "unchanged_before_sha256": before_unchanged,
        "unchanged_after_sha256": after_unchanged,
        "minimum_limit_clearance_deg": min(
            min(row["low_limit_clearance_deg"],
                row["high_limit_clearance_deg"])
            for row in per_dof.values()),
        "per_dof": per_dof,
    }


def _measured_step_config(th, use_split, bod):
    """Build an opt-in additive drive tape from FlyGym's recorded steps."""
    gain = float(th.get("measured_step_gain", 0.0))
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("measured_step_gain must be finite and nonnegative")
    synchronized_adhesion = bool(th.get("measured_step_adhesion", False))
    adhesion_phase = th.get("measured_step_swing_adhesion_phase", "warped")
    if adhesion_phase not in ("warped", "base"):
        raise ValueError(
            "measured_step_swing_adhesion_phase must be 'warped' or 'base'")
    if gain == 0.0:
        if synchronized_adhesion or adhesion_phase != "warped":
            raise ValueError(
                "measured-step adhesion settings require positive gain")
        return None
    if getattr(bod, "tau_deact", None) is not None:
        # The inverse below runs the symmetric filter through arctanh; it
        # would ignore the relaxation constant.
        raise ValueError("the measured-step routine refuses tau_deact_ms")
    if not use_split:
        raise ValueError("measured_step_gain requires the split muscle path")
    if "measured_step_hz" not in th:
        raise ValueError("positive measured_step_gain requires measured_step_hz")
    hz = float(th["measured_step_hz"])
    tick_ms = float(th.get("tick_ms", 5.0))
    if not np.isfinite(hz) or hz <= 0.0:
        raise ValueError("measured_step_hz must be finite and positive")
    if not np.isfinite(tick_ms) or tick_ms <= 0.0:
        raise ValueError("measured-step tick must be finite and positive")
    cycle_ticks = int(round(1000.0 / (hz * tick_ms)))
    if cycle_ticks < 8:
        raise ValueError("measured-step cycle must contain at least eight ticks")

    from flygym_demo.complex_terrain import PreprogrammedSteps

    steps = PreprogrammedSteps()
    tripod_a = {"lf", "rm", "lh"}
    phase_offsets = np.asarray([
        0.0 if leg in tripod_a else np.pi for leg in steps.legs
    ], dtype=float)
    base_phase = 2.0 * np.pi * np.arange(cycle_ticks) / cycle_ticks
    # Lane B's swing fraction, ported verbatim. It warps the phase BEFORE the
    # joint angles are read, so the inverse dynamics below sees the re-timed
    # movement; lane D's `measured_step_swing_stretch` resamples the finished
    # drive tape AFTER that inverse, which re-orders drives computed for the
    # original timing. That difference is the whole reason one moves duty and
    # the other cannot.
    swing_frac = th.get("measured_step_swing_frac")
    if adhesion_phase != "warped" and (
            swing_frac is None or not synchronized_adhesion):
        raise ValueError(
            "base adhesion phase requires swing_frac and synchronized adhesion")
    if swing_frac is None:
        theta = np.asarray([
            steps.get_joint_angles_by_dof_order(
                (phase + phase_offsets) % (2.0 * np.pi))
            for phase in base_phase
        ], dtype=float)
        warped_phases = None
    else:
        sf = float(swing_frac)
        if not np.isfinite(sf) or not (0.0 < sf < 1.0):
            raise ValueError("measured_step_swing_frac must be in (0, 1)")
        raw = (base_phase[:, None] + phase_offsets[None, :]) % (2.0 * np.pi)
        warped_phases = np.empty_like(raw)
        for i, leg in enumerate(steps.legs):
            end = float(steps.swing_period[leg][1])
            if not (0.0 < end < 2.0 * np.pi):
                raise ValueError(
                    f"measured-step swing_period for {leg} is unusable")
            warped_phases[:, i] = _warp_phase_to_swing_frac(
                raw[:, i], end, sf)
        theta = np.asarray([
            steps.get_joint_angles_by_dof_order(warped_phases[tick])
            for tick in range(cycle_ticks)
        ], dtype=float)
    if theta.shape != (cycle_ticks, 42) or not np.isfinite(theta).all():
        raise ValueError("measured step did not resolve to a finite T x 42 tape")
    front_rom_floor_enabled = _resolve_measured_step_front_rom_floor(th)
    front_fti_rom_floor_enabled = (
        _resolve_measured_step_front_fti_rom_floor(th))
    front_rom_motion_gate_enabled = (
        _resolve_measured_step_front_rom_floor_motion_gate(th))
    theta_unfloored = theta.copy() if front_rom_motion_gate_enabled else None
    front_rom_floor = _apply_measured_step_front_rom_floor(
        theta, bod,
        front_rom_floor_enabled or front_fti_rom_floor_enabled,
        joints=(("FTi_pitch",) if front_fti_rom_floor_enabled
                else ("FTi_pitch", "CTr_pitch")))

    dt = tick_ms / 1000.0
    theta_next = np.roll(theta, -1, axis=0)
    theta_prev = np.roll(theta, 1, axis=0)
    velocity = (theta_next - theta_prev) / (2.0 * dt)
    acceleration = (theta_next - 2.0 * theta + theta_prev) / (dt * dt)
    if bod.f0 is not None:
        omega = 2.0 * np.pi * float(bod.f0)
        theta_eq = theta + (
            acceleration + 2.0 * float(bod.zeta) * omega * velocity
        ) / (omega * omega)
    else:
        theta_eq = theta + velocity * float(bod.tau_joint) / 1000.0
    ratio = (theta_eq - bod.mid) / (bod.half + 1e-12)
    activation = float(th["drive_scale"]) * np.arctanh(
        np.clip(ratio, -0.999, 0.999))
    raw_alpha = bod._alpha(bod.tau_act, tick_ms)
    if np.ndim(raw_alpha) == 0:
        # Preserve the legacy scalar arithmetic byte-for-byte.
        alpha = float(raw_alpha)
    else:
        alpha = np.asarray(raw_alpha, dtype=float)
        if (alpha.shape != (42,) or not np.isfinite(alpha).all()
                or np.any(alpha <= 0.0)):
            raise ValueError(
                "measured-step activation alpha must be a positive scalar "
                "or finite positive 42-vector")
    drive = activation + (np.roll(activation, -1, axis=0) - activation) / alpha
    flex = gain * np.maximum(-drive, 0.0)
    ext = gain * np.maximum(drive, 0.0)
    if not np.isfinite(flex).all() or not np.isfinite(ext).all():
        raise ValueError("measured-step inverse produced non-finite drive")
    base_flex = None
    base_ext = None
    if theta_unfloored is not None:
        base_next = np.roll(theta_unfloored, -1, axis=0)
        base_prev = np.roll(theta_unfloored, 1, axis=0)
        base_velocity = (base_next - base_prev) / (2.0 * dt)
        base_acceleration = (
            base_next - 2.0 * theta_unfloored + base_prev) / (dt * dt)
        if bod.f0 is not None:
            base_theta_eq = theta_unfloored + (
                base_acceleration
                + 2.0 * float(bod.zeta) * omega * base_velocity
            ) / (omega * omega)
        else:
            base_theta_eq = theta_unfloored + (
                base_velocity * float(bod.tau_joint) / 1000.0)
        base_ratio = (base_theta_eq - bod.mid) / (bod.half + 1e-12)
        base_activation = float(th["drive_scale"]) * np.arctanh(
            np.clip(base_ratio, -0.999, 0.999))
        base_drive = base_activation + (
            np.roll(base_activation, -1, axis=0) - base_activation) / alpha
        base_flex = gain * np.maximum(-base_drive, 0.0)
        base_ext = gain * np.maximum(base_drive, 0.0)
        if not (np.isfinite(base_flex).all()
                and np.isfinite(base_ext).all()):
            raise ValueError(
                "unfloored measured-step inverse produced non-finite drive")
    # Per-leg scales, 2026-09-01. Absent or all-ones = bit-identical.
    # Order is motormap.LEGS: lf, lm, lh, rf, rm, rh. Named because the
    # selected body shares one scalar across six legs and conflicts on
    # duty; hind clearance is the shallow end.
    leg_gains = th.get("measured_step_leg_gains")
    if leg_gains is not None:
        import motormap as MM
        lg = np.asarray(leg_gains, dtype=float).reshape(-1)
        if lg.shape != (6,):
            raise ValueError(
                "measured_step_leg_gains must be six nonnegative numbers")
        if not np.isfinite(lg).all() or np.any(lg < 0.0):
            raise ValueError(
                "measured_step_leg_gains must be finite and nonnegative")
        by_leg = {leg: float(val) for leg, val in zip(MM.LEGS, lg)}
        scale = np.asarray(
            [by_leg[name.split("_", 1)[0]] for name in MM.DOF_NAMES],
            dtype=float)
        flex = flex * scale
        ext = ext * scale
        if base_flex is not None:
            base_flex = base_flex * scale
            base_ext = base_ext * scale

    adhesion = None
    if synchronized_adhesion:
        # Canonical (main, 8972b4e91): when the fraction re-times the movement
        # the adhesion tape is re-timed with it. Lane D's earlier port dropped
        # this branch and warped the joint angles alone, which is the live
        # deviation the arbiter run found. The explicit base option retains
        # the unwarped adhesion timing for a separately registered experiment.
        if warped_phases is None or adhesion_phase == "base":
            adhesion = np.asarray([
                steps.get_adhesion_onoff_by_phase(
                    (phase + phase_offsets) % (2.0 * np.pi))
                for phase in base_phase
            ], dtype=bool)
        elif th.get("measured_step_hold_adhesion"):
            # Default-off. Absent = warp both (canonical main / 80257).
            adhesion = np.asarray([
                steps.get_adhesion_onoff_by_phase(
                    (phase + phase_offsets) % (2.0 * np.pi))
                for phase in base_phase
            ], dtype=bool)
        else:
            adhesion = np.asarray([
                steps.get_adhesion_onoff_by_phase(warped_phases[tick])
                for tick in range(cycle_ticks)
            ], dtype=bool)
        if adhesion.shape != (cycle_ticks, 6):
            raise ValueError("measured adhesion did not resolve to a T x 6 tape")
    # Swing re-timing, 2026-09-02. The selected body conflicts on duty, feet
    # down 0.889 against a real fly's 0.50-0.83, and per-leg GAIN was measured
    # unable to buy it: gain sets how high a leg peaks, duty is how long it
    # stays up. This stretches the swing part of the cycle against the stance
    # part instead, leaving the trajectory's shape in space untouched and
    # changing only how long the leg spends on each part of it. The template's
    # own adhesion tape says which samples are stance, so the split is the
    # recorded step's, not a guess. 1.0 or absent = bit-identical.
    swing_stretch = th.get("measured_step_swing_stretch")
    if swing_stretch is not None and float(swing_stretch) != 1.0:
        if (type(swing_stretch) not in (int, float)
                or not np.isfinite(swing_stretch)
                or not 0.25 <= float(swing_stretch) <= 6.0):
            raise ValueError(
                "measured_step_swing_stretch must be finite in [0.25, 6]")
        if adhesion is None:
            raise ValueError(
                "measured_step_swing_stretch needs the adhesion tape to say "
                "which samples are stance")
        import motormap as MM3
        stretch = float(swing_stretch)
        leg_of_dof = [name.split("_", 1)[0] for name in MM3.DOF_NAMES]
        warped_flex = flex.copy()
        warped_ext = ext.copy()
        warped_adhesion = adhesion.copy()
        old_index = np.arange(cycle_ticks, dtype=float)
        for leg_i, leg in enumerate(MM3.LEGS):
            stance = adhesion[:, leg_i]
            if stance.all() or not stance.any():
                continue
            # Time spent per original sample: swing samples are stretched,
            # stance samples are left alone; the cumulative is then resampled
            # uniformly, which compresses stance by exactly as much as the
            # cycle length is preserved.
            weight = np.where(stance, 1.0, stretch)
            cumulative = np.concatenate([[0.0], np.cumsum(weight)])
            targets = np.linspace(0.0, cumulative[-1], cycle_ticks,
                                  endpoint=False)
            source = np.interp(targets, cumulative[:-1], old_index)
            take = np.clip(np.rint(source).astype(int), 0, cycle_ticks - 1)
            columns = [i for i, name in enumerate(leg_of_dof) if name == leg]
            warped_flex[:, columns] = flex[np.ix_(take, columns)]
            warped_ext[:, columns] = ext[np.ix_(take, columns)]
            warped_adhesion[:, leg_i] = adhesion[take, leg_i]
        flex, ext, adhesion = warped_flex, warped_ext, warped_adhesion
    return {
        "tick_ms": tick_ms,
        "flex": flex,
        "ext": ext,
        "base_flex": base_flex,
        "base_ext": base_ext,
        "adhesion": adhesion,
        "swing_stretch": (None if swing_stretch is None
                          else float(swing_stretch)),
        "swing_frac": None if swing_frac is None else float(swing_frac),
        "adhesion_phase": adhesion_phase,
        "commanded_stance_fraction": (
            None if adhesion is None else float(np.mean(adhesion))),
        "flex_sha256": hashlib.sha256(flex.tobytes()).hexdigest(),
        "ext_sha256": hashlib.sha256(ext.tobytes()).hexdigest(),
        "adhesion_sha256": (
            None if adhesion is None
            else hashlib.sha256(adhesion.tobytes()).hexdigest()),
        "out_of_range_fraction": float(np.mean(np.abs(ratio) >= 1.0)),
        "front_rom_floor": front_rom_floor,
        "front_rom_floor_motion_gate": ({
            "enabled": True,
            "source": "live_graded_hs_motion_class",
            "motion_active": False,
            "floored_ticks": 0,
            "base_ticks": 0,
        } if front_rom_motion_gate_enabled else None),
    }


def _measured_step_drive(time_ms, config):
    """Return the measured-template additions and optional adhesion for a tick."""
    index = int(round(float(time_ms) / float(config["tick_ms"])))
    index %= len(config["flex"])
    flex = config["flex"]
    ext = config["ext"]
    gate = config.get("front_rom_floor_motion_gate")
    if gate is not None:
        if gate["motion_active"]:
            gate["floored_ticks"] += 1
        else:
            flex = config["base_flex"]
            ext = config["base_ext"]
            gate["base_ticks"] += 1
    adhesion = config["adhesion"]
    onoff = None if adhesion is None else adhesion[index].copy()
    return flex[index].copy(), ext[index].copy(), onoff


def _measured_step_drive_at_phase(phase_rad, config):
    """Sample the measured template at an explicit wrapped phase."""
    phase = float(phase_rad)
    if not np.isfinite(phase):
        raise ValueError("measured-step phase must be finite")
    length = len(config["flex"])
    index = int(round((phase % (2.0 * np.pi)) * length / (2.0 * np.pi)))
    index %= length
    flex = config["flex"]
    ext = config["ext"]
    gate = config.get("front_rom_floor_motion_gate")
    if gate is not None:
        if gate["motion_active"]:
            gate["floored_ticks"] += 1
        else:
            flex = config["base_flex"]
            ext = config["base_ext"]
            gate["base_ticks"] += 1
    adhesion = config["adhesion"]
    onoff = None if adhesion is None else adhesion[index].copy()
    return flex[index].copy(), ext[index].copy(), onoff


def _measured_step_neural_upstream_mix_config(
        th, measured_step, use_split, neural_motor_scale):
    """Resolve a drive-conserving live-MN mix before teacher transforms."""
    value = th.get("measured_step_neural_upstream_mix_fraction", 0.0)
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not 0.0 <= float(value) < 1.0):
        raise ValueError(
            "measured_step_neural_upstream_mix_fraction must be finite "
            "in [0, 1)")
    fraction = float(value)
    if fraction == 0.0:
        return None
    if measured_step is None or not use_split or neural_motor_scale != 1.0:
        raise ValueError(
            "upstream neural mix requires split measured-step drive and "
            "neural_motor_scale=1")
    return {
        "fraction": fraction,
        "active_ticks": 0,
        "silent_neural_ticks": 0,
        "raw_neural_abs_drive_tick_sum": 0.0,
        "source_step_abs_drive_tick_sum": 0.0,
        "retained_step_abs_drive_tick_sum": 0.0,
        "substituted_neural_abs_drive_tick_sum": 0.0,
        "mixed_abs_drive_tick_sum": 0.0,
        "max_tick_budget_abs_error": 0.0,
        "sum_tick_budget_abs_error": 0.0,
    }


def _apply_measured_step_neural_upstream_mix(
        neural_flex, neural_ext, step_flex, step_ext, config):
    """Replace step budget with live-MN pattern before later transforms."""
    if config is None:
        return (np.asarray(step_flex, dtype=float).copy(),
                np.asarray(step_ext, dtype=float).copy())
    arrays = tuple(np.asarray(value, dtype=float) for value in (
        neural_flex, neural_ext, step_flex, step_ext))
    if (any(value.shape != (42,) for value in arrays)
            or not all(np.isfinite(value).all() for value in arrays)
            or any(float(value.min()) < -1e-12 for value in arrays)):
        raise ValueError(
            "upstream neural mix requires finite nonnegative 42-vectors")
    neural_flex, neural_ext, step_flex, step_ext = arrays
    neural_budget = float(np.sum(neural_flex) + np.sum(neural_ext))
    step_budget = float(np.sum(step_flex) + np.sum(step_ext))
    config["raw_neural_abs_drive_tick_sum"] += neural_budget
    config["source_step_abs_drive_tick_sum"] += step_budget
    if neural_budget <= 1e-15 or step_budget <= 1e-15:
        config["silent_neural_ticks"] += int(neural_budget <= 1e-15)
        return step_flex.copy(), step_ext.copy()
    fraction = float(config["fraction"])
    retained_flex = (1.0 - fraction) * step_flex
    retained_ext = (1.0 - fraction) * step_ext
    scale = fraction * step_budget / neural_budget
    substituted_flex = scale * neural_flex
    substituted_ext = scale * neural_ext
    mixed_flex = retained_flex + substituted_flex
    mixed_ext = retained_ext + substituted_ext
    retained_budget = float(np.sum(retained_flex) + np.sum(retained_ext))
    substituted_budget = float(
        np.sum(substituted_flex) + np.sum(substituted_ext))
    mixed_budget = float(np.sum(mixed_flex) + np.sum(mixed_ext))
    error = abs(mixed_budget - step_budget)
    config["active_ticks"] += 1
    config["retained_step_abs_drive_tick_sum"] += retained_budget
    config["substituted_neural_abs_drive_tick_sum"] += substituted_budget
    config["mixed_abs_drive_tick_sum"] += mixed_budget
    config["max_tick_budget_abs_error"] = max(
        config["max_tick_budget_abs_error"], error)
    config["sum_tick_budget_abs_error"] += error
    return mixed_flex, mixed_ext


def _measured_step_neural_sign_gate_config(
        th, measured_step, use_split, neural_motor_scale,
        upstream_mix_config):
    """Resolve per-DOF, budget-preserving live-MN directional authority."""
    value = th.get("measured_step_neural_sign_gate", False)
    if type(value) is not bool:
        raise ValueError("measured_step_neural_sign_gate must be Boolean")
    if not value:
        if ("measured_step_neural_sign_gate_route" in th
                or "measured_step_neural_sign_gate_homeostatic_baseline" in th):
            raise ValueError(
                "neural sign-gate options require the sign gate")
        return None
    if (measured_step is None or not use_split
            or neural_motor_scale != 1.0):
        raise ValueError(
            "measured_step_neural_sign_gate requires split measured step "
            "and neural_motor_scale=1")
    if upstream_mix_config is not None:
        raise ValueError(
            "neural sign gate and upstream fixed-fraction mix are exclusive")
    route = th.get("measured_step_neural_sign_gate_route", "all")
    if route not in ("all", "right_and_detector_zero", "right_only"):
        raise ValueError("unknown measured_step_neural_sign_gate_route")
    homeostatic = th.get(
        "measured_step_neural_sign_gate_homeostatic_baseline", False)
    if type(homeostatic) is not bool:
        raise ValueError(
            "measured_step_neural_sign_gate_homeostatic_baseline must be Boolean")
    return {
        "route": route,
        "homeostatic_baseline": homeostatic,
        "baseline_flex_sum": np.zeros(42, dtype=float),
        "baseline_ext_sum": np.zeros(42, dtype=float),
        "baseline_ticks": 0,
        "baseline_frozen": False,
        "neural_envelope_by_dof": np.zeros(42, dtype=float),
        "ticks": 0,
        "active_dof_ticks": 0,
        "silent_dof_ticks": 0,
        "direction_changed_dof_ticks": 0,
        "strength_min": np.inf,
        "strength_max": 0.0,
        "strength_sum": 0.0,
        "max_dof_budget_abs_error": 0.0,
        "sum_dof_budget_abs_error": 0.0,
        "applied_ticks_by_class0_class1_none": np.zeros(3, dtype=np.int64),
        "bypassed_ticks_by_class0_class1_none": np.zeros(3, dtype=np.int64),
    }


def _apply_measured_step_neural_sign_gate(
        neural_flex, neural_ext, step_flex, step_ext, config,
        eligible=True, class_slot=2):
    """Let each DOF's live MN balance redirect its fixed step-drive budget."""
    if config is None:
        return (np.asarray(step_flex, dtype=float).copy(),
                np.asarray(step_ext, dtype=float).copy())
    if type(eligible) is not bool or class_slot not in (0, 1, 2):
        raise ValueError("neural sign-gate routing state is invalid")
    arrays = tuple(np.asarray(value, dtype=float) for value in (
        neural_flex, neural_ext, step_flex, step_ext))
    if (any(value.shape != (42,) for value in arrays)
            or not all(np.isfinite(value).all() for value in arrays)
            or any(float(value.min()) < -1e-12 for value in arrays)):
        raise ValueError(
            "neural sign gate requires finite nonnegative 42-vectors")
    neural_flex, neural_ext, step_flex, step_ext = arrays
    collecting_baseline = bool(
        config["homeostatic_baseline"]
        and not config["baseline_frozen"]
        and class_slot == 2)
    if (config["homeostatic_baseline"]
            and not config["baseline_frozen"]):
        if class_slot in (0, 1):
            config["baseline_frozen"] = True
        else:
            config["baseline_flex_sum"] += neural_flex
            config["baseline_ext_sum"] += neural_ext
            config["baseline_ticks"] += 1
    if collecting_baseline or not eligible:
        config["bypassed_ticks_by_class0_class1_none"][class_slot] += 1
        return step_flex.copy(), step_ext.copy()
    config["applied_ticks_by_class0_class1_none"][class_slot] += 1
    neural_total = neural_flex + neural_ext
    step_total = step_flex + step_ext
    envelope = np.maximum(config["neural_envelope_by_dof"], neural_total)
    config["neural_envelope_by_dof"][:] = envelope
    active = neural_total > 0.0
    strength = np.divide(neural_total, envelope,
                         out=np.zeros_like(neural_total), where=envelope > 0.0)
    neural_flex_fraction = np.divide(
        neural_flex, neural_total, out=np.zeros_like(neural_flex),
        where=active)
    neural_ext_fraction = np.divide(
        neural_ext, neural_total, out=np.zeros_like(neural_ext),
        where=active)
    directed_flex_fraction = neural_flex_fraction
    if config["homeostatic_baseline"]:
        baseline_total = (config["baseline_flex_sum"]
                          + config["baseline_ext_sum"])
        baseline_flex_fraction = np.divide(
            config["baseline_flex_sum"], baseline_total,
            out=neural_flex_fraction.copy(), where=baseline_total > 0.0)
        teacher_flex_fraction = np.divide(
            step_flex, step_total, out=np.zeros_like(step_flex),
            where=step_total > 0.0)
        directed_flex_fraction = np.clip(
            teacher_flex_fraction
            + neural_flex_fraction - baseline_flex_fraction,
            0.0, 1.0)
    directed_ext_fraction = 1.0 - directed_flex_fraction
    directed_flex = step_total * directed_flex_fraction
    directed_ext = step_total * directed_ext_fraction
    out_flex = (1.0 - strength) * step_flex + strength * directed_flex
    out_ext = (1.0 - strength) * step_ext + strength * directed_ext
    error = np.abs((out_flex + out_ext) - step_total)
    teacher_sign = np.sign(step_ext - step_flex)
    neural_sign = np.sign(directed_ext - directed_flex)
    config["ticks"] += 1
    active_count = int(np.count_nonzero(active))
    config["active_dof_ticks"] += active_count
    config["silent_dof_ticks"] += 42 - active_count
    config["direction_changed_dof_ticks"] += int(np.count_nonzero(
        active & (neural_sign != 0.0) & (neural_sign != teacher_sign)))
    if active_count:
        config["strength_min"] = min(
            float(config["strength_min"]), float(np.min(strength[active])))
        config["strength_max"] = max(
            float(config["strength_max"]), float(np.max(strength[active])))
        config["strength_sum"] += float(np.sum(strength[active]))
    config["max_dof_budget_abs_error"] = max(
        float(config["max_dof_budget_abs_error"]), float(np.max(error)))
    config["sum_dof_budget_abs_error"] += float(np.sum(error))
    return out_flex, out_ext


def _measured_step_contact_gain_config(th, measured_step):
    """Resolve an optional previous-tick stance-load drive multiplier."""
    class_values = th.get("measured_step_contact_gain_by_visual_class")
    if class_values is not None:
        if "measured_step_contact_gain" in th:
            raise ValueError(
                "scalar and visual-class contact gains are mutually exclusive")
        if (not isinstance(class_values, list) or len(class_values) != 2
                or any(type(item) not in (int, float)
                       or not np.isfinite(item)
                       or not 0.0 <= float(item) <= 1.0
                       for item in class_values)):
            raise ValueError(
                "measured_step_contact_gain_by_visual_class must be two "
                "finite gains in [0, 1]")
        gains = np.asarray(class_values, dtype=float)
        if not np.any(gains > 0.0):
            return None
    else:
        gains = None
    value = th.get("measured_step_contact_gain", 0.0)
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not 0.0 <= float(value) <= 1.0):
        raise ValueError(
            "measured_step_contact_gain must be finite in [0, 1]")
    gain = float(value)
    if gains is None and gain == 0.0:
        return None
    if measured_step is None:
        raise ValueError(
            "measured_step_contact_gain requires an active measured step")
    import motormap as MM2
    names = list(MM2.DOF_NAMES)
    leg_order = ("lf", "lm", "lh", "rf", "rm", "rh")
    leg_masks = np.asarray([
        [name.startswith(leg + "_") for name in names]
        for leg in leg_order
    ], dtype=bool)
    if (leg_masks.shape != (6, 42)
            or not np.all(leg_masks.sum(axis=1) == 7)
            or not np.all(leg_masks.sum(axis=0) == 1)):
        raise ValueError("contact-gain masks must split 42 DOFs 7/leg")
    return {
        "gain": gain,
        "gain_by_visual_class": gains,
        "contact_threshold_N": 1.0,
        "leg_order": leg_order,
        "leg_masks": leg_masks,
        "active_ticks_by_leg": np.zeros(6, dtype=np.int64),
        "active_ticks_by_visual_class_leg": np.zeros(
            (2, 6), dtype=np.int64),
        "offered_abs_drive_tick_sum": 0.0,
        "added_abs_drive_tick_sum": 0.0,
    }


def _apply_measured_step_contact_gain(
        flex, ext, prev_loads, config, visual_class=None):
    """Strengthen only the teacher drive of legs loaded on the prior tick."""
    if config is None:
        return
    class_gains = config["gain_by_visual_class"]
    class_index = None
    if class_gains is None:
        gain = float(config["gain"])
    else:
        if visual_class is None:
            return
        if visual_class not in (0, 1):
            raise ValueError("visual class for contact gain must be 0 or 1")
        class_index = int(visual_class)
        gain = float(class_gains[class_index])
        if gain == 0.0:
            return
    factor = 1.0 + gain
    threshold = float(config["contact_threshold_N"])
    loads = {} if prev_loads is None else prev_loads
    for leg_index, (leg, mask) in enumerate(zip(
            config["leg_order"], config["leg_masks"], strict=True)):
        if float(loads.get(leg, 0.0)) <= threshold:
            continue
        offered = float(np.abs(flex[mask]).sum() + np.abs(ext[mask]).sum())
        flex[mask] *= factor
        ext[mask] *= factor
        config["active_ticks_by_leg"][leg_index] += 1
        if class_index is not None:
            config["active_ticks_by_visual_class_leg"][
                class_index, leg_index] += 1
        config["offered_abs_drive_tick_sum"] += offered
        config["added_abs_drive_tick_sum"] += offered * (factor - 1.0)


def _measured_step_hs_outer_contact_gain_config(th, measured_step):
    """Resolve a causal HS-class-gated outer-side stance multiplier."""
    value = th.get("measured_step_hs_outer_contact_gain", 0.0)
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not 0.0 <= float(value) <= 1.0):
        raise ValueError(
            "measured_step_hs_outer_contact_gain must be finite in [0, 1]")
    gain = float(value)
    if gain == 0.0:
        return None
    if measured_step is None:
        raise ValueError(
            "measured_step_hs_outer_contact_gain requires an active "
            "measured step")
    symmetric = th.get("measured_step_contact_gain", 0.0)
    if (type(symmetric) not in (int, float)
            or not np.isfinite(symmetric)):
        raise ValueError("measured_step_contact_gain must be finite")
    if (float(symmetric) != 0.0
            or th.get("measured_step_contact_gain_by_visual_class") is not None):
        raise ValueError(
            "scalar, visual-class and HS-outer contact gains are mutually "
            "exclusive")
    import motormap as MM2
    names = list(MM2.DOF_NAMES)
    leg_order = ("lf", "lm", "lh", "rf", "rm", "rh")
    leg_masks = np.asarray([
        [name.startswith(leg + "_") for name in names]
        for leg in leg_order
    ], dtype=bool)
    if (leg_masks.shape != (6, 42)
            or not np.all(leg_masks.sum(axis=1) == 7)
            or not np.all(leg_masks.sum(axis=0) == 1)):
        raise ValueError("HS-outer contact masks must split 42 DOFs 7/leg")
    return {
        "gain": gain,
        "contact_threshold_N": 1.0,
        "leg_order": leg_order,
        "leg_masks": leg_masks,
        "decoded_ticks_by_class": np.zeros(2, dtype=np.int64),
        "outer_side_ticks_left_right": np.zeros(2, dtype=np.int64),
        "undecoded_ticks": 0,
        "active_ticks_by_leg": np.zeros(6, dtype=np.int64),
        "active_ticks_by_class_leg": np.zeros((2, 6), dtype=np.int64),
        "offered_abs_drive_tick_sum": 0.0,
        "added_abs_drive_tick_sum": 0.0,
    }


def _apply_measured_step_hs_outer_contact_gain(
        flex, ext, prev_loads, decoded_class, config):
    """Boost loaded outer-side legs selected by the current causal HS class."""
    if config is None:
        return
    if decoded_class is None:
        config["undecoded_ticks"] += 1
        return
    if (type(decoded_class) not in (int, np.int64)
            or int(decoded_class) not in (0, 1)):
        raise ValueError("HS outer-load class must be 0, 1 or None")
    class_index = int(decoded_class)
    outer_side = 1 - class_index
    config["decoded_ticks_by_class"][class_index] += 1
    config["outer_side_ticks_left_right"][outer_side] += 1
    factor = 1.0 + float(config["gain"])
    threshold = float(config["contact_threshold_N"])
    loads = {} if prev_loads is None else prev_loads
    first_leg = 0 if outer_side == 0 else 3
    for leg_index in range(first_leg, first_leg + 3):
        leg = config["leg_order"][leg_index]
        if float(loads.get(leg, 0.0)) <= threshold:
            continue
        mask = config["leg_masks"][leg_index]
        offered = float(np.abs(flex[mask]).sum() + np.abs(ext[mask]).sum())
        flex[mask] *= factor
        ext[mask] *= factor
        config["active_ticks_by_leg"][leg_index] += 1
        config["active_ticks_by_class_leg"][class_index, leg_index] += 1
        config["offered_abs_drive_tick_sum"] += offered
        config["added_abs_drive_tick_sum"] += offered * (factor - 1.0)


def _measured_step_hs_conservative_push_pull_config(th, measured_step):
    """Resolve HS push-pull with the C-V29 tick drive budget preserved."""
    value = th.get("measured_step_hs_conservative_push_pull_gain", 0.0)
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not 0.0 <= float(value) <= 1.0):
        raise ValueError(
            "measured_step_hs_conservative_push_pull_gain must be finite "
            "in [0, 1]")
    gain = float(value)
    if gain == 0.0:
        return None
    if measured_step is None:
        raise ValueError(
            "measured_step_hs_conservative_push_pull_gain requires an "
            "active measured step")
    conflicts = (
        th.get("measured_step_contact_gain", 0.0),
        th.get("measured_step_hs_outer_contact_gain", 0.0),
        th.get("measured_step_hs_push_pull_contact_gain", 0.0),
        th.get("measured_step_hs_saturation_aware_push_pull_gain", 0.0),
    )
    if (any(type(item) not in (int, float) or not np.isfinite(item)
            for item in conflicts)
            or any(float(item) != 0.0 for item in conflicts)
            or th.get("measured_step_contact_gain_by_visual_class") is not None):
        raise ValueError(
            "all measured-step contact controllers are mutually exclusive")
    import motormap as MM2
    names = list(MM2.DOF_NAMES)
    leg_order = ("lf", "lm", "lh", "rf", "rm", "rh")
    leg_masks = np.asarray([
        [name.startswith(leg + "_") for name in names]
        for leg in leg_order
    ], dtype=bool)
    if (leg_masks.shape != (6, 42)
            or not np.all(leg_masks.sum(axis=1) == 7)
            or not np.all(leg_masks.sum(axis=0) == 1)):
        raise ValueError("conservative push-pull masks must split 42 DOFs 7/leg")
    return {
        "gain": gain,
        "contact_threshold_N": 1.0,
        "leg_order": leg_order,
        "leg_masks": leg_masks,
        "decoded_ticks_by_class": np.zeros(2, dtype=np.int64),
        "outer_side_ticks_left_right": np.zeros(2, dtype=np.int64),
        "undecoded_ticks": 0,
        "transfer_ticks": 0,
        "outer_active_ticks_by_class_leg": np.zeros((2, 6), dtype=np.int64),
        "inner_active_ticks_by_class_leg": np.zeros((2, 6), dtype=np.int64),
        "source_outer_added_abs_drive_tick_sum": 0.0,
        "inner_removed_abs_drive_tick_sum": 0.0,
        "transferred_abs_drive_tick_sum": 0.0,
        "candidate_outer_added_abs_drive_tick_sum": 0.0,
        "max_outer_factor": 1.0 + gain,
        "max_tick_budget_abs_error": 0.0,
        "sum_tick_budget_abs_error": 0.0,
    }


def _apply_measured_step_hs_conservative_push_pull(
        flex, ext, prev_loads, decoded_class, config):
    """Transfer loaded-inner suppression to loaded outer drive each tick."""
    if config is None:
        return
    if decoded_class is None:
        config["undecoded_ticks"] += 1
        return
    if (type(decoded_class) not in (int, np.int64)
            or int(decoded_class) not in (0, 1)):
        raise ValueError("conservative push-pull class must be 0, 1 or None")
    class_index = int(decoded_class)
    outer_side = 1 - class_index
    config["decoded_ticks_by_class"][class_index] += 1
    config["outer_side_ticks_left_right"][outer_side] += 1
    loads = {} if prev_loads is None else prev_loads
    threshold = float(config["contact_threshold_N"])
    loaded = [
        float(loads.get(leg, 0.0)) > threshold
        for leg in config["leg_order"]
    ]
    offered = np.asarray([
        (float(np.abs(flex[mask]).sum() + np.abs(ext[mask]).sum())
         if is_loaded else 0.0)
        for mask, is_loaded in zip(config["leg_masks"], loaded, strict=True)
    ])
    outer_indices = range(0, 3) if outer_side == 0 else range(3, 6)
    inner_indices = range(3, 6) if outer_side == 0 else range(0, 3)
    outer_offered = float(offered[list(outer_indices)].sum())
    if outer_offered <= 0.0:
        return
    inner_offered = float(offered[list(inner_indices)].sum())
    gain = float(config["gain"])
    source_outer_added = gain * outer_offered
    removed = gain * inner_offered
    outer_factor = 1.0 + gain + removed / outer_offered
    inner_factor = 1.0 - gain
    baseline_total = float(np.abs(flex).sum() + np.abs(ext).sum())
    for leg_index in outer_indices:
        if not loaded[leg_index]:
            continue
        mask = config["leg_masks"][leg_index]
        flex[mask] *= outer_factor
        ext[mask] *= outer_factor
        config["outer_active_ticks_by_class_leg"][
            class_index, leg_index] += 1
    for leg_index in inner_indices:
        if not loaded[leg_index]:
            continue
        mask = config["leg_masks"][leg_index]
        flex[mask] *= inner_factor
        ext[mask] *= inner_factor
        config["inner_active_ticks_by_class_leg"][
            class_index, leg_index] += 1
    candidate_total = float(np.abs(flex).sum() + np.abs(ext).sum())
    error = candidate_total - (baseline_total + source_outer_added)
    if inner_offered > 0.0:
        config["transfer_ticks"] += 1
    config["source_outer_added_abs_drive_tick_sum"] += source_outer_added
    config["inner_removed_abs_drive_tick_sum"] += removed
    config["transferred_abs_drive_tick_sum"] += removed
    config["candidate_outer_added_abs_drive_tick_sum"] += (
        source_outer_added + removed)
    config["max_outer_factor"] = max(
        float(config["max_outer_factor"]), outer_factor)
    config["max_tick_budget_abs_error"] = max(
        float(config["max_tick_budget_abs_error"]), abs(error))
    config["sum_tick_budget_abs_error"] += abs(error)


def _measured_step_hs_saturation_aware_push_pull_config(th, measured_step):
    """Resolve bounded HS push-pull while preserving the C-V29 budget."""
    value = th.get("measured_step_hs_saturation_aware_push_pull_gain", 0.0)
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not 0.0 <= float(value) <= 1.0):
        raise ValueError(
            "measured_step_hs_saturation_aware_push_pull_gain must be "
            "finite in [0, 1]")
    gain = float(value)
    if gain == 0.0:
        return None
    if measured_step is None:
        raise ValueError(
            "measured_step_hs_saturation_aware_push_pull_gain requires an "
            "active measured step")
    conflicts = (
        th.get("measured_step_contact_gain", 0.0),
        th.get("measured_step_hs_outer_contact_gain", 0.0),
        th.get("measured_step_hs_push_pull_contact_gain", 0.0),
        th.get("measured_step_hs_conservative_push_pull_gain", 0.0),
    )
    if (any(type(item) not in (int, float) or not np.isfinite(item)
            for item in conflicts)
            or any(float(item) != 0.0 for item in conflicts)
            or th.get("measured_step_contact_gain_by_visual_class") is not None):
        raise ValueError(
            "all measured-step contact controllers are mutually exclusive")
    import motormap as MM2
    names = list(MM2.DOF_NAMES)
    leg_order = ("lf", "lm", "lh", "rf", "rm", "rh")
    leg_masks = np.asarray([
        [name.startswith(leg + "_") for name in names]
        for leg in leg_order
    ], dtype=bool)
    if (leg_masks.shape != (6, 42)
            or not np.all(leg_masks.sum(axis=1) == 7)
            or not np.all(leg_masks.sum(axis=0) == 1)):
        raise ValueError(
            "saturation-aware push-pull masks must split 42 DOFs 7/leg")
    return {
        "gain": gain,
        # The C-V29 outer addition is 1+gain. At most one equal outer-side
        # budget is accepted from the inner side, so this is architecture,
        # not a separately tuned cap.
        "outer_factor_cap": 1.0 + 2.0 * gain,
        "contact_threshold_N": 1.0,
        "leg_order": leg_order,
        "leg_masks": leg_masks,
        "decoded_ticks_by_class": np.zeros(2, dtype=np.int64),
        "outer_side_ticks_left_right": np.zeros(2, dtype=np.int64),
        "undecoded_ticks": 0,
        "transfer_ticks": 0,
        "saturation_ticks": 0,
        "outer_active_ticks_by_class_leg": np.zeros((2, 6), dtype=np.int64),
        "inner_active_ticks_by_class_leg": np.zeros((2, 6), dtype=np.int64),
        "source_outer_added_abs_drive_tick_sum": 0.0,
        "requested_inner_removal_abs_drive_tick_sum": 0.0,
        "transferred_abs_drive_tick_sum": 0.0,
        "untransferred_abs_drive_tick_sum": 0.0,
        "candidate_outer_added_abs_drive_tick_sum": 0.0,
        "min_inner_factor": 1.0,
        "max_outer_factor": 1.0 + gain,
        "max_tick_budget_abs_error": 0.0,
        "sum_tick_budget_abs_error": 0.0,
    }


def _apply_measured_step_hs_saturation_aware_push_pull(
        flex, ext, prev_loads, decoded_class, config):
    """Transfer only inner drive that fits under the derived outer cap."""
    if config is None:
        return
    if decoded_class is None:
        config["undecoded_ticks"] += 1
        return
    if (type(decoded_class) not in (int, np.int64)
            or int(decoded_class) not in (0, 1)):
        raise ValueError(
            "saturation-aware push-pull class must be 0, 1 or None")
    class_index = int(decoded_class)
    outer_side = 1 - class_index
    config["decoded_ticks_by_class"][class_index] += 1
    config["outer_side_ticks_left_right"][outer_side] += 1
    loads = {} if prev_loads is None else prev_loads
    threshold = float(config["contact_threshold_N"])
    loaded = [
        float(loads.get(leg, 0.0)) > threshold
        for leg in config["leg_order"]
    ]
    offered = np.asarray([
        (float(np.abs(flex[mask]).sum() + np.abs(ext[mask]).sum())
         if is_loaded else 0.0)
        for mask, is_loaded in zip(config["leg_masks"], loaded, strict=True)
    ])
    outer_indices = range(0, 3) if outer_side == 0 else range(3, 6)
    inner_indices = range(3, 6) if outer_side == 0 else range(0, 3)
    outer_offered = float(offered[list(outer_indices)].sum())
    if outer_offered <= 0.0:
        return
    inner_offered = float(offered[list(inner_indices)].sum())
    gain = float(config["gain"])
    source_outer_added = gain * outer_offered
    requested = gain * inner_offered
    transfer_capacity = max(
        0.0, (float(config["outer_factor_cap"]) - (1.0 + gain))
        * outer_offered)
    transferred = min(requested, transfer_capacity)
    untransferred = requested - transferred
    outer_factor = 1.0 + gain + transferred / outer_offered
    inner_factor = (1.0 - transferred / inner_offered
                    if inner_offered > 0.0 else 1.0)
    baseline_total = float(np.abs(flex).sum() + np.abs(ext).sum())
    for leg_index in outer_indices:
        if not loaded[leg_index]:
            continue
        mask = config["leg_masks"][leg_index]
        flex[mask] *= outer_factor
        ext[mask] *= outer_factor
        config["outer_active_ticks_by_class_leg"][
            class_index, leg_index] += 1
    for leg_index in inner_indices:
        if not loaded[leg_index]:
            continue
        mask = config["leg_masks"][leg_index]
        flex[mask] *= inner_factor
        ext[mask] *= inner_factor
        config["inner_active_ticks_by_class_leg"][
            class_index, leg_index] += 1
    candidate_total = float(np.abs(flex).sum() + np.abs(ext).sum())
    error = candidate_total - (baseline_total + source_outer_added)
    if transferred > 0.0:
        config["transfer_ticks"] += 1
    if untransferred > 1e-12:
        config["saturation_ticks"] += 1
    config["source_outer_added_abs_drive_tick_sum"] += source_outer_added
    config["requested_inner_removal_abs_drive_tick_sum"] += requested
    config["transferred_abs_drive_tick_sum"] += transferred
    config["untransferred_abs_drive_tick_sum"] += untransferred
    config["candidate_outer_added_abs_drive_tick_sum"] += (
        source_outer_added + transferred)
    config["min_inner_factor"] = min(
        float(config["min_inner_factor"]), inner_factor)
    config["max_outer_factor"] = max(
        float(config["max_outer_factor"]), outer_factor)
    config["max_tick_budget_abs_error"] = max(
        float(config["max_tick_budget_abs_error"]), abs(error))
    config["sum_tick_budget_abs_error"] += abs(error)


def _visual_teacher_side_phase_config(th, measured_step, use_split):
    """Build the opt-in two-side or six-leg visual phase controller."""
    value = th.get("vision_teacher_phase_asymmetry", 0.0)
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not 0.0 <= float(value) < 1.0):
        raise ValueError(
            "vision_teacher_phase_asymmetry must be finite in [0, 1)")
    asymmetry = float(value)
    outer_load_slowdown = th.get(
        "measured_step_hs_outer_load_phase_slowdown", 0.0)
    if (type(outer_load_slowdown) not in (int, float)
            or not np.isfinite(outer_load_slowdown)
            or not 0.0 <= float(outer_load_slowdown) < 1.0):
        raise ValueError(
            "measured_step_hs_outer_load_phase_slowdown must be finite "
            "in [0, 1)")
    outer_load_slowdown = float(outer_load_slowdown)
    outer_load_self_normalized = th.get(
        "measured_step_hs_outer_load_phase_self_normalized", False)
    if type(outer_load_self_normalized) is not bool:
        raise ValueError(
            "measured_step_hs_outer_load_phase_self_normalized must be Boolean")
    if outer_load_self_normalized and outer_load_slowdown != 0.0:
        raise ValueError(
            "fixed and self-normalized outer-load phase modes are exclusive")
    leg_asymmetry = _resolve_visual_teacher_leg_phase_asymmetry(th)
    if asymmetry != 0.0 and leg_asymmetry is not None:
        raise ValueError(
            "global and per-leg visual phase asymmetry are mutually exclusive")
    if (asymmetry == 0.0
            and (leg_asymmetry is None or not np.any(leg_asymmetry != 0.0))):
        if outer_load_slowdown != 0.0 or outer_load_self_normalized:
            raise ValueError(
                "measured-step HS outer-load phase feedback requires an "
                "active visual teacher phase controller")
        return None
    if not use_split or measured_step is None:
        raise ValueError(
            "vision_teacher_phase_asymmetry requires a split measured step")
    import motormap as MM2
    names = list(MM2.DOF_NAMES)
    left_mask = np.asarray([name.startswith("l") for name in names])
    right_mask = np.asarray([name.startswith("r") for name in names])
    if (left_mask.shape != (42,) or right_mask.shape != (42,)
            or int(left_mask.sum()) != 21 or int(right_mask.sum()) != 21
            or np.any(left_mask & right_mask)
            or not np.all(left_mask | right_mask)):
        raise ValueError("visual side-phase masks must split 42 DOFs 21/21")
    leg_order = ("lf", "lm", "lh", "rf", "rm", "rh")
    leg_masks = np.asarray([
        [name.startswith(leg + "_") for name in names]
        for leg in leg_order
    ], dtype=bool)
    if (leg_masks.shape != (6, 42)
            or not np.all(leg_masks.sum(axis=1) == 7)
            or not np.all(leg_masks.sum(axis=0) == 1)):
        raise ValueError("visual leg-phase masks must split 42 DOFs 7/leg")
    common = {
        "tick_ms": float(measured_step["tick_ms"]),
        "cycle_length": len(measured_step["flex"]),
        "asymmetric_ticks": 0,
        "decoded_ticks": np.zeros(2, dtype=np.int64),
        "leg_order": leg_order,
        "leg_masks": leg_masks,
        "hs_outer_load_phase_slowdown": outer_load_slowdown,
        "hs_outer_load_phase_self_normalized": outer_load_self_normalized,
        "hs_outer_load_threshold_N": 1.0,
        "hs_outer_load_envelope_N_by_leg": np.zeros(6, dtype=float),
        "hs_outer_load_strength_min": np.inf,
        "hs_outer_load_strength_max": 0.0,
        "hs_outer_load_strength_sum": 0.0,
        "hs_outer_load_strength_ticks": 0,
        "hs_outer_phase_initialized": False,
        "hs_outer_phase_rad": np.zeros(6, dtype=float),
        "hs_outer_decoded_ticks_by_class": np.zeros(2, dtype=np.int64),
        "hs_outer_undecoded_ticks": 0,
        "hs_outer_loaded_slow_ticks_by_class_leg": np.zeros(
            (2, 6), dtype=np.int64),
        "hs_outer_delayed_phase_rad_by_class_leg": np.zeros(
            (2, 6), dtype=float),
        "memoryless_ticks": 0,
        "memoryless_target_abs_min_rad": np.inf,
        "memoryless_target_abs_max_rad": 0.0,
    }
    if leg_asymmetry is None:
        return {
            **common,
            "scope": "side",
            "asymmetry": asymmetry,
            "phase_offsets": np.zeros(2, dtype=float),
            "left_mask": left_mask,
            "right_mask": right_mask,
        }
    return {
        **common,
        "scope": "leg",
        "asymmetry": 0.0,
        "leg_asymmetry": np.tile(leg_asymmetry, 2),
        "leg_order": leg_order,
        "leg_masks": leg_masks,
        "phase_offsets": np.zeros(6, dtype=float),
    }


def _visual_teacher_side_phase_drive(time_ms, measured_step, config,
                                     decoded_class, prev_loads=None,
                                     hs_load_class=None,
                                     decoded_strength=1.0,
                                     decoded_memoryless=False,
                                     strength=1.0):
    """Sample every joint/adhesion channel from persistent phase states."""
    tick_ms = float(config["tick_ms"])
    length = int(config["cycle_length"])
    clock_index = int(round(float(time_ms) / tick_ms)) % length
    nominal_phase = 2.0 * np.pi * clock_index / length
    slowdown = float(config.get("hs_outer_load_phase_slowdown", 0.0))
    self_normalized = bool(config.get(
        "hs_outer_load_phase_self_normalized", False))
    if slowdown > 0.0 or self_normalized:
        phase_state = config["hs_outer_phase_rad"]
        if not config["hs_outer_phase_initialized"]:
            phase_state[:] = nominal_phase
            config["hs_outer_phase_initialized"] = True
        else:
            nominal_delta = 2.0 * np.pi / length
            class_index = None
            if hs_load_class is None:
                config["hs_outer_undecoded_ticks"] += 1
            else:
                if (type(hs_load_class) not in (int, np.int64)
                        or int(hs_load_class) not in (0, 1)):
                    raise ValueError(
                        "HS outer load-phase class must be 0, 1 or None")
                class_index = int(hs_load_class)
                config["hs_outer_decoded_ticks_by_class"][class_index] += 1
            outer_side = None if class_index is None else 1 - class_index
            loads = {} if prev_loads is None else prev_loads
            for leg_index, leg in enumerate(config["leg_order"]):
                load = max(0.0, float(loads.get(leg, 0.0)))
                if self_normalized:
                    envelope = max(
                        float(config["hs_outer_load_envelope_N_by_leg"][
                            leg_index]), load)
                    config["hs_outer_load_envelope_N_by_leg"][
                        leg_index] = envelope
                else:
                    envelope = 0.0
                on_outer_side = (
                    outer_side is not None
                    and (0 if leg_index < 3 else 1) == outer_side)
                loaded = (load > 0.0 if self_normalized
                          else load > float(
                              config["hs_outer_load_threshold_N"]))
                slow = bool(on_outer_side and loaded)
                strength = ((load / envelope) if self_normalized and slow
                            and envelope > 0.0 else slowdown if slow else 0.0)
                scale = 1.0 - strength
                phase_state[leg_index] = (
                    phase_state[leg_index] + nominal_delta * scale
                ) % (2.0 * np.pi)
                if slow:
                    config["hs_outer_load_strength_min"] = min(
                        float(config["hs_outer_load_strength_min"]), strength)
                    config["hs_outer_load_strength_max"] = max(
                        float(config["hs_outer_load_strength_max"]), strength)
                    config["hs_outer_load_strength_sum"] += strength
                    config["hs_outer_load_strength_ticks"] += 1
                    config["hs_outer_loaded_slow_ticks_by_class_leg"][
                        class_index, leg_index] += 1
                    config["hs_outer_delayed_phase_rad_by_class_leg"][
                        class_index, leg_index] += nominal_delta * strength
        offsets = (np.repeat(config["phase_offsets"], 3)
                   if config["scope"] == "side"
                   else config["phase_offsets"])
        phases = phase_state + offsets
        if config["scope"] == "side":
            config["sampled_phases_rad"] = np.asarray(
                [np.mean(phases[:3]), np.mean(phases[3:])], dtype=float)
        else:
            config["sampled_phases_rad"] = np.asarray(
                phases, dtype=float).copy()
        flex = np.empty(42, dtype=float)
        ext = np.empty(42, dtype=float)
        adhesion = (None if measured_step["adhesion"] is None
                    else np.empty(6, dtype=bool))
        for leg_index, (phase, mask) in enumerate(
                zip(phases, config["leg_masks"], strict=True)):
            leg_flex, leg_ext, leg_adhesion = _measured_step_drive_at_phase(
                phase, measured_step)
            flex[mask] = leg_flex[mask]
            ext[mask] = leg_ext[mask]
            if adhesion is not None:
                adhesion[leg_index] = leg_adhesion[leg_index]
    else:
        phases = nominal_phase + config["phase_offsets"]
        config["sampled_phases_rad"] = np.asarray(phases, dtype=float).copy()
    if slowdown <= 0.0 and config["scope"] == "side":
        left_flex, left_ext, left_adhesion = _measured_step_drive_at_phase(
            phases[0], measured_step)
        right_flex, right_ext, right_adhesion = _measured_step_drive_at_phase(
            phases[1], measured_step)
        flex = left_flex
        ext = left_ext
        flex[config["right_mask"]] = right_flex[config["right_mask"]]
        ext[config["right_mask"]] = right_ext[config["right_mask"]]
        adhesion = left_adhesion
        if adhesion is not None:
            adhesion[3:] = right_adhesion[3:]
    elif slowdown <= 0.0 and config["scope"] == "leg":
        flex = np.empty(42, dtype=float)
        ext = np.empty(42, dtype=float)
        adhesion = (None if measured_step["adhesion"] is None
                    else np.empty(6, dtype=bool))
        for leg_index, (phase, mask) in enumerate(
                zip(phases, config["leg_masks"])):
            leg_flex, leg_ext, leg_adhesion = _measured_step_drive_at_phase(
                phase, measured_step)
            flex[mask] = leg_flex[mask]
            ext[mask] = leg_ext[mask]
            if adhesion is not None:
                adhesion[leg_index] = leg_adhesion[leg_index]
    elif slowdown > 0.0:
        pass
    else:
        raise ValueError("unknown visual teacher phase scope")
    if decoded_class is not None:
        if type(decoded_class) not in (int, np.int64) or decoded_class not in (0, 1):
            raise ValueError("visual side-phase class must be 0, 1 or None")
        # The brake corrects a body rate it measures, but its correction has
        # been the same size whatever that rate is. `strength` lets a caller
        # scale one tick's correction; every other caller passes 1.0 and the
        # arithmetic is unchanged.
        if (type(strength) not in (int, float, np.float64)
                or not np.isfinite(strength) or float(strength) < 0.0):
            raise ValueError("side-phase strength must be finite and >= 0")
        if (type(decoded_strength) not in (int, float)
                or not np.isfinite(decoded_strength)
                or not 0.0 <= float(decoded_strength) <= 2.0):
            raise ValueError(
                "visual side-phase decoded strength must be in [0, 2]")
        sign = -1.0 if int(decoded_class) == 0 else 1.0
        effective_strength = float(decoded_strength) * float(strength)
        if type(decoded_memoryless) is not bool:
            raise ValueError("visual side-phase memoryless flag must be Boolean")
        if decoded_memoryless and config["scope"] == "side":
            target = (2.0 * np.pi * float(config["asymmetry"])
                      * effective_strength)
            phase_delta = np.asarray([sign * target, -sign * target])
        elif decoded_memoryless:
            phase_delta = (2.0 * np.pi * config["leg_asymmetry"]
                           * effective_strength
                           * np.asarray([sign, sign, sign,
                                         -sign, -sign, -sign]))
        elif config["scope"] == "side":
            delta = ((2.0 * np.pi / length) * float(config["asymmetry"])
                     * effective_strength)
            phase_delta = np.asarray([sign * delta, -sign * delta])
        else:
            phase_delta = ((2.0 * np.pi / length)
                           * config["leg_asymmetry"]
                           * effective_strength
                           * np.asarray([sign, sign, sign,
                                         -sign, -sign, -sign]))
        if decoded_memoryless:
            config["phase_offsets"][:] = phase_delta
            target_abs = float(np.max(np.abs(phase_delta)))
            config["memoryless_ticks"] += 1
            config["memoryless_target_abs_min_rad"] = min(
                float(config["memoryless_target_abs_min_rad"]), target_abs)
            config["memoryless_target_abs_max_rad"] = max(
                float(config["memoryless_target_abs_max_rad"]), target_abs)
        else:
            config["phase_offsets"] += phase_delta
        config["asymmetric_ticks"] += 1
        config["decoded_ticks"][int(decoded_class)] += 1
    return flex, ext, adhesion


def _visual_teacher_hs_thc_yaw_phase_config(
        th, measured_step, use_split, visual_teacher_phase, visual_bridge):
    """Build the opt-in raw-HS front/hind ThC-yaw phase modifier.

    Ported from lane A's C-V29 candidate. Lane A required detector-selected
    side phase; the raw-HS class this modifier reads is produced identically
    when live HS activity selects the phase instead, so C-V41's neural phase
    control is accepted as well.
    """
    # Lane A capped the magnitude at 90 degrees. That bound was inherited,
    # not measured, and lane D's C-V41 sweep put the best angle exactly on it,
    # so the half-cycle the template actually spans is opened instead.
    value = th.get("vision_teacher_hs_thc_yaw_phase_deg", 0.0)
    if (type(value) not in (int, float) or not np.isfinite(value)
            or not -180.0 <= float(value) <= 180.0):
        raise ValueError(
            "vision_teacher_hs_thc_yaw_phase_deg must be finite in "
            "[-180, 180]")
    magnitude_deg = float(value)
    # One magnitude per decoded class. The composition helps the right turn
    # and costs the left one by a fixed amount on every seed, so the two
    # directions are given separate magnitudes rather than one shared number.
    # Absent, this is the scalar above on both classes and bit-identical.
    by_class = th.get("vision_teacher_hs_thc_yaw_phase_deg_by_class")
    if by_class is not None:
        if ("vision_teacher_hs_thc_yaw_phase_deg" in th
                and magnitude_deg != 0.0):
            raise ValueError(
                "scalar and per-class ThC-yaw phase magnitudes are mutually "
                "exclusive")
        if (not isinstance(by_class, list) or len(by_class) != 2
                or any(type(item) not in (int, float)
                       or not np.isfinite(item)
                       or not -180.0 <= float(item) <= 180.0
                       for item in by_class)):
            raise ValueError(
                "vision_teacher_hs_thc_yaw_phase_deg_by_class must be two "
                "finite degrees in [-180, 180]")
        magnitude_by_class = (float(by_class[0]), float(by_class[1]))
        magnitude_deg = max(magnitude_by_class, key=abs)
    else:
        magnitude_by_class = (magnitude_deg, magnitude_deg)
    if magnitude_by_class == (0.0, 0.0):
        return None
    if not use_split or measured_step is None:
        raise ValueError(
            "HS ThC-yaw phase requires the split measured-step path")
    if (visual_teacher_phase is None
            or visual_teacher_phase.get("scope") != "side"):
        raise ValueError(
            "HS ThC-yaw phase requires two-side teacher phase")
    if (visual_bridge is None
            or visual_bridge.get("relay_code") not in _HS_CLASS_RELAY_CODES
            or not (visual_bridge.get("phase_from_detector")
                    or visual_bridge.get("phase_neural_motion_control"))):
        raise ValueError(
            "HS ThC-yaw phase requires raw HS with detector or live-HS "
            "side phase")
    # Default-off control: pinning the class removes the visual dependence
    # while leaving the four-DOF shift and its magnitude intact, which is what
    # separates a phase mechanism driven by HS activity from a static retune
    # of the same joints.
    override = th.get("vision_teacher_hs_thc_yaw_phase_class_override")
    if override is not None and (type(override) is not int
                                 or override not in (0, 1)):
        raise ValueError(
            "vision_teacher_hs_thc_yaw_phase_class_override must be the "
            "integer 0 or 1, or absent")
    # Per-leg scope. The shape this modifier was ported with advances the
    # front legs, retards the hind legs and leaves the middle legs alone,
    # which is one chosen front-to-hind gradient with its node placed on the
    # middle leg. The gradient varies per leg in the animal, so it is a
    # six-element vector here rather than a hardcoded pair of signs. Absent,
    # it is that same shape and the four DOFs, receipt included, are what the
    # modifier already produced.
    legs = ("lf", "lm", "lh", "rf", "rm", "rh")
    weights = th.get("vision_teacher_hs_thc_yaw_phase_leg_weights")
    if weights is None:
        leg_weights = (1.0, 0.0, -1.0, 1.0, 0.0, -1.0)
    else:
        if (not isinstance(weights, list) or len(weights) != 6
                or any(type(item) not in (int, float)
                       or not np.isfinite(item)
                       or not -1.0 <= float(item) <= 1.0
                       for item in weights)):
            raise ValueError(
                "vision_teacher_hs_thc_yaw_phase_leg_weights must be six "
                "finite weights in [-1, 1], ordered lf lm lh rf rm rh")
        leg_weights = tuple(float(item) for item in weights)
        if all(item == 0.0 for item in leg_weights):
            raise ValueError(
                "vision_teacher_hs_thc_yaw_phase_leg_weights must not be "
                "all zero: the modifier would be a silent no-op")
    import motormap as MM2
    entries = tuple(
        (f"{leg}_ThC_yaw", 0 if leg[0] == "l" else 1, weight)
        for leg, weight in zip(legs, leg_weights) if weight != 0.0)
    resolved = tuple(
        (name, int(MM2.DOF_INDEX[name]), side, sign)
        for name, side, sign in entries)
    indices = np.asarray([item[1] for item in resolved], dtype=np.int64)
    if (len(MM2.DOF_NAMES) != 42
            or len(np.unique(indices)) != len(resolved)
            or [MM2.DOF_NAMES[index] for index in indices]
            != [item[0] for item in resolved]):
        raise ValueError("HS ThC-yaw phase DOFs did not resolve exactly")
    other_indices = np.asarray([
        index for index in range(42) if index not in set(indices.tolist())
    ], dtype=np.int64)
    # How fast the phase returns to neutral when no class is decoded. The
    # pinned-class controls kept both turns and lost only the static-image
    # clause, which is the clause this release governs, so it is separated
    # from the relay's own time constant. Absent, it is that constant and the
    # arithmetic is unchanged.
    release_tau_ms = th.get("vision_teacher_hs_thc_yaw_phase_release_tau_ms")
    if release_tau_ms is not None and (
            type(release_tau_ms) not in (int, float)
            or not np.isfinite(release_tau_ms)
            or not 0.5 <= float(release_tau_ms) <= 1000.0):
        raise ValueError(
            "vision_teacher_hs_thc_yaw_phase_release_tau_ms must be finite "
            "in [0.5, 1000]")
    return {
        "magnitude_deg": magnitude_deg,
        "magnitude_rad": float(np.deg2rad(magnitude_deg)),
        "release_tau_ms": (float(visual_bridge["relay_tau_ms"])
                           if release_tau_ms is None
                           else float(release_tau_ms)),
        "separate_release_tau": release_tau_ms is not None,
        "release_ticks": 0,
        "magnitude_deg_by_class": magnitude_by_class,
        "magnitude_rad_by_class": (
            float(np.deg2rad(magnitude_by_class[0])),
            float(np.deg2rad(magnitude_by_class[1]))),
        "per_class_magnitudes": by_class is not None,
        "tick_ms": float(measured_step["tick_ms"]),
        "tau_ms": float(visual_bridge["relay_tau_ms"]),
        "phase_source": (
            f"constant_class_{int(override)}" if override is not None
            else "detector" if visual_bridge.get("phase_from_detector")
            else "live_hs"),
        "class_override": None if override is None else int(override),
        "override_ticks": 0,
        "command_state": 0.0,
        "command_state_min": 0.0,
        "command_state_max": 0.0,
        "entries": resolved,
        "modified_indices": indices,
        "other_indices": other_indices,
        "applied_ticks": 0,
        "decoded_ticks": np.zeros(2, dtype=np.int64),
        "target_ticks_class0_class1_none": np.zeros(3, dtype=np.int64),
        "leg_weights": leg_weights,
        "per_leg_weights": weights is not None,
        "changed_ticks_by_dof": np.zeros(len(resolved), dtype=np.int64),
        "abs_drive_delta_sum_by_dof": np.zeros(len(resolved), dtype=float),
        "other_channel_violation_ticks": 0,
        "adhesion_violation_ticks": 0,
        "max_absolute_phase_deg": 0.0,
        "dna02_amplitude_control": bool(
            visual_bridge.get("phase_dna02_amplitude", False)),
        "dna02_amplitude_mode": visual_bridge.get(
            "phase_dna02_amplitude_mode", "off"),
        "dna02_amplitude": 1.0,
        "dna02_amplitude_ticks": 0,
        "dna02_amplitude_max": 1.0,
        "dna02_rate_hz_max": 0.0,
        "dna02_max_rate_hz": visual_bridge.get("dna02_max_rate_hz"),
        "dna02_max_rate_hz_by_side": visual_bridge.get(
            "dna02_max_rate_hz_by_side"),
        "other_flex_before_sha256": hashlib.sha256(),
        "other_flex_after_sha256": hashlib.sha256(),
        "other_ext_before_sha256": hashlib.sha256(),
        "other_ext_after_sha256": hashlib.sha256(),
        "adhesion_before_sha256": hashlib.sha256(),
        "adhesion_after_sha256": hashlib.sha256(),
    }


def _apply_visual_teacher_hs_thc_yaw_phase(
        phase_step, measured_step, base_phases_rad, decoded_class, config):
    """Compose raw-HS front/hind yaw phase onto the sampled side phases."""
    if config is None:
        return phase_step
    if phase_step is None:
        raise ValueError("HS ThC-yaw phase requires an existing phase sample")
    flex, ext, adhesion = phase_step
    flex_before = np.asarray(flex, dtype=float).copy()
    ext_before = np.asarray(ext, dtype=float).copy()
    adhesion_before = (None if adhesion is None
                       else np.asarray(adhesion, dtype=bool).copy())
    base_phases = np.asarray(base_phases_rad, dtype=float)
    if (base_phases.shape != (2,) or not np.isfinite(base_phases).all()
            or np.asarray(flex).shape != (42,)
            or np.asarray(ext).shape != (42,)):
        raise ValueError("HS ThC-yaw phase input shape is invalid")

    if config["class_override"] is not None:
        decoded_class = int(config["class_override"])
        config["override_ticks"] += 1
    if decoded_class is None:
        target = 0.0
        config["target_ticks_class0_class1_none"][2] += 1
    else:
        if (type(decoded_class) not in (int, np.int64)
                or int(decoded_class) not in (0, 1)):
            raise ValueError("HS ThC-yaw class must be 0, 1 or None")
        class_index = int(decoded_class)
        amplitude = (float(config["dna02_amplitude"])
                     if config["dna02_amplitude_control"] else 1.0)
        if not 1.0 <= amplitude <= 2.0:
            raise ValueError("DNa02 phase amplitude must be in [1, 2]")
        target = amplitude if class_index == 0 else -amplitude
        config["decoded_ticks"][class_index] += 1
        config["target_ticks_class0_class1_none"][class_index] += 1
    if target == 0.0:
        tau = float(config["release_tau_ms"])
        config["release_ticks"] += 1
    else:
        tau = float(config["tau_ms"])
    alpha = 1.0 - np.exp(-float(config["tick_ms"]) / tau)
    state = float(config["command_state"])
    state += alpha * (target - state)
    config["command_state"] = state
    config["command_state_min"] = min(
        float(config["command_state_min"]), state)
    config["command_state_max"] = max(
        float(config["command_state_max"]), state)
    # A non-negative command state is class 0's half of the cycle, so each
    # class scales its own side of the same continuous state; the two agree at
    # zero, and a single magnitude on both classes is the scalar case exactly.
    magnitudes = config["magnitude_rad_by_class"]
    phase_delta = state * float(magnitudes[0] if state >= 0.0
                                else magnitudes[1])

    for receipt_index, (_, dof_index, side_index, sign) in enumerate(
            config["entries"]):
        sampled_flex, sampled_ext, _ = _measured_step_drive_at_phase(
            base_phases[side_index] + sign * phase_delta, measured_step)
        flex[dof_index] = sampled_flex[dof_index]
        ext[dof_index] = sampled_ext[dof_index]
        delta = (abs(float(flex[dof_index] - flex_before[dof_index]))
                 + abs(float(ext[dof_index] - ext_before[dof_index])))
        if delta > 0.0:
            config["changed_ticks_by_dof"][receipt_index] += 1
            config["abs_drive_delta_sum_by_dof"][receipt_index] += delta

    other = config["other_indices"]
    config["other_flex_before_sha256"].update(
        np.ascontiguousarray(flex_before[other]).tobytes())
    config["other_flex_after_sha256"].update(
        np.ascontiguousarray(np.asarray(flex)[other]).tobytes())
    config["other_ext_before_sha256"].update(
        np.ascontiguousarray(ext_before[other]).tobytes())
    config["other_ext_after_sha256"].update(
        np.ascontiguousarray(np.asarray(ext)[other]).tobytes())
    if (not np.array_equal(np.asarray(flex)[other], flex_before[other])
            or not np.array_equal(np.asarray(ext)[other], ext_before[other])):
        config["other_channel_violation_ticks"] += 1
        raise ValueError("HS ThC-yaw phase modified an unregistered channel")
    adhesion_before_bytes = (b"NONE" if adhesion_before is None
                             else np.ascontiguousarray(
                                 adhesion_before).tobytes())
    adhesion_after_bytes = (b"NONE" if adhesion is None
                            else np.ascontiguousarray(adhesion).tobytes())
    config["adhesion_before_sha256"].update(adhesion_before_bytes)
    config["adhesion_after_sha256"].update(adhesion_after_bytes)
    if ((adhesion is None) != (adhesion_before is None)
            or (adhesion is not None
                and not np.array_equal(adhesion, adhesion_before))):
        config["adhesion_violation_ticks"] += 1
        raise ValueError("HS ThC-yaw phase modified adhesion")
    if abs(phase_delta) > 1e-12:
        config["applied_ticks"] += 1
        config["max_absolute_phase_deg"] = max(
            float(config["max_absolute_phase_deg"]),
            float(abs(np.degrees(phase_delta))))
    return flex, ext, adhesion


def _visual_teacher_hs_thc_yaw_phase_receipt(config):
    """Return the JSON-safe four-DOF phase-composition receipt."""
    return {
        "magnitude_deg": float(config["magnitude_deg"]),
        "magnitude_deg_by_class": [
            float(item) for item in config["magnitude_deg_by_class"]],
        "per_class_magnitudes": bool(config["per_class_magnitudes"]),
        "tau_ms": float(config["tau_ms"]),
        "release_tau_ms": float(config["release_tau_ms"]),
        "separate_release_tau": bool(config["separate_release_tau"]),
        "release_ticks": int(config["release_ticks"]),
        "phase_source": str(config["phase_source"]),
        "class_override": config["class_override"],
        "override_ticks": int(config["override_ticks"]),
        "command_state": float(config["command_state"]),
        "command_state_min": float(config["command_state_min"]),
        "command_state_max": float(config["command_state_max"]),
        "leg_weights_lf_lm_lh_rf_rm_rh": [
            float(item) for item in config["leg_weights"]],
        "per_leg_weights": bool(config["per_leg_weights"]),
        "modified_dof_names": [item[0] for item in config["entries"]],
        "modified_dof_indices": (
            config["modified_indices"].astype(int).tolist()),
        "other_dof_count": int(len(config["other_indices"])),
        "applied_ticks": int(config["applied_ticks"]),
        "decoded_ticks_by_class": config["decoded_ticks"].astype(int).tolist(),
        "target_ticks_class0_class1_none": (
            config["target_ticks_class0_class1_none"].astype(int).tolist()),
        "changed_ticks_by_dof": (
            config["changed_ticks_by_dof"].astype(int).tolist()),
        "abs_drive_delta_sum_by_dof": (
            config["abs_drive_delta_sum_by_dof"].astype(float).tolist()),
        "other_channel_violation_ticks": int(
            config["other_channel_violation_ticks"]),
        "adhesion_violation_ticks": int(config["adhesion_violation_ticks"]),
        "other_flex_before_sha256": (
            config["other_flex_before_sha256"].hexdigest()),
        "other_flex_after_sha256": (
            config["other_flex_after_sha256"].hexdigest()),
        "other_ext_before_sha256": (
            config["other_ext_before_sha256"].hexdigest()),
        "other_ext_after_sha256": (
            config["other_ext_after_sha256"].hexdigest()),
        "adhesion_before_sha256": (
            config["adhesion_before_sha256"].hexdigest()),
        "adhesion_after_sha256": (
            config["adhesion_after_sha256"].hexdigest()),
        "max_absolute_phase_deg": float(config["max_absolute_phase_deg"]),
        "dna02_amplitude_control": bool(
            config["dna02_amplitude_control"]),
        "dna02_amplitude_ticks": int(config["dna02_amplitude_ticks"]),
        "dna02_amplitude_max": float(config["dna02_amplitude_max"]),
        "dna02_rate_hz_max": float(config["dna02_rate_hz_max"]),
        "dna02_max_rate_hz": (None if config["dna02_max_rate_hz"] is None
                               else float(config["dna02_max_rate_hz"])),
        "dna02_max_rate_hz_by_side": config[
            "dna02_max_rate_hz_by_side"],
        "dna02_amplitude_mode": config["dna02_amplitude_mode"],
        "class_source": "raw_hs_centroid_before_detector_override",
        "application_order": "after_side_phase_before_contact_control",
    }


STRUCTURED_BLOCKS = frozenset({
    "type_out_gain",
    "type_out_gain_id",
    "threshold_span_mv",
    "delay_type_ms",
})


def _require_structured_spec(spec):
    if not isinstance(spec, dict) or not spec:
        raise ValueError("structured spec must be a non-empty dict")
    unknown = sorted(set(spec) - STRUCTURED_BLOCKS)
    if unknown:
        raise ValueError(f"unknown structured coordinates: {unknown}")
    return spec


def sample_structured(rng, spec):
    """Sample bounded named maps. Does not touch SPACE or CATS."""
    spec = _require_structured_spec(spec)
    out = {}
    tog = spec.get("type_out_gain")
    if tog:
        names = list(tog["names"])
        lo, hi = (float(tog["bounds"][0]), float(tog["bounds"][1]))
        default = float(tog.get("default", 1.0))
        k = min(int(tog.get("k", len(names))), len(names))
        chosen = set()
        if k and names:
            idx = rng.choice(len(names), size=k, replace=False)
            chosen = {names[int(i)] for i in np.atleast_1d(idx)}
        out["type_out_gain"] = {
            n: float(rng.uniform(lo, hi)) if n in chosen else default
            for n in names
        }
    togid = spec.get("type_out_gain_id")
    if togid:
        ids = [str(x) for x in togid["ids"]]
        lo, hi = (float(togid["bounds"][0]), float(togid["bounds"][1]))
        default = float(togid.get("default", 1.0))
        k = min(int(togid.get("k", len(ids))), len(ids))
        chosen = set()
        if k and ids:
            idx = rng.choice(len(ids), size=k, replace=False)
            chosen = {ids[int(i)] for i in np.atleast_1d(idx)}
        out["type_out_gain_id"] = {
            i: float(rng.uniform(lo, hi)) if i in chosen else default
            for i in ids
        }
    span = spec.get("threshold_span_mv")
    if span:
        lo, hi = (float(span["bounds"][0]), float(span["bounds"][1]))
        p_on = float(span.get("p_on", 1.0))
        if rng.random() < p_on:
            out["threshold_mode"] = span.get("mode", "volume_rank")
            out["threshold_mean_mv"] = float(span.get("mean_mv", -45.0))
            out["threshold_span_mv"] = float(rng.uniform(lo, hi))
    dtm = spec.get("delay_type_ms")
    if dtm:
        names = list(dtm["names"])
        lo, hi = (float(dtm["bounds"][0]), float(dtm["bounds"][1]))
        if lo < 0.1 or hi > 100.0:
            raise ValueError("delay_type_ms bounds must sit inside [0.1, 100]")
        default = float(dtm.get("default", 1.0))
        k = min(int(dtm.get("k", len(names))), len(names))
        chosen = set()
        if k and names:
            idx = rng.choice(len(names), size=k, replace=False)
            chosen = {names[int(i)] for i in np.atleast_1d(idx)}
        out["delay_type_ms"] = {
            n: float(rng.uniform(lo, hi)) if n in chosen else default
            for n in names
        }
    return out


def neighbour_structured(th, rng, spec):
    """Step named maps already on `th`. Missing maps are sampled fresh."""
    spec = _require_structured_spec(spec)
    out = {}
    tog = spec.get("type_out_gain")
    if tog:
        names = list(tog["names"])
        lo, hi = (float(tog["bounds"][0]), float(tog["bounds"][1]))
        cur = dict(th.get("type_out_gain") or {})
        default = float(tog.get("default", 1.0))
        width = hi - lo
        nxt = {}
        for n in names:
            v = float(cur[n]) if n in cur else default
            nxt[n] = float(np.clip(v + rng.normal(0, 0.15) * width, lo, hi))
        out["type_out_gain"] = nxt
    togid = spec.get("type_out_gain_id")
    if togid:
        ids = [str(x) for x in togid["ids"]]
        lo, hi = (float(togid["bounds"][0]), float(togid["bounds"][1]))
        cur = dict(th.get("type_out_gain_id") or {})
        default = float(togid.get("default", 1.0))
        width = hi - lo
        nxt = {}
        for i in ids:
            v = float(cur[i]) if i in cur else default
            nxt[i] = float(np.clip(v + rng.normal(0, 0.15) * width, lo, hi))
        out["type_out_gain_id"] = nxt
    span = spec.get("threshold_span_mv")
    if span:
        lo, hi = (float(span["bounds"][0]), float(span["bounds"][1]))
        if "threshold_span_mv" in th:
            width = hi - lo
            out["threshold_mode"] = th.get(
                "threshold_mode", span.get("mode", "volume_rank"))
            out["threshold_mean_mv"] = float(
                th.get("threshold_mean_mv", span.get("mean_mv", -45.0)))
            out["threshold_span_mv"] = float(np.clip(
                float(th["threshold_span_mv"]) + rng.normal(0, 0.15) * width,
                lo, hi))
        elif rng.random() < float(span.get("p_on", 1.0)):
            out["threshold_mode"] = span.get("mode", "volume_rank")
            out["threshold_mean_mv"] = float(span.get("mean_mv", -45.0))
            out["threshold_span_mv"] = float(rng.uniform(lo, hi))
    dtm = spec.get("delay_type_ms")
    if dtm:
        names = list(dtm["names"])
        lo, hi = (float(dtm["bounds"][0]), float(dtm["bounds"][1]))
        if lo < 0.1 or hi > 100.0:
            raise ValueError("delay_type_ms bounds must sit inside [0.1, 100]")
        cur = dict(th.get("delay_type_ms") or {})
        default = float(dtm.get("default", 1.0))
        width = hi - lo
        nxt = {}
        for n in names:
            v = float(cur[n]) if n in cur else default
            nxt[n] = float(np.clip(v + rng.normal(0, 0.15) * width, lo, hi))
        out["delay_type_ms"] = nxt
    return out


def apply_structured(th, rng, spec, step=False):
    """Overlay structured maps on an existing theta. Scalar keys stay put."""
    out = dict(th)
    maps = (neighbour_structured(out, rng, spec) if step
            else sample_structured(rng, spec))
    out.update(maps)
    return out


def structured_vector(th, spec):
    """Every declared coordinate, including defaults left at 1.0."""
    spec = _require_structured_spec(spec)
    vec = {}
    tog = spec.get("type_out_gain")
    if tog:
        cur = dict(th.get("type_out_gain") or {})
        default = float(tog.get("default", 1.0))
        vec["type_out_gain"] = {
            n: float(cur[n]) if n in cur else default for n in tog["names"]
        }
    togid = spec.get("type_out_gain_id")
    if togid:
        cur = dict(th.get("type_out_gain_id") or {})
        default = float(togid.get("default", 1.0))
        vec["type_out_gain_id"] = {
            str(i): float(cur[str(i)]) if str(i) in cur else default
            for i in togid["ids"]
        }
    if spec.get("threshold_span_mv"):
        vec["threshold_mode"] = th.get("threshold_mode")
        vec["threshold_mean_mv"] = th.get("threshold_mean_mv")
        vec["threshold_span_mv"] = th.get("threshold_span_mv")
    dtm = spec.get("delay_type_ms")
    if dtm:
        cur = dict(th.get("delay_type_ms") or {})
        default = float(dtm.get("default", 1.0))
        vec["delay_type_ms"] = {
            n: float(cur[n]) if n in cur else default for n in dtm["names"]
        }
    return vec


def sample(rng, blocks=None):
    th = {k: float(rng.uniform(*v)) for k, v in SPACE.items()}
    for k, opts in CATS.items():
        th[k] = opts[rng.integers(len(opts))]
    if blocks:
        th.update(sample_structured(rng, blocks))
    return th


def neighbour(th, rng, blocks=None):
    out = dict(th)
    for k, (lo, hi) in SPACE.items():
        out[k] = float(np.clip(th[k] + rng.normal(0, 0.15) * (hi - lo), lo, hi))
    for k, opts in CATS.items():
        if rng.random() < 0.25:
            out[k] = opts[rng.integers(len(opts))]
    if blocks:
        out.update(neighbour_structured(out, rng, blocks))
    return out


def _quantise_delays(d, groups):
    """Quantise a delay vector to `groups` equal-occupancy levels (each
    neuron gets its group's median) so the delay ring stays shallow and the
    per-tick delivery cost is bounded by the group count. Deterministic."""
    g = int(groups)
    if g > 0:
        qs = np.quantile(d, np.linspace(0.0, 1.0, g + 1))
        idx = np.clip(np.searchsorted(qs, d, side="right") - 1, 0, g - 1)
        for gi in range(g):
            m = idx == gi
            if m.any():
                d[m] = float(np.median(d[m]))
    return d


def _cable_lengths(meta):
    L = meta["l2_cable_length_um"].to_numpy(dtype=np.float64)
    finite = np.isfinite(L) & (L > 0.0)
    med = float(np.median(L[finite]))
    return np.where(finite, L, med)


def _site_rows(meta):
    """Design 394ad83c: the recorded-site rows gpu/escape/two_site.py appends
    after every BANC row, cell_type "<type>:M" (no BANC type ends so)."""
    ct = meta["cell_type"]
    return ct.notna().to_numpy() & ct.astype(str).str.endswith(":M").to_numpy()


def _with_site_rows(meta, th):
    """Design 394ad83c, retained into the build body at team 0e33912583: the
    body's explicit gap pairs name ten recorded-site rows S, each a DLM row's
    id behind a 1 (gpu/escape/two_site.py), which no BANC table holds. A theta
    whose gap_pair_ids name such ids walks on the table with them appended by
    two_site.extend, in the order the pairs first name them, which is the
    table the design's own walkers (gpu/escape/e05c.py, e05c_walk.py) build.
    A table that already holds them and a theta naming none are returned
    untouched = bit-identical; an id that is neither a row nor "1" + a row is
    left for _explicit_gap_pair_ids to refuse.
    """
    pairs = th.get("gap_pair_ids")
    if th.get("gap") != "explicit_pairs" or not isinstance(pairs, list):
        return meta
    from gpu.escape import two_site as _two_site
    have = set(meta["banc_888_id"].astype(str))
    dlm = []
    for pair in pairs:
        for x in (map(str, pair) if isinstance(pair, (list, tuple)) else ()):
            a = x[len(_two_site.PREFIX):]
            if x not in have and x.startswith(_two_site.PREFIX) and a in have and a not in dlm:
                dlm.append(a)
    if not dlm:
        return meta
    return _two_site.extend(meta, dlm)[0]


class CampRiseLaw:
    """Design 6cc949c1, Build B1 (study 779063c1): camp_src "load_rise", C1's
    lagged load law in 86a4f135's form, for one leg. Each tick reads F(t - n)
    and F(t - n - 1) from the ring of the last n + 1 inputs, both 0 until the
    ring is full (so the first n + 1 ticks of a walk read 0), then pushes this
    tick's input. dF = F(t - n) - F(t - n - 1); the tick is gated when
    dF < -g_fall x tick x F_ref; L(F) = G x clip(F / F_ref, 0, 1); the rate is
    s x f x L(F(t - n)) gated and s x clip(L(F(t - n)) + k_rise x max(0, q),
    0, G) otherwise, q = dF / (tick x F_ref) in F_ref per ms. The walk and the
    build note's unit check both step this class."""

    def __init__(self, lag_ticks, G, F_ref, k_rise, floor, gate, tick_ms, s=1.0):
        import collections
        self.G, self.F_ref, self.k_rise = float(G), float(F_ref), float(k_rise)
        self.floor, self.gate = float(floor), float(gate)
        self.tick_ms, self.s = float(tick_ms), float(s)
        self.ring = collections.deque(maxlen=int(lag_ticks) + 1)

    def step(self, F):
        """-> (rate Hz, clip(F(t - n) / F_ref, 0, 1), q, gated, rising)."""
        if len(self.ring) == self.ring.maxlen:
            f_lag, f_prev = self.ring[1], self.ring[0]
        else:
            f_lag = f_prev = 0.0
        self.ring.append(float(F))
        dF = f_lag - f_prev
        u = float(np.clip(f_lag / self.F_ref, 0.0, 1.0))
        L = self.G * u
        q = dF / (self.tick_ms * self.F_ref)
        gated = dF < -self.gate * self.tick_ms * self.F_ref
        if gated:
            rate = self.s * self.floor * L
        else:
            rate = self.s * float(np.clip(L + self.k_rise * max(0.0, q), 0.0, self.G))
        return rate, u, q, gated, (not gated) and dF > 0.0


def length_delay_vector(meta, mean_ms, cap_ms, groups):
    """MEASURED-STRUCTURE DIRECTIVE (Robin, 2026-09-01): per-neuron
    presynaptic delay proportional to BANC whole-neuron cable length,
    scaled so its MEAN equals `mean_ms` (default 1.8, the verified scalar
    operating point, so the dial isolates delay HETEROGENEITY from a mean
    shift), clipped to [0.1, cap_ms], then quantised via _quantise_delays.
    Semantics from lane B's 2026-09-01 implementation, canonicalised here;
    whole-neuron cable is a labelled PROXY for spike-initiation-to-synapse
    path, which BANC does not carry.

    Design 394ad83c: a site row (_site_rows) is not a neuron, so the law is
    computed over the other rows alone, which then carry exactly the values
    a table without site rows gives them, and a site row takes mean_ms. A
    site row has no chemical edge, so its delay reaches no engine term. A
    table without site rows runs the law as before.
    """
    site = _site_rows(meta)
    if site.any():
        d = np.full(len(meta), float(mean_ms))
        d[~site] = length_delay_vector(meta.loc[~site, ["l2_cable_length_um", "cell_type"]],
                                       mean_ms, cap_ms, groups)
        return d
    L = _cable_lengths(meta)
    d = L * (float(mean_ms) / float(L.mean()))
    d = _quantise_delays(np.clip(d, 0.1, float(cap_ms)), groups)
    # quantisation to group medians drags the mean off the anchor (measured
    # 1.551 for 1.8 at 8 groups), so restore it exactly after quantising;
    # the level count is unchanged and the bounds stay inside [0.1, cap]
    # for any anchor this model plausibly searches.
    d *= float(mean_ms) / float(d.mean())
    return d


def type_delay_vector(meta, base_ms, type_ms, cap_ms):
    """Per-neuron presynaptic delay set by cell type.

    C-M55 (2026-09-04) measured that the T4/T5 wiring offset does not
    produce direction selectivity in this model, on the male where the
    offset is present at 148 to 168 degrees of opposition. The reason is
    arithmetic: a T4 cell sums linearly and every synapse carries the same
    delay, so an offset with no temporal difference is direction-blind.
    This gives the delayed arm an actual delay.

    `type_ms` maps a cell type to that type's presynaptic delay in
    milliseconds; every other neuron keeps `base_ms`, a scalar or a
    per-neuron vector (the `length_by_type` mode passes the length vector,
    because CD1-SC measured that the cord's 5.8 Hz line needs the
    length-derived delays and a flat base has no rhythm to test on). Tied at cell type,
    per Arie's 2026-09-03 rule that a free parameter is tied at cell type
    or coarser and never searched per individual cell. Empty map = every
    neuron at base = the scalar path, bit-identical.
    """
    ct = meta["cell_type"].astype("string")
    if "fafb_alignment_cell_type" in meta.columns:
        fa = meta["fafb_alignment_cell_type"].astype("string")
        ct = ct.where(ct.notna(), fa)
    labels = np.array([x if isinstance(x, str) else ""
                       for x in ct.to_numpy()], dtype=object)
    if np.ndim(base_ms):
        d = np.array(base_ms, dtype=np.float64, copy=True)
        if d.shape != (len(labels),):
            raise ValueError("type_delay_vector: per-neuron base needs one "
                             f"value per neuron: got {d.shape}, meta {len(labels)}")
    else:
        d = np.full(len(labels), float(base_ms), dtype=np.float64)
    for name, value in type_ms.items():
        d[labels == str(name)] = float(value)
    return np.clip(d, 0.1, float(cap_ms))


def type_out_gain_vector(meta, type_map, id_map=None):
    """Post-balance outgoing gain by cell type, with optional per-id override.

    Labels match `type_delay_vector`: `cell_type`, then
    `fafb_alignment_cell_type`. Exceptional-cell ids in `id_map` replace the
    type value for that row. Gains must be finite and >= 0. Unnamed types
    and ids stay 1.0. Every named type and id must match at least one row.
    Empty maps are not called; absent theta keys stay bit-identical.
    """
    ct = meta["cell_type"].astype("string")
    if "fafb_alignment_cell_type" in meta.columns:
        fa = meta["fafb_alignment_cell_type"].astype("string")
        ct = ct.where(ct.notna(), fa)
    labels = np.array([x if isinstance(x, str) else ""
                       for x in ct.to_numpy()], dtype=object)
    vec = np.ones(len(labels), dtype=np.float64)
    type_rows = {}
    # One pass groups the row numbers by label. The name-by-name equality scan
    # this replaces compared every named type against every row, and a champion
    # theta names 11,578 types over 188,508 object-dtype rows: 2.2e9 string
    # comparisons, 9.0 s of a 22.3 s seed, paid again on every seed (profiled
    # 2026-09-19, gpu seat). The rows a name selects are the rows its mask
    # selected, so every assignment and every count below is unchanged.
    rows_by_label = {}
    if type_map:
        for _row, _label in enumerate(labels):
            rows_by_label.setdefault(_label, []).append(_row)
    for name, value in (type_map or {}).items():
        g = float(value)
        if not np.isfinite(g) or g < 0.0:
            raise ValueError(
                f"type_out_gain: {name} must be finite and >= 0, got {value!r}")
        rows = rows_by_label.get(str(name), ())
        n = len(rows)
        if n == 0:
            raise ValueError(f"type_out_gain: {name} matches no row")
        vec[rows] = g
        type_rows[str(name)] = n
    id_rows = []
    if id_map:
        ids = meta["banc_888_id"].astype(str).to_numpy()
        pos = {v: i for i, v in enumerate(ids)}
        for k, value in id_map.items():
            if str(k) not in pos:
                raise ValueError(f"type_out_gain_id: {k} is not a BANC row")
            g = float(value)
            if not np.isfinite(g) or g < 0.0:
                raise ValueError(
                    f"type_out_gain_id: {k} must be finite and >= 0, got "
                    f"{value!r}")
            vec[pos[str(k)]] = g
            id_rows.append(pos[str(k)])
    return vec, {"type_rows": type_rows, "id_rows": id_rows}


def length_bv_delay_vector(meta, base_ms, velocity_um_ms, cap_ms, groups):
    """The base-plus-velocity form: delay_i = base_ms +
    l2_cable_length_um_i / velocity_um_ms, clipped and quantised as above.
    velocity 300 um/ms = 0.3 m/s, the thin-unmyelinated-fibre range; both
    dials are searchable and the velocity is pinnable by published
    conduction measurements (ledger row cns.delay_ms).
    """
    L = _cable_lengths(meta)
    d = float(base_ms) + L / max(float(velocity_um_ms), 1e-9)
    return _quantise_delays(np.clip(d, 0.1, float(cap_ms)), groups)


def volume_rank_threshold_vector(meta, mean_mv=-45.0, span_mv=4.0):
    """Per-neuron spike thresholds from measured morphology rank.

    This is a HYPOTHESIS, not a measured voltage law.  It removes the
    biologically implausible universal threshold while preserving the
    historical population mean.  Positive finite ``volume_nm3`` values keep
    their measured ordering; incomplete rows receive the median valid volume
    before average-tie ranking.  Graded rows remain thresholdless in CNS.run.
    """
    mean_mv = float(mean_mv)
    span_mv = float(span_mv)
    if not np.isfinite(mean_mv):
        raise ValueError("threshold_mean_mv must be finite")
    if not np.isfinite(span_mv) or not 0.0 < span_mv <= 20.0:
        raise ValueError("threshold_span_mv must be finite in (0, 20]")
    if "volume_nm3" not in meta:
        raise ValueError("threshold_mode='volume_rank' needs volume_nm3")
    volume = np.asarray(meta["volume_nm3"], dtype=np.float64)
    valid = np.isfinite(volume) & (volume > 0.0)
    if not np.any(valid):
        raise ValueError("volume_nm3 has no positive finite rows")
    filled = volume.copy()
    filled[~valid] = float(np.median(volume[valid]))
    ranks = pd.Series(filled).rank(method="average").to_numpy(dtype=float)
    centred = ranks - float(ranks.mean())
    max_abs = float(np.max(np.abs(centred)))
    if not max_abs > 0.0:
        raise ValueError("volume_nm3 cannot produce threshold heterogeneity")
    threshold = mean_mv + 0.5 * span_mv * centred / max_abs
    # Remove the last floating-point residue without changing ordering.
    threshold += mean_mv - float(threshold.mean())
    if (threshold.shape != (len(meta),)
            or not np.all(np.isfinite(threshold))):
        raise ValueError("invalid per-neuron threshold vector")
    return threshold.astype(np.float32)


def type_threshold_vector(meta, base_mv, type_mv, *, exact=False):
    """Assign thresholds by type, optionally forbidding alignment fallback.

    Historical ``by_type`` semantics fill a missing BANC ``cell_type`` from
    ``fafb_alignment_cell_type``.  ``exact=True`` is the evidence-preserving
    path: only the BANC field may select a row, and every requested type must
    match.  The returned audit makes that distinction machine-checkable.
    """
    labels_exact = meta["cell_type"].astype("string")
    labels = labels_exact
    if not exact and "fafb_alignment_cell_type" in meta.columns:
        labels = labels.where(
            labels.notna(),
            meta["fafb_alignment_cell_type"].astype("string"))
    labels_array = np.array(
        [value if isinstance(value, str) else ""
         for value in labels.to_numpy()], dtype=object)
    exact_array = np.array(
        [value if isinstance(value, str) else ""
         for value in labels_exact.to_numpy()], dtype=object)
    vector = np.full(len(labels_array), float(base_mv), dtype=np.float64)
    matched = {}
    selected = np.zeros(len(labels_array), dtype=bool)
    for name, value in type_mv.items():
        mask = labels_array == str(name)
        if exact and not np.any(mask):
            raise ValueError(
                f"threshold exact cell type {name!r} matched no BANC rows")
        vector[mask] = float(value)
        selected |= mask
        matched[str(name)] = int(mask.sum())
    requested = np.isin(labels_array, list(map(str, type_mv)))
    exact_requested = np.isin(exact_array, list(map(str, type_mv)))
    fallback_only = requested & ~exact_requested
    audit = {
        "mode": "by_exact_type" if exact else "by_type",
        "matched_rows_by_type": matched,
        "selected_rows": int(selected.sum()),
        "exact_banc_rows": int(exact_requested.sum()),
        "fallback_only_rows": int(fallback_only.sum()),
        "fallback_only_banc_ids": (
            meta.loc[fallback_only, "banc_888_id"].astype(str).tolist()
            if "banc_888_id" in meta.columns else []),
    }
    return vector, audit


def exact_type_vrest_overrides(meta, type_mv):
    """Build exact-BANC-type resting-potential overrides and an audit.

    IDs, rather than an N-vector, preserve the scalar conductance-normalizing
    reference used by the existing model.  Only the selected cells' effective
    rest changes; size-derived motor overrides and other exact-ID mechanisms
    can compose through the same dictionary.
    """
    labels = np.array(
        [value if isinstance(value, str) else ""
         for value in meta["cell_type"].astype("string").to_numpy()],
        dtype=object)
    identifiers = meta["banc_888_id"].to_numpy()
    overrides = {}
    matched = {}
    selected_ids = []
    for name, value in type_mv.items():
        rows = np.flatnonzero(labels == str(name))
        if not len(rows):
            raise ValueError(
                f"v_rest exact cell type {name!r} matched no BANC rows")
        matched[str(name)] = int(len(rows))
        for row in rows:
            identifier = identifiers[row]
            overrides[identifier] = float(value)
            selected_ids.append(str(identifier))
    return overrides, {
        "mode": "by_exact_type",
        "matched_rows_by_type": matched,
        "selected_rows": len(overrides),
        "selected_banc_ids": selected_ids,
        "fallback_only_rows": 0,
        "fallback_only_banc_ids": [],
    }


def membrane_annotation_groups(meta):
    """Use the finest available biological annotation for each MCNS row.

    MCNS ``cell_class`` is blank on 97% of rows, whereas ``cell_type`` carries
    the connectome's actual type identity for most neurons.  Fall back through
    subclass/class/superclass only when the finer key is absent.  Rows with no
    annotation stay tied together instead of receiving invented independent
    identities.
    """
    priority = (
        ("super_class", "super:"),
        ("mcns_class_raw", "rawclass:"),
        ("mcns_subclass_raw", "rawsub:"),
        ("cell_sub_class", "sub:"),
        ("cell_type", "type:"),
    )
    if not any(column in meta for column, _ in priority):
        raise ValueError("class_draw needs a biological annotation column")
    groups = np.full(len(meta), "unannotated", dtype=object)
    for column, prefix in priority:
        if column not in meta:
            continue
        values = meta[column].fillna("").astype(str).str.strip()
        mask = values.ne("").to_numpy()
        groups[mask] = prefix + values.to_numpy(dtype=str)[mask]
    return groups.astype(str)


def class_draw_membrane_vectors(meta, threshold, seed, v_rest_override=None):
    """Return one tied heterogeneous membrane draw.

    The selected closed-loop model otherwise gives every neuron the same
    resting/reset potential, membrane time constant, and refractory period.
    That synchronises unrelated cell classes for no biological reason.  This
    hypothesis draws one value per finest available annotated biological group
    (rather than one value per neuron), then population-anchors the log means
    at the historical operating point so the experiment changes heterogeneity
    rather than global gain.

    Ranges are the pre-existing Stage-1 plausible bounds, not measurements of
    individual MCNS cells.  The returned threshold preserves the caller's
    measured-structure ordering as an offset in each cell's rest-relative
    firing margin.  This is therefore a labelled hypothesis, never a claim
    that BANC/MCNS measured these membrane values.
    """
    groups = membrane_annotation_groups(meta)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("class_draw needs at least two cell classes")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ValueError("membrane_seed must be an integer") from error
    rng = np.random.default_rng(seed)

    # Moderate class spread, bounded by the already registered Stage-1 ranges.
    rest_by_class = np.clip(-52.0 + rng.normal(0.0, 2.5, len(unique)),
                            -60.0, -45.0)
    log_tau = np.clip(np.log(20.0) + rng.normal(0.0, 0.35, len(unique)),
                      np.log(5.0), np.log(50.0))
    log_refrac = np.clip(np.log(2.2) + rng.normal(0.0, 0.30, len(unique)),
                         np.log(0.5), np.log(5.0))
    rest = rest_by_class[inverse].astype(np.float64)
    tau = np.exp(log_tau[inverse]).astype(np.float64)
    refrac = np.exp(log_refrac[inverse]).astype(np.float64)

    # Preserve the selected model's population operating point.  These are
    # population anchors, not universal per-cell values.
    rest += -52.0 - float(rest.mean())
    tau *= 20.0 / float(np.exp(np.mean(np.log(tau))))
    refrac *= 2.2 / float(np.exp(np.mean(np.log(refrac))))

    if v_rest_override:
        id_to_row = {str(value): row for row, value in enumerate(
            meta["banc_888_id"].astype(str).to_numpy())}
        for neuron_id, value in v_rest_override.items():
            row = id_to_row.get(str(neuron_id))
            if row is not None:
                rest[row] = float(value)

    old_threshold = np.broadcast_to(
        np.asarray(threshold, dtype=np.float64), (len(meta),))
    if not np.isfinite(old_threshold).all():
        raise ValueError("class_draw received non-finite thresholds")
    structural_offset = old_threshold - float(old_threshold.mean())
    margin = np.clip(7.0 + structural_offset, 3.0, 12.0)
    new_threshold = rest + margin
    values = (rest, tau, refrac, new_threshold)
    if any(value.shape != (len(meta),) or not np.isfinite(value).all()
           for value in values):
        raise ValueError("invalid class-draw membrane vectors")
    return tuple(value.astype(np.float32) for value in values)


def class_draw_synaptic_decay_vector(meta, seed):
    """Return one annotation-tied, area-normalized decay hypothesis.

    The historical model assigns 5 ms to every chemical synapse.  Here the
    postsynaptic neuron receives one value tied to its finest available
    biological annotation.  The 2--25 ms interval is the engine's existing
    explored range, not a per-cell measurement.  A clipped location shift
    keeps the population geometric mean exactly at 5 ms, so the paired body
    test changes heterogeneity rather than the global time-scale anchor.
    """
    groups = membrane_annotation_groups(meta)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("synaptic class_draw needs at least two cell classes")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ValueError("synaptic_decay_seed must be an integer") from error
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53594E]))
    offsets = rng.normal(0.0, 0.45, len(unique))
    target = np.log(5.0)
    lower, upper = np.log(2.0), np.log(25.0)

    # Monotone clipped-location solve; unlike a post-hoc rescale it cannot
    # move a value back outside the registered bounds.
    lo, hi = -4.0, 4.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        clipped = np.clip(target + offsets + mid, lower, upper)
        mean = float(clipped[inverse].mean())
        if mean < target:
            lo = mid
        else:
            hi = mid
    by_class = np.exp(np.clip(
        target + offsets + 0.5 * (lo + hi), lower, upper))
    vector = by_class[inverse]
    if (vector.shape != (len(meta),)
            or not np.all(np.isfinite(vector))
            or np.any(vector < 2.0)
            or np.any(vector > 25.0)
            or not np.isclose(np.exp(np.mean(np.log(vector))), 5.0,
                              atol=2e-6, rtol=0.0)):
        raise ValueError("invalid class-draw synaptic decay vector")
    return vector.astype(np.float32)


def membrane_range_graded_vectors(v_rest, threshold):
    """Derive heterogeneous graded-release curves from each membrane range.

    The sigmoid is fixed by two dimensionless endpoint identities rather than
    one universal voltage: release is 10% at that row's rest potential and 90%
    at its counterfactual firing threshold. The midpoint is therefore the
    row's membrane-range midpoint and the slope is range/(2*ln(9)).
    """
    rest = np.asarray(v_rest, dtype=np.float64)
    firing = np.asarray(threshold, dtype=np.float64)
    if (rest.shape != firing.shape or rest.ndim != 1
            or not np.all(np.isfinite(rest))
            or not np.all(np.isfinite(firing))):
        raise ValueError("graded membrane range needs matching finite vectors")
    span = firing - rest
    if np.any(span <= 0.0):
        raise ValueError("graded membrane range must be positive")
    midpoint = rest + 0.5 * span
    slope = span / (2.0 * np.log(9.0))
    return midpoint.astype(np.float32), slope.astype(np.float32)


def membrane_coupled_adaptation_vectors(
        v_rest, threshold, tau_mem, population_adapt_b, population_tau_w):
    """Replace universal adaptation with membrane-coupled neuron vectors.

    The active population anchors are preserved: adapt_b arithmetically and
    tau_w geometrically.  No new scale or random draw is introduced.
    """
    rest = np.asarray(v_rest, dtype=np.float64)
    firing = np.asarray(threshold, dtype=np.float64)
    membrane_tau = np.asarray(tau_mem, dtype=np.float64)
    if (rest.ndim != 1 or firing.shape != rest.shape
            or membrane_tau.shape != rest.shape
            or not np.all(np.isfinite(rest))
            or not np.all(np.isfinite(firing))
            or not np.all(np.isfinite(membrane_tau))
            or np.any(membrane_tau <= 0.0)):
        raise ValueError(
            "membrane-coupled adaptation needs matching finite vectors")
    gap = firing - rest
    if np.any(gap <= 0.0):
        raise ValueError(
            "membrane-coupled adaptation needs positive threshold gaps")
    if (type(population_adapt_b) not in (int, float)
            or not np.isfinite(population_adapt_b)
            or float(population_adapt_b) <= 0.0
            or type(population_tau_w) not in (int, float)
            or not np.isfinite(population_tau_w)
            or float(population_tau_w) <= 0.0):
        raise ValueError(
            "membrane-coupled adaptation needs positive scalar anchors")
    adapt_b = float(population_adapt_b) * gap / float(gap.mean())
    tau_w = (float(population_tau_w) * membrane_tau
             / float(np.exp(np.mean(np.log(membrane_tau)))))
    # Correct only floating reduction error, without adding a parameter.
    adapt_b *= float(population_adapt_b) / float(adapt_b.mean())
    tau_w *= (float(population_tau_w)
              / float(np.exp(np.mean(np.log(tau_w)))))
    if (not np.all(np.isfinite(adapt_b)) or np.any(adapt_b <= 0.0)
            or not np.all(np.isfinite(tau_w)) or np.any(tau_w <= 0.0)
            or not np.isclose(adapt_b.mean(), float(population_adapt_b),
                              rtol=0.0, atol=2e-7)
            or not np.isclose(np.exp(np.mean(np.log(tau_w))),
                              float(population_tau_w),
                              rtol=0.0, atol=2e-5)):
        raise ValueError("invalid membrane-coupled adaptation vectors")
    return adapt_b.astype(np.float32), tau_w.astype(np.float32)


def _align_mcns_active_structure(meta, structure):
    """Align the frozen R40 male structure table to network row order."""
    required = {
        "banc_888_id", "active_graph", "delay_proxy_ms",
        "threshold_proxy_mv"}
    missing = required - set(structure.columns)
    if missing:
        raise ValueError(
            f"MCNS active structure missing columns: {sorted(missing)}")
    ids = structure["banc_888_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("MCNS active structure has duplicate neuron IDs")
    network_ids = meta["banc_888_id"].astype(str)
    aligned = structure.assign(banc_888_id=ids).set_index(
        "banc_888_id").reindex(network_ids)
    if aligned.index.duplicated().any() or aligned["active_graph"].isna().any():
        raise ValueError("MCNS active structure does not cover network IDs")
    if not pd.api.types.is_bool_dtype(aligned["active_graph"]):
        raise ValueError("MCNS active_graph must be Boolean")
    delay = aligned["delay_proxy_ms"].to_numpy(dtype=np.float64)
    threshold = aligned["threshold_proxy_mv"].to_numpy(dtype=np.float64)
    active = aligned["active_graph"].to_numpy(dtype=bool)
    if (delay.shape != (len(meta),) or threshold.shape != (len(meta),)
            or not np.isfinite(delay).all() or not np.isfinite(threshold).all()
            or np.any(delay < 0.0)):
        raise ValueError("MCNS active structure vectors are invalid")
    return delay, threshold, active


def _load_mcns_active_structure(meta):
    if not os.path.exists(_MCNS_ACTIVE_STRUCTURE_PATH):
        raise ValueError("frozen MCNS active structure table is absent")
    digest = hashlib.sha256()
    with open(_MCNS_ACTIVE_STRUCTURE_PATH, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != _MCNS_ACTIVE_STRUCTURE_SHA256:
        raise ValueError("frozen MCNS active structure bytes drift")
    return _align_mcns_active_structure(
        meta, pd.read_feather(_MCNS_ACTIVE_STRUCTURE_PATH))


def neuron_type_hemilineage_groups(meta):
    """Finest available biological identity for tied neural parameters."""
    needed = {"cell_type", "hemilineage", "cell_sub_class", "super_class",
              "neurotransmitter_predicted"}
    if not needed.intersection(meta.columns):
        raise ValueError("type/hemilineage prior needs biological annotations")

    def values(column):
        if column not in meta:
            return np.full(len(meta), "", dtype=str)
        return (meta[column].fillna("").astype(str).str.strip()
                .to_numpy(dtype=str))

    super_class = values("super_class")
    transmitter = values("neurotransmitter_predicted")
    groups = np.char.add(
        np.char.add("fallback:", np.where(super_class == "", "unknown",
                                          super_class)),
        np.char.add(":", np.where(transmitter == "", "unknown", transmitter)))
    for column, prefix in (("cell_sub_class", "subclass:"),
                           ("hemilineage", "hemilineage:"),
                           ("cell_type", "type:")):
        annotated = values(column)
        mask = annotated != ""
        groups[mask] = np.char.add(prefix, annotated[mask])
    return groups.astype(str)


def premotor_13a_21a_gain_vector(meta, ratio):
    """Relative inhibitory-output gain for premotor hemilineages 13A/21A.

    ``ratio`` is gain(13A) / gain(21A).  The inhibitory rows' arithmetic
    mean remains exactly one, so this is a lineage contrast rather than a
    disguised global inhibitory-gain change.  All other rows remain one.
    """
    try:
        ratio = float(ratio)
    except (TypeError, ValueError) as error:
        raise ValueError("inh_13a_21a_ratio must be numeric") from error
    if not np.isfinite(ratio) or not 0.25 <= ratio <= 4.0:
        raise ValueError("inh_13a_21a_ratio must be within [0.25, 4.0]")
    if "hemilineage" not in meta:
        raise ValueError("13A/21A gain contrast requires hemilineage metadata")
    nt = cns._resolve_nt(meta, cns.SimParams())
    sign = cns._sign_of(nt, cns.SimParams())
    lineage = meta["hemilineage"].fillna("").astype(str).to_numpy()
    is_inhibitory = sign < 0
    rows_13a = is_inhibitory & (lineage == "13A")
    rows_21a = is_inhibitory & (lineage == "21A")
    n13, n21 = int(rows_13a.sum()), int(rows_21a.sum())
    if n13 == 0 or n21 == 0:
        raise ValueError("13A/21A gain contrast resolved an empty lineage")
    gain_21a = (n13 + n21) / (n13 * ratio + n21)
    gain_13a = ratio * gain_21a
    gain = np.ones(len(meta), dtype=np.float64)
    gain[rows_13a] = gain_13a
    gain[rows_21a] = gain_21a
    selected = is_inhibitory & np.isin(lineage, ("13A", "21A"))
    if not np.isclose(float(gain[selected].mean()), 1.0):
        raise ValueError("13A/21A gain contrast failed to preserve its mean")
    return gain


def type_hemilineage_inhibitory_reversal_vector(
        meta, seed, permutation_seed=None):
    """Type-tied chloride reversals hung between two published adult values.

    The two anchors are real but not comparable (lane D R205, 2026-09-14,
    resolving the 2026-09-13 citation flag): l-LNv -48 +/- 0.6 mV is McCarthy
    et al. 2011 (whole-cell, brain explant, 17 mM NaCl internal, so the value
    is set by the pipette: solution ECl -48.1 mV; SRC-MCCARTHY-2011,
    MEAS-LIT-MCCARTHY-2011-LLNV-REVERSALS-R205); MN5 -74 +/- 4 mV is
    Ryglewski et al. 2017 (sharp electrode, VNC exposed in the animal, adult
    flight MN5; SRC-RYGLEWSKI-2017, MEAS-LIT-RYGLEWSKI-2017-MN5-ADULT-R205).
    The 26 mV spread between them is therefore not shown to be biological.
    Neither gives an adult leg premotor or leg motor value; -70 mV is ASSUMED.

    The exact anchors below select by cell_type. `cell_type == "MN5"` matches
    two PROBOSCIS motor neurons in the pinned BANC table, not the flight
    power cells, which are typed `DLM5`; so the MN5 anchor has never applied
    to a flight cell here. No number changes with this docstring (D R205).
    Other groups receive reproducible offsets clipped to that envelope; a
    common shift preserves the historical population mean of -70 mV.
    """
    groups = neuron_type_hemilineage_groups(meta)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("inhibitory reversal prior needs multiple groups")
    try:
        seed = int(seed)
    except (TypeError, ValueError) as error:
        raise ValueError("inhibitory_reversal_seed must be an integer") from error
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x45494E48]))
    offsets = rng.normal(0.0, 4.5, len(unique))
    cell_type = (meta["cell_type"].fillna("").astype(str).str.strip()
                 .to_numpy(dtype=str))
    exact_lnv = cell_type == "l-LNv"
    exact_mn5 = cell_type == "MN5"
    target, lo_bound, hi_bound = -70.0, -74.0, -48.0
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        values = np.clip(target + offsets[inverse] + mid,
                         lo_bound, hi_bound)
        values[exact_lnv] = -48.0
        values[exact_mn5] = -74.0
        if float(values.mean()) < target:
            lo = mid
        else:
            hi = mid
    values = np.clip(target + offsets[inverse] + 0.5 * (lo + hi),
                     lo_bound, hi_bound)
    values[exact_lnv] = -48.0
    values[exact_mn5] = -74.0
    if (values.shape != (len(meta),) or not np.all(np.isfinite(values))
            or np.any(values < lo_bound) or np.any(values > hi_bound)
            or not np.isclose(float(values.mean()), target,
                              atol=2e-6, rtol=0.0)):
        raise ValueError("invalid inhibitory reversal prior")
    values = values.astype(np.float32)
    if permutation_seed is None:
        return values

    # Equal-capacity negative control: reassign realised group values only
    # among biological groups with the same row count. This preserves the
    # complete row-weighted distribution, mean, bounds, number of parameters,
    # and within-group tying while deleting label-to-value identity.
    try:
        permutation_seed = int(permutation_seed)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "inhibitory_reversal_permutation_seed must be an integer") from error
    counts = np.bincount(inverse, minlength=len(unique))
    group_values = np.empty(len(unique), dtype=np.float32)
    for group_index in range(len(unique)):
        group_values[group_index] = values[np.flatnonzero(
            inverse == group_index)[0]]
    rng_perm = np.random.default_rng(np.random.SeedSequence(
        [permutation_seed, 0x5045524D]))
    reassigned = group_values.copy()
    for count in np.unique(counts):
        same_size = np.flatnonzero(counts == count)
        if len(same_size) > 1:
            reassigned[same_size] = group_values[
                rng_perm.permutation(same_size)]
    permuted = reassigned[inverse]
    if (permuted.shape != values.shape
            or not np.array_equal(np.sort(permuted), np.sort(values))
            or not np.isclose(float(permuted.mean()), -70.0,
                              atol=2e-6, rtol=0.0)):
        raise ValueError("invalid label-permuted inhibitory reversal control")
    return permuted


_TYPED_12B_VREST_IDS = {
    # R163 whole-BANC direct-motor inventory: these two T3 IN12B023 cells
    # contribute 930 direct synapses to RH motor neurons and none to LH.
    "rh_t3_direct": (
        "720575941461633220",
        "720575941576828753",
    ),
    # Equal-capacity identity control: the two T2 cells of the same BANC type.
    "t2_same_type_twin": (
        "720575941352944944",
        "720575941451665991",
    ),
}


def typed_12b_vrest_overrides(meta, mode, value_mv):
    """Return a two-cell, identity-checked IN12B023 intrinsic override.

    This parameter is deliberately tied to an anatomical cell type/segment,
    not applied as a universal neuronal constant.  Harris et al. 2015 found
    that adult 12B activation drives tonic T2/T3 extension and freezing; R164
    tests whether reducing the resting/reset depolarisation of the two T3
    cells most directly coupled to RH motor neurons relieves the RH bottleneck.
    """
    if mode not in _TYPED_12B_VREST_IDS:
        raise ValueError("unknown typed_12b_v_rest_mode")
    try:
        value_mv = float(value_mv)
    except (TypeError, ValueError) as error:
        raise ValueError("typed_12b_v_rest_mv must be numeric") from error
    if not np.isfinite(value_mv) or not -80.0 <= value_mv <= -45.0:
        raise ValueError("typed_12b_v_rest_mv must be within [-80, -45]")
    required = {"banc_888_id", "cell_type", "hemilineage",
                "neurotransmitter_predicted", "neuromere"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"typed 12B prior missing metadata: {sorted(missing)}")
    indexed = meta.assign(
        banc_888_id=meta["banc_888_id"].astype(str)).set_index("banc_888_id")
    ids = _TYPED_12B_VREST_IDS[mode]
    absent = [body_id for body_id in ids if body_id not in indexed.index]
    if absent:
        raise ValueError(f"typed 12B prior missing IDs: {absent}")
    rows = indexed.loc[list(ids)]
    expected_neuromere = "T3" if mode == "rh_t3_direct" else "T2"
    identity_ok = (
        rows["cell_type"].fillna("").astype(str).eq("IN12B023").all()
        and rows["hemilineage"].fillna("").astype(str).eq("12B").all()
        and rows["neurotransmitter_predicted"].fillna("").astype(str)
        .str.lower().eq("gaba").all()
        and rows["neuromere"].fillna("").astype(str)
        .eq(expected_neuromere).all())
    if not identity_ok:
        raise ValueError("typed 12B prior identity mismatch")
    return {body_id: value_mv for body_id in ids}


def typed_21a_inhibitory_gain_vector(meta, mode, gain):
    """Tie inhibitory presynaptic gain to 21A and one thoracic segment.

    Adult 21A activation drives femur-tibia flexion. R165 removes R141's
    body-wide 13A/21A contrast and asks whether its small 21A gain, restricted
    to the hind-leg T3 module, improves hind timing. T2 is the equal-size
    same-hemilineage control.
    """
    if mode not in ("t3_hind", "t2_middle_twin"):
        raise ValueError("unknown typed_21a_inhibitory_gain_mode")
    try:
        gain = float(gain)
    except (TypeError, ValueError) as error:
        raise ValueError("typed_21a_inhibitory_gain must be numeric") from error
    if not np.isfinite(gain) or not 0.25 <= gain <= 4.0:
        raise ValueError("typed_21a_inhibitory_gain must be within [0.25, 4]")
    required = {"hemilineage", "neuromere"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"typed 21A gain missing metadata: {sorted(missing)}")
    nt = cns._resolve_nt(meta, cns.SimParams())
    sign = cns._sign_of(nt, cns.SimParams())
    segment = "T3" if mode == "t3_hind" else "T2"
    selected = (
        (sign < 0)
        & meta["hemilineage"].fillna("").astype(str).eq("21A").to_numpy()
        & meta["neuromere"].fillna("").astype(str).eq(segment).to_numpy())
    rows = np.flatnonzero(selected)
    if len(rows) != 128:
        raise ValueError(f"typed 21A gain expected 128 rows, got {len(rows)}")
    vector = np.ones(len(meta), dtype=np.float64)
    vector[rows] = gain
    return vector, rows


_TYPED_19B_GAIN_TYPES = {
    "in19b012_structured": "IN19B012",
    "in19b003_twin": "IN19B003",
}


def typed_19b_excitatory_gain_vector(meta, mode, gain):
    """Tie presynaptic strength to one six-cell cholinergic 19B type.

    Lesser et al. 2024 place IN19B012 in a serial premotor set upstream of
    trochanter flexor and tibia extensor motor modules. IN19B003 is the
    equal-size, same-hemilineage/transmitter label twin. The numeric gain is a
    searched hypothesis; membership is anatomical.
    """
    if mode not in _TYPED_19B_GAIN_TYPES:
        raise ValueError("unknown typed_19b_excitatory_gain_mode")
    try:
        gain = float(gain)
    except (TypeError, ValueError) as error:
        raise ValueError("typed_19b_excitatory_gain must be numeric") from error
    if not np.isfinite(gain) or not 0.25 <= gain <= 4.0:
        raise ValueError("typed_19b_excitatory_gain must be within [0.25, 4]")
    required = {"banc_888_id", "cell_type", "hemilineage",
                "neurotransmitter_predicted"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"typed 19B gain missing metadata: {sorted(missing)}")
    cell_type = _TYPED_19B_GAIN_TYPES[mode]
    rows = meta[meta["cell_type"].fillna("").astype(str).eq(cell_type)]
    identity_ok = (len(rows) == 6
                   and rows["hemilineage"].fillna("").astype(str).eq("19B").all()
                   and rows["neurotransmitter_predicted"].fillna("")
                   .astype(str).str.lower().eq("acetylcholine").all())
    if not identity_ok:
        raise ValueError("typed 19B prior identity/count mismatch")
    values = np.ones(len(meta), dtype=np.float64)
    values[rows.index.to_numpy(dtype=int)] = gain
    return values, rows["banc_888_id"].astype(str).tolist()


def typed_21a_excitatory_gain_vector(meta, mode, gain):
    """One-cell IN21A010 presynaptic gain with a same-type segment twin.

    Adult 21A activation drives femur-tibia flexion, while the BANC right-T3
    IN21A010 cell supplies a highly RH-enriched direct motor input. The T2 arm
    holds type, side, transmitter, hemilineage, and parameter count fixed.
    Numeric gain remains a search hypothesis.
    """
    segments = {"right_t3": "T3", "right_t2_twin": "T2"}
    if mode not in segments:
        raise ValueError("unknown typed_21a_excitatory_gain_mode")
    try:
        gain = float(gain)
    except (TypeError, ValueError) as error:
        raise ValueError("typed_21a_excitatory_gain must be numeric") from error
    if not np.isfinite(gain) or not 0.25 <= gain <= 4.0:
        raise ValueError("typed_21a_excitatory_gain must be within [0.25, 4]")
    required = {"banc_888_id", "cell_type", "hemilineage", "side",
                "neuromere", "neurotransmitter_predicted"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(
            f"typed 21A excitatory gain missing metadata: {sorted(missing)}")
    rows = meta[
        meta["cell_type"].fillna("").astype(str).eq("IN21A010")
        & meta["hemilineage"].fillna("").astype(str).eq("21A")
        & meta["side"].fillna("").astype(str).eq("right")
        & meta["neuromere"].fillna("").astype(str).eq(segments[mode])
        & meta["neurotransmitter_predicted"].fillna("").astype(str)
        .str.lower().eq("acetylcholine")]
    if len(rows) != 1:
        raise ValueError(
            f"typed 21A excitatory gain expected one row, got {len(rows)}")
    values = np.ones(len(meta), dtype=np.float64)
    values[rows.index.to_numpy(dtype=int)] = gain
    return values, rows["banc_888_id"].astype(str).tolist()


def presyn_gain_by_id_vector(meta, mapping):
    """Per-neuron presynaptic gain from a theta dict of BANC id -> gain.

    2026-09-03 (judge, from lane E's R229/R230 nominee): a candidate body
    whose only change from the champion is the outgoing gain of a few named
    rows must be expressible in its theta of record, so the vector lane E
    built in its runner is built here from `presyn_gain_by_id`. Every id must
    be a BANC row and every gain finite and positive; unnamed rows stay 1.0.
    Composed multiplicatively with any typed or command presynaptic gain.
    Absent key = bit-identical (this function is not called)."""
    ids = meta["banc_888_id"].astype(str).to_numpy()
    pos = {v: i for i, v in enumerate(ids)}
    vec = np.ones(len(ids), dtype=np.float64)
    rows = []
    for k, g in mapping.items():
        if str(k) not in pos:
            raise ValueError(f"presyn_gain_by_id: {k} is not a BANC row")
        g = float(g)
        if not np.isfinite(g) or g <= 0.0:
            raise ValueError(f"presyn_gain_by_id: gain for {k} must be finite and positive")
        vec[pos[str(k)]] = g
        rows.append(pos[str(k)])
    return vec, rows


def cmd_presyn_gain_vector(meta, cmd_type, gain):
    """Scale outgoing synapses of one descending command type.

    Absent from theta = bit-identical. Used to test whether a quiet
    cord (lower bal_target) can still be driven if DNg100's own
    weights are stronger. Membership is BANC cell_type.
    """
    try:
        gain = float(gain)
    except (TypeError, ValueError) as error:
        raise ValueError("cmd_presyn_gain must be numeric") from error
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("cmd_presyn_gain must be finite and > 0")
    if "cell_type" not in meta.columns:
        raise ValueError("cmd_presyn_gain needs cell_type")
    if not cmd_type:
        raise ValueError("cmd_presyn_gain requires th['cmd']")
    rows = meta[meta["cell_type"].fillna("").astype(str).eq(str(cmd_type))]
    if len(rows) < 1:
        raise ValueError(
            f"cmd_presyn_gain found no rows for cmd {cmd_type!r}")
    values = np.ones(len(meta), dtype=np.float64)
    values[rows.index.to_numpy(dtype=int)] = gain
    ids = []
    if "banc_888_id" in meta.columns:
        ids = rows["banc_888_id"].astype(str).tolist()
    return values, ids


# scalar adapt_a from the theta (lane D, 2026-09-05; ported by the judge for
# the wb4 match). cns.py has carried SimParams.adapt_a since 2026-08; no
# theta could set it until this resolver. Absent = None = bit-identical.
def theta_adapt_a(th):
    """Resolve an explicit scalar subthreshold-adaptation theta value."""
    if "adapt_a" not in th:
        return None
    value = th["adapt_a"]
    if type(value) not in (int, float):
        raise ValueError("adapt_a must be finite in [0, 16]")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("adapt_a must be finite in [0, 16]") from error
    if not np.isfinite(value) or not 0.0 <= value <= 16.0:
        raise ValueError("adapt_a must be finite in [0, 16]")
    if th.get("adaptation_parameter_mode") not in (None, ""):
        raise ValueError(
            "scalar adapt_a cannot compose with adaptation_parameter_mode")
    return 0.0 if value == 0.0 else value


# Design 6d0042d5, Build 1: the per-row adaptation law. Form codes as the
# audit's per-row array stores them; -1 marks a row no group wrote.
ADAPT_LAW_FORMS = {"sym": 0, "rect_pos": 1, "rect_neg": 2}


def adapt_law_rows_vectors(th, meta, p):
    """Apply th["adapt_law_rows"] to p's adaptation law and return its audit.

    The key is a list of groups {rows, a, b_mv, tau_w_ms, form} with an
    optional except, applied in order over the law every other path left.
    rows is "all" or a list of banc_888_id and except removes ids from it.
    Per row the law is w <- w + (dt/tau_w)(a f(v - v_rest) - w), w += b at
    each spike, with f(x) = x for sym, max(x, 0) for rect_pos and min(x, 0)
    for rect_neg, written as adapt_a above rest and adapt_a_below below it
    ((a, a), (a, 0) and (0, a)). adapt_a, adapt_a_below, adapt_b and tau_w
    become float64 vectors, which the engines cast through the float32 path
    a scalar takes. Rows mn_membrane_by_id names keep its values: "all" skips
    them and a list naming one is refused. membrane_mode and
    adaptation_parameter_mode set every row and are refused; so is a group
    whose rows meet crossing_tau_w_mult's, adapt_zero_ids' or
    mn_adapt_b_zero_ids'."""
    groups = th["adapt_law_rows"]
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("adapt_law_rows must be a non-empty list of groups")
    for _clash in ("membrane_mode", "adaptation_parameter_mode"):
        if th.get(_clash) not in (None, ""):
            raise ValueError(
                f"adapt_law_rows cannot compose with {_clash}, which sets every row")
    ids = meta["banc_888_id"].astype(str).to_numpy()
    n = len(ids)
    pos = {b: i for i, b in enumerate(ids)}

    def _rows(where, lst):
        if not isinstance(lst, (list, tuple)) or not lst:
            raise ValueError(f"adapt_law_rows{where} must be \"all\" or a non-empty id list")
        out = []
        for b in lst:
            i = pos.get(str(b))
            if i is None:
                raise ValueError(f"adapt_law_rows{where} id {b} is not a row of meta")
            out.append(i)
        if len(set(out)) != len(out):
            raise ValueError(f"adapt_law_rows{where} names a row twice")
        return np.asarray(out, dtype=np.int64)

    kept = np.zeros(n, dtype=bool)
    if th.get("mn_membrane_by_id") is not None:
        kept[[pos[str(b)] for b in th["mn_membrane_by_id"]]] = True
    guarded = {}
    if th.get("crossing_tau_w_mult") is not None:
        guarded["crossing_tau_w_mult"] = [str(x) for x in th["crossing_tau_w_mult"]["rows"]]
    for _k in ("adapt_zero_ids", "mn_adapt_b_zero_ids"):
        if th.get(_k) is not None:
            guarded[_k] = [str(x) for x in th[_k]]
    vec = lambda x: np.broadcast_to(  # noqa: E731
        np.asarray(x, dtype=np.float64), (n,)).copy()
    old = {"adapt_a": vec(p.adapt_a), "adapt_b": vec(p.adapt_b), "tau_w": vec(p.tau_w)}
    old["adapt_a_below"] = (old["adapt_a"].copy()
                            if getattr(p, "adapt_a_below", None) is None
                            else vec(p.adapt_a_below))
    new = {k: v.copy() for k, v in old.items()}
    form = np.full(n, -1, dtype=np.int8)
    written = np.zeros(n, dtype=bool)
    # Design 0af6d1c5 Build 1: a's bound widened from 16.0 to 32.0 for L1's laws
    # (a 18.5 to 22); the bound only refuses, so a law inside 16 is unchanged.
    # Design fbcd0f61 B2: tau_w_ms's ceiling raised from 5,000 to 100,000 ms for the LHN1
    # law's grid (to 20,000 ms) and knockdown (60,000 ms); the bound only refuses.
    fields = (("a", 0.0, 32.0), ("b_mv", 0.0, 20.0), ("tau_w_ms", 1.0, 100000.0))
    audit_groups = []
    for g, spec in enumerate(groups):
        if (not isinstance(spec, dict)
                or not {"rows", "a", "b_mv", "tau_w_ms", "form"} <= set(spec)
                or not set(spec) <= {"rows", "except", "a", "b_mv", "tau_w_ms", "form"}):
            raise ValueError(
                f"adapt_law_rows[{g}] must carry rows, a, b_mv, tau_w_ms and form, "
                "and may carry except")
        f = spec["form"]
        if f not in ADAPT_LAW_FORMS:
            raise ValueError(
                f"adapt_law_rows[{g}] form must be one of {sorted(ADAPT_LAW_FORMS)}, not {f!r}")
        vals = {}
        for _key, _lo, _hi in fields:
            _x = spec[_key]
            if (type(_x) not in (int, float) or not np.isfinite(float(_x))
                    or not _lo <= float(_x) <= _hi):
                raise ValueError(
                    f"adapt_law_rows[{g}][{_key}] must be finite and in [{_lo}, {_hi}], "
                    f"not {_x!r}")
            vals[_key] = float(_x)
        if spec["rows"] == "all":
            mask = ~kept
        else:
            _r = _rows(f"[{g}] rows", spec["rows"])
            if kept[_r].any():
                raise ValueError(
                    f"adapt_law_rows[{g}] names {int(kept[_r].sum())} rows "
                    "mn_membrane_by_id sets, which keep its values")
            mask = np.zeros(n, dtype=bool)
            mask[_r] = True
        if "except" in spec:
            mask[_rows(f"[{g}] except", spec["except"])] = False
        if not mask.any():
            raise ValueError(f"adapt_law_rows[{g}] selects no row")
        for _k, _lst in guarded.items():
            _hit = [b for b in _lst if b in pos and mask[pos[b]]]
            if _hit:
                raise ValueError(
                    f"adapt_law_rows[{g}] meets {_k} on {len(_hit)} rows, first {_hit[0]}")
        _a = vals["a"]
        above, below = {"sym": (_a, _a), "rect_pos": (_a, 0.0), "rect_neg": (0.0, _a)}[f]
        new["adapt_a"][mask] = above
        new["adapt_a_below"][mask] = below
        new["adapt_b"][mask] = vals["b_mv"]
        new["tau_w"][mask] = vals["tau_w_ms"]
        form[mask] = ADAPT_LAW_FORMS[f]
        written |= mask
        audit_groups.append({
            "form": f, **vals,
            "rows": "all" if spec["rows"] == "all" else len(spec["rows"]),
            "except": len(spec.get("except", ())),
            "rows_written": int(mask.sum()),
            "rows_skipped_mn_membrane_by_id": int(kept.sum()) if spec["rows"] == "all" else 0})
    p.adapt_a = new["adapt_a"]
    p.adapt_a_below = new["adapt_a_below"]
    p.adapt_b = new["adapt_b"]
    p.tau_w = new["tau_w"]
    on_old = np.ones(n, dtype=bool)
    for _k in new:
        on_old &= new[_k] == old[_k]
    arrays = {**new, "form": form}
    return {
        "groups": audit_groups, "rows": n, "rows_written": int(written.sum()),
        "rows_on_old_law": int(on_old.sum()),
        "rows_kept_by_mn_membrane_by_id": int(kept.sum()),
        "form_rows": {_f: int(np.count_nonzero(form == _c))
                      for _f, _c in ADAPT_LAW_FORMS.items()},
        "arrays": arrays,
        "sha256": {_k: hashlib.sha256(np.ascontiguousarray(_v).tobytes()).hexdigest()
                   for _k, _v in arrays.items()},
    }



def _neural_physics_receipt(net):
    """Read-only receipt for explicitly enabled neural-fidelity mechanisms."""
    delay = getattr(net.p, "delay_per_neuron", None)
    structure_active = getattr(net.p, "structure_active_mask", None)
    threshold = np.broadcast_to(
        np.asarray(net.p.v_threshold, dtype=np.float64), (net.N,))
    delay_array = (None if delay is None
                   else np.broadcast_to(np.asarray(delay, dtype=np.float64),
                                        (net.N,)))
    active = (None if structure_active is None else np.broadcast_to(
        np.asarray(structure_active, dtype=bool), (net.N,)))
    tau_mem = np.broadcast_to(
        np.asarray(getattr(net.p, "tau_mem", 20.0), dtype=np.float64),
        (net.N,))
    v_rest = np.broadcast_to(
        np.asarray(getattr(net.p, "v_rest", -52.0), dtype=np.float64),
        (net.N,)).copy()
    if net.p.v_rest_override:
        row_by_id = pd.Series(np.arange(net.N), index=net.ids)
        for identifier, value in net.p.v_rest_override.items():
            row = row_by_id.get(identifier)
            if row is not None and not np.isnan(row):
                v_rest[int(row)] = float(value)
    v_reset = np.broadcast_to(
        np.asarray(getattr(net.p, "v_reset", -52.0), dtype=np.float64),
        (net.N,))
    t_refrac = np.broadcast_to(
        np.asarray(getattr(net.p, "t_refrac", 2.2), dtype=np.float64),
        (net.N,))
    adapt_b = np.broadcast_to(
        np.asarray(getattr(net.p, "adapt_b", 0.0), dtype=np.float64),
        (net.N,))
    adapt_a = np.broadcast_to(
        np.asarray(getattr(net.p, "adapt_a", 0.0), dtype=np.float64),
        (net.N,))
    tau_w = np.broadcast_to(
        np.asarray(getattr(net.p, "tau_w", 100.0), dtype=np.float64),
        (net.N,))
    e_inh = np.broadcast_to(
        np.asarray(getattr(net.p, "e_inh", -70.0), dtype=np.float64),
        (net.N,))
    tau_syn = np.broadcast_to(
        np.asarray(getattr(net.p, "tau_syn", 5.0), dtype=np.float64),
        (net.N,))
    serotonin_source = getattr(net.p, "serotonin_source_rows", None)
    serotonin_target = getattr(net.p, "serotonin_target_rows", None)
    presynaptic_inhibition = getattr(
        net.p, "presynaptic_inhibition_rows", None)
    spike_reg_vector = getattr(net.p, "spike_reg_per_neuron", None)
    graded_idx = np.asarray(getattr(
        net, "graded_idx", np.empty(0, dtype=np.int64)), dtype=np.int64)
    graded_v50 = np.broadcast_to(
        np.asarray(getattr(net.p, "graded_v50", -48.0), dtype=np.float64),
        (net.N,))
    graded_slope = np.broadcast_to(
        np.asarray(getattr(net.p, "graded_slope", 2.5), dtype=np.float64),
        (net.N,))
    if graded_idx.size:
        rest_release = 1.0 / (1.0 + np.exp(
            -(v_rest[graded_idx] - graded_v50[graded_idx])
            / graded_slope[graded_idx]))
        threshold_release = 1.0 / (1.0 + np.exp(
            -(threshold[graded_idx] - graded_v50[graded_idx])
            / graded_slope[graded_idx]))
    else:
        rest_release = threshold_release = np.empty(0, dtype=np.float64)
    return {
        "neuron_rows": int(net.N),
        "graded_rows": int(len(getattr(net, "graded_idx", ()))),
        "graded_cells_thresholdless": True,
        "graded_parameter_mode": str(getattr(
            net.p, "graded_parameter_mode", "scalar")),
        "graded_hemilineage_permutation_seed": getattr(
            net.p, "graded_hemilineage_permutation_seed", None),
        "graded_delay_delivery": str(getattr(
            net.p, "graded_delay_delivery", "bucket")),
        "graded_v50_unique_values": int(np.unique(
            graded_v50[graded_idx]).size) if graded_idx.size else 0,
        "graded_slope_unique_values": int(np.unique(
            graded_slope[graded_idx]).size) if graded_idx.size else 0,
        "graded_rest_release_min": (None if not graded_idx.size
                                    else float(rest_release.min())),
        "graded_rest_release_max": (None if not graded_idx.size
                                    else float(rest_release.max())),
        "graded_threshold_release_min": (
            None if not graded_idx.size else float(threshold_release.min())),
        "graded_threshold_release_max": (
            None if not graded_idx.size else float(threshold_release.max())),
        "graded_edge_equilibrium": dict(getattr(
            net, "graded_edge_equilibrium_audit", {
                "mode": "off", "requested_pair_count": 0,
                "selected_raw_row_count": 0, "selected_total_count": 0,
                "targets": [], "pairs": [],
            })),
        "conductance_based": bool(net.p.conductance_based),
        "split_conductances": bool(net.p.split_conductances),
        "conductance_integrator": str(getattr(
            net.p, "conductance_integrator", "euler")),
        "inhibitory_reversal_mode": str(getattr(
            net.p, "inhibitory_reversal_mode", "scalar")),
        "inhibitory_reversal_group_count": int(getattr(
            net.p, "inhibitory_reversal_group_count", 0)),
        "split_synaptic_decay_mode": str(getattr(
            net.p, "split_synaptic_decay_mode", "scalar")),
        "tau_syn_exc_ms": (None if getattr(net.p, "tau_syn_exc", None) is None
                           else float(net.p.tau_syn_exc)),
        "tau_syn_inh_ms": (None if getattr(net.p, "tau_syn_inh", None) is None
                           else float(net.p.tau_syn_inh)),
        "premotor_inh_parameter_mode": str(getattr(
            net.p, "premotor_inh_parameter_mode", "scalar")),
        "premotor_inh_13a_21a_ratio": getattr(
            net.p, "premotor_inh_13a_21a_ratio", None),
        "e_inh_mean_mv": float(e_inh.mean()),
        "e_inh_min_mv": float(e_inh.min()),
        "e_inh_max_mv": float(e_inh.max()),
        "e_inh_unique_values": int(np.unique(e_inh).size),
        "balance_target": float(net.p.balance_target),
        "membrane_mode": str(getattr(net.p, "membrane_mode", "scalar")),
        "membrane_seed": getattr(net.p, "membrane_seed", None),
        "membrane_class_count": int(getattr(
            net.p, "membrane_class_count", 0)),
        "tau_mem_mean_ms": float(tau_mem.mean()),
        "tau_mem_unique_values": int(np.unique(tau_mem).size),
        "v_rest_mean_mv": float(v_rest.mean()),
        "v_rest_unique_values": int(np.unique(v_rest).size),
        "graded_premotor_v_rest_mode": str(getattr(
            net.p, "graded_premotor_v_rest_mode", "absent")),
        "graded_premotor_v_rest_population": str(getattr(
            net.p, "graded_premotor_v_rest_population", "absent")),
        "graded_premotor_v_rest_rows": int(getattr(
            net.p, "graded_premotor_v_rest_rows", 0)),
        "graded_premotor_v_rest_mv": getattr(
            net.p, "graded_premotor_v_rest_mv", None),
        "graded_premotor_v_rest_values_mv": getattr(
            net.p, "graded_premotor_v_rest_values_mv", None),
        "graded_premotor_v_rest_unique_values": int(getattr(
            net.p, "graded_premotor_v_rest_unique_values", 0)),
        "typed_12b_v_rest_mode": str(getattr(
            net.p, "typed_12b_v_rest_mode", "absent")),
        "typed_12b_v_rest_rows": int(getattr(
            net.p, "typed_12b_v_rest_rows", 0)),
        "typed_12b_v_rest_mv": getattr(
            net.p, "typed_12b_v_rest_mv", None),
        "typed_21a_inhibitory_gain_mode": str(getattr(
            net.p, "typed_21a_inhibitory_gain_mode", "absent")),
        "typed_21a_inhibitory_gain_rows": int(getattr(
            net.p, "typed_21a_inhibitory_gain_rows", 0)),
        "typed_21a_inhibitory_gain": getattr(
            net.p, "typed_21a_inhibitory_gain", None),
        "typed_19b_excitatory_gain_mode": str(getattr(
            net.p, "typed_19b_excitatory_gain_mode", "absent")),
        "typed_19b_excitatory_gain_rows": int(getattr(
            net.p, "typed_19b_excitatory_gain_rows", 0)),
        "typed_19b_excitatory_gain_ids": list(getattr(
            net.p, "typed_19b_excitatory_gain_ids", [])),
        "typed_19b_excitatory_gain": getattr(
            net.p, "typed_19b_excitatory_gain", None),
        "typed_21a_excitatory_gain_mode": str(getattr(
            net.p, "typed_21a_excitatory_gain_mode", "absent")),
        "typed_21a_excitatory_gain_rows": int(getattr(
            net.p, "typed_21a_excitatory_gain_rows", 0)),
        "typed_21a_excitatory_gain_ids": list(getattr(
            net.p, "typed_21a_excitatory_gain_ids", [])),
        "typed_21a_excitatory_gain": getattr(
            net.p, "typed_21a_excitatory_gain", None),
        "v_reset_unique_values": int(np.unique(v_reset).size),
        "t_refrac_geomean_ms": float(np.exp(np.mean(np.log(t_refrac)))),
        "t_refrac_unique_values": int(np.unique(t_refrac).size),
        "adaptation_parameter_mode": str(getattr(
            net.p, "adaptation_parameter_mode", "scalar")),
        "adapt_a_mean_dimensionless": float(adapt_a.mean()),
        "adapt_a_unique_values": int(np.unique(adapt_a).size),
        "adapt_b_mean_mv": float(adapt_b.mean()),
        "adapt_b_unique_values": int(np.unique(adapt_b).size),
        "tau_w_geomean_ms": float(np.exp(np.mean(np.log(tau_w)))),
        "tau_w_unique_values": int(np.unique(tau_w).size),
        "synaptic_decay_mode": str(getattr(
            net.p, "synaptic_decay_mode", "scalar")),
        "synaptic_decay_seed": getattr(net.p, "synaptic_decay_seed", None),
        "synaptic_decay_class_count": int(getattr(
            net.p, "synaptic_decay_class_count", 0)),
        "tau_syn_geomean_ms": float(np.exp(np.mean(np.log(tau_syn)))),
        "tau_syn_min_ms": float(tau_syn.min()),
        "tau_syn_max_ms": float(tau_syn.max()),
        "tau_syn_unique_values": int(np.unique(tau_syn).size),
        "synaptic_kernel_area_normalized": bool(getattr(
            net.p, "syn_normalised", False)),
        "serotonin_5ht7_08b_configured": bool(
            serotonin_source is not None and serotonin_target is not None),
        "serotonin_source_rows": (0 if serotonin_source is None
                                   else int(len(serotonin_source))),
        "serotonin_target_rows": (0 if serotonin_target is None
                                   else int(len(serotonin_target))),
        "serotonin_tau_ms": (None if serotonin_source is None else float(
            net.p.serotonin_tau_ms)),
        "serotonin_gain_mv_per_hz": (
            None if serotonin_target is None else float(
                net.p.serotonin_gain_mv_per_hz)),
        "presynaptic_inhibition_rows": (
            0 if presynaptic_inhibition is None
            else int(len(presynaptic_inhibition))),
        "spike_reg_parameter_mode": (
            "per_neuron" if spike_reg_vector is not None else "scalar"),
        "spike_reg_unique_values": (
            1 if spike_reg_vector is None
            else int(np.unique(np.asarray(spike_reg_vector)).size)),
        "spike_reg_poisson_rows": (
            0 if spike_reg_vector is None else int(
                (np.asarray(spike_reg_vector) <= 1.0).sum())),
        "spike_reg_renewal_rows": (
            0 if spike_reg_vector is None else int(
                (np.asarray(spike_reg_vector) > 1.0).sum())),
        "threshold_margin_min_mv": float(np.min(threshold - v_rest)),
        "threshold_margin_max_mv": float(np.max(threshold - v_rest)),
        "delay_per_neuron": delay_array is not None,
        "delay_mean_ms": (None if delay_array is None
                          else float(delay_array.mean())),
        "delay_unique_values": (0 if delay_array is None
                                else int(np.unique(delay_array).size)),
        "structure_active_rows": (0 if active is None else int(active.sum())),
        "delay_active_mean_ms": (
            None if delay_array is None or active is None or not active.any()
            else float(delay_array[active].mean())),
        "delay_active_unique_values": (
            0 if delay_array is None or active is None or not active.any()
            else int(np.unique(delay_array[active]).size)),
        "threshold_mean_mv": float(threshold.mean()),
        "threshold_unique_values": int(np.unique(threshold).size),
        "threshold_active_mean_mv": (
            None if active is None or not active.any()
            else float(threshold[active].mean())),
        "threshold_active_unique_values": (
            0 if active is None or not active.any()
            else int(np.unique(threshold[active]).size)),
        "edges_dropped_missing_endpoint": int(net.n_edges_dropped),
        "unknown_transmitter_rows": int(net.n_unknown_nt),
        "gap_pairs": int(getattr(net, "n_gap_pairs", 0)),
        "gap_pair_weight_unique_values": int(np.unique(getattr(
            net, "gap_pair_weights", np.empty(0))).size),
        "gap_pair_weight_min": (None if not len(getattr(
            net, "gap_pair_weights", ())) else float(
                np.min(net.gap_pair_weights))),
        "gap_pair_weight_max": (None if not len(getattr(
            net, "gap_pair_weights", ())) else float(
                np.max(net.gap_pair_weights))),
        "feco_13b_analog_mode": str(getattr(
            net.p, "feco_13b_analog_mode", "absent")),
        "feco_13b_analog_edge_pairs": int(getattr(
            net.p, "feco_13b_analog_edge_pairs", 0)),
        "feco_13b_analog_chemical_synapses_replaced": int(getattr(
            net.p, "feco_13b_analog_chemical_synapses_replaced", 0)),
        "feco_13b_analog_target_rows": int(getattr(
            net.p, "feco_13b_analog_target_rows", 0)),
        "graded_13b_gain": (None if not hasattr(net.p, "graded_13b_gain")
                            else float(net.p.graded_13b_gain)),
        "graded_13b_gain_rows": int(getattr(
            net.p, "graded_13b_gain_rows", 0)),
        "graded_13b_gain_mode": str(getattr(
            net.p, "graded_13b_gain_mode", "absent")),
    }


def _graded_drive_clamp_receipt(
        net, rows, command_min_by_row, command_max_by_row,
        saturation_samples_by_row):
    """Describe the exact externally clamped membrane boundary for graded rows."""
    rows = np.asarray(rows, dtype=np.int64)
    command_min = np.asarray(command_min_by_row, dtype=np.float64)
    command_max = np.asarray(command_max_by_row, dtype=np.float64)
    saturation = np.asarray(saturation_samples_by_row, dtype=np.int64)
    if (rows.ndim != 1 or command_min.shape != rows.shape
            or command_max.shape != rows.shape or saturation.shape != rows.shape
            or np.any(rows < 0) or np.any(rows >= net.N)):
        raise ValueError("invalid graded clamp receipt rows")

    rest = np.broadcast_to(
        np.asarray(net.p.v_rest, dtype=np.float64), (net.N,)).copy()
    if net.p.v_rest_override:
        row_by_id = pd.Series(np.arange(net.N), index=net.ids)
        for identifier, value in net.p.v_rest_override.items():
            row = row_by_id.get(identifier)
            if row is not None and not np.isnan(row):
                rest[int(row)] = float(value)
    use_exp = float(getattr(net.p, "delta_T", 0.0)) > 0.0
    upper_name = "v_peak" if use_exp else "v_threshold"
    upper = np.broadcast_to(np.asarray(
        getattr(net.p, upper_name), dtype=np.float64), (net.N,))
    if np.any(upper[rows] <= rest[rows]):
        raise ValueError("graded clamp upper anchor must exceed rest")
    lower = rest - (upper - rest)
    sampled = np.isfinite(command_min) & np.isfinite(command_max)
    voltage_min = np.full(len(rows), np.nan, dtype=np.float64)
    voltage_max = np.full(len(rows), np.nan, dtype=np.float64)
    voltage_min[sampled] = (
        rest[rows[sampled]] + command_min[sampled]
        * (upper[rows[sampled]] - rest[rows[sampled]]))
    voltage_max[sampled] = (
        rest[rows[sampled]] + command_max[sampled]
        * (upper[rows[sampled]] - rest[rows[sampled]]))
    return {
        "external_voltage_clamp": True,
        "command_units": "dimensionless_legacy_vision_hz_normalization",
        "clamp_law": "v_rest_i + command_i * (upper_i - v_rest_i)",
        "negative_command_meaning": "hyperpolarization_not_negative_firing_rate",
        "upper_anchor": upper_name,
        "added_threshold": False,
        "existing_upper_anchor_unique_values": int(
            np.unique(upper[rows]).size),
        "input_ids": [str(net.ids[row]) for row in rows],
        "lower_envelope_mv_by_input_row": lower[rows].astype(float).tolist(),
        "rest_mv_by_input_row": rest[rows].astype(float).tolist(),
        "upper_envelope_mv_by_input_row": upper[rows].astype(float).tolist(),
        "actual_command_min_by_input_row": [
            None if not sampled[i] else float(command_min[i])
            for i in range(len(rows))],
        "actual_command_max_by_input_row": [
            None if not sampled[i] else float(command_max[i])
            for i in range(len(rows))],
        "actual_voltage_min_mv_by_input_row": [
            None if not sampled[i] else float(voltage_min[i])
            for i in range(len(rows))],
        "actual_voltage_max_mv_by_input_row": [
            None if not sampled[i] else float(voltage_max[i])
            for i in range(len(rows))],
        "saturation_samples_by_input_row": saturation.astype(int).tolist(),
    }


_SIZE_FACTORS_CACHE = {}


def afferent_regularity_vector(meta, sensory_k):
    """Apply the chordotonal renewal prior only to sensory neurons."""
    k = float(sensory_k)
    if not np.isfinite(k) or k <= 1.0:
        raise ValueError("afferent sensory renewal shape must be > 1")
    out = np.ones(len(meta), dtype=np.float64)
    # `sensory_ascending` rows are primary sensory axons that continue into
    # the brain, not central ascending neurons; keep the afferent prior there.
    sensory = meta["super_class"].fillna("").astype(str).str.startswith(
        "sensory")
    out[sensory.to_numpy()] = k
    return out


def _frame_fingerprint(meta, edges):
    """Cheap, content-sensitive key for memoising pure functions of the two
    frames: the addresses of the immutable Arrow (or numpy) buffers behind
    edges["post"], edges["count"] and meta["banc_888_id"], their lengths,
    and the count column's sum. A shuffle null assigns a new post column
    (new buffer), a cut builds new frames, a different meta is a different
    buffer, so all of those miss; repeated calls on unchanged frames hit in
    microseconds. The cache entry keeps references to the frames so no
    address can be recycled while its key is live."""
    parts = [len(meta), len(edges), int(edges["count"].to_numpy().sum())]
    for col, frame in (("post", edges), ("count", edges), ("banc_888_id", meta)):
        ea = frame[col].array
        arr = getattr(ea, "_pa_array", None)
        if arr is not None:
            for chunk in arr.chunks:
                parts.extend(b.address if b is not None else 0 for b in chunk.buffers())
        else:
            parts.append(frame[col].to_numpy().__array_interface__["data"][0])
    return tuple(parts)


def _size_pool_sha(mapping):
    """The key a size_pools mapping is cached under (design 171d6f1a B2): the
    sha256 of its live rows' (id, muscle, leg) in row order, everything the
    rank reads from a mapping (row order breaks ties in input synapses)."""
    live = mapping[~mapping["hypothesised"]].drop_duplicates("banc_888_id")
    rows = list(zip(live["banc_888_id"].astype(str), live["muscle"].astype(str), live["leg"].astype(str)))
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _size_factors_entry(meta, edges, pools):
    key = _frame_fingerprint(meta, edges)
    if pools is not None:
        key = key + ("size_pools", _size_pool_sha(pools))
    hit = _SIZE_FACTORS_CACHE.get(key)
    if hit is None:
        ranks = {}
        hit = (_size_factors_uncached(meta, edges, mapping=pools, ranks=ranks), (meta, edges), ranks)
        if len(_SIZE_FACTORS_CACHE) >= 4:
            _SIZE_FACTORS_CACHE.pop(next(iter(_SIZE_FACTORS_CACHE)))
        _SIZE_FACTORS_CACHE[key] = hit
    return hit


def size_factors(meta, edges, *, pools=None):
    """Memoised on the frames' content: the groupby over all 13.6M edges cost
    0.75 s and evaluate() calls this up to three times per seed with the
    same frames (compute seat, 2026-09-02). Returns fresh dict copies.

    pools (design 171d6f1a B2, th["size_pools"] "transferred"): a mapping
    whose live rows set the (muscle, leg) pools in place of BANC's labels,
    cached apart under _size_pool_sha. None is BANC's pools, as before."""
    return tuple(dict(d) for d in _size_factors_entry(meta, edges, pools)[0])


def size_pool_ranks(meta, edges, *, pools=None):
    """Each pooled cell's (muscle, leg, rank, pool size) under size_factors'
    rule for the same pools; rank 0 is the most input synapses."""
    return dict(_size_factors_entry(meta, edges, pools)[2])


def _size_pool_receipt(meta, edges, rule, transferred):
    """LAST_EXTRAS["size_pools"] (design 171d6f1a B2). For each moved cell and
    each other cell of a pool whose members differ between BANC's labels and
    the transferred mapping: its pool, rank, pool size, gain (the 770x value,
    before size_span), input-resistance factor and rest under both rules. A
    rest of None is no override, so the default applies (singletons); a side
    of None is a cell in no pool under that rule."""
    tables, members = {}, {}
    for name, kw in (("banc", {}), ("transferred", {"pools": transferred})):
        tables[name] = (size_pool_ranks(meta, edges, **kw), *size_factors(meta, edges, **kw))
        members[name] = {}
        for i, (m, l, _k, _n) in tables[name][0].items():
            members[name].setdefault((m, l), set()).add(i)
    changed = {p for p in set(members["banc"]) | set(members["transferred"])
               if members["banc"].get(p) != members["transferred"].get(p)}
    ids = {i for name in members for p in changed for i in members[name].get(p, ())}

    def side(name, i):
        ranks, g, r, v = tables[name]
        if i not in ranks:
            return None
        m, l, k, n = ranks[i]
        return {"pool": [str(m), str(l)], "rank": int(k), "pool_size": int(n),
                "gain": float(g[i]), "factor": float(r[i]), "rest_mV": v.get(i)}
    cells = [{"id": str(i), "banc": side("banc", i), "transferred": side("transferred", i)}
             for i in sorted(ids, key=str)]
    blob = json.dumps(cells, sort_keys=True, separators=(",", ":")).encode()
    return {"rule": rule, "cells": cells, "sha256": hashlib.sha256(blob).hexdigest()}


def _size_factors_uncached(meta, edges, *, mapping=None, ranks=None):
    """The measured MN size principle (Azevedo 2020; recipe MUSCLE-SPLIT.md):
    within each muscle pool, rank MNs by total input synapse count (the
    r=0.94 anatomical-size proxy, Lesser 2024). Per-spike force gains
    log-spaced over the measured ~770x span (10 / 1 / 0.013 uN); input-
    resistance factors log-spaced over the measured 150-700 MOhm (largest
    MN 0.5x, smallest 2.33x -- geometric-mean-neutral). Singleton pools
    stay neutral: no gradient information exists for them.
    Returns (gain, r_factor) dicts keyed by banc id.

    mapping (design 171d6f1a B2): the pools' mapping, None = MM.build_map(meta).
    ranks, when given, receives each pooled cell's (muscle, leg, rank, n)."""
    import motormap as MM
    mapping = MM.build_map(meta) if mapping is None else mapping
    base = mapping[~mapping["hypothesised"]].drop_duplicates("banc_888_id")
    insyn = edges.groupby("post")["count"].sum()
    gains, rfac, vrest = {}, {}, {}
    for pool_key, pool in base.groupby(["muscle", "leg"]):
        ids = sorted(pool["banc_888_id"], key=lambda i: -insyn.get(i, 0))
        n = len(ids)
        for k, i in enumerate(ids):
            if ranks is not None:
                ranks[i] = (*pool_key, k, n)
            if n == 1:
                gains[i], rfac[i] = 1.0, 1.0
            else:
                f = k / (n - 1)               # 0 = largest/fast, 1 = smallest/slow
                gains[i] = 770.0 ** (-f)
                rfac[i] = 0.5 * (700.0 / 150.0) ** f
                # measured resting-Vm gradient: fast -68, slow -48 mV
                vrest[i] = -68.0 + 20.0 * f
    return gains, rfac, vrest


# Design adcf11c7 (study 1e3d790f): th["r_in_law"] names one input-resistance
# law, its coefficient and gm = g x m. Absent = untouched = bit-identical.
_R_IN_LAW_CACHE = {}
_R_IN_LAW_WSS = 1.0113670590845238       # wm1's w_syn_scale: gm is taken over it
_R_IN_LAW_GF = ("720575941509145950", "720575941451068597")   # giant fibre, fixed
_R_IN_LAW_FORMS = {"count_power": ("exponent", "clip", "gm"),
                   "conductance": ("c", "gm"),
                   "cable_power": ("exponent", "clip", "gm")}
# Design d033eefd B1 (study 5d547f05): three optional keys on every form, each
# absent = untouched = bit-identical. hold names a rule-derived row set kept at
# the build body's factor, f_mult multiplies f on free cord and free descending
# rows, base_wss replaces wm1's w_syn_scale in the installer's gm check.
_R_IN_LAW_OPTIONAL = ("hold", "f_mult", "base_wss", "fragment_rule")
# Design 80b3da76 B6 (study 3ab5c294): BC and BO, hold B plus every row whose
# region is central_brain, or optic_lobe.
_R_IN_LAW_HOLDS = ("B", "B3", "PD", "PDm", "P", "all", "BC", "BO")


def r_in_law_hold_mask(meta, edges, name):
    """Design d033eefd B1: the hold set `name` as a boolean mask in meta order,
    by the rules of the design's hold_ids.py (blob fe181296) on the frames
    given. brain is region central_brain or optic_lobe (a missing region is
    cord); leg MN is super_class motor with a leg effector; fixed is the leg
    MNs and the two giant fibres; P is every non-brain row with a synapse onto
    a leg MN from a row that is not one, plus fixed. B = ~brain | descending |
    fixed (33,372 rows on BANC 888); B3 = ~brain | fixed (32,058); PD = P |
    descending (11,815); PDm = PD | non-brain motor (12,097); P (10,501); all,
    every row (188,508). Design 80b3da76 B6: BC = B | region central_brain
    (79,744) and BO = B | region optic_lobe (142,136)."""
    if name not in _R_IN_LAW_HOLDS:
        raise ValueError(f"r_in_law hold set must be one of {_R_IN_LAW_HOLDS}, not {name!r}")
    ids = meta["banc_888_id"].astype(np.int64).to_numpy()
    N = len(ids)
    if name == "all":
        return np.ones(N, dtype=bool)
    sc = meta["super_class"].fillna("").astype(str).to_numpy()
    brain = np.isin(meta["region"].fillna("none").astype(str).to_numpy(), ["central_brain", "optic_lobe"])
    eff = meta["body_part_effector"].fillna("").astype(str).to_numpy()
    legmn = (sc == "motor") & np.isin(eff, ["front_leg", "middle_leg", "hind_leg"])
    fixed = legmn | np.isin(ids, [int(b) for b in _R_IN_LAW_GF])
    dn = sc == "descending"
    if name == "B":
        return ~brain | dn | fixed
    if name in ("BC", "BO"):
        reg = "central_brain" if name == "BC" else "optic_lobe"
        return ~brain | dn | fixed | (meta["region"].fillna("none").astype(str).to_numpy() == reg)
    if name == "B3":
        return ~brain | fixed
    pos = pd.Series(np.arange(N), index=ids)
    pre = pos.reindex(edges["pre"].to_numpy().astype(np.int64)).to_numpy()
    post = pos.reindex(edges["post"].to_numpy().astype(np.int64)).to_numpy()
    keep = ~(np.isnan(pre) | np.isnan(post))
    pre, post = pre[keep].astype(np.int64), post[keep].astype(np.int64)
    cnt = edges["count"].to_numpy()[keep].astype(float)
    s_mn = legmn[post] & ~legmn[pre]
    syn_mn = np.bincount(pre[s_mn], weights=cnt[s_mn], minlength=N)
    P = (~brain & (syn_mn > 0)) | fixed
    if name == "P":
        return P
    if name == "PD":
        return P | dn
    return P | dn | (~brain & (sc == "motor"))


def r_in_law_idhash(meta, mask):
    """gpu/rin_law/plan.py's idhash: sha256 of the sorted decimal ids of the
    masked rows joined by newlines."""
    ids = meta["banc_888_id"].astype(np.int64).to_numpy()
    sel = sorted(int(i) for i in ids[np.asarray(mask, dtype=bool)])
    return hashlib.sha256("\n".join(str(i) for i in sel).encode()).hexdigest()


def _r_in_law_rfac_map(spec):
    """mn_rfac_by_id composes with r_in_law as any non-empty map of finite
    factors above 0 (design 80b3da76 B1, lifting d033eefd B2's every-value-1.0
    guard); r_in_law_override refuses an id that is not a size-rule row. A
    size-rule row is fixed under every hold set (fixed = leg MN | giant fibre,
    and a hold writes no fixed row), so the law writes it rfac/gm, the install
    check "size-rule row moved" verifies float32(rfac/gm) per row, and its
    absolute weight is the body's base_wss x r_scale x rfac: no hold frees one
    today, and an arm of a later design that frees such a row does not apply
    the law to it."""
    return (isinstance(spec, dict) and bool(spec)
            and all(type(v) in (int, float) and np.isfinite(float(v)) and float(v) > 0.0
                    for v in spec.values()))


def r_in_law_override(meta, edges, spec, r_in_alpha=0.10, r_in_max=8.0, *, pools=None,
                      rfac_by_id=None, gf_total_by_id=None):
    """The override r_in_law merges into rfac before p.r_in_override.

    n is the engine's own input synapse count per row (cns.CNS: the bincount
    of the kept edges' counts behind r_scale), rs the engine's r_scale
    clip((med/n)^alpha, 1/r_in_max, r_in_max) in float32, med its median over
    rows with input (110 on BANC 888). The law's factor f is
      count_power  clip((med/n)^exponent, lo, hi), f = 1 where n = 0
      conductance  (med + c)/(n + c)
      cable_power  clip((L0/L)^exponent, lo, hi) on l2_cable_length_um, f = 1
                   where L is missing or not positive; L0 the median of L over
                   rows with a cable and input (392.33 um on the pinned meta)
    Written per row: f/rs on free rows, rfac/gm on the size rule's leg motor
    neurons and 1/gm on the two giant-fibre rows, so r_scale x override x gm
    is f x gm on free rows and the build body's r_scale x rfac on fixed rows
    once w_syn_scale carries gm over wm1's. Memoised on the frames' content
    and the spec. Returns (override dict, sha256 of the int64 ids and float32
    values in row order, audit).

    Design d033eefd: hold {"set", "n", "ids_sha256"} writes every row of the
    rule-derived set (r_in_law_hold_mask) 1/gm, as the giant fibres are, after
    checking its count and idhash; a size-rule row inside it keeps rfac/gm.
    f_mult {"cord": x, "descending": y} multiplies f on free rows whose region
    is neither central_brain nor optic_lobe (cord, a missing region included)
    or whose super_class is descending, before the division by rs. base_wss
    is read by the installer only. rfac_by_id, mn_rfac_by_id's map (design
    80b3da76 B1: any finite factors above 0), writes those size-rule rows
    before they are fixed, so the override carries the rfac the build body
    runs.

    Design 80b3da76 B2 (port 3): gf_total_by_id, the theta's r_in_total_by_id
    restricted to the two giant-fibre rows, writes each named row
    f / (gm x r_scale) in place of 1/gm, so w_syn_scale x r_scale x override
    is base_wss x f, the body's absolute weight; the audit records f, the
    entry written and that absolute factor over base_wss per row.

    Design 80b3da76 B3: fragment_rule true (cable_power with a clip only) sets
    f = 1 on every row whose unclipped (L0/L)^exponent reaches the ceiling hi,
    the rows with L at or below L0 x hi^(-1/exponent), as the rows without a
    cable are; the audit records the threshold and the rows set to 1."""
    if not isinstance(spec, dict) or spec.get("law") not in _R_IN_LAW_FORMS:
        raise ValueError(f"r_in_law must name one of {sorted(_R_IN_LAW_FORMS)}")
    need = _R_IN_LAW_FORMS[spec["law"]]
    if set(spec) - set(_R_IN_LAW_OPTIONAL) != {"law", *need}:
        raise ValueError(f"r_in_law {spec['law']} takes exactly {need} and optionally "
                         f"{_R_IN_LAW_OPTIONAL}, got {sorted(spec)}")
    gm = float(spec["gm"])
    if not np.isfinite(gm) or gm <= 0.0:
        raise ValueError("r_in_law gm must be finite and > 0")
    if "clip" in spec:
        lo, hi = (float(x) for x in spec["clip"])
        if not (0.0 < lo <= 1.0 <= hi and np.isfinite(hi)):
            raise ValueError("r_in_law clip must be [lo, hi] with 0 < lo <= 1 <= hi")

    def _positive(v):
        return type(v) in (int, float) and np.isfinite(float(v)) and float(v) > 0.0

    hold = spec.get("hold")
    if "hold" in spec and (
            not isinstance(hold, dict) or set(hold) != {"set", "n", "ids_sha256"}
            or hold["set"] not in _R_IN_LAW_HOLDS or type(hold["n"]) is not int
            or not isinstance(hold["ids_sha256"], str) or len(hold["ids_sha256"]) != 64):
        raise ValueError(f"r_in_law hold must be {{set, n, ids_sha256}} with set among "
                         f"{_R_IN_LAW_HOLDS}, n an int and ids_sha256 64 hex digits")
    fmult = spec.get("f_mult")
    if "f_mult" in spec and (
            not isinstance(fmult, dict) or not fmult or not set(fmult) <= {"cord", "descending"}
            or not all(_positive(v) for v in fmult.values())):
        raise ValueError("r_in_law f_mult must be a non-empty map of cord and/or descending "
                         "to a finite factor > 0")
    if "base_wss" in spec and not _positive(spec["base_wss"]):
        raise ValueError("r_in_law base_wss must be finite and > 0")
    if rfac_by_id is not None and not _r_in_law_rfac_map(rfac_by_id):
        raise ValueError("r_in_law composes with mn_rfac_by_id only as a non-empty map of "
                         "finite factors > 0")
    if "fragment_rule" in spec and (spec["fragment_rule"] is not True
                                    or spec["law"] != "cable_power"):
        raise ValueError("r_in_law fragment_rule is true on cable_power only")
    if gf_total_by_id is not None:
        if (not isinstance(gf_total_by_id, dict) or not gf_total_by_id
                or not {str(b) for b in gf_total_by_id} <= set(_R_IN_LAW_GF)
                or len({str(b) for b in gf_total_by_id}) != len(gf_total_by_id)
                or not all(_positive(v) for v in gf_total_by_id.values())):
            raise ValueError("r_in_law gf_total_by_id must map giant-fibre ids, each once, "
                             "to a finite factor > 0")
        gf_total_by_id = {str(b): float(v) for b, v in gf_total_by_id.items()}
    ck = (_frame_fingerprint(meta, edges), json.dumps(spec, sort_keys=True),
          float(r_in_alpha), float(r_in_max))
    if rfac_by_id is not None:
        ck = ck + ("rfac_by_id", json.dumps({str(b): float(v) for b, v in rfac_by_id.items()},
                                            sort_keys=True))
    if gf_total_by_id is not None:
        ck = ck + ("gf_total_by_id", json.dumps(gf_total_by_id, sort_keys=True))
    if pools is not None:                         # size_pools "transferred" (design 171d6f1a B2)
        ck = ck + ("size_pools", _size_pool_sha(pools))
    hit = _R_IN_LAW_CACHE.get(ck)
    if hit is not None:
        return dict(hit[0]), hit[1], json.loads(json.dumps(hit[2]))
    ids = meta["banc_888_id"].to_numpy()
    N = len(ids)
    pre, post, keep = cns._endpoint_rows(pd.Series(np.arange(N), index=ids), ids, edges)
    cnt = edges["count"].to_numpy()[keep].astype(np.float32)
    n = np.bincount(post, weights=cnt, minlength=N)
    med = float(np.median(n[n > 0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = (med / n) ** float(r_in_alpha)
    rs[~np.isfinite(rs)] = 1.0
    rs = np.clip(rs, 1.0 / float(r_in_max), float(r_in_max)).astype(np.float32)
    f = np.ones(N, dtype=np.float64)
    audit = {"law": spec["law"], "gm": gm, "median_input_synapses": med,
             "rows": int(N), "rows_without_input": int((n == 0).sum())}
    if spec["law"] == "count_power":
        m = n > 0
        f[m] = np.clip((med / n[m]) ** float(spec["exponent"]), lo, hi)
    elif spec["law"] == "conductance":
        c = float(spec["c"])
        if not np.isfinite(c) or c <= 0.0:
            raise ValueError("r_in_law conductance c must be finite and > 0")
        f = (med + c) / (n + c)
    else:
        L = pd.to_numeric(meta["l2_cable_length_um"], errors="coerce").to_numpy(dtype=np.float64)
        has = np.isfinite(L) & (L > 0.0)
        L0 = float(np.median(L[has & (n > 0)]))
        f[has] = np.clip((L0 / L[has]) ** float(spec["exponent"]), lo, hi)
        audit.update({"L0_um": L0, "rows_without_cable": int((~has).sum()),
                      "rows_without_cable_with_input": int((~has & (n > 0)).sum())})
        if spec.get("fragment_rule"):
            frag = np.zeros(N, dtype=bool)
            frag[has] = (L0 / L[has]) ** float(spec["exponent"]) >= hi
            f[frag] = 1.0
            audit["fragment_rule"] = {
                "threshold_um": L0 * hi ** (-1.0 / float(spec["exponent"])),
                "rows_set_to_one": int(frag.sum()),
                "rows_set_to_one_with_input": int((frag & (n > 0)).sum()),
                "input_synapses_of_rows_set_to_one": float(n[frag].sum()),
                "ids_sha256": hashlib.sha256("\n".join(
                    str(i) for i in sorted(int(x) for x in ids[frag])).encode()).hexdigest()}
    if "clip" in spec:
        audit.update({"rows_on_floor": int((f == lo).sum()), "rows_on_ceiling": int((f == hi).sum())})
    _, rfac, _ = size_factors(meta, edges, **({} if pools is None else {"pools": pools}))
    if rfac_by_id is not None:                    # size_factors returns fresh copies
        rkey = {str(b): b for b in rfac}
        for b, v in rfac_by_id.items():
            if str(b) not in rkey:
                raise ValueError(f"r_in_law: mn_rfac_by_id id {b} is not a size-rule row")
            rfac[rkey[str(b)]] = float(v)
    sid = np.array([str(i) for i in ids], dtype=object)
    row = {s: r for r, s in enumerate(sid)}
    fixed = {}
    for b, v in rfac.items():
        if str(b) not in row:
            raise ValueError(f"r_in_law: size-rule row {b} is not in meta")
        fixed[row[str(b)]] = float(v) / gm
    gf_total = {}
    for b in _R_IN_LAW_GF:
        if b not in row:
            raise ValueError(f"r_in_law: giant-fibre row {b} is not in meta")
        if row[b] in fixed:
            raise ValueError(f"r_in_law: giant-fibre row {b} is also a size-rule row")
        fixed[row[b]] = 1.0 / gm
        if gf_total_by_id is not None and b in gf_total_by_id:
            fixed[row[b]] = gf_total_by_id[b] / (gm * float(rs[row[b]]))
            gf_total[b] = row[b]
    held = {}
    if hold is not None:
        hmask = r_in_law_hold_mask(meta, edges, hold["set"])
        hsha = r_in_law_idhash(meta, hmask)
        if int(hmask.sum()) != hold["n"] or hsha != hold["ids_sha256"]:
            raise ValueError(f"r_in_law hold {hold['set']}: {int(hmask.sum())} rows with idhash "
                             f"{hsha[:16]} on these frames, the theta names {hold['n']} and "
                             f"{hold['ids_sha256'][:16]}")
        held = {int(r): 1.0 / gm for r in np.flatnonzero(hmask) if int(r) not in fixed}
    if fmult is not None:
        free = np.ones(N, dtype=bool)
        free[sorted({*fixed, *held})] = False
        cord = free & ~np.isin(meta["region"].fillna("none").astype(str).to_numpy(),
                               ["central_brain", "optic_lobe"])
        desc = free & (meta["super_class"].fillna("").astype(str).to_numpy() == "descending")
        f = f.copy()
        if "cord" in fmult:
            f[cord] *= float(fmult["cord"])
        if "descending" in fmult:
            f[desc] *= float(fmult["descending"])
        audit["f_mult"] = {**{k: float(v) for k, v in fmult.items()}, "free_cord_rows": int(cord.sum()),
                           "free_descending_rows": int(desc.sum()),
                           "free_cord_descending_rows": int((cord & desc).sum())}
    vals = (f / rs.astype(np.float64)).astype(np.float32)
    for r, v in fixed.items():
        vals[r] = np.float32(v)
    for r, v in held.items():
        vals[r] = np.float32(v)
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0.0):
        raise ValueError("r_in_law produced a non-finite or non-positive override")
    ov = {s: float(v) for s, v in zip(sid, vals)}
    sha = hashlib.sha256(np.asarray([int(s) for s in sid], dtype=np.int64).tobytes()
                         + vals.tobytes()).hexdigest()
    audit.update({"override_sha256": sha, "free_rows": int(N - len(fixed) - len(held)),
                  "fixed_leg_mn_rows": int(len(rfac)), "fixed_gf_rows": len(_R_IN_LAW_GF),
                  "factor_sum_free_rows": float(np.delete(f, sorted({*fixed, *held})).sum())})
    if hold is not None:
        audit["hold"] = {**hold, "held_rows_written": len(held),
                         "held_rows_fixed": int(hmask.sum()) - len(held)}
    if rfac_by_id is not None:
        audit["rfac_by_id"] = {str(b): float(v) for b, v in rfac_by_id.items()}
    if "base_wss" in spec:
        audit["base_wss"] = float(spec["base_wss"])
    if gf_total:
        audit["gf_total_by_id"] = {
            b: {"row": int(r), "f": gf_total_by_id[b], "r_scale": float(rs[r]),
                "override": float(vals[r]),
                "absolute_over_base_wss": float(gm * float(rs[r]) * float(vals[r]))}
            for b, r in gf_total.items()}
    if len(_R_IN_LAW_CACHE) >= 4:
        _R_IN_LAW_CACHE.pop(next(iter(_R_IN_LAW_CACHE)))
    _R_IN_LAW_CACHE[ck] = (ov, sha, audit)
    return dict(ov), sha, json.loads(json.dumps(audit))


def r_in_total_by_id_apply(p, meta, spec, *, r_in_law=None):
    """Design d27f82df B1 (study eb9ff316): validate th["r_in_total_by_id"],
    {banc_888_id: f}, and put it on p as p.r_in_total_by_id.

    cns.py then writes each named row's override entry as f / r_scale after
    r_in_override's entries, so the row's input scale r_scale x override is f
    at the body's w_syn_scale. Called after every path that writes
    p.r_in_override (the size rule, mn_rfac_by_id, wing_power_mn_r_in_factor).
    Refused: a spec that is not a non-empty map; an id not in meta (cns.py's
    r_in_override loop skips one without a word); an id named twice; an id
    that already has an r_in_override entry; composition with r_in_law, which
    writes every row's factor; an f that is not a finite number > 0.
    Absent = never called = bit-identical.

    Design 80b3da76 B2 (port 3): with r_in_law, a map naming only the two
    giant-fibre rows composes. The law has already written each named row
    f / (gm x r_scale) (r_in_law_override's gf_total_by_id), so the entry the
    apply step finds is its own: it checks the law's audit names the same f
    and leaves p.r_in_total_by_id unset, so cns.py's loop never writes over
    the law's entry. Every other id still refuses."""
    if r_in_law is not None:
        _audit = getattr(p, "r_in_law_audit", None) or {}
        _gf = (_audit.get("gf_total_by_id") or {}) if isinstance(spec, dict) else {}
        if (not isinstance(spec, dict) or not spec
                or not {str(b) for b in spec} <= set(_R_IN_LAW_GF)):
            raise ValueError("r_in_total_by_id cannot compose with r_in_law outside the "
                             "two giant-fibre rows")
        out = {}
        for k, f in spec.items():
            b = str(k)
            if b in out:
                raise ValueError(f"r_in_total_by_id names {b} twice")
            if (isinstance(f, bool) or not isinstance(f, (int, float))
                    or not np.isfinite(f) or float(f) <= 0.0):
                raise ValueError(f"r_in_total_by_id[{b}] must be a finite number > 0")
            if b not in _gf or _gf[b]["f"] != float(f):
                raise ValueError(f"r_in_total_by_id id {b}: r_in_law did not write its total "
                                 f"factor {float(f)!r}")
            out[b] = float(f)
        return out
    if not isinstance(spec, dict) or not spec:
        raise ValueError(
            "r_in_total_by_id must be a non-empty map of banc_888_id to a factor")
    meta_ids = set(meta["banc_888_id"].astype(str))
    have = {str(b) for b in (p.r_in_override or {})}
    out = {}
    for k, f in spec.items():
        b = str(k)
        if b in out:
            raise ValueError(f"r_in_total_by_id names {b} twice")
        if b not in meta_ids:
            raise ValueError(f"r_in_total_by_id id {b} is not in meta")
        if b in have:
            raise ValueError(
                f"r_in_total_by_id id {b} already has an r_in_override entry")
        if (isinstance(f, bool) or not isinstance(f, (int, float))
                or not np.isfinite(f) or float(f) <= 0.0):
            raise ValueError(f"r_in_total_by_id[{b}] must be a finite number > 0")
        out[b] = float(f)
    p.r_in_total_by_id = out
    return out


def r_in_total_by_id_readback(net, spec, *, r_in_law=None, w_syn_scale=None):
    """B1's output: per named row f, r_scale and the override entry read from
    the built network's arrays, and their product against f. Raises when a
    product is more than 1e-6 relative from f, which a network built for
    another map would give.

    Design 80b3da76 B2: under r_in_law the product compared with f is the
    absolute form w_syn_scale x r_scale x override / base_wss, the quantity
    port 3 fixes."""
    pos = {str(b): r for r, b in enumerate(net.ids)}
    ov = net.r_in_override_scale
    scale = 1.0
    if r_in_law is not None:
        scale = float(w_syn_scale) / float(r_in_law.get("base_wss", _R_IN_LAW_WSS))
    rows = []
    for k, f in spec.items():
        r = pos[str(k)]
        rs = 1.0 if net.r_scale is None else float(net.r_scale[r])
        o = 1.0 if ov is None else float(ov[r])
        prod = float(np.float32(rs) * np.float32(o)) * scale
        if abs(prod - float(f)) > 1e-6 * abs(float(f)):
            raise ValueError(
                f"r_in_total_by_id: row {k} reads "
                f"{'w_syn_scale x r_scale x override / base_wss' if r_in_law is not None else 'r_scale x override'}"
                f" {prod!r}, not {float(f)!r}")
        row_ = {"banc_888_id": str(k), "row": int(r), "f": float(f),
                "r_scale": rs, "ov": o, "product": prod}
        if r_in_law is not None:
            row_["form"] = "w_syn_scale x r_scale x override / base_wss"
        rows.append(row_)
    return {"n_rows": len(rows), "rows": rows}


# Design d27f82df B2: theta field -> (SimParams attribute, floor, ceiling).
# v_threshold's floor is the row's own v_reset, exclusive.
ROW_MEMBRANE_FIELDS = {"tau_mem_ms": ("tau_mem", 0.1, 200.0),   # efe88957 Build 6: was 100.0
                       "adapt_a": ("adapt_a", 0.0, 10.0),
                       "adapt_b_mv": ("adapt_b", 0.0, 20.0),
                       "tau_w_ms": ("tau_w", 1.0, 100000.0),  # fbcd0f61 B2: was 5000.0
                       "v_threshold_mv": ("v_threshold", None, 0.0)}


def row_membrane_by_id_apply(th, meta, p, graded_rows):
    """Design d27f82df B2: th["row_membrane_by_id"] gives named rows their own
    membrane constant, threshold and adaptation, and optionally the exact
    update, {banc_888_id: {tau_mem_ms, adapt_a, adapt_b_mv, tau_w_ms,
    v_threshold_mv, exact}}. Applied after every other membrane, threshold
    and adaptation path: tau_mem, adapt_a, adapt_b, tau_w and v_threshold each
    become a float64 vector over every row at the value it has here and the
    named rows are written, so an unnamed row keeps what those paths left (a
    float32 value survives the float64 copy exactly). exact becomes
    p.row_exact_mask, a uint8 vector, set only when some row is exact, so a
    map with every exact false runs the kernel path it ran before. v_rest is
    not touched. It composes with adapt_law_rows, which runs first
    (experiments-2, note b1adeebb): a row any group writes is refused, and
    while the law is split (p.adapt_a_below set) a named row's adapt_a is
    written below rest too, so it runs the unsplit law at its own a. Refused:
    membrane_mode or adaptation_parameter_mode present; adapt_law_rows
    present without its audit; a row that silence_rows, adapt_zero_ids,
    mn_adapt_b_zero_ids, mn_membrane_by_id or crossing_tau_w_mult names; a
    motor-map id; an id not in meta or named twice; a missing, extra,
    non-finite or out-of-range field; exact true without the conductance law.
    Every engine runs the mask since B8; NumPy and Metal refuse the opsin
    beside it. A graded row is accepted (design efe88957 Build 3): every
    engine integrates it with the per-row decay this writes, threshold
    detection is off on graded rows, and release follows r(V) as before; the
    audit then counts them under n_graded.
    Sets p.row_membrane_by_id_audit and returns it."""
    spec = th["row_membrane_by_id"]
    for _clash in ("membrane_mode", "adaptation_parameter_mode"):
        if th.get(_clash) not in (None, ""):
            raise ValueError(f"row_membrane_by_id cannot compose with {_clash}")
    law_rows = None
    if th.get("adapt_law_rows") is not None:
        _law = getattr(p, "adapt_law_rows_audit", None)
        if _law is None:
            raise ValueError(
                "row_membrane_by_id composes with adapt_law_rows only after "
                "adapt_law_rows_vectors has run")
        law_rows = np.asarray(_law["arrays"]["form"]) >= 0
    if not isinstance(spec, dict) or not spec:
        raise ValueError(
            "row_membrane_by_id must be a non-empty map of banc_888_id to "
            f"{sorted(ROW_MEMBRANE_FIELDS) + ['exact']}")
    ids = meta["banc_888_id"].astype(str).to_numpy()
    n = len(ids)
    pos = {b: i for i, b in enumerate(ids)}
    import motormap as _mm
    motor = {str(x) for x in _mm.to_matrix(_mm.build_map(meta))[1]}
    guarded = {}
    for _k in ("silence_rows", "adapt_zero_ids", "mn_adapt_b_zero_ids",
               "mn_membrane_by_id"):
        if th.get(_k) is not None:
            guarded[_k] = {str(x) for x in th[_k]}
    if th.get("crossing_tau_w_mult") is not None:
        guarded["crossing_tau_w_mult"] = {
            str(x) for x in th["crossing_tau_w_mult"]["rows"]}
    graded = {int(r) for r in graded_rows}
    fields = set(ROW_MEMBRANE_FIELDS) | {"exact"}
    vec = {a: np.broadcast_to(np.asarray(getattr(p, a), dtype=np.float64), (n,)).copy()
           for a, _lo, _hi in ROW_MEMBRANE_FIELDS.values()}
    below = (None if getattr(p, "adapt_a_below", None) is None else np.broadcast_to(
        np.asarray(p.adapt_a_below, dtype=np.float64), (n,)).copy())
    v_reset = np.broadcast_to(np.asarray(p.v_reset, dtype=np.float64), (n,))
    mask = np.zeros(n, dtype=np.uint8)
    rows = []
    for b, s in spec.items():
        bid = str(b)
        r = pos.get(bid)
        if r is None:
            raise ValueError(f"row_membrane_by_id id {bid} is not a row of meta")
        if r in rows:
            raise ValueError(f"row_membrane_by_id names row {bid} twice")
        if bid in motor:
            raise ValueError(
                f"row_membrane_by_id id {bid} is a motor-map row; mn_membrane_by_id "
                "covers those")
        for _k, _ids in guarded.items():
            if bid in _ids:
                raise ValueError(f"row_membrane_by_id id {bid} is also named by {_k}")
        if law_rows is not None and law_rows[r]:
            raise ValueError(f"row_membrane_by_id id {bid} is also written by adapt_law_rows")
        if not isinstance(s, dict) or set(s) != fields:
            raise ValueError(
                f"row_membrane_by_id[{bid}] must carry exactly {sorted(fields)}")
        if type(s["exact"]) is not bool:
            raise ValueError(f"row_membrane_by_id[{bid}][exact] must be true or false")
        for f, (a, lo, hi) in ROW_MEMBRANE_FIELDS.items():
            x = s[f]
            ok = type(x) in (int, float) and np.isfinite(float(x))
            if ok and lo is None:
                ok = float(v_reset[r]) < float(x) <= hi
            elif ok:
                ok = lo <= float(x) <= hi
            if not ok:
                floor = f"above v_reset {float(v_reset[r])}" if lo is None else f">= {lo}"
                raise ValueError(
                    f"row_membrane_by_id[{bid}][{f}] must be finite, {floor} and "
                    f"<= {hi}, not {x!r}")
        if s["exact"] and not p.conductance_based:
            raise ValueError(
                f"row_membrane_by_id[{bid}] exact needs the conductance law (cbased)")
        for f, (a, _lo, _hi) in ROW_MEMBRANE_FIELDS.items():
            vec[a][r] = float(s[f])
        if below is not None:
            below[r] = float(s["adapt_a"])
        mask[r] = 1 if s["exact"] else 0
        rows.append(r)
    for a, v in vec.items():
        setattr(p, a, v)
    if below is not None:
        p.adapt_a_below = below
    p.row_exact_mask = mask if mask.any() else None
    sel = np.asarray(sorted(rows), dtype=np.int64)
    p.row_membrane_by_id_audit = {
        "n_rows": len(rows), "n_exact": int(mask.sum()),
        "min": {f: float(vec[a][sel].min()) for f, (a, _l, _h) in ROW_MEMBRANE_FIELDS.items()},
        "max": {f: float(vec[a][sel].max()) for f, (a, _l, _h) in ROW_MEMBRANE_FIELDS.items()},
        "sha256": {a: hashlib.sha256(np.ascontiguousarray(v, dtype=np.float64).tobytes()
                                     ).hexdigest() for a, v in vec.items()},
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
    }
    n_graded = sum(1 for r in rows if r in graded)
    if n_graded:   # only a graded row adds the key, so an audit without one is unchanged
        p.row_membrane_by_id_audit["n_graded"] = n_graded
    if below is not None:   # only a split law adds the key, so an unsplit audit is unchanged
        p.row_membrane_by_id_audit["adapt_a_below"] = {
            "min": float(below[sel].min()), "max": float(below[sel].max()),
            "sha256": hashlib.sha256(np.ascontiguousarray(below).tobytes()).hexdigest()}
    return p.row_membrane_by_id_audit


def row_membrane_by_id_readback(net, spec):
    """B2's output: per named row the engine's decay_v, dt_over_tau, spike_at,
    adapt_a, adapt_b, dt_over_tauw and exact-update byte, read from the arrays
    the engine was built with (net._metal_engine), or from p's vectors when
    no engine ran; while the law is split, adapt_a_below as well."""
    eng = getattr(net, "_metal_engine", None)
    keep = None if eng is None else eng._keep
    pos = {str(b): r for r, b in enumerate(net.ids)}
    out = []
    for b in spec:
        r = pos[str(b)]
        row = {"banc_888_id": str(b), "row": int(r)}
        if keep is not None:
            for k in ("decay_v", "dt_over_tau", "spike_at", "adapt_a", "adapt_b",
                      "dt_over_tauw"):
                a = np.asarray(keep[k])
                row[k] = float(a if a.ndim == 0 else a[r])
            m = keep.get("row_exact_mask")
            row["exact"] = int(0 if m is None else m[r])
            below = getattr(eng, "_adapt_below", None)
        else:
            for f, (a, _l, _h) in ROW_MEMBRANE_FIELDS.items():
                x = np.asarray(getattr(net.p, a), dtype=np.float64)
                row[f] = float(x if x.ndim == 0 else x[r])
            m = getattr(net.p, "row_exact_mask", None)
            row["exact"] = int(0 if m is None else m[r])
            below = getattr(net.p, "adapt_a_below", None)
        if below is not None:
            x = np.asarray(below)
            row["adapt_a_below"] = float(x if x.ndim == 0 else x[r])
        out.append(row)
    return {"engine": keep is not None, "rows": out}


def mn_vrest_by_id_readback(p, ids, keep=None):
    """Study 7f1b4a1e B1's check and output: per row th["mn_vrest_by_id"] names,
    the size rule's rest, the rest asked for (None deletes the entry, so the
    row rests at p.v_rest) and the rest built: from the engine's v_rest array
    when keep (net._metal_engine._keep) is given, otherwise from p as cns.CNS
    folds p.v_rest_override into p.v_rest. Raises when the built rest is not
    the one asked for, as when a rest path after the size rule (v_rest_type_mv,
    membrane_mode) wrote a named row."""
    audit = getattr(p, "mn_vrest_by_id_audit", None)
    if audit is None:
        raise ValueError("mn_vrest_by_id needs mn_gains 'size': no size rule rested a row")
    pos = {str(b): r for r, b in enumerate(ids)}
    base = np.broadcast_to(np.asarray(p.v_rest, dtype=np.float32), (len(pos),))
    ov = p.v_rest_override or {}
    out = []
    for bid, want in audit["requested_mV"].items():
        r = pos[bid]
        if keep is not None:
            built = np.float32(np.asarray(keep["v_rest"])[r])
        else:
            built = np.float32(ov[bid]) if bid in ov else base[r]
        if (want is None and bid in ov) or built != (base[r] if want is None else np.float32(want)):
            raise ValueError(f"mn_vrest_by_id: row {bid} is built at {float(built)} mV, "
                             f"not the {want} asked for")
        out.append({"banc_888_id": bid, "row": int(r), "size_rule_mV": audit["size_rule_mV"][bid],
                    "requested_mV": want, "built_mV": float(built)})
    return {"engine": keep is not None, "n_deleted": audit["n_deleted"],
            "n_override_entries": audit["n_override_entries"], "rows": out}


def mn_reset_offset_by_id_apply(th, meta, p, vrest):
    """Study 9e0afa22 Build 1 (design 7c57b886): th["mn_reset_offset_by_id"],
    {banc_888_id: offset_mV}, sets each named leg motor row's reset to its own
    rest plus the offset, v_reset[i] = v_rest_eff[i] + offset_i, at the
    mn_vreset seam. v_rest_eff is the rest the size rule leaves the row after
    mn_vrest_by_id (vrest, the map that becomes p.v_rest_override), or the
    shared p.v_rest for a motor row the rule does not rest (the two null rows
    of the build body). Unnamed rows keep the reset they have. Refuses an empty
    or malformed map, an id that is not one of the motor-map rows, an id named
    twice, an offset that is not a finite number, and mn_vreset in the same
    theta (both set the reset of the motor rows). mn_reset_offset_by_id_check
    refuses v_rest_type_reset_follows per row, where it resets a named row
    (note 82bdf65f), and any later writer that moved a named row's rest or
    reset. Absent = never called = no array built."""
    req = th["mn_reset_offset_by_id"]
    if not isinstance(req, dict) or not req:
        raise ValueError(
            "mn_reset_offset_by_id must be a non-empty map of banc_888_id to a "
            "finite reset offset in mV")
    if th.get("mn_vreset"):
        raise ValueError(
            "mn_reset_offset_by_id cannot compose with mn_vreset: both set the "
            "reset of the motor rows")
    import motormap as _MMr
    motor = {str(x) for x in _MMr.to_matrix(_MMr.build_map(meta))[1]}
    ids = meta["banc_888_id"].astype(str).to_numpy()
    pos = {b: i for i, b in enumerate(ids)}
    rest0 = np.broadcast_to(np.asarray(p.v_rest, dtype=np.float64), (len(ids),))
    reset = np.broadcast_to(np.asarray(p.v_reset, dtype=np.float64),
                            (len(ids),)).astype(np.float64).copy()
    rows = []
    for _b, _off in req.items():
        bid = str(_b)
        if any(r["banc_888_id"] == bid for r in rows):
            raise ValueError("mn_reset_offset_by_id names an id twice")
        if bid not in motor:
            raise ValueError(
                f"mn_reset_offset_by_id id {bid} is not one of the "
                f"{len(motor)} leg motor-map rows")
        if type(_off) not in (int, float) or not np.isfinite(float(_off)):
            raise ValueError(
                f"mn_reset_offset_by_id[{bid}] must be a finite number of mV, "
                f"not {_off!r}")
        i = pos[bid]
        rest = float(vrest[bid]) if bid in vrest else float(rest0[i])
        rows.append({"banc_888_id": bid, "row": int(i), "rest_mV": rest,
                     "rest_from": "size_rule" if bid in vrest else "shared",
                     "offset_mV": float(_off),
                     "reset_before_mV": float(reset[i]),
                     "reset_mV": rest + float(_off)})
        reset[i] = rest + float(_off)
    p.v_reset = reset
    p.mn_reset_offset_by_id_audit = {
        "n_named": len(rows), "n_motor_rows": len(motor), "rows": rows}


def mn_reset_offset_by_id_check(p, ids, keep=None):
    """Study 9e0afa22 Build 1's check and the gate's output: per named row the
    rest the offset was taken from, the offset, and the rest and reset built,
    in float32, from the engine's received v_rest and v_reset row arrays when
    keep (net._metal_engine._keep) is given, otherwise from p as cns.CNS folds
    p.v_rest_override into p.v_rest and broadcasts p.v_reset. Raises per row
    when v_rest_type_reset_follows reset a named row, and when a writer after
    the seam (v_rest_type_mv, wing power, typed 12B, membrane_mode) moved a
    named row's rest or reset: the built reset must equal float32(rest +
    offset) and the built rest float32(rest)."""
    audit = getattr(p, "mn_reset_offset_by_id_audit", None)
    if audit is None:
        raise ValueError(
            "mn_reset_offset_by_id needs mn_gains 'size': the seam that reads it "
            "did not run")
    follows = {r["banc_888_id"] for r in
               (getattr(p, "v_rest_type_reset_audit", None) or {}).get("rows", [])}
    both = sorted(follows & {r["banc_888_id"] for r in audit["rows"]})
    if both:
        raise ValueError(
            f"mn_reset_offset_by_id and v_rest_type_reset_follows both reset "
            f"row(s) {both}")
    n = len(ids)
    if keep is not None:
        rest_b = np.asarray(keep["v_rest"], dtype=np.float32)
        reset_b = np.asarray(keep["v_reset"], dtype=np.float32)
    else:
        rest_b = np.broadcast_to(np.asarray(p.v_rest, dtype=np.float32), (n,)).copy()
        ov = p.v_rest_override or {}
        pos = {str(b): r for r, b in enumerate(ids)}
        for bid, mv in ov.items():
            r = pos.get(str(bid))
            if r is not None:
                rest_b[r] = mv
        reset_b = np.broadcast_to(np.asarray(p.v_reset, dtype=np.float32), (n,))
    out = []
    for r in audit["rows"]:
        i = r["row"]
        if str(ids[i]) != r["banc_888_id"]:
            raise ValueError(f"mn_reset_offset_by_id: row {i} is not {r['banc_888_id']}")
        want_reset = np.float32(r["reset_mV"])
        want_rest = np.float32(r["rest_mV"])
        if reset_b[i] != want_reset or rest_b[i] != want_rest:
            raise ValueError(
                f"mn_reset_offset_by_id: row {r['banc_888_id']} is built at rest "
                f"{float(rest_b[i])} and reset {float(reset_b[i])} mV, not rest "
                f"{float(want_rest)} and reset {float(want_reset)}")
        out.append({**r, "built_rest_mV": float(rest_b[i]),
                    "built_reset_mV": float(reset_b[i])})
    named = np.zeros(n, dtype=bool)
    named[[r["row"] for r in audit["rows"]]] = True
    unnamed = np.asarray(reset_b, dtype=np.float32)[~named]
    return {"engine": keep is not None, "n_named": audit["n_named"],
            "n_motor_rows": audit["n_motor_rows"],
            "unnamed_reset_sha256": hashlib.sha256(unnamed.tobytes()).hexdigest(),
            "unnamed_reset_values_mV": sorted({float(x) for x in np.unique(unnamed)}),
            "rows": out}


# Design d27f82df B3: the hold-down's theta fields per form.
HOLDDOWN_FIELDS = {
    "A": ("dg", "tau_g_ms", "e_rel_mv", "latch_mv", "latch_ms"),
    "pool": ("gmax", "e_rel_mv", "c_spk", "k_per_mv_ms", "va_mv", "tau_c_ms",
             "latch_mv", "latch_ms"),
}


def mn_exact_by_id_apply(th, meta, p):
    """Design 83177ced Build 1 (study b6890108): th["mn_exact_by_id"], a map of
    banc_888_id to true, puts named leg motor rows on the exact membrane
    update by ORing them into p.row_exact_mask, so holddown_by_id can put a
    hold-down on them. The rows keep the membrane mn_membrane_by_id gives them.
    Applied after row_membrane_by_id_apply, which sets the mask afresh, and
    before holddown_by_id_apply, whose exact set is the union of the two keys'
    rows. Refused: a map that is not a non-empty dict; a value other than
    true; an id that is not a motor-map row; an id mn_membrane_by_id does not
    name; an id row_membrane_by_id names; the current-based membrane (cbased
    false). Absent = never called = bit-identical. Sets p.mn_exact_by_id_audit
    and returns it."""
    spec = th["mn_exact_by_id"]
    if not isinstance(spec, dict) or not spec:
        raise ValueError("mn_exact_by_id must be a non-empty map of banc_888_id to true")
    if not p.conductance_based:
        raise ValueError("mn_exact_by_id needs the conductance law (cbased)")
    ids = meta["banc_888_id"].astype(str).to_numpy()
    n = len(ids)
    pos = {b: i for i, b in enumerate(ids)}
    import motormap as _mm
    motor = {str(x) for x in _mm.to_matrix(_mm.build_map(meta))[1]}
    membrane = {str(b) for b in (th.get("mn_membrane_by_id") or {})}
    named = {str(b) for b in (th.get("row_membrane_by_id") or {})}
    old = getattr(p, "row_exact_mask", None)
    mask = np.zeros(n, dtype=np.uint8) if old is None else np.asarray(old, dtype=np.uint8).copy()
    if mask.shape != (n,):
        raise ValueError("mn_exact_by_id: row_exact_mask needs one byte per row")
    n_before = int(np.count_nonzero(mask))
    rows = []
    for b, v in spec.items():
        bid = str(b)
        if v is not True:
            raise ValueError(f"mn_exact_by_id[{bid}] must be true, not {v!r}")
        if bid not in motor:
            raise ValueError(f"mn_exact_by_id id {bid} is not one of the {len(motor)} motor-map rows")
        if bid not in membrane:
            raise ValueError(f"mn_exact_by_id id {bid} is not named by mn_membrane_by_id")
        if bid in named:
            raise ValueError(f"mn_exact_by_id id {bid} is also named by row_membrane_by_id")
        r = pos[bid]
        if r in rows:
            raise ValueError(f"mn_exact_by_id names row {bid} twice")
        mask[r] = 1
        rows.append(r)
    p.row_exact_mask = mask
    p.mn_exact_by_id_audit = {
        "n_named": len(rows), "n_exact_before": n_before, "n_exact": int(np.count_nonzero(mask)),
        "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()}
    return p.mn_exact_by_id_audit


def mn_exact_by_id_readback(net, spec):
    """Build 1's output: the exact-update mask the engine was built with
    (net._metal_engine._keep["row_exact_mask"]), or p's when no engine ran:
    rows on it, the named rows on it, its hash."""
    eng = getattr(net, "_metal_engine", None)
    m = (getattr(net.p, "row_exact_mask", None) if eng is None
         else eng._keep.get("row_exact_mask"))
    if m is None:
        raise ValueError("mn_exact_by_id: the engine carries no row_exact_mask")
    m = np.asarray(m, dtype=np.uint8)
    pos = {str(b): r for r, b in enumerate(net.ids)}
    return {"engine": eng is not None, "n_exact": int(np.count_nonzero(m)),
            "n_named_exact": int(sum(1 for b in spec if m[pos[str(b)]] != 0)),
            "n_named": len(spec), "mask_sha256": hashlib.sha256(m.tobytes()).hexdigest()}


def holddown_by_id_apply(th, meta, p):
    """Design d27f82df B3: th["holddown_by_id"] puts a hold-down conductance
    on named rows, {banc_888_id: {"form": "A", dg, tau_g_ms, e_rel_mv,
    latch_mv, latch_ms}} or {"form": "pool", gmax, e_rel_mv, c_spk,
    k_per_mv_ms, va_mv, tau_c_ms, latch_mv, latch_ms}. Each row must be one
    that row_membrane_by_id or mn_exact_by_id (design 83177ced Build 1) puts
    on the exact update. The per-slot arrays the
    CUDA engine takes go on p.holddown_slots: rows, form (0 A, 1 pool),
    e_rel, decay = exp(-dt/tau_g) in float32 (1 for the pool), dg, gmax,
    c_spk, k, va, tau_c (1 for A), latch_mv and latch_ticks =
    latch_ms/dt. Refused: a non-empty map that is not a dict; an id not in
    meta or named twice; a row without exact; an unknown form; a missing,
    extra or non-finite field; a time constant not above zero; a negative
    dg, gmax, c_spk or k; a latch that is not a whole number of steps; the
    current-based membrane. Every engine runs it since B8; NumPy and Metal
    refuse the opsin beside it, and a state branch refuses it. Sets p.holddown_by_id_audit and returns
    it."""
    spec = th["holddown_by_id"]
    if not isinstance(spec, dict) or not spec:
        raise ValueError("holddown_by_id must be a non-empty map of banc_888_id to a form")
    if not p.conductance_based:
        raise ValueError("holddown_by_id needs the conductance law (cbased)")
    exact = {str(b) for b, s in (th.get("row_membrane_by_id") or {}).items()
             if isinstance(s, dict) and s.get("exact") is True}
    exact |= {str(b) for b, v in (th.get("mn_exact_by_id") or {}).items() if v is True}
    ids = meta["banc_888_id"].astype(str).to_numpy()
    pos = {b: i for i, b in enumerate(ids)}
    dt = float(p.dt)
    cols = {k: [] for k in ("rows", "form", "e_rel", "decay", "dg", "gmax", "c_spk",
                            "k", "va", "tau_c", "latch_mv", "latch_ticks")}
    audit_rows = []
    seen = set()
    for b, s in spec.items():
        bid = str(b)
        r = pos.get(bid)
        if r is None:
            raise ValueError(f"holddown_by_id id {bid} is not a row of meta")
        if r in seen:
            raise ValueError(f"holddown_by_id names row {bid} twice")
        seen.add(r)
        if bid not in exact:
            raise ValueError(
                f"holddown_by_id id {bid} is not on the exact update "
                "(row_membrane_by_id with exact true, or mn_exact_by_id)")
        form = s.get("form") if isinstance(s, dict) else None
        if form not in HOLDDOWN_FIELDS:
            raise ValueError(
                f"holddown_by_id[{bid}] form must be one of {sorted(HOLDDOWN_FIELDS)}")
        want = set(HOLDDOWN_FIELDS[form]) | {"form"}
        if set(s) != want:
            raise ValueError(f"holddown_by_id[{bid}] must carry exactly {sorted(want)}")
        v = {}
        for f in HOLDDOWN_FIELDS[form]:
            x = s[f]
            if type(x) not in (int, float) or not np.isfinite(float(x)):
                raise ValueError(f"holddown_by_id[{bid}][{f}] must be finite, not {x!r}")
            v[f] = float(x)
        for f in ("tau_g_ms", "tau_c_ms", "latch_ms"):
            if f in v and not v[f] > 0.0:
                raise ValueError(f"holddown_by_id[{bid}][{f}] must be above zero")
        for f in ("dg", "gmax", "c_spk", "k_per_mv_ms"):
            if f in v and v[f] < 0.0:
                raise ValueError(f"holddown_by_id[{bid}][{f}] must not be negative")
        lt = v["latch_ms"] / dt
        if abs(lt - round(lt)) > 1e-9 * max(1.0, lt) or round(lt) < 1:
            raise ValueError(
                f"holddown_by_id[{bid}][latch_ms] must be a whole number of {dt} ms steps")
        pool = form == "pool"
        cols["rows"].append(r)
        cols["form"].append(1 if pool else 0)
        cols["e_rel"].append(v["e_rel_mv"])
        cols["decay"].append(1.0 if pool else float(np.float32(np.exp(-dt / v["tau_g_ms"]))))
        cols["dg"].append(0.0 if pool else v["dg"])
        cols["gmax"].append(v["gmax"] if pool else 0.0)
        cols["c_spk"].append(v["c_spk"] if pool else 0.0)
        cols["k"].append(v["k_per_mv_ms"] if pool else 0.0)
        cols["va"].append(v["va_mv"] if pool else 0.0)
        cols["tau_c"].append(v["tau_c_ms"] if pool else 1.0)
        cols["latch_mv"].append(v["latch_mv"])
        cols["latch_ticks"].append(int(round(lt)))
        audit_rows.append({"banc_888_id": bid, "row": int(r), "form": form, **v,
                           "decay": cols["decay"][-1], "latch_ticks": int(round(lt))})
    dtypes = {"rows": np.int32, "form": np.uint32, "latch_ticks": np.uint32}
    p.holddown_slots = {k: np.ascontiguousarray(c, dtype=dtypes.get(k, np.float32))
                        for k, c in cols.items()}
    p.holddown_by_id_audit = {"n_rows": len(audit_rows), "dt_ms": dt, "rows": audit_rows}
    return p.holddown_by_id_audit


def holddown_by_id_readback(net):
    """B3's output: per named row the run's peak g_h and its tick, the state
    at the run's end, and every field read back from the arrays the engine
    was configured with: the last NumPy run's (net._holddown_numpy, B8) when
    it carried the hold-down, else the native engine's."""
    nh = getattr(net, "_holddown_numpy", None)
    if nh is not None:
        st, keep = nh["state"], {"holddown_" + k: v for k, v in nh["slots"].items()}
    else:
        eng = getattr(net, "_metal_engine", None)
        if eng is None or getattr(eng, "holddown_n", 0) == 0:
            raise ValueError("holddown_by_id: no engine carried the hold-down")
        st = eng.holddown_state()
        keep = eng._keep
    out = []
    for q, a in enumerate(net.p.holddown_by_id_audit["rows"]):
        row = {"banc_888_id": a["banc_888_id"], "row": int(keep["holddown_rows"][q]),
               "peak_g_h": float(st["peak"][q]), "peak_tick": int(st["peak_tick"][q])}
        for k in ("c", "g_h", "L", "b"):
            row["end_" + k] = float(st[k][q])
        for k in ("form", "e_rel", "decay", "dg", "gmax", "c_spk", "k", "va", "tau_c",
                  "latch_mv", "latch_ticks"):
            x = keep["holddown_" + k][q]
            row[k] = int(x) if k in ("form", "latch_ticks") else float(x)
        out.append(row)
    return {"n_rows": len(out), "rows": out}


def pitch_only_idx():
    """DOF indices that are not pitch. Default-off lesion, not a model."""
    import motormap as MM
    return np.array([
        i for i, n in enumerate(MM.DOF_NAMES)
        if n.endswith("_roll") or n.endswith("_yaw")
    ], dtype=np.int64)


def feti_force_overlay(gvec, mn_ids, meta, scope=None):
    """Set named FETi gain to 1.0. Size rank is otherwise kept.

    scope=None / absent: every FETi (the confirmed overlay).
    scope='hind': only body_part_effector==hind_leg. Mid-right
    FETi stays at the size floor; that cell fired 190 Hz on the
    confirm stander after the all-FETi overlay unmuted it.
    """
    out = np.asarray(gvec, dtype=float).copy()
    types = dict(zip(meta["banc_888_id"].astype(str),
                     meta["cell_type"].astype(str)))
    parts = dict(zip(meta["banc_888_id"].astype(str),
                     meta["body_part_effector"].astype(str)))
    for k, i in enumerate(mn_ids):
        sid = str(i)
        if "FETi" not in types.get(sid, ""):
            continue
        if scope == "hind" and parts.get(sid) != "hind_leg":
            continue
        out[k] = 1.0
    return out


# Homology guess, not a BANC label. Left T2 Fast cable is 4317.39 um.
# Closest right-T2 tibia_flexor on right_mesothoracic_leg_nerve is
# this cell at 4453.19 um (volume also closest: 2.26e10 vs 2.37e10).
# Longest cable on that nerve (5496 um) is a worse size match.
AZEVEDO_RT2_FAST_HOMOLOGY_ID = "720575941474499377"
AZEVEDO_RT2_FAST_HOMOLOGY_RULE = "closest_l2_cable_to_left_T2_Fast"


def azevedo_rt2_fast_homology_id(meta):
    """Pick the unlabeled right-T2 fast flexor by homology.

    BANC names tibia_flexor_Fast on five legs, not right T2. Right T2
    has five tibia_flexor cells. This returns the one whose
    l2_cable_length_um is closest to left T2 Fast, same nerve. Frozen
    id is checked so a BANC refresh cannot silently retarget the gain.
    """
    left = meta[
        (meta["cell_type"].astype(str) == "tibia_flexor_Fast")
        & (meta["side"].astype(str) == "left")
        & (meta["neuromere"].astype(str) == "T2")
    ]
    if len(left) != 1:
        raise ValueError(f"expected 1 left T2 Fast, got {len(left)}")
    target = float(left.iloc[0]["l2_cable_length_um"])
    nerve = "right_mesothoracic_leg_nerve"
    right = meta[
        (meta["cell_type"].astype(str) == "tibia_flexor")
        & (meta["side"].astype(str) == "right")
        & (meta["neuromere"].astype(str) == "T2")
        & (meta["nerve"].astype(str) == nerve)
    ]
    if right.empty:
        raise ValueError("no right T2 tibia_flexor on matching nerve")
    d = (right["l2_cable_length_um"].astype(float) - target).abs()
    picked = str(right.loc[d.idxmin()]["banc_888_id"])
    if picked != AZEVEDO_RT2_FAST_HOMOLOGY_ID:
        raise ValueError(
            f"homology pick {picked} != frozen {AZEVEDO_RT2_FAST_HOMOLOGY_ID}")
    return picked


def azevedo_label_overlay(gvec, mn_ids, meta, extra_fast_ids=(),
                          skip_legs=()):
    """Azevedo 2020 force ratios on the 18 BANC-named speed MNs.

    Fast (tibia_flexor_Fast + tibia_extensor_FETi, n=11): 10.0
    Slow (accessory_tibia_flexor_A_slow + tibia_extensor_SETi, n=7):
    0.013. No intermediate label exists in BANC. Every other MN
    keeps its incoming gain (size rank on the operating fly).
    Unmuting unlabeled cells collapsed standing tonight; this
    does not do that. Default-off. Absent = bit-identical.
    extra_fast_ids: homology-supplied Fast cells (right T2). Empty
    default is bit-identical to the named-only overlay.
    skip_legs: (side, neuromere) pairs left at size rank. Used by
    the five-leg control that leaves right T2 untouched.
    """
    out = np.asarray(gvec, dtype=float).copy()
    types = dict(zip(meta["banc_888_id"].astype(str),
                     meta["cell_type"].astype(str)))
    sides = dict(zip(meta["banc_888_id"].astype(str),
                     meta["side"].astype(str)))
    neuromeres = dict(zip(meta["banc_888_id"].astype(str),
                          meta["neuromere"].astype(str)))
    extra = {str(x) for x in extra_fast_ids}
    skip = {(str(s), str(n)) for s, n in skip_legs}
    for k, i in enumerate(mn_ids):
        sid = str(i)
        if (sides.get(sid, ""), neuromeres.get(sid, "")) in skip:
            continue
        name = types.get(sid, "")
        if name in ("tibia_flexor_Fast", "tibia_extensor_FETi") or sid in extra:
            out[k] = 10.0
        elif name in ("accessory_tibia_flexor_A_slow", "tibia_extensor_SETi"):
            out[k] = 0.013
    return out


def roll_mn_force_overlay(gvec, mn_ids, meta):
    """Set anatomical ThC_roll MN gain to 1.0. Size rank otherwise kept.

    Favoured-axis roll muscles are pleural_remotor_and_abductor and
    sternal_adductor (12 MNs). Two front remotor cells sit at size
    gain 0.0013; the rest are already 1.0. Sitters fail on body roll.
    Absent = bit-identical. Not a FETi-scope change.
    """
    import motormap as MM
    mapping = MM.build_map(meta)
    roll = mapping[
        mapping["dof"].astype(str).str.endswith("_ThC_roll")
        & ~mapping["hypothesised"]
    ]
    ids = set(roll["banc_888_id"].astype(str))
    out = np.asarray(gvec, dtype=float).copy()
    for k, i in enumerate(mn_ids):
        if str(i) in ids:
            out[k] = 1.0
    return out


def reductor_force_overlay(gvec, mn_ids, meta):
    """Set femur-reductor (CTr_roll) MN gain to 1.0.

    34 anatomical CTr_roll MNs; 22 sit below size gain 0.1.
    Placement is by elimination (motormap: TrF fused, FANC
    function unknown). Hypothesis, default-off. Absent =
    bit-identical. Not a ThC_roll / roll_force change.
    """
    import motormap as MM
    mapping = MM.build_map(meta)
    red = mapping[
        mapping["dof"].astype(str).str.endswith("_CTr_roll")
        & ~mapping["hypothesised"]
    ]
    ids = set(red["banc_888_id"].astype(str))
    out = np.asarray(gvec, dtype=float).copy()
    for k, i in enumerate(mn_ids):
        if str(i) in ids:
            out[k] = 1.0
    return out


def flexor_force_overlay(gvec, mn_ids, meta):
    """Set anatomical FTi flexor MN gain to 1.0.

    101 tibia_flexor + accessory_tibia_flexor MNs; 64 sit
    below size gain 0.1. Pugliese: tibia flexors stay
    silent or non-rhythmic without proprioception. This
    rig already has FeCO + campaniform; the flexors
    remain size-muted. Hypothesis, default-off. Absent
    = bit-identical. Not a FETi / roll / reductor change.
    """
    import motormap as MM
    mapping = MM.build_map(meta)
    flex = mapping[
        mapping["dof"].astype(str).str.endswith("_FTi_pitch")
        & ~mapping["hypothesised"]
        & mapping["muscle"].astype(str).str.contains("flexor")
    ]
    ids = set(flex["banc_888_id"].astype(str))
    out = np.asarray(gvec, dtype=float).copy()
    for k, i in enumerate(mn_ids):
        if str(i) in ids:
            out[k] = 1.0
    return out


def seti_force_overlay(gvec, mn_ids, meta):
    """Set named SETi gain to 1.0. Size rank otherwise kept.

    Six BANC-named slow tibia extensors. Four sit at
    size gain 0.0013; two are already 1.0. Fast FETi
    is already overlaid. Azevedo 2020: slow units
    hold force. Sitters fail posture. Absent =
    bit-identical. Not a FETi-scope or flexor change.
    """
    out = np.asarray(gvec, dtype=float).copy()
    types = dict(zip(meta["banc_888_id"].astype(str),
                     meta["cell_type"].astype(str)))
    for k, i in enumerate(mn_ids):
        if "SETi" in types.get(str(i), ""):
            out[k] = 1.0
    return out


def tita_force_overlay(gvec, mn_ids, meta):
    """Set anatomical TiTa_pitch MN gain to 1.0.

    78 tarsus MNs; 45 sit below size gain 0.1. Unused
    joint. Plant and stance hypothesis. Absent =
    bit-identical. Not an FTi / roll / reductor /
    flexor / SETi change.
    """
    import motormap as MM
    mapping = MM.build_map(meta)
    tita = mapping[
        mapping["dof"].astype(str).str.endswith("_TiTa_pitch")
        & ~mapping["hypothesised"]
    ]
    ids = set(tita["banc_888_id"].astype(str))
    out = np.asarray(gvec, dtype=float).copy()
    for k, i in enumerate(mn_ids):
        if str(i) in ids:
            out[k] = 1.0
    return out


CTR_EXT_MUSCLES = {
    "tergotrochanter_extensor_muscle",
    "sternotrochanter_extensor_muscle",
    "trochanter_extensor_muscle",
}


def ctr_ext_force_overlay(gvec, mn_ids, meta):
    """Set anatomical CTr_pitch extensor MN gain to 1.0.

    48 levator MNs (tergotrochanter, sternotrochanter,
    trochanter extensor); 28 sit below size gain 0.1.
    Extensor-only, like FETi. Not both CTr antagonists.
    Absent = bit-identical. Not an FTi / TiTa / roll
    / reductor / flexor / SETi change.
    """
    import motormap as MM
    mapping = MM.build_map(meta)
    ext = mapping[
        mapping["dof"].astype(str).str.endswith("_CTr_pitch")
        & ~mapping["hypothesised"]
        & mapping["muscle"].astype(str).isin(CTR_EXT_MUSCLES)
    ]
    ids = set(ext["banc_888_id"].astype(str))
    out = np.asarray(gvec, dtype=float).copy()
    for k, i in enumerate(mn_ids):
        if str(i) in ids:
            out[k] = 1.0
    return out


def apply_muscle_gain(M_fixed, mapping, mn_ids, gains):
    """Apply sparse, bounded per-muscle gains to a fixed-normalised map.

    ``th["muscle_gain"]`` is a mapping from any represented muscle name to a
    relative scale in motormap's fixed physical-plausibility box. Missing
    names remain exactly 1.0, so all 17 coordinates stay separately
    expressible without making every run spell out a 17-entry vector. The
    multiplication follows any per-MN force overlay and the map's baseline
    column normalisation. It therefore preserves exceptional fast/slow
    motor-neuron gains as a separate factor without recomputing a denominator
    that would cancel sole-contributor gains or couple antagonists.

    Absent is handled by the caller. An explicit all-ones mapping returns the
    input array unchanged, which makes the default identity check exact rather
    than tolerance-based.
    """
    import motormap as MM2

    if not isinstance(gains, dict):
        raise ValueError("th['muscle_gain'] must be absent or a dict")
    unknown = sorted(set(gains) - set(MM2.MUSCLE_GAIN_NAMES))
    if unknown:
        raise ValueError(
            f"muscle_gain names unrepresented muscles: {unknown}")

    effective = {name: float(MM2.MUSCLE_GAIN_DEFAULT)
                 for name in MM2.MUSCLE_GAIN_NAMES}
    for name, value in gains.items():
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError("muscle_gain values must be finite numbers")
        value = float(value)
        if not MM2.MUSCLE_GAIN_MIN <= value <= MM2.MUSCLE_GAIN_MAX:
            raise ValueError(
                "muscle_gain values must be within "
                f"[{MM2.MUSCLE_GAIN_MIN}, {MM2.MUSCLE_GAIN_MAX}]")
        effective[name] = value

    # A motor neuron can have several moment-arm rows, but build_map assigns it
    # one peripheral muscle. Assert that invariant before scaling its matrix
    # row, including after an optional mn_transfer rewrite.
    ids = {str(i) for i in mn_ids}
    live = mapping[
        mapping["banc_888_id"].astype(str).isin(ids)
        & mapping["muscle"].notna()
    ]
    muscle_sets = live.groupby(
        live["banc_888_id"].astype(str))["muscle"].agg(
            lambda x: tuple(sorted(set(str(v) for v in x))))
    bad = muscle_sets[muscle_sets.map(len) != 1]
    if len(bad):
        raise ValueError(
            "muscle_gain requires one peripheral muscle per motor neuron")
    muscle_of = {str(k): values[0] for k, values in muscle_sets.items()}
    missing = [str(i) for i in mn_ids if str(i) not in muscle_of]
    if missing:
        raise ValueError(
            f"muscle_gain found {len(missing)} mapped motor neurons without "
            f"a muscle, first {missing[0]!r}")
    unrepresented = sorted(set(muscle_of.values()) - set(effective))
    if unrepresented:
        raise ValueError(
            f"muscle_gain mapping contains unregistered muscles: {unrepresented}")

    row_scale = np.asarray(
        [effective[muscle_of[str(i)]] for i in mn_ids], dtype=np.float64)
    changed = np.flatnonzero(row_scale != MM2.MUSCLE_GAIN_DEFAULT)
    if changed.size:
        out = np.asarray(M_fixed).copy()
        out[changed] *= row_scale[changed, None]
    else:
        out = M_fixed
    changed_counts = {}
    for row in changed:
        muscle = muscle_of[str(mn_ids[int(row)])]
        changed_counts[muscle] = changed_counts.get(muscle, 0) + 1
    matched_counts = {name: 0 for name in MM2.MUSCLE_GAIN_NAMES}
    for muscle in muscle_of.values():
        matched_counts[muscle] += 1
    receipt = {
        "bounds": [float(MM2.MUSCLE_GAIN_MIN),
                   float(MM2.MUSCLE_GAIN_MAX)],
        "default": float(MM2.MUSCLE_GAIN_DEFAULT),
        "specified": {name: float(effective[name]) for name in gains},
        "effective": {name: float(effective[name])
                      for name in MM2.MUSCLE_GAIN_NAMES},
        "represented_coordinate_count": len(MM2.MUSCLE_GAIN_NAMES),
        "matched_motor_neuron_rows": len(mn_ids),
        "matched_rows_by_muscle": matched_counts,
        "changed_motor_neuron_rows": int(changed.size),
        "changed_rows_by_muscle": changed_counts,
        "composition": "multiplies_existing_per_motor_neuron_gain",
        "normalization": "fixed_pre_muscle_column_l1",
    }
    return out, receipt


def finalize_motor_matrix(M_raw, mapping, mn_ids, muscle_gain=None):
    """Freeze the baseline map denominator, then apply physical muscle gain.

    The denominator never sees ``muscle_gain``. This is the sole supported
    ordering because normalising a candidate-scaled column makes a muscle that
    is the only contributor to a DOF mathematically unidentifiable and makes
    antagonist coordinates change one another's delivered coefficients.
    """
    fixed = M_raw / np.maximum(
        np.abs(M_raw).sum(axis=0, keepdims=True), 1.0)
    if muscle_gain is None:
        return fixed, None
    return apply_muscle_gain(fixed, mapping, mn_ids, muscle_gain)


# Compound body_part_sensory labels the exact-leg selector drops.
# BANC names these front_leg_hair_plate_neuron. Joint from the part
# string, not the old "everything on ThC_pitch" assumption.
DROPPED_HP_DOF = {
    "coxa,front_leg": "ThC_pitch",
    "front_leg,trochanter": "CTr_pitch",
}


def abdomen_bristle_channel(meta):
    """781 abdomen bristles, split by BANC side. Default-off.

    Silent when |body roll| <= 16 deg, the same frozen band
    as haltere_rest. Left fires at 5 Hz when roll < -16
    (left-side down under the existing quaternion-to-roll
    formula); right fires when roll > +16. Signed tactile
    righting, not a continuous CHO, not planted-tonic, not
    an unsigned gyro. Not a gain or threshold search.
    """
    cc = meta["cell_class"].fillna("").astype(str)
    bp = meta["body_part_sensory"].fillna("").astype(str)
    side = meta["side"].fillna("").astype(str)
    is_abd = ((bp == "abdomen") & (cc == "bristle_neuron")).to_numpy()
    left = np.flatnonzero(is_abd & (side == "left").to_numpy()).astype(np.int64)
    right = np.flatnonzero(is_abd & (side == "right").to_numpy()).astype(np.int64)
    return left, right


def abdomen_bristle_rates(roll_deg, n_left, n_right):
    """Frozen 5 Hz contact gate, signed by roll. Matches evaluate."""
    hz_l = 5.0 if float(roll_deg) < -16.0 else 0.0
    hz_r = 5.0 if float(roll_deg) > 16.0 else 0.0
    return (np.full(int(n_left), hz_l), np.full(int(n_right), hz_r))


def thorax_bristle_channel(meta):
    """298 thorax bristles. Default-off.

    Silent unless thorax or abdomen geoms carry more
    than 1 N ground contact. Sit-detector, not the
    planted-tonic encoding that collapsed thorax CS.
    Not a roll-gate. Not a gain.
    """
    cc = meta["cell_class"].fillna("").astype(str)
    bp = meta["body_part_sensory"].fillna("").astype(str)
    return np.flatnonzero(
        ((bp == "thorax") & (cc == "bristle_neuron")).to_numpy()
    ).astype(np.int64)


def body_contact_geoms(model):
    """Thorax and abdomen geoms only. Not coxa or shin."""
    import mujoco as mj
    out = set()
    for g in range(model.ngeom):
        nm = (mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if "thorax" in nm or "abdomen" in nm:
            out.add(int(g))
    return out


def body_contact_force(sim, geoms):
    """Sum contact force on the given geoms."""
    import mujoco as mj
    if not geoms:
        return 0.0
    tot = 0.0
    buf = np.zeros(6)
    for c in range(sim.mj_data.ncon):
        con = sim.mj_data.contact[c]
        if con.geom1 in geoms or con.geom2 in geoms:
            mj.mj_contactForce(sim.mj_model, sim.mj_data, c, buf)
            tot += float(np.linalg.norm(buf[:3]))
    return tot


def orphan_club_channel(meta):
    """141 exact-leg orphan neurons, nerve-mapped. Default-off.

    They are not in Proprioceptors (cell_class filter). Exact-leg
    CHO already inherit the club |velocity| law. Orphans get the
    same inherited law on FTi_pitch. Not a contact extra. Not a
    gain.
    """
    import motormap as MM
    from body import FECO_DOF, NERVE_LEG, Proprioceptors as _P  # noqa: F401

    cc = meta["cell_class"].fillna("").astype(str)
    bp = meta["body_part_sensory"].fillna("").astype(str)
    nv = meta["nerve"].fillna("").astype(str)
    dof_ix = {n: i for i, n in enumerate(MM.DOF_NAMES)}
    is_or = (
        bp.isin(["front_leg", "middle_leg", "hind_leg"])
        & (cc == "orphan_neuron")
    )
    rows, dofs = [], []
    for row in np.flatnonzero(is_or.to_numpy()):
        name = nv.iat[int(row)]
        leg = None
        for (sd, seg), code in NERVE_LEG.items():
            if name.startswith(sd) and seg in name:
                leg = code
                break
        if leg is None:
            continue
        key = f"{leg}_{FECO_DOF}"
        if key not in dof_ix:
            continue
        rows.append(int(row))
        dofs.append(dof_ix[key])
    if not rows:
        return None, None
    return (np.asarray(rows, dtype=np.int64),
            np.asarray(dofs, dtype=np.int64))


def orphan_club_rates(bod, dofs):
    """Frozen club |velocity| law. Matches body.Proprioceptors.rates."""
    from body import Proprioceptors as P
    r = P.R_REST + P.R_MAX * np.abs(bod.vel[dofs]) / P.V_SCALE
    return np.clip(r, 0.0, P.R_MAX)


def wing_base_cs_channel(meta):
    """125 wing-base campaniforms. Default-off.

    Phasic roll-velocity, not the |roll| threshold used
    by haltere_rest and not planted load. Frozen club
    law: 5 + 120 * clip(|droll_deg_s| / (V_SCALE in
    deg/s), 0, 1). Not a gain or scale search.
    """
    cc = meta["cell_class"].fillna("").astype(str)
    bp = meta["body_part_sensory"].fillna("").astype(str)
    return np.flatnonzero(
        ((bp == "wing_base") & (cc == "campaniform_sensillum_neuron")).to_numpy()
    ).astype(np.int64)


def body_roll_deg(qpos):
    qw, qx, qy, qz = (float(qpos[k]) for k in range(3, 7))
    return float(np.degrees(np.arctan2(
        2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))))


def wing_cs_rollvel_rate(droll_deg_s):
    """Frozen club |velocity| law in body-roll degrees/s."""
    from body import Proprioceptors as P
    vscale = P.V_SCALE * 180.0 / np.pi
    return float(np.clip(
        P.R_REST + P.R_MAX * abs(float(droll_deg_s)) / vscale,
        0.0, P.R_MAX))


def dropped_hp_channel(meta):
    """52 named coxa/trochanter hair plates. Default-off membership.

    The 217 exact-leg HPs already live in Proprioceptors. These 52
    sit at coxa,front_leg and front_leg,trochanter and never enter
    that selector. Same frozen limit law as body.Proprioceptors
    hair plates (R_REST=5, R_MAX=120, extreme past 0.7). Not a gain.
    """
    import motormap as MM
    from body import NERVE_LEG

    cc = meta["cell_class"].fillna("").astype(str)
    bp = meta["body_part_sensory"].fillna("").astype(str)
    nv = meta["nerve"].fillna("").astype(str)
    dof_ix = {n: i for i, n in enumerate(MM.DOF_NAMES)}
    rows, dofs = [], []
    for part, joint in DROPPED_HP_DOF.items():
        for row in np.flatnonzero(((cc == "hair_plate_neuron")
                                   & (bp == part)).to_numpy()):
            name = nv.iat[int(row)]
            leg = None
            for (sd, seg), code in NERVE_LEG.items():
                if name.startswith(sd) and seg in name:
                    leg = code
                    break
            if leg is None:
                continue
            key = f"{leg}_{joint}"
            if key not in dof_ix:
                continue
            rows.append(int(row))
            dofs.append(dof_ix[key])
    if not rows:
        return None, None
    return (np.asarray(rows, dtype=np.int64),
            np.asarray(dofs, dtype=np.int64))


def dropped_hp_rates(bod, dofs):
    """Frozen hair-plate limit law. Matches body.Proprioceptors.rates."""
    p = bod.normalised()[dofs]
    return 5.0 + 120.0 * np.maximum(0.0, np.abs(2.0 * p - 1.0) - 0.7) / 0.3


def cohp8_channel(meta):
    """16 BANC CxHP8 hair-plate afferents. Default-off membership.

    Nature Communications 2026: CxHP8 / CoHP8 fire at the
    anterior coxa extreme and drive posterior movement
    (swing-to-stance). BANC identifies the organ with the exact
    `CoHP8` token in `peripheral_target_type`: eight afferents per
    front leg, all on the ventral prothoracic nerve. The legacy
    `cell_type == CoHP8` selector found ten rows, but only one was
    in that organ; it omitted fifteen and added nine trochanteral
    or unplaced rows. Every exact organ row rides that front leg's
    ThC_pitch, not CTr and not the both-ends law `hp_dropped` uses.
    Not a gain. Not `hp_dropped`.
    """
    import motormap as MM
    from body import NERVE_LEG

    target = meta["peripheral_target_type"].fillna("").astype(str)
    cell_class = meta["cell_class"].fillna("").astype(str)
    nv = meta["nerve"].fillna("").astype(str)
    exact_organ = target.map(
        lambda value: "CoHP8" in {part.strip() for part in value.split(",")}
    ) & cell_class.eq("hair_plate_neuron")
    dof_ix = {n: i for i, n in enumerate(MM.DOF_NAMES)}
    rows, dofs = [], []
    for row in np.flatnonzero(exact_organ.to_numpy()):
        name = nv.iat[int(row)]
        leg = None
        for (sd, seg), code in NERVE_LEG.items():
            if name.startswith(sd) and seg in name:
                leg = code
                break
        if leg is None:
            continue
        key = f"{leg}_ThC_pitch"
        if key not in dof_ix:
            continue
        rows.append(int(row))
        dofs.append(dof_ix[key])
    if not rows:
        return None, None
    return (np.asarray(rows, dtype=np.int64),
            np.asarray(dofs, dtype=np.int64))


def cohp8_rates(bod, dofs):
    """Anterior-only coxa limit. Frozen R_REST=5, R_MAX=120, 0.7.

    High normalised ThC_pitch is promotion / anterior
    (motormap ThC_pitch +1). Fires only past 0.7 toward 1.
    Not the two-ended |2p-1| law.
    """
    p = bod.normalised()[dofs]
    return 5.0 + 120.0 * np.maximum(0.0, p - 0.7) / 0.3


def hp_row_code_channel(meta, code):
    """Hair-plate rows each on its own joint and side (design 2163ba7d Build 5, H08). Default off.

    `code["rows"]` maps banc_888_id to {"dof": "CTr_pitch" or "ThC_pitch", "side": -1, 0 or +1}. The leg
    is read from the row's nerve, as dropped_hp_channel reads it. A row the channel cannot place raises,
    so the receipt names every listed row. `code["lev_threshold"]` gives each leg's CTr side -1 hinge t
    (default 0.15, lane A's).
    """
    import motormap as MM
    from body import NERVE_LEG

    pos = {b: i for i, b in enumerate(meta["banc_888_id"].astype(str))}
    nv = meta["nerve"].fillna("").astype(str)
    dof_ix = {n: i for i, n in enumerate(MM.DOF_NAMES)}
    t_leg = code.get("lev_threshold", {})
    rows, dofs, sides, hinges, legs = [], [], [], [], []
    for bid, entry in code["rows"].items():
        row = pos.get(str(bid))
        name = "" if row is None else nv.iat[row]
        leg = next((c for (sd, seg), c in NERVE_LEG.items() if name.startswith(sd) and seg in name), None)
        key = f"{leg}_{entry['dof']}"
        if row is None or leg is None or key not in dof_ix or entry["dof"] not in ("CTr_pitch", "ThC_pitch"):
            raise ValueError(f"hp_row_code row {bid} ({name!r}) has no {entry['dof']}")
        rows.append(row)
        dofs.append(dof_ix[key])
        sides.append(float(entry["side"]))
        hinges.append(float(t_leg.get(leg, 0.15)) if entry["dof"] == "CTr_pitch" else 0.15)
        legs.append(leg)
    return {"rows": np.asarray(rows, dtype=np.int64), "dofs": np.asarray(dofs, dtype=np.int64),
            "sides": np.asarray(sides), "hinges": np.asarray(hinges), "legs": legs}


def hp_row_code_rates(p, sides, hinges):
    """side -1: 5 + min(120, 800 max(0, h - p)); side +1: 5 + min(120, 800 max(0, p - 0.85));
    side 0: 5 + 120 max(0, |2p - 1| - 0.7) / 0.3. At h = 0.15 each equals lane A's law
    5 + 120 max(0, side (2p - 1) - 0.7) / 0.3 with p on [0, 1], and side 0 is dropped_hp_rates."""
    minus = 5.0 + np.minimum(120.0, 800.0 * np.maximum(0.0, hinges - p))
    plus = 5.0 + np.minimum(120.0, 800.0 * np.maximum(0.0, p - 0.85))
    both = 5.0 + 120.0 * np.maximum(0.0, np.abs(2.0 * p - 1.0) - 0.7) / 0.3
    return np.where(sides < 0, minus, np.where(sides > 0, plus, both))


def _explicit_gap_pair_rules(meta, rules):
    """Resolve type/side rules to heterogeneous undirected gap pairs.

    Evidence ownership stays with the caller: the engine accepts exact cell
    types, a same/opposite-side rule and one conductance for that relation. It
    never turns qualitative annotation into a universal numeric coupling.
    """
    if not isinstance(rules, list) or not rules:
        raise ValueError("gap='explicit_rules' requires nonempty gap_pair_rules")
    required = {"name", "cell_types", "side_rule", "weight"}
    cell_type = meta["cell_type"].fillna("").astype(str)
    side = meta["side"].fillna("").astype(str).str.lower()
    pair_to_weight = {}
    receipt = []
    names = set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != required:
            raise ValueError(
                "each gap pair rule needs exactly name, cell_types, "
                "side_rule, weight")
        name = rule["name"]
        types = rule["cell_types"]
        side_rule = rule["side_rule"]
        weight = rule["weight"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("gap pair rule names must be unique nonempty strings")
        names.add(name)
        if (not isinstance(types, list) or len(types) != 2
                or not all(isinstance(value, str) and value for value in types)
                or types[0] == types[1]):
            raise ValueError("cell_types must be two distinct nonempty strings")
        if side_rule not in ("same", "opposite"):
            raise ValueError("gap pair side_rule must be 'same' or 'opposite'")
        if (type(weight) not in (int, float) or not np.isfinite(weight)
                or float(weight) <= 0.0):
            raise ValueError("gap pair rule weight must be finite and positive")
        rows_a = np.flatnonzero(cell_type.eq(types[0]).to_numpy())
        rows_b = np.flatnonzero(cell_type.eq(types[1]).to_numpy())
        made = []
        for row_a in rows_a:
            for row_b in rows_b:
                if side.iat[row_a] not in ("left", "right"):
                    continue
                if side.iat[row_b] not in ("left", "right"):
                    continue
                side_match = side.iat[row_a] == side.iat[row_b]
                if side_match != (side_rule == "same"):
                    continue
                pair = (min(int(row_a), int(row_b)),
                        max(int(row_a), int(row_b)))
                if pair in pair_to_weight:
                    raise ValueError(
                        f"gap pair {pair} is selected by more than one rule")
                pair_to_weight[pair] = float(weight)
                made.append(pair)
        if not made:
            raise ValueError(f"gap pair rule {name!r} resolves zero pairs")
        receipt.append({"name": name, "cell_types": list(types),
                        "side_rule": side_rule, "weight": float(weight),
                        "resolved_pairs": len(made)})
    ordered = sorted(pair_to_weight)
    return (tuple(ordered), tuple(pair_to_weight[pair] for pair in ordered),
            tuple(receipt))


def _explicit_gap_pair_ids(meta, pairs):
    """Resolve named (pre, post) BANC ids to DIRECTED gap pairs.

    The rules path above stores every pair as (min, max) of the two row
    indices, so under gap_rectify the presynaptic side is decided by row
    order and is recovered only by re-reading the rule's first cell type.
    That recovery needs a cell type both ends share, and the escape
    topology has none: its eight junctions run from two giant fibres onto
    two TTM motor neurons of DIFFERENT exact types and two PSI, and the
    same substring pool also selects a third, untyped TTM row
    (BANC 720575941460002368, `manc_cell_type` auto:TTMn) that the
    topology excludes. Naming the eight pairs cell by cell states the
    direction in the theta instead of inferring it, and it is checked
    here: unknown id, self-pair or duplicate is an error, never a
    silently smaller junction set. Absent = no pairs = bit-identical.
    """
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("gap='explicit_pairs' requires a nonempty gap_pair_ids")
    row_of = {str(i): r for r, i in enumerate(meta["banc_888_id"].to_numpy())}
    out, seen = [], set()
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2):
            raise ValueError("each gap_pair_ids entry is [pre_id, post_id]")
        pre, post = (str(pair[0]), str(pair[1]))
        if pre not in row_of or post not in row_of:
            missing = [x for x in (pre, post) if x not in row_of]
            raise ValueError(f"gap_pair_ids names rows this meta lacks: {missing}")
        rp, rq = row_of[pre], row_of[post]
        if rp == rq:
            raise ValueError(f"gap_pair_ids contains the self-pair {pre}")
        if (rp, rq) in seen:
            raise ValueError(f"gap_pair_ids repeats the pair {pre} -> {post}")
        seen.add((rp, rq))
        out.append((rp, rq))
    return tuple(out)


def graded_map_plus_19a006_13ba_rows(meta):
    """Union known graded types with the carried premotor hypothesis set.

    ``graded='19a006_13ba'`` predates the evidence map and therefore omits
    known graded L1/L2/APL/HS cells.  The union preserves that locomotor
    hypothesis while satisfying the standing rule that evidence-mapped cells
    may not fall back to spike-gated release merely because another graded
    hypothesis is active.
    """
    import json as _json
    with open("graded_map.json") as handle:
        enabled = set(_json.load(handle).get("enabled", {}))
    hypothesis = {"IN19A006", "IN13B005", "IN13B013", "IN13B010"}
    cell_type = meta["cell_type"].fillna("").astype(str)
    return tuple(int(row) for row in np.flatnonzero(
        cell_type.isin(enabled | hypothesis).to_numpy()))


def graded_map_plus_pugliese_osc_rows(meta):
    """Union known graded types with the Pugliese oscillator hypothesis.

    ``graded='pugliese_osc'`` replaces the evidence map. On R93 that
    drops HS cells the analog motion-class feedback must read as graded.
    The union keeps mapped cells graded and adds IN17A001 / INXXX466 /
    IN16B036 / IN19A007. Hypothesis, default-off, not live enabled.
    """
    import json as _json
    with open("graded_map.json") as handle:
        enabled = set(_json.load(handle).get("enabled", {}))
    hypothesis = {"IN17A001", "INXXX466", "IN16B036", "IN19A007"}
    cell_type = meta["cell_type"].fillna("").astype(str)
    return tuple(int(row) for row in np.flatnonzero(
        cell_type.isin(enabled | hypothesis).to_numpy()))


def serotonin_vnc_5ht7_rows(meta):
    """Resolve and identity-gate the accepted R134 source/target map."""
    map_path = os.path.join(
        os.path.dirname(__file__), "lanes", "E",
        "serotonin-vnc-walking-map.json")
    with open(map_path) as handle:
        record = json.load(handle)
    nt = cns._resolve_nt(meta, cns.SimParams())
    core = ((meta["region"] == "ventral_nerve_cord")
            | (meta["super_class"] == "descending"))
    source_mask = (
        core
        & meta["neurotransmitter_predicted"].eq("serotonin")
        & meta["super_class"].isin(
            ["ascending", "ventral_nerve_cord_intrinsic"])
        & nt.eq("serotonin"))
    target_mask = (
        core
        & meta["hemilineage"].fillna("").eq("08B")
        & nt.eq("acetylcholine"))
    source = np.flatnonzero(source_mask.to_numpy())
    target = np.flatnonzero(target_mask.to_numpy())

    def id_hash(rows):
        ids = sorted(meta.iloc[rows]["banc_888_id"].astype(str), key=int)
        return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()

    source_record = record["source_population"]
    target_record = record["target_population"]
    source_ids = sorted(
        meta.iloc[source]["banc_888_id"].astype(str), key=int)
    if (len(source) != source_record["resolved_rows"]
            or id_hash(source) != source_record["resolved_ids_newline_sha256"]
            or source_ids != source_record["banc_888_ids"]):
        raise ValueError("R134 serotonin source identity gate failed")
    if (len(target) != target_record["resolved_rows"]
            or id_hash(target) != target_record["resolved_ids_newline_sha256"]):
        raise ValueError("R134 5-HT7/08B target identity gate failed")
    return tuple(int(row) for row in source), tuple(int(row) for row in target)


def serotonin_target_set_rows(meta, spec):
    """Design b1746728, Build 2: a named target set in place of the R134 437,
    {name, ids, ids_sha256}, hashed R134's way (the ids sorted as integers,
    joined by newlines, with a trailing newline). Returns the rows in meta
    order and the (name, sha256) the cache key and the receipt carry."""
    if not isinstance(spec, dict) or set(spec) != {"name", "ids", "ids_sha256"}:
        raise ValueError("serotonin_target_set is {name, ids, ids_sha256}")
    name = spec["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("serotonin_target_set name must be a non-empty string")
    if not isinstance(spec["ids"], (list, tuple)):
        raise ValueError("serotonin_target_set ids must be a list")
    ids = [str(x) for x in spec["ids"]]
    if not ids or len(set(ids)) != len(ids) or not all(x.isdigit() for x in ids):
        raise ValueError("serotonin_target_set ids must be distinct integer ids")
    got = hashlib.sha256(
        ("\n".join(sorted(ids, key=int)) + "\n").encode()).hexdigest()
    if got != spec["ids_sha256"]:
        raise ValueError(
            f"serotonin_target_set {name}: the ids hash to {got[:16]}, "
            f"the set names {str(spec['ids_sha256'])[:16]}")
    pos = pd.Series(np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
    missing = [x for x in ids if x not in pos.index]
    if missing:
        raise ValueError(
            f"serotonin_target_set {name}: {len(missing)} ids are absent from meta")
    rows = np.sort(pos.loc[ids].to_numpy())
    return tuple(int(r) for r in rows), (name, got)


def octopamine_set_rows(meta, spec, key):
    """Design 8b6e0441: a named row set for the theta key `key`, {name, ids,
    ids_sha256}, hashed as gpu/a3f66049/rows/sets.json hashes its sets (the
    ids sorted as integers and joined by newlines, no trailing newline).
    Returns the rows in meta order and the (name, sha256) the cache key and
    the receipt carry."""
    if not isinstance(spec, dict) or set(spec) != {"name", "ids", "ids_sha256"}:
        raise ValueError(f"{key} is {{name, ids, ids_sha256}}")
    name = spec["name"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"{key} name must be a non-empty string")
    if not isinstance(spec["ids"], (list, tuple)):
        raise ValueError(f"{key} ids must be a list")
    ids = [str(x) for x in spec["ids"]]
    if not ids or len(set(ids)) != len(ids) or not all(x.isdigit() for x in ids):
        raise ValueError(f"{key} ids must be distinct integer ids")
    got = hashlib.sha256("\n".join(sorted(ids, key=int)).encode()).hexdigest()
    if got != spec["ids_sha256"]:
        raise ValueError(f"{key} {name}: the ids hash to {got[:16]}, "
                         f"the set names {str(spec['ids_sha256'])[:16]}")
    pos = pd.Series(np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
    missing = [x for x in ids if x not in pos.index]
    if missing:
        raise ValueError(f"{key} {name}: {len(missing)} ids are absent from meta")
    rows = np.sort(pos.loc[ids].to_numpy())
    return tuple(int(r) for r in rows), (name, got)


def graded_walking_map_rows(meta):
    """Evidence-map rows whose committed entry participates in walking."""
    import json as _json
    with open("graded_map.json") as handle:
        enabled = _json.load(handle).get("enabled", {})
    walking_types = {
        name for name, record in enabled.items()
        if record.get("in_walking_rig") is True
    }
    cell_type = meta["cell_type"].fillna("").astype(str)
    return tuple(int(row) for row in np.flatnonzero(
        cell_type.isin(walking_types).to_numpy()))


def graded_hs_plus_19a006_13ba_rows(meta):
    """Carried premotor hypothesis union evidence-mapped walking HS rows."""
    hypothesis = {"IN19A006", "IN13B005", "IN13B013", "IN13B010"}
    hs_rows = set(graded_walking_map_rows(meta))
    cell_type = meta["cell_type"].fillna("").astype(str)
    hypothesis_rows = set(int(row) for row in np.flatnonzero(
        cell_type.isin(hypothesis).to_numpy()))
    return tuple(sorted(hs_rows | hypothesis_rows))


def cell_type_patch_for(sel):
    """th["cell_type_patch"]: absent = no patch = bit-identical. "patch" loads the
    judge-owned cell_type_patch.json (per-row identity corrections with a
    citation each) into a dict banc_888_id -> {cell_type, neurotransmitter}
    for cns.SimParams.cell_type_patch. Any other value is an error."""
    import json as _json
    if sel in (None, "", False):
        return None
    if sel != "patch":
        raise ValueError("cell_type_patch must be 'patch' (cell_type_patch.json)")
    cp = _json.load(open("cell_type_patch.json"))
    out = {}
    for rid, entry in (cp.get("enabled") or {}).items():
        if not isinstance(entry, dict) or not entry.get("cell_type"):
            raise ValueError(f"cell_type_patch.json: {rid} has no cell_type")
        if not entry.get("citation"):
            raise ValueError(f"cell_type_patch.json: {rid} has no citation")
        out[str(rid)] = {"cell_type": str(entry["cell_type"]),
                         "neurotransmitter": entry.get("neurotransmitter")}
    if not out:
        raise ValueError("cell_type_patch.json has no enabled entries")
    return out


_META_PATCH_CACHE = {}


def _apply_meta_patch(meta, path):
    """th["meta_patch"] (design 543c8da2 Build 1, study b99fb985): absent = the
    caller's meta object, untouched = bit-identical. Otherwise a committed JSON
    list of per-row label corrections {banc_888_id, field, before, after, flip,
    label, citation}, applied to a copy of meta before any selector, type map
    or side-keyed group reads it. Every entry's `before` must equal the table's
    value, or the run refuses; an entry whose `after` equals its `before` is
    kept for the record and changes nothing. `flip` marks a side flip and is
    true exactly when the field is side and the value changes. The patched
    table is built once per (meta object, file sha256) in a process, so every
    later call reads the same object; the sha256 enters the network cache key.
    Returns (meta, receipt or None)."""
    if path in (None, "", False):
        return meta, None
    fp = path if os.path.isabs(path) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), path)
    with open(fp, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    hit = _META_PATCH_CACHE.get((id(meta), sha))
    if hit is not None and hit[0] is meta:
        return hit[1], hit[2]
    doc = json.loads(raw)
    entries = doc["entries"] if isinstance(doc, dict) else doc
    pos = {s: i for i, s in enumerate(meta["banc_888_id"].astype(str).to_numpy())}
    out = meta.copy(deep=False)
    fields = sorted({str(e["field"]) for e in entries})
    for f in fields:
        if f not in meta.columns:
            raise ValueError(f"meta_patch: {f} is not a meta column")
        out[f] = meta[f].copy()
    applied, kept, flips = 0, 0, 0
    for e in entries:
        rid, f = str(e["banc_888_id"]), str(e["field"])
        if rid not in pos:
            raise ValueError(f"meta_patch: {rid} is not a meta row")
        i = pos[rid]
        cur = meta[f].iat[i]
        cur = None if pd.isna(cur) else str(cur)
        before = None if e.get("before") is None else str(e["before"])
        after = None if e.get("after") is None else str(e["after"])
        if cur != before:
            raise ValueError(f"meta_patch: {rid} {f} is {cur!r}, the file says {before!r}")
        if bool(e.get("flip")) != (f == "side" and after != before):
            raise ValueError(f"meta_patch: {rid} flip must be true exactly for a side change")
        if not e.get("citation"):
            raise ValueError(f"meta_patch: {rid} {f} has no citation")
        if after == before:
            if not e.get("reason"):
                raise ValueError(f"meta_patch: {rid} {f} changes nothing and gives no reason")
            kept += 1
            continue
        if e.get("label") not in ("TRANSFERRED", "HYPOTHESIS"):
            raise ValueError(f"meta_patch: {rid} {f} needs a TRANSFERRED or HYPOTHESIS label")
        out.iloc[i, out.columns.get_loc(f)] = after if after is not None else np.nan
        applied += 1
        flips += bool(e.get("flip"))
    receipt = {"file": str(path), "sha256": sha, "entries": len(entries),
               "applied": applied, "side_flips": flips, "kept_unchanged": kept,
               "fields": fields}
    _META_PATCH_CACHE[(id(meta), sha)] = (meta, out, receipt)
    return out, receipt


def transmitter_map_for(sel):
    """Citation-gated per-cell-type transmitter overrides, from the
    judge-owned transmitter_map.json (measured-structure directive,
    2026-09-03). None/absent = {} = bit-identical; "map" = every entry in
    the file's "enabled" section, as {cell_type: transmitter token}. One
    way only: a lane that wants a subset proposes the file change."""
    if not sel:
        return {}
    if sel != "map":
        raise ValueError("transmitter_map must be 'map' (transmitter_map.json)")
    import json as _json
    tm = _json.load(open("transmitter_map.json"))
    out = {}
    for cell_type, entry in tm.get("enabled", {}).items():
        if not entry.get("citations"):
            raise ValueError(f"transmitter_map.json: {cell_type} has no citation")
        out[str(cell_type)] = str(entry["transmitter"])
    if not out:
        raise ValueError("transmitter_map.json has no enabled entries")
    return out


_BAL_TYPE_LR_CACHE = {}


def balance_type_log_ratio(meta, edges):
    """Centred log of each neuron's exact-type median E/I input ratio.

    Measured, not searched: the excitatory and inhibitory synaptic weight each
    neuron receives, read from the same edge list the network is built from and
    signed by cns.py's own resolution, then the median log ratio of the
    neuron's exact cell type. Neurons whose type has no usable member, and
    neurons with no exact type, get 0.0, which leaves them at the flat target.

    R339: 72.1% of this quantity's variance lies between exact cell types
    against a 17.0% permutation null, so it is a cell-type property and one
    scalar flattens it on 159,472 neurons.
    """
    key = (id(meta), len(edges))
    if key in _BAL_TYPE_LR_CACHE:
        return _BAL_TYPE_LR_CACHE[key]
    import numpy as _np
    _UNK = {"unclear", "unknown", "not_known", "none", "na", "nan",
            "undefined", "unassigned", "ambiguous"}
    _SIGN = {"acetylcholine": +1.0, "glutamate": -1.0, "gaba": -1.0,
             "histamine": -1.0}
    ver, pred = meta["neurotransmitter_verified"], meta["neurotransmitter_predicted"]
    nt = ver.where(ver.notna(), pred).astype(object).map(
        lambda v: "" if v is None or (isinstance(v, float) and _np.isnan(v))
        else str(v).strip().lower())
    sign = nt.map(lambda v: _SIGN.get(
        v, +1.0 if (v == "" or v in _UNK) else 0.0)).to_numpy()

    ec = list(edges.columns)
    pre_c, post_c = ec[0], ec[1]
    w_c = next((c for c in ("syn_count", "count", "weight", "n_syn")
                if c in ec), None)
    ids = meta["banc_888_id"].astype(str).to_numpy()
    row_of = {i: r for r, i in enumerate(ids)}
    pre = edges[pre_c].astype(str).map(row_of).to_numpy()
    post = edges[post_c].astype(str).map(row_of).to_numpy()
    w = (edges[w_c].to_numpy(dtype=_np.float64) if w_c
         else _np.ones(len(edges), dtype=_np.float64))
    ok = ~(pd.isna(pre) | pd.isna(post))
    pre, post, w = pre[ok].astype(_np.int64), post[ok].astype(_np.int64), w[ok]
    sg = sign[pre]
    n = len(meta)
    exc = _np.bincount(post[sg > 0], weights=w[sg > 0], minlength=n)
    inh = _np.bincount(post[sg < 0], weights=w[sg < 0], minlength=n)
    usable = (exc > 0) & (inh > 0)
    lr_row = _np.zeros(n, dtype=_np.float64)
    lr_row[usable] = _np.log(exc[usable] / inh[usable])

    ctype = meta["cell_type"].fillna("").astype(str).to_numpy()
    out = _np.zeros(n, dtype=_np.float64)
    df = pd.DataFrame({"t": ctype[usable], "lr": lr_row[usable]})
    med = df[df["t"] != ""].groupby("t")["lr"].median()
    if len(med):
        mapped = pd.Series(ctype).map(med)
        out = mapped.fillna(0.0).to_numpy(dtype=_np.float64)
    out = out - out.mean()
    _BAL_TYPE_LR_CACHE[key] = out
    return out


# Design e2b1a20d B4: L3 to L5 by exact BANC cell_type (hypothesis 8567af00).
LAMINA_L3_L5_COUNTS = {"L3": 783, "L4": 877, "L5": 1683}
# Design e2b1a20d B5: S, the 5,585 untyped rows FAFB's alignment labels as
# Tm1, Tm2, Tm3, T4 or T5 (hypothesis 767a6d44). The committed list is pinned
# by hash and re-derived from meta at every use.
S5585_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "gpu", "lamina_walk", "s5585.json")
S5585_SHA256 = "0ca1955d256f22855e13bc1701027c948c02be5a3f208930932278a9bfb9432f"
S5585_FAMILIES = {"Tm1": ("Tm1",), "Tm2": ("Tm2",), "Tm3": ("Tm3",),
                  "T4": ("T4a", "T4b", "T4c", "T4d"),
                  "T5": ("T5a", "T5b", "T5c", "T5d")}
S5585_COUNTS = {"Tm1": (588, 42), "Tm2": (702, 47), "Tm3": (721, 34),
                "T4": (1769, 63), "T5": (1517, 102)}


def lamina_l3_l5_rows(meta):
    ct_ = meta["cell_type"].fillna("").astype(str)
    have = {t: int(ct_.eq(t).sum()) for t in LAMINA_L3_L5_COUNTS}
    if have != LAMINA_L3_L5_COUNTS:
        raise ValueError(f"lamina_l3_l5 expects {LAMINA_L3_L5_COUNTS}, meta "
                         f"has {have}")
    return tuple(int(r) for r in np.flatnonzero(
        ct_.isin(tuple(LAMINA_L3_L5_COUNTS)).to_numpy()))


def alignment_s5585(meta):
    """S's rows and its committed list, refused unless the rule re-derives it."""
    import hashlib as _hl
    import json as _json
    raw = open(S5585_PATH, "rb").read()
    if _hl.sha256(raw).hexdigest() != S5585_SHA256:
        raise ValueError(f"{S5585_PATH} does not hash to {S5585_SHA256}")
    doc = _json.loads(raw)
    ct_ = meta["cell_type"].fillna("").astype(str).to_numpy()
    fa_ = meta["fafb_alignment_cell_type"].fillna("").astype(str).to_numpy()
    fam_of = {t: f for f, ts in S5585_FAMILIES.items() for t in ts}
    enabled = set(_json.load(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "graded_map.json")))
        .get("enabled", {}))
    rows = np.flatnonzero((ct_ == "") & np.isin(fa_, list(fam_of))
                          & ~np.isin(ct_, list(enabled)))
    ids = [str(x) for x in meta["banc_888_id"].to_numpy()[rows]]
    if ids != doc["ids"]:
        raise ValueError("alignment_s5585: the rule on this meta does not "
                         "return the committed rows")
    side = meta["side"].astype(str).to_numpy()[rows]
    fam = np.asarray([fam_of[x] for x in fa_[rows]])
    have = {f: (int(((fam == f) & (side == "left")).sum()),
                int(((fam == f) & (side == "right")).sum()))
            for f in S5585_FAMILIES}
    if have != S5585_COUNTS or len(rows) != 5585:
        raise ValueError(f"alignment_s5585 expects {S5585_COUNTS}, meta has {have}")
    return tuple(int(r) for r in rows), doc


def nt_transfer_receipt(net, spec):
    """B5's TRANSFERRED label on every S row, with its resolved transmitter
    before and after the transfer. The hook refuses verified and typed rows,
    so before resolution is the predicted call."""
    import cns as _cns
    import json as _json
    import pandas as _pd
    aud = net.nt_transfer_audit
    rows = np.asarray(aud["rows"], dtype=np.int64)
    doc = _json.loads(open(S5585_PATH, "rb").read())
    before = _pd.Series(aud["predicted_before"], dtype=object)
    after = _pd.Series(np.asarray(net.nt, dtype=object)[rows], dtype=object)

    def _effective(series):
        raw = _cns._sign_of(series, net.p)
        return np.where(np.isnan(raw),
                        float(net.p.sign_unknown) * float(net.p.w_unknown_scale), raw)

    def _cls(x):
        return ("none" if not isinstance(x, str) or not x else
                {1.0: "excitatory", -1.0: "inhibitory", 0.0: "modulatory"}.get(
                    float(net.p.nt_sign.get(x, np.nan)), "other"))

    changed = _effective(before) != _effective(after)
    btok = [x if isinstance(x, str) else "none" for x in before]
    return {
        "label": "TRANSFERRED", "label_source": spec["label_source"],
        "transmitter": spec["transmitter"], "citation": spec["citation"],
        "transmitter_source": "verified acetylcholine on the family's typed rows",
        "ids_sha256": spec["ids_sha256"], "n_rows": int(len(rows)),
        "family_counts": {f: int(sum(1 for x in doc["family"] if x == f))
                          for f in S5585_FAMILIES},
        "resolved_before_counts": {k: int(v) for k, v in
                                   _pd.Series(btok).value_counts().items()},
        "resolved_after_counts": {k: int(v) for k, v in
                                  _pd.Series(after).fillna("none").value_counts().items()},
        "sign_changed": int(changed.sum()),
        "sign_changed_from": {k: int(v) for k, v in _pd.Series(
            [f"{_cls(b)}:{t}" for b, t, c in zip(before, btok, changed) if c]
        ).value_counts().items()},
        "resolved_before": btok,
        "resolved_after": [str(x) for x in after],
    }


def sorted_ids_sha256(meta, rows):
    """sha256 of the rows' banc_888_ids, sorted as integers, one per line."""
    import hashlib as _hl
    ids = sorted(int(x) for x in meta["banc_888_id"].to_numpy()[list(rows)])
    return _hl.sha256("\n".join(str(x) for x in ids).encode()).hexdigest()


def graded_rows_for(sel, meta):
    """Rows a graded selector names, as a tuple of core-row indices.

    Lifted out of evaluate() unchanged (lane A, 2026-09-02) so a second
    selector can extend a theta's graded set: th["graded_add"] names rows
    added beside th["graded"], and they carry th["graded_gain_add"] while
    the base rows keep th["graded_gain"]. A champion's graded rows are
    then untouched by the extension, which is what a paired test needs.
    A list of selectors returns their union in order (design e2b1a20d B4),
    so lanes/judge/graded_guard.py reads the set evaluate builds.
    """
    if isinstance(sel, (list, tuple)):
        seen, out = set(), []
        for one in sel:
            if not isinstance(one, str):
                raise ValueError(f"graded selector list holds {one!r}")
            for r in graded_rows_for(one, meta):
                if r not in seen:
                    seen.add(r)
                    out.append(r)
        return tuple(out)
    graded_rows = ()
    if sel == "lamina_l3_l5":
        graded_rows = lamina_l3_l5_rows(meta)
    elif sel == "alignment_s5585":
        graded_rows = alignment_s5585(meta)[0]
    elif sel == "hyp":
        # CEO-approved 2026-08-19: cross-species HYPOTHESIS set -- the
        # locust/stick-insect non-spiking premotor homologues, by
        # hemilineage (13A, 13B, 09A). Matches the hemilineage column
        # where present (BANC), else hemilineage-bearing type names
        # (MANC: IN13A001 etc). Results carry the HYPOTHESIS label.
        import json as _json
        _hl_set = _json.load(open("graded_map.json"))[
            "enabled_cross_species_hypothesis"]["hemilineages"]
        _src = (meta["hemilineage"].astype(str) if "hemilineage" in meta
                else meta["cell_type"].astype(str))
        _m = _src.str.contains("|".join(_hl_set), na=False, regex=True)
        graded_rows = tuple(int(r) for r in np.flatnonzero(_m.to_numpy()))
    elif sel == "map":
        import json as _json
        gm = _json.load(open("graded_map.json"))
        types = set(gm.get("enabled", {}))
        ct_ = meta["cell_type"].fillna("").astype(str)
        graded_rows = tuple(int(r) for r in
                            np.flatnonzero(ct_.isin(types).to_numpy()))
    elif sel == "walking_map":
        graded_rows = graded_walking_map_rows(meta)
    elif sel == "19a006":
        # HYPOTHESIS, default-off. IN19A006 is the intersegmental
        # projection type that SPEAKS to the neighbor (76.9% outgoing)
        # and is DEAF in spikes (~52 Hz tonic). Not adult-Drosophila
        # release evidence. Pugliese used a rate model because walking
        # premotor cells are often non-spiking. Results carry HYPOTHESIS.
        ct_ = meta["cell_type"].fillna("").astype(str)
        graded_rows = tuple(int(r) for r in
                            np.flatnonzero(ct_.eq("IN19A006").to_numpy()))
    elif sel == "map_19a006_13ba":
        graded_rows = graded_map_plus_19a006_13ba_rows(meta)
    elif sel == "walking_map_19a006_13ba":
        # Walking-rig evidence rows (6 HS) plus ng1's 24. Default-off.
        # Attribution control for map_19a006_13ba (3376).
        graded_rows = graded_hs_plus_19a006_13ba_rows(meta)
    elif sel == "19a006_13ba":
        # HYPOTHESIS, default-off. Keep analog IN19A006 (the R1i
        # wait) and add the three 13B types that actually receive
        # claw FeCO input (IN13B005/013/010, 18 GABA cells, all
        # six neuromeres). Agrawal 2020: 13Bα is nonspiking, encodes
        # FTi angle, and changes tibia posture. BANC has no 13Bα
        # name; membership is the measured claw-hearing 13B types,
        # not the 440-cell 13B dump. No graded-gain search.
        ct_ = meta["cell_type"].fillna("").astype(str)
        graded_rows = tuple(int(r) for r in np.flatnonzero(
            ct_.isin(("IN19A006", "IN13B005", "IN13B013", "IN13B010")).to_numpy()
        ))
    elif sel == "19a006_13ba_disinhibition":
        # HYPOTHESIS, default-off. Syed et al. 2025 eLife RP106446 map a
        # 13B-4i -> 13A disinhibitory route that releases flexor MNs. BANC
        # has no exact 13B-4i cross-dataset match, but its six IN13B004 rows
        # are the frozen type-level homology candidate: the right-T1 row is
        # the only local 13B candidate with both claw input and output onto
        # mapped flexor-inhibiting 13A targets. Keep the earlier 24 rows and
        # add only this six-row, equal-twin-tested hypothesis.
        ct_ = meta["cell_type"].fillna("").astype(str)
        graded_rows = tuple(int(r) for r in np.flatnonzero(
            ct_.isin(("IN19A006", "IN13B005", "IN13B013", "IN13B010",
                      "IN13B004")).to_numpy()
        ))
    elif sel == "pugliese_wait":
        # HYPOTHESIS, default-off. IN19A006 plus Pugliese E1/E2/I1
        # (IN17A001, INXXX466, IN16B036). 19A006 alone writes R1i and
        # leaves tip cadence at 3-5 Hz. This asks whether analog local
        # CPG plus the wait cell also writes real-fly cadence.
        ct_ = meta["cell_type"].fillna("").astype(str)
        graded_rows = tuple(int(r) for r in np.flatnonzero(
            ct_.isin(("IN19A006", "IN17A001", "INXXX466", "IN16B036")).to_numpy()
        ))
    elif sel == "19a006_e1e2":
        # HYPOTHESIS, default-off. IN19A006 plus Pugliese E1/E2 only
        # (IN17A001, INXXX466). The full motif including I1 wrote
        # cadence and killed R1i. E1/E2 are the necessary oscillator;
        # I1 is not unique. This asks whether dropping I1 keeps
        # neighbor turns while still writing tempo.
        ct_ = meta["cell_type"].fillna("").astype(str)
        graded_rows = tuple(int(r) for r in np.flatnonzero(
            ct_.isin(("IN19A006", "IN17A001", "INXXX466")).to_numpy()
        ))
    elif sel == "19a006_t2":
        # HYPOTHESIS, default-off. Only the two IN19A006 cells in T2
        # (left and right), the hub whose neighbor set is T1 and T3.
        ct_ = meta["cell_type"].fillna("").astype(str)
        nm = meta["neuromere"].fillna("").astype(str) if "neuromere" in meta else ct_.map(lambda _: "")
        graded_rows = tuple(int(r) for r in np.flatnonzero(
            (ct_.eq("IN19A006") & nm.eq("T2")).to_numpy()
        ))
    elif sel == "pugliese_osc":
        # HYPOTHESIS, default-off. Pugliese/Tuthill oscillator types.
        ct_ = meta["cell_type"].fillna("").astype(str)
        graded_rows = tuple(int(r) for r in np.flatnonzero(
            ct_.isin(("IN17A001", "INXXX466", "IN16B036", "IN19A007")).to_numpy()
        ))
    elif sel == "map_pugliese_osc":
        graded_rows = graded_map_plus_pugliese_osc_rows(meta)
    return graded_rows


def graded_hs_dna02_direct_edge_plan(meta, edges):
    """Resolve the six measured, same-side HS -> DNa02 BANC edges.

    This is anatomy only.  The returned IDs name existing chemical edges; the
    C-V68 conductance hypothesis changes neither membership nor any of the six
    synapse-count ratios. Counts and identities are frozen to BANC v888 so a
    dataset or annotation drift cannot silently retarget the mechanism.
    """
    required_meta = {
        "banc_888_id", "cell_type", "side", "super_class",
        "neurotransmitter_verified",
    }
    required_edges = {"pre", "post", "count"}
    if missing := required_meta.difference(meta.columns):
        raise ValueError(
            f"HS-DNa02 edge plan missing metadata columns: {sorted(missing)}")
    if missing := required_edges.difference(edges.columns):
        raise ValueError(
            f"HS-DNa02 edge plan missing edge columns: {sorted(missing)}")
    cell_type = meta["cell_type"].fillna("").astype(str)
    side = meta["side"].fillna("").astype(str)
    super_class = meta["super_class"].fillna("").astype(str)
    hs = meta[
        cell_type.isin(_OPTOMOTOR_HS_TYPES)
        & super_class.eq("visual_projection")
        & side.isin(("left", "right"))
    ]
    dna = meta[
        cell_type.eq("DNa02")
        & super_class.eq("descending")
        & side.isin(("left", "right"))
    ]
    expected_hs = sorted(
        (which, flank) for flank in ("left", "right")
        for which in _OPTOMOTOR_HS_TYPES)
    observed_hs = sorted(zip(
        hs["cell_type"].astype(str), hs["side"].astype(str)))
    if observed_hs != expected_hs:
        raise ValueError(
            f"HS-DNa02 edge plan HS identity drift: {observed_hs}")
    if sorted(dna["side"].astype(str).tolist()) != ["left", "right"]:
        raise ValueError("HS-DNa02 edge plan needs one DNa02 per side")
    if not hs["neurotransmitter_verified"].fillna("").astype(
            str).str.lower().eq("acetylcholine").all():
        raise ValueError(
            "HS-DNa02 edge plan HS source lacks verified acetylcholine")

    hs_by_key = {
        (str(row.cell_type), str(row.side)): row.banc_888_id
        for row in hs.itertuples(index=False)
    }
    dna_by_side = {
        str(row.side): row.banc_888_id
        for row in dna.itertuples(index=False)
    }
    expected_ids = {
        ("HSS", "left"): "720575941433747415",
        ("HSN", "left"): "720575941561098222",
        ("HSE", "left"): "720575941515173196",
        ("HSS", "right"): "720575941574535240",
        ("HSN", "right"): "720575941446781354",
        ("HSE", "right"): "720575941501878667",
        ("DNa02", "left"): "720575941510475536",
        ("DNa02", "right"): "720575941456897005",
    }
    observed_ids = {
        key: str(value) for key, value in hs_by_key.items()
    }
    observed_ids.update({
        ("DNa02", flank): str(value)
        for flank, value in dna_by_side.items()
    })
    if observed_ids != expected_ids:
        raise ValueError(
            f"HS-DNa02 edge plan stable-ID drift: {observed_ids}")
    expected_counts = {
        ("HSS", "left"): 12,
        ("HSN", "left"): 3,
        ("HSE", "left"): 1,
        ("HSS", "right"): 29,
        ("HSN", "right"): 5,
        ("HSE", "right"): 8,
    }
    hs_ids = {str(value) for value in hs_by_key.values()}
    dna_ids = {str(value) for value in dna_by_side.values()}
    induced = edges[
        edges["pre"].astype(str).isin(hs_ids)
        & edges["post"].astype(str).isin(dna_ids)
    ]
    induced_counts = induced["count"].to_numpy(dtype=np.float64)
    if (not np.all(np.isfinite(induced_counts))
            or np.any(induced_counts <= 0.0)
            or not np.array_equal(induced_counts, np.rint(induced_counts))):
        raise ValueError("HS-DNa02 induced edge count is not a positive integer")
    expected_induced = {
        (expected_ids[(which, flank)], expected_ids[("DNa02", flank)], count)
        for (which, flank), count in expected_counts.items()
    }
    observed_induced = {
        (str(row.pre), str(row.post), int(row.count))
        for row in induced.itertuples(index=False)
    }
    if len(induced) != 6 or observed_induced != expected_induced:
        raise ValueError(
            f"HS-DNa02 induced edge-set drift: {sorted(observed_induced)}")
    pairs = []
    records = []
    for flank in ("left", "right"):
        for which in _OPTOMOTOR_HS_TYPES:
            pre_id = hs_by_key[(which, flank)]
            post_id = dna_by_side[flank]
            selected = edges[
                edges["pre"].eq(pre_id) & edges["post"].eq(post_id)]
            selected_counts = selected["count"].to_numpy(dtype=np.float64)
            if (not np.all(np.isfinite(selected_counts))
                    or np.any(selected_counts <= 0.0)
                    or not np.array_equal(
                        selected_counts, np.rint(selected_counts))):
                raise ValueError(
                    f"HS-DNa02 direct edge count is not a positive integer "
                    f"for {which}/{flank}")
            count = float(selected_counts.sum())
            expected = expected_counts[(which, flank)]
            if len(selected) != 1 or count != expected or count <= 0:
                raise ValueError(
                    "HS-DNa02 direct edge drift for "
                    f"{which}/{flank}: rows={len(selected)}, count={count}, "
                    f"expected={expected}")
            pair = (str(pre_id), str(post_id))
            pairs.append(pair)
            records.append({
                "source_cell_type": which,
                "side": flank,
                "pre_banc_888_id": pair[0],
                "post_banc_888_id": pair[1],
                "synapse_count": int(count),
            })
    if len(set(pairs)) != 6 or sum(row["synapse_count"] for row in records) != 58:
        raise ValueError("HS-DNa02 direct edge plan aggregate drift")
    return tuple(pairs), {
        "pair_count": 6,
        "total_synapse_count": 58,
        "left_synapse_count": 16,
        "right_synapse_count": 42,
        "pairs": records,
    }


def feco_13b_analog_plan(meta, edges, mode, permutation_seed):
    """Replace claw->13B chemical edges with a continuous analog boundary.

    Agrawal et al. 2020 found nonspiking 13Balpha responses survive nicotinic
    and muscarinic blockade and proposed electrical coupling from claw
    afferents as one explanation. BANC does not trace gap junctions, so the
    existing chemical contacts provide the anatomical pair/relative-weight
    proxy. Each target receives the count-weighted mean of its claw partners'
    body-encoded rates. The permuted arm preserves sources, weights, degrees,
    and changed-row count while replacing only developmental identity.
    """
    if mode not in ("structured_13b", "permuted_13b"):
        raise ValueError("FeCO-13B analog mode must be structured or permuted")
    required_meta = {"banc_888_id", "cell_type", "cell_sub_class",
                     "hemilineage"}
    if not required_meta.issubset(meta.columns):
        raise ValueError("FeCO-13B analog plan needs BANC identity annotations")
    if not {"pre", "post", "count"}.issubset(edges.columns):
        raise ValueError("FeCO-13B analog plan needs pre/post/count edges")
    cell_type = meta["cell_type"].fillna("").astype(str)
    subclass = meta["cell_sub_class"].fillna("").astype(str)
    actual_rows = np.flatnonzero(cell_type.isin(
        ("IN13B005", "IN13B010", "IN13B013")).to_numpy())
    if len(actual_rows) != 18:
        raise ValueError(f"FeCO-13B analog expected 18 targets, got {len(actual_rows)}")
    claw_rows = np.flatnonzero(subclass.str.contains(
        "claw_chordotonal_organ_neuron", regex=False).to_numpy())
    ids = meta["banc_888_id"].to_numpy()
    claw_ids = set(ids[claw_rows].tolist())
    actual_ids = set(ids[actual_rows].tolist())
    edge_mask = (edges["pre"].isin(claw_ids).to_numpy()
                 & edges["post"].isin(actual_ids).to_numpy())
    edge_positions = np.flatnonzero(edge_mask)
    selected = edges.iloc[edge_positions]
    if len(selected) == 0 or np.any(selected["count"].to_numpy() <= 0):
        raise ValueError("FeCO-13B analog resolved no positive contacts")
    row_by_id = {identifier: row for row, identifier in enumerate(ids)}
    source_rows = np.asarray(
        [row_by_id[value] for value in selected["pre"]], dtype=np.int64)
    original_target_rows = np.asarray(
        [row_by_id[value] for value in selected["post"]], dtype=np.int64)
    if mode == "structured_13b":
        target_map = {int(row): int(row) for row in actual_rows}
    else:
        premotor_rows = graded_rows_for("19a006_13ba", meta)
        _, permuted_rows = graded_premotor_vrest_overrides(
            meta, premotor_rows, "permuted_13b", -52.0,
            permutation_seed)
        target_map = dict(zip(sorted(int(row) for row in actual_rows),
                              sorted(int(row) for row in permuted_rows)))
    target_rows_per_pair = np.asarray(
        [target_map[int(row)] for row in original_target_rows], dtype=np.int64)
    counts = selected["count"].to_numpy(dtype=np.float64)
    totals = {}
    for row, count in zip(original_target_rows, counts):
        totals[int(row)] = totals.get(int(row), 0.0) + float(count)
    weights = np.asarray([
        count / totals[int(row)]
        for row, count in zip(original_target_rows, counts)], dtype=np.float64)
    drive_rows = np.unique(target_rows_per_pair)
    slot_by_row = {int(row): slot for slot, row in enumerate(drive_rows)}
    target_slots = np.asarray(
        [slot_by_row[int(row)] for row in target_rows_per_pair], dtype=np.int64)
    for original_row in np.unique(original_target_rows):
        if not np.isclose(weights[original_target_rows == original_row].sum(),
                          1.0, atol=1e-12):
            raise ValueError("FeCO-13B analog target weights do not sum to one")
    return {
        "mode": mode,
        "source_rows": source_rows,
        "target_rows_per_pair": target_rows_per_pair,
        "target_slots": target_slots,
        "weights": weights,
        "drive_rows": drive_rows,
        "drop_edge_positions": edge_positions,
        "edge_pairs": int(len(selected)),
        "chemical_synapses_replaced": int(selected["count"].sum()),
        "claw_source_rows": int(len(np.unique(source_rows))),
        "actual_13b_rows": int(len(actual_rows)),
        "driven_target_rows": int(len(drive_rows)),
    }


def graded_13b_command_rows(meta, mode, permutation_seed):
    """Left-T1 13B rows or their equal-count frozen label permutation.

    Agrawal et al. 2020 stimulated the left prothoracic 13Balpha population.
    BANC has no literal 13Balpha type, so the structured hypothesis uses the
    two left-T1 rows among the claw-hearing IN13B005/010/013 set. The twin
    applies the frozen 18-of-24 19A/13B permutation and preserves two rows.
    """
    if mode not in ("structured_13b_t1_left", "permuted_13b_t1_left"):
        raise ValueError("unknown graded 13B command mode")
    required = {"cell_type", "hemilineage", "neuromere", "side"}
    if not required.issubset(meta.columns):
        raise ValueError("graded 13B command needs identity annotations")
    cell_type = meta["cell_type"].fillna("").astype(str)
    actual_all = np.flatnonzero(cell_type.isin(
        ("IN13B005", "IN13B010", "IN13B013")).to_numpy())
    actual = np.flatnonzero((
        cell_type.isin(("IN13B005", "IN13B010", "IN13B013"))
        & meta["hemilineage"].fillna("").astype(str).eq("13B")
        & meta["neuromere"].fillna("").astype(str).eq("T1")
        & meta["side"].fillna("").astype(str).eq("left")).to_numpy())
    if len(actual_all) != 18 or len(actual) != 2:
        raise ValueError(
            f"graded 13B command expected 18/3 rows, got "
            f"{len(actual_all)}/{len(actual)}")
    if mode == "structured_13b_t1_left":
        return tuple(int(row) for row in np.sort(actual))
    premotor_rows = graded_rows_for("19a006_13ba", meta)
    _, permuted = graded_premotor_vrest_overrides(
        meta, premotor_rows, "permuted_13b", -52.0, permutation_seed)
    mapping = dict(zip(sorted(int(row) for row in actual_all),
                       sorted(int(row) for row in permuted)))
    rows = tuple(sorted(mapping[int(row)] for row in actual))
    if len(rows) != 2 or len(set(rows)) != 2:
        raise ValueError("graded 13B command permutation changed cardinality")
    return rows


def graded_13b_gain_vector(
        meta, base_gain, gain, mode="structured_13b", permutation_seed=0):
    """Per-row graded gain on 13B or its equal-count label permutation."""
    base_gain, gain = float(base_gain), float(gain)
    if (not np.isfinite(base_gain) or base_gain <= 0.0
            or not np.isfinite(gain) or gain <= 0.0):
        raise ValueError("graded 13B gains must be finite and positive")
    if mode == "structured_13b":
        cell_type = meta["cell_type"].fillna("").astype(str)
        rows = np.flatnonzero(cell_type.isin(
            ("IN13B005", "IN13B010", "IN13B013")).to_numpy())
    elif mode == "permuted_13b":
        premotor_rows = graded_rows_for("19a006_13ba", meta)
        _, rows = graded_premotor_vrest_overrides(
            meta, premotor_rows, mode, -52.0, permutation_seed)
        rows = np.asarray(rows, dtype=np.int64)
    else:
        raise ValueError("unknown graded 13B gain mode")
    if len(rows) != 18:
        raise ValueError(f"graded_13b_gain expected 18 rows, got {len(rows)}")
    vector = np.full(len(meta), base_gain, dtype=np.float32)
    vector[rows] = gain
    return vector, tuple(int(row) for row in rows)


def hemilineage_graded_parameter_vectors(
        meta, graded_rows, base_v50, base_slope, delta_mv, slope_ratio,
        permutation_seed=None):
    """Tie the 19A/13B graded curves by developmental hemilineage.

    ``delta_mv`` is V50(19A) - V50(13B), while ``slope_ratio`` is
    slope(19A) / slope(13B).  The cell-weighted V50 mean and geometric-mean
    slope remain at the crowned scalar values, so the dials change only the
    biologically motivated between-hemilineage structure.  Non-graded rows
    retain the anchors but never read these vectors.
    """
    try:
        base_v50 = float(base_v50)
        base_slope = float(base_slope)
        delta_mv = float(delta_mv)
        slope_ratio = float(slope_ratio)
    except (TypeError, ValueError) as error:
        raise ValueError("hemilineage graded parameters must be numeric") from error
    if (not np.isfinite(base_v50) or not np.isfinite(base_slope)
            or base_slope <= 0.0 or not np.isfinite(delta_mv)
            or not -8.0 <= delta_mv <= 8.0
            or not np.isfinite(slope_ratio)
            or not 0.25 <= slope_ratio <= 4.0):
        raise ValueError("hemilineage graded parameters outside bounds")
    rows = np.asarray(graded_rows, dtype=np.int64)
    if (rows.ndim != 1 or not len(rows) or np.any(rows < 0)
            or np.any(rows >= len(meta)) or len(np.unique(rows)) != len(rows)):
        raise ValueError("graded rows must be unique valid meta rows")
    lineage = meta["hemilineage"].fillna("").astype(str).to_numpy()[rows]
    if permutation_seed is not None:
        try:
            permutation_seed = int(permutation_seed)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "graded hemilineage permutation seed must be an integer"
            ) from error
        # Equal-capacity attribution control: preserve the exact 18/6 group
        # sizes and both parameter values, but break their association with
        # the biological 13B/19A labels.
        lineage = np.random.default_rng(permutation_seed).permutation(lineage)
    is_19a, is_13b = lineage == "19A", lineage == "13B"
    if (not is_19a.any() or not is_13b.any()
            or not np.all(is_19a | is_13b)):
        raise ValueError("hemilineage graded mode requires only 19A and 13B")
    frac_19a = float(is_19a.mean())
    v50_19a = base_v50 + (1.0 - frac_19a) * delta_mv
    v50_13b = base_v50 - frac_19a * delta_mv
    slope_19a = base_slope * slope_ratio ** (1.0 - frac_19a)
    slope_13b = base_slope * slope_ratio ** (-frac_19a)
    v50 = np.full(len(meta), base_v50, dtype=np.float32)
    slope = np.full(len(meta), base_slope, dtype=np.float32)
    v50[rows[is_19a]], v50[rows[is_13b]] = v50_19a, v50_13b
    slope[rows[is_19a]], slope[rows[is_13b]] = slope_19a, slope_13b
    if (not np.isclose(float(v50[rows].mean()), base_v50, atol=2e-6)
            or not np.isclose(float(np.exp(np.log(slope[rows]).mean())),
                              base_slope, atol=2e-6)
            or np.any(slope <= 0.0)):
        raise ValueError("hemilineage graded anchors drifted")
    return v50, slope


def graded_premotor_vrest_overrides(
        meta, graded_rows, mode, value_mv, permutation_seed):
    """Resting-potential overlay for ng1's 13B graded premotor rows.

    ``structured_13b`` selects the 13B-labelled rows. ``permuted_13b``
    assigns the same value to the same number of rows after a frozen label
    permutation within the identical 19A/13B graded population.
    ``structured_19a`` and ``matched_13b005`` are an equal-six-row contrast
    between the two cell types implicated by that permutation. Each twin has
    the same parameter count, changed-row count, and magnitude; only
    developmental identity differs.

    The caller owns the literature prior. R148 uses only the four resting
    baselines printed beside adult 13Balpha recordings in Agrawal et al. 2020
    Fig. 2E/F (-51, -50, -46, -44 mV); this helper does not invent a global
    plausible bracket around them.
    """
    if mode not in ("structured_13b", "permuted_13b",
                    "structured_19a", "matched_13b005"):
        raise ValueError("unknown graded premotor v_rest mode")
    try:
        value_mv = float(value_mv)
        permutation_seed = int(permutation_seed)
    except (TypeError, ValueError) as error:
        raise ValueError("graded premotor v_rest inputs must be numeric") from error
    if not np.isfinite(value_mv):
        raise ValueError("graded premotor v_rest value must be finite")
    rows = np.asarray(graded_rows, dtype=np.int64)
    if (rows.ndim != 1 or not len(rows) or np.any(rows < 0)
            or np.any(rows >= len(meta)) or len(np.unique(rows)) != len(rows)):
        raise ValueError("graded premotor rows must be unique valid meta rows")
    lineage = meta["hemilineage"].fillna("").astype(str).to_numpy()[rows]
    structured = lineage == "13B"
    if (not structured.any() or not np.any(lineage == "19A")
            or not np.all(np.isin(lineage, ("19A", "13B")))):
        raise ValueError("graded premotor v_rest requires only 19A and 13B")
    n_selected = int(structured.sum())
    if mode == "structured_13b":
        selected = rows[structured]
    elif mode == "permuted_13b":
        rng = np.random.default_rng(np.random.SeedSequence(
            [permutation_seed, 0x13505245]))
        selected = np.sort(rng.choice(rows, size=n_selected, replace=False))
        structured_rows = np.sort(rows[structured])
        if np.array_equal(selected, structured_rows):
            selected = np.sort(np.roll(rows, 1)[:n_selected])
    elif mode == "structured_19a":
        selected = rows[lineage == "19A"]
        if len(selected) != 6:
            raise ValueError(
                f"structured_19a expected 6 rows, got {len(selected)}")
    else:
        cell_type = meta["cell_type"].fillna("").astype(str).to_numpy()[rows]
        selected = rows[cell_type == "IN13B005"]
        if (len(selected) != 6
                or not np.all(lineage[cell_type == "IN13B005"] == "13B")):
            raise ValueError(
                f"matched_13b005 expected 6 13B rows, got {len(selected)}")
    n_selected = len(selected)
    ids = meta.iloc[selected]["banc_888_id"].tolist()
    overrides = {banc_id: value_mv for banc_id in ids}
    if len(overrides) != n_selected:
        raise ValueError("graded premotor v_rest identity collision")
    return overrides, tuple(int(row) for row in selected)


def graded_premotor_empirical_vrest_overrides(
        meta, graded_rows, mode, values_mv, assignment_seed,
        permutation_seed):
    """Assign an empirical v_rest sample across 13B or permuted twin rows.

    Both arms receive the same multiset of published values and the same
    number of changed rows. ``assignment_seed`` freezes which selected neuron
    receives which measurement; ``permutation_seed`` freezes only the label
    twin's membership.
    """
    if mode not in ("structured_13b_empirical", "permuted_13b_empirical"):
        raise ValueError("invalid empirical graded premotor v_rest mode")
    values = np.asarray(values_mv, dtype=np.float64)
    if (values.ndim != 1 or not len(values)
            or not np.all(np.isfinite(values))):
        raise ValueError("empirical graded premotor v_rest values invalid")
    scalar_mode = ("structured_13b" if mode == "structured_13b_empirical"
                   else "permuted_13b")
    _, selected_rows = graded_premotor_vrest_overrides(
        meta, graded_rows, scalar_mode, float(values[0]), permutation_seed)
    selected = np.asarray(selected_rows, dtype=np.int64)
    assigned = np.resize(values, len(selected)).copy()
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(assignment_seed), 0x454D5052]))
    rng.shuffle(assigned)
    ids = meta.iloc[selected]["banc_888_id"].tolist()
    overrides = {banc_id: float(value)
                 for banc_id, value in zip(ids, assigned)}
    if (len(overrides) != len(selected)
            or sorted(overrides.values()) != sorted(assigned.tolist())):
        raise ValueError("empirical graded premotor assignment drift")
    return overrides, tuple(int(row) for row in selected)


_ENGINE_KEYS = None
_THETA_META_KEYS = {"id", "lineage", "winner", "theta", "raw_winner"}


# Design 2163ba7d, Build 1: the pulse keys beside pulse_exc_ids and their
# defaults (None = required); the ids and their hash have none.
_PULSE_EXC_KEYS = (("pulse_exc_g", None), ("pulse_on_ms", None), ("pulse_off_ms", None),
                   ("pulse_start_ms", None), ("pulse_stop_ms", None),
                   ("pulse_tau_open_ms", 24.6), ("pulse_tau_close_ms", 24.6))


def _pulse_exc_values(th):
    """Design 2163ba7d, Build 1: the pulse keys as floats, or a ValueError
    naming a required key the theta leaves out."""
    miss = [k for k, d in _PULSE_EXC_KEYS if d is None and th.get(k) is None]
    if miss:
        raise ValueError(f"pulse_exc_ids needs {miss}")
    return {k: float(th[k] if th.get(k) is not None else d) for k, d in _PULSE_EXC_KEYS}


def _pulse_exc_sets(th):
    """Design 568d05a8, Build B5: pulse_exc_sets, a list of {ids, ids_sha256, g},
    one opsin set per entry at its own conductance, sharing the pulse keys
    above except pulse_exc_g. Returns (sets, schedule): each set's ids as
    strings, its hash and g as a float, and the shared keys as floats; a
    ValueError on a malformed entry, a hash that differs, an id in two sets,
    a missing schedule key, or pulse_exc_ids or pulse_exc_g beside it."""
    if th.get("pulse_exc_ids") is not None or th.get("pulse_exc_g") is not None:
        raise ValueError("pulse_exc_sets replaces pulse_exc_ids and pulse_exc_g; give one form")
    raw = th["pulse_exc_sets"]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("pulse_exc_sets must be a non-empty list of {ids, ids_sha256, g}")
    sets = []
    for j, s in enumerate(raw):
        if not isinstance(s, dict) or set(s) != {"ids", "ids_sha256", "g"}:
            raise ValueError(f"pulse_exc_sets[{j}] must be {{ids, ids_sha256, g}}")
        ids = [str(x) for x in s["ids"]]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"pulse_exc_sets[{j}]: ids must be a non-empty list of distinct ids")
        sha = hashlib.sha256("\n".join(sorted(ids, key=int)).encode()).hexdigest()
        if s["ids_sha256"] != sha:
            raise ValueError(f"pulse_exc_sets[{j}]: ids hash {sha} differs from "
                             f"ids_sha256 {s['ids_sha256']!r}")
        sets.append({"ids": ids, "ids_sha256": sha, "g": float(s["g"])})
    every = [x for s in sets for x in s["ids"]]
    if len(set(every)) != len(every):
        raise ValueError("pulse_exc_sets: an id is in more than one set")
    keys = [(k, d) for k, d in _PULSE_EXC_KEYS if k != "pulse_exc_g"]
    miss = [k for k, d in keys if d is None and th.get(k) is None]
    if miss:
        raise ValueError(f"pulse_exc_sets needs {miss}")
    return sets, {k: float(th[k] if th.get(k) is not None else d) for k, d in keys}


def _opsin_tap_carried(p):
    """Design 4ee8b7f3, Build B1: whether the engine p builds records the
    opsin tap. The Mac's Metal library carries the conductance on its pulse
    train and refuses the tap (gpu/cns_engine.m, design 244fb801 Build 1);
    numpy and the CUDA library behind backend "metal" off the Mac carry it."""
    return not (str(getattr(p, "backend", "numpy")) == "metal"
                and os.uname().sysname == "Darwin")


# Design 2163ba7d, Build 2: slow_inh_out's fields, exactly these.
_SLOW_INH_FIELDS = ("frac", "ids", "tau_ms")

# Design 1e8f1a64, Build B1 (study 48570d4d): th["jo_rest"], the resting
# Johnston's organ encoder. The selection names its organ rows and their
# sorted-integer-id sha256 (newline joined): BANC's chordotonal rows whose
# subclass names Johnston's organ, or MCNS's rows typed "JO-".
_JO_REST_FIELDS = ("A_hz", "nonA_hz", "A_sigma_log", "A_floor_hz", "draw_seed",
                   "A_extra_ids", "coupled_left", "coupled_right")
_JO_REST_OPTIONAL = ("A_sigma_log_by_group", "rows")
# The BANC hash is the computed one: the design's text carries
# ...4719afd5... at characters 21-24, a one-character slip of this string
# (row count, subclass counts and the critique's 16-character prefix agree).
_JO_REST_ROWS = {
    "banc_johnston": (1193, "9b8a5487dcccab584719acd5b39bba16d7afb553a23d6f4adfbd364906d27a8d"),
    "mcns_jo_prefix": (672, "c65a6de3c23e73178c5fd3278c72ff541c9f5b83b6d4a4489e583d75f6621913"),
}


def _jo_ids_sha256(ids):
    return hashlib.sha256("\n".join(
        str(x) for x in sorted(int(i) for i in ids)).encode()).hexdigest()


def jo_rest_plan(meta, spec):
    """Design 1e8f1a64, Build B1: th["jo_rest"] as the organ's rows (meta
    order), one fixed rate per row and the receipt, or a ValueError.

    A-law rows are the JO-A rows plus spec["A_extra_ids"]; the others take
    nonA_hz. The A-law rows fall in groups, coupled left, coupled right (the
    spec's lists, in their order) and the other A rows (sorted by integer
    id); on MCNS, whose organ has no coupled list, the groups are the A rows
    by side. In a group of m rows q_i = (i - 0.5) / m, z_i = Phi^-1(q_i),
    X_i = exp(s z_i) / mean_j exp(s z_j) and the rates are A_hz * X_i, so the
    group's arithmetic mean is A_hz; one permutation per group from
    numpy.random.default_rng(draw_seed), groups in the order above, assigns
    the k-th row rates[perm[k]]. s is the group's entry in
    A_sigma_log_by_group when present, else A_sigma_log, and 0 when A_hz is
    at or below 10 Hz. A group whose s > 0 puts its lowest rate under
    A_floor_hz is refused: the floor is a build check, never a truncation."""
    from scipy.stats import norm as _norm
    if not isinstance(spec, dict):
        raise ValueError("jo_rest must be a dict")
    miss = [k for k in _JO_REST_FIELDS if k not in spec]
    extra = [k for k in spec if k not in _JO_REST_FIELDS + _JO_REST_OPTIONAL]
    if miss or extra:
        raise ValueError(f"jo_rest needs exactly {_JO_REST_FIELDS} plus optional "
                         f"{_JO_REST_OPTIONAL}; missing {miss}, unknown {extra}")
    num = {}
    for k in ("A_hz", "nonA_hz", "A_sigma_log", "A_floor_hz"):
        v = spec[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError(f"jo_rest {k} must be a finite number >= 0")
        num[k] = float(v)
    seed = spec["draw_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("jo_rest draw_seed must be an integer >= 0")
    mode = spec.get("rows", "banc_johnston")
    if mode not in _JO_REST_ROWS:
        raise ValueError(f"jo_rest rows must be one of {tuple(_JO_REST_ROWS)}")
    ids_all = meta["banc_888_id"].astype(np.int64).to_numpy()
    if mode == "banc_johnston":
        cc = meta["cell_class"].fillna("").astype(str)
        sub = meta["cell_sub_class"].fillna("").astype(str)
        is_jo = ((cc == "chordotonal_organ_neuron")
                 & sub.str.lower().str.contains("johnston")).to_numpy()
        is_a = is_jo & (sub == "johnstons_organ_A_neuron").to_numpy()
        label = sub.to_numpy()
    else:
        ct = meta["cell_type"].fillna("").astype(str)
        is_jo = ct.str.startswith("JO-").to_numpy()
        is_a = is_jo & ct.str.startswith("JO-A").to_numpy()
        label = ct.to_numpy()
    rows = np.flatnonzero(is_jo).astype(np.int64)
    n_want, sha_want = _JO_REST_ROWS[mode]
    ids_sha = _jo_ids_sha256(ids_all[rows])
    if len(rows) != n_want or ids_sha != sha_want:
        raise ValueError(f"jo_rest {mode}: {len(rows)} rows, sha256 {ids_sha[:16]}; "
                         f"the design's are {n_want}, {sha_want[:16]}")
    pos = {int(i): int(r) for i, r in zip(ids_all[rows], rows)}

    def _rows_of(name):
        v = spec[name]
        if not isinstance(v, list) or any(not isinstance(x, (str, int)) or isinstance(x, bool)
                                          for x in v):
            raise ValueError(f"jo_rest {name} must be a list of ids")
        r = []
        for x in v:
            if int(x) not in pos:
                raise ValueError(f"jo_rest {name}: {x} is not an organ row")
            r.append(pos[int(x)])
        if len(set(r)) != len(r):
            raise ValueError(f"jo_rest {name} repeats an id")
        return np.asarray(r, dtype=np.int64)
    extra_rows = _rows_of("A_extra_ids")
    if is_a[extra_rows].any():
        raise ValueError("jo_rest A_extra_ids names a row that is already A")
    law = is_a.copy()
    law[extra_rows] = True
    left, right = _rows_of("coupled_left"), _rows_of("coupled_right")
    if np.intersect1d(left, right).size or not law[np.concatenate([left, right])].all():
        raise ValueError("jo_rest coupled lists must be disjoint A-law rows")
    law_rows = np.flatnonzero(law)
    if mode == "banc_johnston":
        other = np.setdiff1d(law_rows, np.concatenate([left, right]))
        other = other[np.argsort(ids_all[other], kind="stable")]
        groups = [("left", left), ("right", right), ("other", other)]
    else:
        if len(left) or len(right) or len(extra_rows):
            raise ValueError("jo_rest on MCNS has no coupled lists and no extra ids")
        side = meta["side"].fillna("").astype(str).to_numpy()
        groups = []
        for g in ("left", "right"):
            r = law_rows[side[law_rows] == g]
            groups.append((g, r[np.argsort(ids_all[r], kind="stable")]))
        if sum(len(r) for _, r in groups) != len(law_rows):
            raise ValueError("jo_rest on MCNS: an A row has no side")
    by_group = spec.get("A_sigma_log_by_group") or {}
    names = [g for g, _ in groups]
    if not isinstance(by_group, dict) or any(k not in names for k in by_group):
        raise ValueError(f"jo_rest A_sigma_log_by_group keys must be among {names}")
    rate = np.full(len(meta), num["nonA_hz"], dtype=np.float64)
    nominal = rate.copy()
    group_of = np.full(len(meta), "nonA", dtype=object)
    rng = np.random.default_rng(seed)
    receipt_groups = []
    for g, r in groups:
        s = float(by_group.get(g, num["A_sigma_log"]))
        if not np.isfinite(s) or s < 0:
            raise ValueError(f"jo_rest sigma for {g} must be finite and >= 0")
        if num["A_hz"] <= 10.0:
            s = 0.0
        m = len(r)
        z = _norm.ppf((np.arange(1, m + 1) - 0.5) / m) if m else np.empty(0)
        x = np.exp(s * z)
        q = num["A_hz"] * x / x.mean() if m else x
        perm = rng.permutation(m)
        if s > 0 and m and q.min() < num["A_floor_hz"]:
            raise ValueError(f"jo_rest group {g}: sigma {s} puts the lowest rate at "
                             f"{q.min():.4f} Hz, under the floor {num['A_floor_hz']}")
        rate[r] = q[perm]
        nominal[r] = num["A_hz"]
        group_of[r] = g
        receipt_groups.append({"group": g, "rows": int(m), "sigma_log": s,
                               "lowest_hz": float(q.min()) if m else None,
                               "highest_hz": float(q.max()) if m else None,
                               "mean_hz": float(q.mean()) if m else None})
    rates = rate[rows]
    receipt = {
        "design": "1e8f1a64", "rows_mode": mode, "spec": json.loads(json.dumps(spec)),
        "n_rows": int(len(rows)), "ids_sha256": ids_sha,
        "rates_sha256": hashlib.sha256(np.ascontiguousarray(rates, dtype="<f8").tobytes()).hexdigest(),
        "a_law_rows": int(len(law_rows)), "groups": receipt_groups,
        "per_row": [{"id": str(ids_all[r]), "label": str(label[r]),
                     "side": str(meta["side"].iat[int(r)]), "group": str(group_of[r]),
                     "nominal_hz": float(nominal[r]), "rate_hz": float(rate[r])} for r in rows],
    }
    return rows, rates, receipt


# Design 0bec278a (study 34bd411b): the labellar and pharyngeal taste encoder.
# The three classes, the row pins of each selection (count, sha256 of the
# sorted integer ids, newline joined) and, for MCNS, whose cell_class reads
# bristle_neuron on 161 of these rows, the class and function by cell_type
# (section 1's table; LB3, untyped, goes to sugar and is labelled so).
_LABELLAR_CLASSES = ("taste_bristle_gustatory_neuron", "taste_peg_gustatory_neuron",
                     "internal_taste_sensillum_gustatory_neuron")
_LABELLAR_ROWS = {
    "banc": (340, "419c18401553a53adcef2d447bdbf4fb36cd00ec20ef2f58907b99302d7c9a30"),
    "mcns_type": (273, "947a50252d6f8280ee87feab59af6cf0cf167528223655a92cf005f637c23028"),
}
_LABELLAR_MCNS_TYPES = {
    **{t: ("taste_bristle_gustatory_neuron", "bitter, Gr33a") for t in ("LB1a", "LB1b", "LB1c", "LB1d")},
    "LB1e": ("taste_bristle_gustatory_neuron", "amino_acids, Ir94e"),
    **{t: ("taste_bristle_gustatory_neuron", "aversive") for t in ("LB2a", "LB2b", "LB2c")},
    **{t: ("taste_bristle_gustatory_neuron", "water, ppk28") for t in ("LB2d", "LB3a")},
    "LB3": ("taste_bristle_gustatory_neuron", "sugar, LB3 untyped (labelled)"),
    "LB3b": ("taste_bristle_gustatory_neuron", "sugar, low_salt, Gr64f, Ir56b"),
    "LB3c": ("taste_bristle_gustatory_neuron", "sugar, Gr64f"),
    "LB3d": ("taste_bristle_gustatory_neuron", "heavy_metal, Ir47a"),
    **{t: ("taste_bristle_gustatory_neuron", "unannotated") for t in ("LB4a", "LB4b")},
    "claw_tpGRN": ("taste_peg_gustatory_neuron", "fatty_acids, Ir56d, Gr64e"),
    "dorsal_tpGRN": ("taste_peg_gustatory_neuron", "sugar, Gr5a, Ir60d"),
}
_LABELLAR_STIM_FIELDS = ("rows", "onsets_ms", "contact_ms", "bins_hz", "label")


def labellar_rest_plan(meta, spec):
    """Design 0bec278a: th["labellar_rest"] as the encoder's rows (meta order),
    one resting rate per row and the receipt, or a ValueError.

    spec = {"rates_hz": {cell_class: {cell_function_detailed or "default":
    spikes/s}}, "rows": "banc" (absent) or "mcns_type"}. On BANC the rows are
    super_class sensory, cell_class one of the three, cell_sub_class naming
    labellum or pharynx; the function is cell_function_detailed, blank read as
    "unannotated". On MCNS they are cell_type LB1a to LB4b, claw_tpGRN,
    dorsal_tpGRN and PhG*, class and function by type. Every class needs a
    default, and a function key no selected row carries is refused, so a
    misspelt function cannot fall back to the default unseen."""
    if not isinstance(spec, dict) or set(spec) - {"rates_hz", "rows"} or "rates_hz" not in spec:
        raise ValueError("labellar_rest is {'rates_hz': {...}} plus an optional 'rows'")
    mode = spec.get("rows", "banc")
    if mode not in _LABELLAR_ROWS:
        raise ValueError(f"labellar_rest rows must be one of {tuple(_LABELLAR_ROWS)}")
    ct = meta["cell_type"].fillna("").astype(str)
    if mode == "banc":
        if "cell_function_detailed" not in meta.columns:
            raise ValueError("labellar_rest rows 'banc' reads cell_function_detailed, which this meta lacks; "
                             "a male graph takes rows 'mcns_type'")
        cc = meta["cell_class"].fillna("").astype(str)
        sub = meta["cell_sub_class"].fillna("").astype(str)
        sel = ((meta["super_class"].fillna("").astype(str) == "sensory") & cc.isin(_LABELLAR_CLASSES)
               & (sub.str.contains("labellum") | sub.str.contains("pharynx"))).to_numpy()
        fn = meta["cell_function_detailed"].fillna("").astype(str).replace("", "unannotated")
        klass, func = cc.to_numpy(), fn.to_numpy()
    else:
        phg = ct.str.fullmatch(r"PhG\d+[a-c]?").to_numpy()
        sel = ct.isin(list(_LABELLAR_MCNS_TYPES)).to_numpy() | phg
        klass = np.asarray([_LABELLAR_MCNS_TYPES.get(t, ("internal_taste_sensillum_gustatory_neuron",))[0]
                            for t in ct], dtype=object)
        func = np.asarray([_LABELLAR_MCNS_TYPES[t][1] if t in _LABELLAR_MCNS_TYPES else "by type, PhG"
                           for t in ct], dtype=object)
    rows = np.flatnonzero(sel).astype(np.int64)
    ids_all = meta["banc_888_id"].astype(np.int64).to_numpy()
    n_want, sha_want = _LABELLAR_ROWS[mode]
    ids_sha = _jo_ids_sha256(ids_all[rows])
    if len(rows) != n_want or ids_sha != sha_want:
        raise ValueError(f"labellar_rest {mode}: {len(rows)} rows, sha256 {ids_sha[:16]}; "
                         f"the design's are {n_want}, {sha_want[:16]}")
    table = spec["rates_hz"]
    if not isinstance(table, dict) or set(table) != set(_LABELLAR_CLASSES):
        raise ValueError(f"labellar_rest rates_hz needs exactly the classes {_LABELLAR_CLASSES}")
    rate = np.empty(len(rows), dtype=np.float64)
    for c in _LABELLAR_CLASSES:
        tc = table[c]
        if not isinstance(tc, dict) or "default" not in tc:
            raise ValueError(f"labellar_rest rates_hz[{c!r}] needs a 'default'")
        for k, v in tc.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise ValueError(f"labellar_rest rate {c}/{k} must be a finite number >= 0")
        in_c = klass[rows] == c
        have = set(func[rows][in_c])
        stray = sorted(set(tc) - have - {"default"})
        if stray:
            raise ValueError(f"labellar_rest rates_hz[{c!r}] names functions no row carries: {stray}")
        rate[in_c] = [float(tc.get(f, tc["default"])) for f in func[rows][in_c]]
    by = {}
    for c, f, r in zip(klass[rows], func[rows], rate):
        e = by.setdefault(c, {}).setdefault(f, {"rows": 0, "rate_hz": float(r)})
        e["rows"] += 1
    receipt = {
        "design": "0bec278a", "rows_mode": mode, "spec": json.loads(json.dumps(spec)),
        "n_rows": int(len(rows)), "ids_sha256": ids_sha,
        "rates_sha256": hashlib.sha256(np.ascontiguousarray(rate, dtype="<f8").tobytes()).hexdigest(),
        "rest_sum_hz": float(rate.sum()), "by_class": by,
        "ids": [str(i) for i in ids_all[rows]]}
    return rows, rate, receipt


# Design a146ad7f (study 7285a98d), Build B1: th["gust_rest"] as a per-leg
# rate table. The six legs, the leg a row's side and body part name, the keys.
_GUST_LEGS = ("lf", "rf", "lm", "rm", "lh", "rh")
_GUST_BODY_LEG = {("left", "front_leg"): "lf", ("right", "front_leg"): "rf",
                  ("left", "middle_leg"): "lm", ("right", "middle_leg"): "rm",
                  ("left", "hind_leg"): "lh", ("right", "hind_leg"): "rh"}
_GUST_KEYS = ("rates_hz", "rows", "leg_fallback", "load_n", "n_rows", "ids_sha256")


def gust_rest_plan(meta, spec):
    """Design a146ad7f, Build B1: th["gust_rest"] as a dict. Returns the rows
    by leg and one rate per row (legs in first-row order, meta order within a
    leg, as the true path builds them), the load a leg's rows fire above, and
    the receipt; or a ValueError.

    spec = {"rates_hz": {leg: {cell_function_detailed or "default":
    spikes/s}} for the six legs, "rows": "banc", "leg_fallback": "drop" or
    "body_part_side", "load_n": the contact load, "n_rows" and "ids_sha256"
    of the driven rows (sorted integer ids, newline joined)}. The rows are the
    true path's: taste_bristle_gustatory_neuron on the three leg body parts,
    the leg by NERVE_LEG on the nerve; a row no nerve places is dropped, or
    under "body_part_side" placed by side and body_part_sensory. A blank
    function reads "unannotated". Every leg needs a default, and a function
    key no driven row of that leg carries is refused, so a misspelt function
    cannot fall back to the default unseen."""
    if not isinstance(spec, dict) or set(spec) != set(_GUST_KEYS):
        raise ValueError(f"gust_rest is true or a dict with exactly the keys {_GUST_KEYS}")
    if spec["rows"] != "banc":
        raise ValueError("gust_rest rows must be 'banc'")
    if "cell_function_detailed" not in meta.columns:
        raise ValueError("gust_rest rows 'banc' reads cell_function_detailed, which this meta lacks")
    mode = spec["leg_fallback"]
    if mode not in ("drop", "body_part_side"):
        raise ValueError("gust_rest leg_fallback must be 'drop' or 'body_part_side'")
    load_n = spec["load_n"]
    if isinstance(load_n, bool) or not isinstance(load_n, (int, float)) or not np.isfinite(load_n) or load_n < 0:
        raise ValueError("gust_rest load_n must be a finite number >= 0")
    if isinstance(spec["n_rows"], bool) or not isinstance(spec["n_rows"], int) \
            or not isinstance(spec["ids_sha256"], str):
        raise ValueError("gust_rest n_rows is an integer and ids_sha256 a hex string")
    table = spec["rates_hz"]
    if not isinstance(table, dict) or set(table) != set(_GUST_LEGS):
        raise ValueError(f"gust_rest rates_hz needs exactly the legs {_GUST_LEGS}")
    for leg in _GUST_LEGS:
        tl = table[leg]
        if not isinstance(tl, dict) or "default" not in tl:
            raise ValueError(f"gust_rest rates_hz[{leg!r}] needs a 'default'")
        for k, v in tl.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise ValueError(f"gust_rest rate {leg}/{k} must be a finite number >= 0")
    from body import NERVE_LEG
    cc = meta["cell_class"].fillna("").astype(str)
    bp = meta["body_part_sensory"].fillna("").astype(str)
    nv = meta["nerve"].fillna("").astype(str)
    sd = meta["side"].fillna("").astype(str)
    fn = meta["cell_function_detailed"].fillna("").astype(str).replace("", "unannotated").to_numpy()
    ids_all = meta["banc_888_id"].astype(np.int64).to_numpy()
    sel = bp.isin(["front_leg", "middle_leg", "hind_leg"]) & (cc == "taste_bristle_gustatory_neuron")
    by_leg, fallback, disagree = {}, [], []
    for row in np.flatnonzero(sel.to_numpy()):
        row = int(row)
        name = nv.iat[row]
        leg = None
        for (s, seg), code in NERVE_LEG.items():
            if name.startswith(s) and seg in name:
                leg = code
                break
        leg_sb = _GUST_BODY_LEG.get((sd.iat[row], bp.iat[row]))
        if leg is None:
            if mode != "body_part_side" or leg_sb is None:
                continue
            leg = leg_sb
            fallback.append({"id": str(ids_all[row]), "leg": leg, "nerve": name,
                             "label": f"leg by body part and side; nerve column {name}"})
        elif leg_sb is not None and leg_sb != leg:
            disagree.append({"id": str(ids_all[row]), "leg_by_nerve": leg, "leg_by_body_part_side": leg_sb,
                             "label": "leg by nerve; body part and side disagree"})
        by_leg.setdefault(leg, []).append(row)
    rows = np.sort(np.asarray([r for rl in by_leg.values() for r in rl], dtype=np.int64))
    ids_sha = _jo_ids_sha256(ids_all[rows])
    if len(rows) != spec["n_rows"] or ids_sha != spec["ids_sha256"]:
        raise ValueError(f"gust_rest {mode}: {len(rows)} rows, sha256 {ids_sha}; "
                         f"the theta pins {spec['n_rows']}, {spec['ids_sha256']}")
    rows_by_leg, rates_by_leg, by = {}, {}, {}
    rate = np.empty(len(rows), dtype=np.float64)
    for leg in _GUST_LEGS:
        tl = table[leg]
        rl = np.asarray(by_leg.get(leg, []), dtype=np.int64)
        stray = sorted(set(tl) - set(fn[rl]) - {"default"})
        if stray:
            raise ValueError(f"gust_rest rates_hz[{leg!r}] names functions no row of that leg carries: {stray}")
        if not len(rl):
            continue
        r = np.asarray([float(tl.get(f, tl["default"])) for f in fn[rl]], dtype=np.float64)
        rate[np.searchsorted(rows, rl)] = r
        for f, x in zip(fn[rl], r):
            e = by.setdefault(leg, {}).setdefault(str(f), {"rows": 0, "rate_hz": float(x)})
            e["rows"] += 1
    for leg in by_leg:
        rows_by_leg[leg] = np.asarray(by_leg[leg], dtype=np.int64)
        rates_by_leg[leg] = rate[np.searchsorted(rows, rows_by_leg[leg])]
    receipt = {
        "design": "a146ad7f", "spec": json.loads(json.dumps(spec)),
        "n_rows": int(len(rows)), "ids_sha256": ids_sha,
        "ids_serialisation": "sha256 of the driven rows' banc_888_id as decimal integers, sorted, "
                             "newline joined, no trailing newline (_jo_ids_sha256)",
        "rates_sha256": hashlib.sha256(np.ascontiguousarray(rate, dtype="<f8").tobytes()).hexdigest(),
        "rates_order": "the driven rows in meta row order, little-endian float64",
        "rest_sum_hz": float(rate.sum()), "by_leg": by,
        "fallback_rows": fallback, "nerve_vs_body_part_rows": disagree,
        "load_n": float(load_n), "leg_fallback": mode,
        "ids": [str(i) for i in ids_all[rows]]}
    return rows_by_leg, rates_by_leg, float(load_n), receipt


def labellar_stimulus_plan(meta, lab_rows, stim, tick_ms, dur_ms, force_spikes=None, run_force=None,
                           what="labellar_stimulus"):
    """Design 0bec278a and amendment 1ec8cb82: evaluate's labellar_stimulus as
    a per-tick table, or a ValueError.

    stim = {"rows": [ids], "onsets_ms": [...], "contact_ms", "bins_hz": [...],
    "label", and exactly one of "bin_ms" or "bin_edges_ms"}. Inside contact p,
    at drive tick time t, a stimulus row's rate is the bin's rate, the bin
    being floor((t - onset_p) / bin_ms) or the edge interval holding
    t - onset_p; elsewhere its resting rate. Every time is a whole number of
    control ticks, the contacts do not overlap and end inside the run, the
    rows are encoder rows, and neither force_spikes schedule names one. what
    names the caller's argument in each refusal (design a146ad7f's
    tarsal_stimulus validates its schedules here, on the gust_rest rows)."""
    from fractions import Fraction as _F
    if not isinstance(stim, dict):
        raise ValueError(f"{what} must be a dict")
    has_ms, has_edges = "bin_ms" in stim, "bin_edges_ms" in stim
    if has_ms == has_edges:
        raise ValueError(f"{what} takes exactly one of bin_ms and bin_edges_ms")
    extra = set(stim) - set(_LABELLAR_STIM_FIELDS) - {"bin_ms", "bin_edges_ms"}
    miss = [k for k in _LABELLAR_STIM_FIELDS if k not in stim]
    if extra or miss:
        raise ValueError(f"{what}: missing {miss}, unknown {sorted(extra)}")
    if not isinstance(stim["label"], str) or not stim["label"]:
        raise ValueError(f"{what} needs its label")

    def _ticks(name, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError(f"{what} {name} must be a finite time >= 0")
        q = _F(str(float(v))) / _F(str(float(tick_ms)))
        if q.denominator != 1:
            raise ValueError(f"{what} {name} {v} ms is not a whole number of {tick_ms} ms ticks")
        return int(q)
    contact = _ticks("contact_ms", stim["contact_ms"])
    hz = stim["bins_hz"]
    if (not isinstance(hz, list) or not hz or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                    or not np.isfinite(x) or x < 0 for x in hz)):
        raise ValueError(f"{what} bins_hz must be a non-empty list of rates >= 0")
    if has_ms:
        width = _ticks("bin_ms", stim["bin_ms"])
        if width == 0 or width * len(hz) != contact:
            raise ValueError(f"{what} bins_hz times bin_ms must equal contact_ms")
        edges = [width * i for i in range(len(hz) + 1)]
    else:
        e = stim["bin_edges_ms"]
        if not isinstance(e, list) or len(e) != len(hz) + 1:
            raise ValueError(f"{what} bin_edges_ms needs one more entry than bins_hz")
        edges = [_ticks("bin_edges_ms", x) for x in e]
        if edges[0] != 0 or edges[-1] != contact or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError(f"{what} bin_edges_ms rise strictly from 0 to contact_ms")
    on = stim["onsets_ms"]
    if not isinstance(on, list) or not on:
        raise ValueError(f"{what} onsets_ms must be a non-empty list")
    onsets = [_ticks("onsets_ms", x) for x in on]
    n_ticks = _ticks("dur_ms", dur_ms)
    if any(b < a + contact for a, b in zip(onsets, onsets[1:])) or onsets[-1] + contact > n_ticks:
        raise ValueError(f"{what} contacts must rise without overlap and end inside the run")
    ids_all = meta["banc_888_id"].astype(np.int64).to_numpy()
    pos_of = {int(ids_all[r]): j for j, r in enumerate(lab_rows)}
    if (not isinstance(stim["rows"], list)
            or any(isinstance(x, bool) or not isinstance(x, (str, int)) for x in stim["rows"])):
        raise ValueError(f"{what} rows must be a list of ids")
    want = [int(x) for x in stim["rows"]]
    if not want or len(set(want)) != len(want):
        raise ValueError(f"{what} rows must be distinct and not empty")
    outside = [x for x in want if x not in pos_of]
    if outside:
        raise ValueError(f"{what} names {len(outside)} rows outside the encoder, first {outside[0]}")
    if force_spikes and {int(b) for b in force_spikes} & set(want):
        raise ValueError(f"{what} and force_spikes name the same rows")
    stim_rows = {int(lab_rows[pos_of[x]]) for x in want}
    if run_force and any(stim_rows & {int(r) for r in rs} for rs in run_force.values()):
        raise ValueError(f"{what} and run_kwargs force_spikes name the same rows")
    rate_by_tick = np.full(n_ticks, np.nan)
    for o in onsets:
        for b, r in enumerate(hz):
            rate_by_tick[o + edges[b]:o + edges[b + 1]] = float(r)
    pos = np.asarray(sorted(pos_of[x] for x in want), dtype=np.int64)
    trace_id = min(want)
    return {"schedule": json.loads(json.dumps(stim)),
            "schedule_sha256": hashlib.sha256(json.dumps(
                stim, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "tick_ms": float(tick_ms), "rate_by_tick": rate_by_tick, "pos": pos,
            "trace_id": str(trace_id), "trace_pos": int(pos_of[trace_id]),
            "ids_sha256": _jo_ids_sha256(want), "log": []}


def labellar_stimulus_receipt(plan):
    """The stimulus receipt: the schedule and its sha256, and per drive call the
    tick, the trace row's rate and the lowest and highest rate over every
    stimulus row (equal on a call where every row follows the schedule)."""
    k, tr, lo, hi = (np.asarray(c) for c in zip(*plan["log"])) if plan["log"] else ([],) * 4
    inside = ~np.isnan(plan["rate_by_tick"])
    return {"design": "0bec278a", "schedule": plan["schedule"], "schedule_sha256": plan["schedule_sha256"],
            "canonical": "json.dumps(schedule, sort_keys=True, separators=(',', ':'))",
            "tick_ms": plan["tick_ms"], "rows": int(len(plan["pos"])), "ids_sha256": plan["ids_sha256"],
            "contact_ticks": int(inside.sum()), "trace_id": plan["trace_id"],
            "per_call": {"tick": np.asarray(k, dtype=np.int64), "trace_hz": np.asarray(tr, dtype=np.float64),
                         "min_hz": np.asarray(lo, dtype=np.float64), "max_hz": np.asarray(hi, dtype=np.float64)}}


# Design 1e20a949, Build C1: sez_rest's fields, and the fixed constant its own
# generator is derived with, default_rng([seed, SEZ_REST_STREAM]); the
# constant is the design id read as hex.
_SEZ_REST_FIELDS = ("hz", "ids", "ids_sha256", "kick_fraction")
SEZ_REST_STREAM = 0x1E20A949


def _sez_rest_spec(th, meta, cmd_mode):
    """Design 1e20a949, Build C1: th["sez_rest"] as CNS.run's rest_drive, the
    ids' sha256 and the row count, or a ValueError. The ids are distinct BANC
    ids whose sha256, sorted as integers and newline joined (design A's map
    form), must equal ids_sha256; hz is finite and >= 0, kick_fraction finite
    and > 0. Refused beside hook_membrane_drive_fraction or
    cmd_mode='membrane', which take the engine's one membrane drive."""
    s = th["sez_rest"]
    if not isinstance(s, dict) or set(s) != set(_SEZ_REST_FIELDS):
        raise ValueError(f"sez_rest must be a dict with exactly {_SEZ_REST_FIELDS}")
    if th.get("hook_membrane_drive_fraction") is not None or cmd_mode == "membrane":
        raise ValueError("sez_rest refuses hook_membrane_drive_fraction and "
                         "cmd_mode='membrane': both kick through the engine's "
                         "one membrane drive")
    ids = [str(x) for x in s["ids"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("sez_rest ids must be a non-empty list of distinct ids")
    sha = hashlib.sha256("\n".join(sorted(ids, key=int)).encode()).hexdigest()
    if sha != s["ids_sha256"]:
        raise ValueError(f"sez_rest ids hash {sha} differs from ids_sha256 "
                         f"{s['ids_sha256']!r}")
    for k in ("hz", "kick_fraction"):
        if type(s[k]) not in (int, float) or not np.isfinite(s[k]):
            raise ValueError(f"sez_rest {k} must be a finite number")
    if s["hz"] < 0 or s["kick_fraction"] <= 0:
        raise ValueError("sez_rest needs hz >= 0 and kick_fraction > 0")
    pos = pd.Series(np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
    missing = [x for x in ids if x not in pos.index]
    if missing:
        raise ValueError(f"sez_rest names {len(missing)} ids absent from meta, "
                         f"first {missing[0]!r}")
    rows = np.sort(np.asarray([int(pos.loc[x]) for x in ids], dtype=np.int64))
    return ({"rows": rows, "hz": float(s["hz"]), "kick_fraction": float(s["kick_fraction"]),
             "stream": SEZ_REST_STREAM}, sha, int(len(rows)))


def _slow_inh_spec(th):
    """Design 2163ba7d, Build 2: slow_inh_out as its ids (distinct id strings,
    sorted by value), their sha256 (the design's hash rule), frac and tau_ms,
    or a ValueError for a missing or extra field, an empty or repeated id list,
    or a value that is not a finite number."""
    s = th["slow_inh_out"]
    if not isinstance(s, dict) or set(s) != set(_SLOW_INH_FIELDS):
        raise ValueError(f"slow_inh_out must be a dict with exactly {_SLOW_INH_FIELDS}")
    ids = [str(x) for x in s["ids"]]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("slow_inh_out ids must be a non-empty list of distinct ids")
    for k in ("frac", "tau_ms"):
        if type(s[k]) not in (int, float) or not np.isfinite(s[k]):
            raise ValueError(f"slow_inh_out {k} must be a finite number")
    ids = sorted(ids, key=int)
    return {"ids": ids, "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "frac": float(s["frac"]), "tau_ms": float(s["tau_ms"])}


def _front_leg_mn_rows(meta):
    """Design 2163ba7d, Build 2: the sorted rows of the front-leg motor
    neurons, every id motormap.build_map places on leg lf or rf."""
    import motormap as MM2
    mm = MM2.build_map(meta)
    ids = set(mm.loc[mm["leg"].isin(["lf", "rf"]), "banc_888_id"].astype(str))
    pos = pd.Series(np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
    return np.sort(np.asarray([int(pos.loc[i]) for i in ids], dtype=np.int64))


_ROW_SET_VALUES = ("v_rest_mv", "v50_mv", "slope_mv", "gain")
_ROW_SET_BOUNDS = {"v_rest_mv": (-80.0, -30.0), "v50_mv": (-80.0, -20.0),
                   "slope_mv": (1.8, 9.0)}
_ROW_SET_CURVE = (("v50_mv", "graded_v50"), ("slope_mv", "graded_slope"),
                  ("gain", "graded_gain"))


def _row_set_params_parse(spec, meta, graded_rows):
    """Design 707773e7 Build 2 (study 0fc789d7): th["row_set_params"], a list of
    {ids, ids_sha256, v_rest_mv, v50_mv, slope_mv, gain}, the last four each
    optional. Every row of a set rests at v_rest_mv, its reset following it as
    v_rest_type_reset_follows does for types, and each curve field present
    replaces the inherited scalar and any per-type value on the set's rows:
    release_i(v) = 1 / (1 + exp(-(v - v50_mv) / slope_mv)) delivered at gain.
    ids are exact-decimal strings and ids_sha256 is their list_sha256 (sorted
    as integers, joined by newline). Refused: an empty list, a field outside
    the six, a set without ids, ids_sha256 or any value, a hash that differs,
    an id absent from meta, a row in two sets, a value outside the per-type
    dials' bounds (rest -80 to -30 mV as v_rest_type_mv, v50 -80 to -20 mV,
    slope 1.8 to 9.0 mV, gain above 0), and a curve field on a row outside the
    graded set. Returns one dict per set with its sorted rows. Absent = never
    called."""
    if not isinstance(spec, list) or not spec:
        raise ValueError("row_set_params must be a non-empty list of sets")
    pos = pd.Series(np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
    bids = meta["banc_888_id"].astype(str).to_numpy()
    graded = {int(r) for r in graded_rows}
    seen, out = {}, []
    for k, s in enumerate(spec):
        if (not isinstance(s, dict) or "ids" not in s or "ids_sha256" not in s
                or not set(s) <= {"ids", "ids_sha256", *_ROW_SET_VALUES}):
            raise ValueError(
                f"row_set_params[{k}] must carry ids and ids_sha256 and no field "
                f"outside {('ids', 'ids_sha256') + _ROW_SET_VALUES}")
        vals = {f: s[f] for f in _ROW_SET_VALUES if f in s}
        if not vals:
            raise ValueError(f"row_set_params[{k}] sets no value")
        ids = s["ids"]
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                or not all(isinstance(x, str) and x.isdigit() for x in ids)):
            raise ValueError(
                f"row_set_params[{k}] ids must be a non-empty list of distinct "
                "exact-decimal strings")
        sha = hashlib.sha256("\n".join(sorted(ids, key=int)).encode()).hexdigest()
        if sha != s["ids_sha256"]:
            raise ValueError(
                f"row_set_params[{k}] ids hash {sha} differs from ids_sha256 "
                f"{s['ids_sha256']!r}")
        missing = [x for x in ids if x not in pos.index]
        if missing:
            raise ValueError(
                f"row_set_params[{k}] names {len(missing)} ids absent from meta, "
                f"first {missing[0]!r}")
        rows = sorted(int(pos.loc[x]) for x in ids)
        for r in rows:
            if r in seen:
                raise ValueError(
                    f"row_set_params: row {bids[r]} is in sets {seen[r]} and {k}")
            seen[r] = k
        for f, v in vals.items():
            if type(v) not in (int, float) or not np.isfinite(v):
                raise ValueError(f"row_set_params[{k}] {f} must be a finite number")
            if f == "gain":
                if not v > 0.0:
                    raise ValueError(f"row_set_params[{k}] gain must be above 0")
            elif not _ROW_SET_BOUNDS[f][0] <= v <= _ROW_SET_BOUNDS[f][1]:
                raise ValueError(
                    f"row_set_params[{k}] {f} = {v} is outside "
                    f"{_ROW_SET_BOUNDS[f][0]} to {_ROW_SET_BOUNDS[f][1]} mV")
        curve = [f for f, _ in _ROW_SET_CURVE if f in vals]
        outside = [r for r in rows if r not in graded]
        if curve and outside:
            raise ValueError(
                f"row_set_params[{k}] sets {curve} on {len(outside)} rows outside "
                f"the graded set, first {bids[outside[0]]}")
        out.append({"rows": tuple(rows), "ids_sha256": sha,
                    **{f: float(v) for f, v in vals.items()}})
    return out


def _row_set_rest_apply(sets, meta, p):
    """Build 2's rest: each set's v_rest_mv into p.v_rest_override by id and
    the same value into p.v_reset on its rows, after v_rest_type_mv. A list
    with no v_rest_mv leaves p as it was."""
    if not any("v_rest_mv" in s for s in sets):
        return
    ids = meta["banc_888_id"].astype(str).to_numpy()
    merged = dict(p.v_rest_override or {})
    reset = np.broadcast_to(np.asarray(p.v_reset, dtype=np.float64),
                            (len(meta),)).astype(np.float64).copy()
    for s in sets:
        if "v_rest_mv" in s:
            for r in s["rows"]:
                merged[str(ids[r])] = s["v_rest_mv"]
                reset[r] = s["v_rest_mv"]
    p.v_rest_override = merged
    p.v_reset = reset


def _row_set_curve_apply(sets, n, p):
    """Build 2's curve: each field present becomes an (N,) float64 vector of the
    inherited value (scalar or per-row) with the set's value on its rows."""
    for f, attr in _ROW_SET_CURVE:
        if any(f in s for s in sets):
            vec = np.broadcast_to(np.asarray(getattr(p, attr), dtype=np.float64),
                                  (n,)).astype(np.float64).copy()
            for s in sets:
                if f in s:
                    vec[list(s["rows"])] = s[f]
            setattr(p, attr, vec)


def _row_set_receipt(sets, meta, p):
    """Build 2's output: per set its hash, row count, the values asked and the
    values built, read from p as cns.CNS folds p.v_rest_override into p.v_rest
    (rest and reset in float32, the engines' dtype). Raises when a later writer
    moved a value on a set row."""
    ids = meta["banc_888_id"].astype(str).to_numpy()
    n = len(ids)
    base = np.broadcast_to(np.asarray(p.v_rest, dtype=np.float32), (n,))
    ov = p.v_rest_override or {}
    reset = np.broadcast_to(np.asarray(p.v_reset, dtype=np.float32), (n,))
    out = []
    for k, s in enumerate(sets):
        rows = list(s["rows"])
        built = {}
        if "v_rest_mv" in s:
            rest_b = np.asarray([np.float32(ov[str(ids[r])]) if str(ids[r]) in ov
                                 else base[r] for r in rows], dtype=np.float32)
            built["v_rest_mv"] = sorted({float(x) for x in rest_b})
            built["v_reset_mv"] = sorted({float(x) for x in reset[rows]})
            if (built["v_rest_mv"] != [float(np.float32(s["v_rest_mv"]))]
                    or built["v_reset_mv"] != [float(np.float32(s["v_rest_mv"]))]):
                raise ValueError(
                    f"row_set_params[{k}] rest is built at {built['v_rest_mv']} and "
                    f"reset {built['v_reset_mv']} mV, not {s['v_rest_mv']}")
        for f, attr in _ROW_SET_CURVE:
            if f in s:
                vec = np.broadcast_to(np.asarray(getattr(p, attr), dtype=np.float64), (n,))
                built[f] = sorted({float(x) for x in vec[rows]})
                if built[f] != [s[f]]:
                    raise ValueError(
                        f"row_set_params[{k}] {f} is built at {built[f]}, not {s[f]}")
        out.append({"ids_sha256": s["ids_sha256"], "rows": len(rows),
                    "asked": {f: s[f] for f in _ROW_SET_VALUES if f in s},
                    "built": built})
    return out


def _coincidence_triples_parse(spec, labels, meta, graded_rows):
    """Design efe88957 Build 2 (study ff3ee4ee): th["coincidence_triples"], the
    coincidence seam's list form. Each triple is {targets, fast, slow, gain,
    lag_ms, tau_ms}: targets, fast and slow are lists of family names, each
    resolving the rows the single key's label resolves (cell_type, or
    fafb_alignment_cell_type where cell_type is null), and a pool may instead
    be given as fast_ids or slow_ids with its list_sha256 (sorted as
    integers, joined by newline) in fast_ids_sha256 or slow_ids_sha256. The
    engine reads a graded source by its release and any other source by its
    spike; `read` counts each pool's rows by that mode and by side, and the
    engine checks its own counts against it. Refused: an empty list, a field
    outside the triple's, a family name that resolves no row, an id absent
    from meta, a hash that differs, a target in two triples, a gain that is
    zero or not finite, lag_ms outside (0, 500] or tau_ms outside (0, 200],
    the single key's bounds. Absent = never called."""
    if not isinstance(spec, list) or not spec:
        raise ValueError("coincidence_triples must be a non-empty list of triples")
    pos = pd.Series(np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
    side = meta["side"].astype("string").fillna("none").to_numpy().astype(str)
    graded = np.zeros(len(meta), dtype=bool)
    graded[np.asarray(graded_rows, dtype=np.int64)] = True
    claimed = np.full(len(meta), -1, dtype=np.int64)

    def by_side(rows):
        s, c = np.unique(side[rows], return_counts=True)
        return {str(a): int(b) for a, b in zip(s, c)}

    out = []
    for k, tr in enumerate(spec):
        if not isinstance(tr, dict):
            raise ValueError(f"coincidence_triples[{k}] must be a dict")
        want = {"targets", "gain", "lag_ms", "tau_ms"}
        for nm in ("fast", "slow"):
            want |= {nm} if nm in tr else {nm + "_ids", nm + "_ids_sha256"}
        if set(tr) != want:
            raise ValueError(
                f"coincidence_triples[{k}] carries {sorted(tr)}; a triple carries "
                "targets, gain, lag_ms, tau_ms and each pool as names (fast, slow) "
                "or as ids with their hash (fast_ids and fast_ids_sha256)")
        got, named = {}, {}
        for nm in ("targets", "fast", "slow"):
            if nm in tr:
                names = tr[nm]
                if (not isinstance(names, list) or not names
                        or not all(isinstance(x, str) and x for x in names)
                        or len(set(names)) != len(names)):
                    raise ValueError(
                        f"coincidence_triples[{k}] {nm} must be a non-empty list "
                        "of distinct family names")
                none = [x for x in names if not (labels == x).any()]
                if none:
                    raise ValueError(
                        f"coincidence_triples[{k}] {nm} names {none}, which "
                        "resolve no row in this core")
                got[nm] = np.flatnonzero(np.isin(labels, names))
                named[nm] = list(names)
                continue
            ids = tr[nm + "_ids"]
            if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                    or not all(isinstance(x, str) and x.isdigit() for x in ids)):
                raise ValueError(
                    f"coincidence_triples[{k}] {nm}_ids must be a non-empty list "
                    "of distinct exact-decimal strings")
            sha = hashlib.sha256("\n".join(sorted(ids, key=int)).encode()).hexdigest()
            if sha != tr[nm + "_ids_sha256"]:
                raise ValueError(
                    f"coincidence_triples[{k}] {nm}_ids hash {sha} differs from "
                    f"{nm}_ids_sha256 {tr[nm + '_ids_sha256']!r}")
            missing = [x for x in ids if x not in pos.index]
            if missing:
                raise ValueError(
                    f"coincidence_triples[{k}] {nm}_ids names {len(missing)} ids "
                    f"absent from meta, first {missing[0]!r}")
            got[nm] = np.asarray(sorted(int(pos.loc[x]) for x in ids), dtype=np.int64)
            named[nm] = {"ids_sha256": sha, "ids": len(ids)}
        twice = got["targets"][claimed[got["targets"]] >= 0]
        if len(twice):
            raise ValueError(
                f"coincidence_triples[{k}] shares {len(twice)} targets with triple "
                f"{int(claimed[twice[0]])}; each target carries one product")
        claimed[got["targets"]] = k
        gain, lag, tau = tr["gain"], tr["lag_ms"], tr["tau_ms"]
        if type(gain) not in (int, float) or not np.isfinite(gain) or gain == 0:
            raise ValueError(f"coincidence_triples[{k}] gain must be finite and nonzero")
        if type(lag) not in (int, float) or not np.isfinite(lag) or not 0.0 < lag <= 500.0:
            raise ValueError(f"coincidence_triples[{k}] lag_ms must be finite in (0, 500]")
        if type(tau) not in (int, float) or not np.isfinite(tau) or not 0.0 < tau <= 200.0:
            raise ValueError(f"coincidence_triples[{k}] tau_ms must be finite in (0, 200]")
        read = {"targets": {"rows": int(len(got["targets"])),
                            "by_side": by_side(got["targets"])}}
        for nm in ("fast", "slow"):
            rows = got[nm]
            g = graded[rows]
            read[nm] = {"rows": int(len(rows)),
                        "release": int(g.sum()), "spike": int((~g).sum()),
                        "release_by_side": by_side(rows[g]),
                        "spike_by_side": by_side(rows[~g])}
        out.append({"targets": got["targets"], "fast": got["fast"], "slow": got["slow"],
                    "gain": float(gain), "lag_ms": float(lag), "tau_ms": float(tau),
                    "names": named, "read": read})
    return tuple(out)


def _refuse_unknown_theta_keys(th):
    """A theta key this tree does not read is a claim the run does not carry.

    Every key in evaluate is read with th.get(...), so a key the running tree has
    not merged does NOTHING and raises nothing: the lane runs a different body
    from the theta it was handed and gets a plausible number. Measured
    2026-09-06 11:30 (lane E, 213 commits behind main): the judge's grip-15
    theta's `delay_region_scale` was ignored, arm A stood 0 of 12 against the
    judge's 9 of 12, and a contradiction was nearly filed.

    The read-key set is THIS tree's own sources (walk_search, body, cns,
    regression_walk, and the modules evaluate hands the whole theta to that
    read keys from it: abdomen, gpu/window_hooks, gpu/fall_controls), the same
    quoted-string read lanes/judge/theta_keys_gate.py makes, so a lane's
    unported mechanism is known on the lane's tree and a judge theta with a
    newer key is unknown on a stale one. Underscore-prefixed keys and the
    theta-file wrapper keys are metadata and tolerated.

    Every run gets a stderr line. Only a decision-grade run, marked by
    FLY_WBE_DECISION_GRADE=1 or by the older FLY_WBE_METAL_LOCK_MODE=exclusive,
    is refused: a routine BUILD screen may
    carry a lane's orchestration key that its runner translates before this
    call, and refusing those would break screens to catch a case that only
    matters where a number supports a verdict.
    """
    global _ENGINE_KEYS
    if _ENGINE_KEYS is None:
        import re as _re
        here = os.path.dirname(os.path.abspath(__file__))
        keys = set()
        # abdomen.AbdomenAdapter(th, net) reads abd_signs, abd_range_deg,
        # abd_tau_act_ms and abd_tau_joint_ms, which the first four files never
        # quote, so until abdomen.py was scanned every decision-grade run carrying
        # one was refused: 32 runs of standing sweep S1 on 2026-09-28 (knowledge,
        # note eb6bac90). A module evaluate hands th to joins this list once it
        # reads a key from it; lanes/B/escape_ttm_evaluate_bridge.py reads none.
        for fn in ("walk_search.py", "body.py", "cns.py", "regression_walk.py",
                   "abdomen.py", "gpu/window_hooks.py", "gpu/fall_controls.py"):
            fp = os.path.join(here, fn)
            if os.path.exists(fp):
                with open(fp) as fh:
                    keys |= set(_re.findall(r'["\']([A-Za-z0-9_]+)["\']', fh.read()))
        _ENGINE_KEYS = keys
    # A key the tree builds by concatenation is read but never appears whole in
    # the source, so the literal scan cannot see it and calls it unknown. That
    # is the false absence this guard exists to prevent, in the guard itself:
    # it refused lane C's ordered confirmation on 2026-09-06 over
    # vision_carrier_peak_hz, which walk_search reads as
    # th.get("vision_carrier_" + key) with the three suffixes quoted beside it
    # (found by the compute seat, 15:1x). A key counts as read when it splits
    # into a quoted prefix ending in an underscore plus a quoted remainder,
    # which is the construction and nothing wider.
    def _built_by_concat(k):
        for i in range(1, len(k)):
            if k[i - 1] == "_" and k[:i] in _ENGINE_KEYS and k[i:] in _ENGINE_KEYS:
                return True
        return False
    unknown = sorted(k for k in th if isinstance(k, str) and not k.startswith("_")
                     and k not in _THETA_META_KEYS and k not in _ENGINE_KEYS
                     and not _built_by_concat(k))
    if not unknown:
        return
    msg = ("theta carries keys this tree does not read, so they would be silently "
           f"ignored: {unknown}. Merge origin/main, or run lanes/judge/theta_keys_gate.py.")
    import sys as _sys
    print("WARNING (walk_search.evaluate): " + msg, file=_sys.stderr, flush=True)
    # "Decision-grade" was read off the Metal lock mode because until
    # 2026-09-07 the two always travelled together: a run supporting a verdict
    # held the lock exclusive. They came apart when the exclusive lock stopped
    # being a correctness measure (engine rule, 08:20) and the title
    # instruments moved to the shared lock to stop starving the fleet, so a
    # decision-grade run now takes the shared lock and would have been let
    # through with a warning. FLY_WBE_DECISION_GRADE states the meaning
    # directly; the lock mode still implies it, so every lane and vision
    # runner that asserts exclusive is unaffected.
    if (os.environ.get("FLY_WBE_METAL_LOCK_MODE") == "exclusive"
            or os.environ.get("FLY_WBE_DECISION_GRADE") == "1"):
        raise ValueError("REFUSED, decision-grade run: " + msg)


def hook_rows_encoded(meta, th):
    """Design 302c1bfb, Builds 8 and 9: the hook rows the run encodes, the
    union of the production encoder's hook_flex and hook_ext groups built
    with the theta's two FeCO modes as evaluate builds it, less the
    campaniform sensillum the exact-type route files as hook. It includes
    the one hook-subclass / hair-class conflict body.Proprioceptors has
    always routed as hook. With both modes absent these are R144's 142 rows,
    none of them campaniform; on the build body they are H168."""
    enc = B.Proprioceptors(
        meta, None, claw_polarity=th["claw"], hook_polarity=th["hook"],
        feco_polarity_mode=th.get("feco_polarity_mode"),
        feco_modality_mode=th.get("feco_modality_mode"))
    rows = np.union1d(enc.groups["hook_flex"], enc.groups["hook_ext"])
    return rows[meta["cell_class"].to_numpy()[rows] != "campaniform_sensillum_neuron"]


def apply_edge_count_set(edges, entries):
    """Design 478fe148, Build B1: the edge table with each entry's synapse
    count set on its pair, the row BANC has, or appended as a new row where
    BANC has none. entries are (pre_id, post_id, count, label, citation)
    tuples of canonical id strings. A new row's norm is its count over the
    post row's summed input count in the file and its pre_count and
    post_count are the file's totals; cns.CNS reads only pre, post and
    count, so norm is carried for the table's consistency and never enters a
    weight. Existing rows keep their position, so positional edge plans stay
    valid. Returns (table, receipt)."""
    pre = edges["pre"].to_numpy()
    post = edges["post"].to_numpy()
    out = edges.copy()
    col = out.columns.get_loc("count")
    rows, new = [], []
    for pre_id, post_id, count, label, citation in entries:
        hit = np.flatnonzero((pre == pre_id) & (post == post_id))
        if len(hit) > 1:
            raise ValueError(f"edge table holds {pre_id} -> {post_id} twice")
        post_in = int(edges["count"].to_numpy()[post == post_id].sum())
        pre_out = int(edges["count"].to_numpy()[pre == pre_id].sum())
        if post_in <= 0:
            raise ValueError(f"edge_count_set post {post_id} has no input in the file")
        row = {"pre_id": pre_id, "post_id": post_id, "label": label,
               "citation": citation, "count_set": int(count),
               "post_input_count_in_file": post_in,
               "pre_output_count_in_file": pre_out}
        if len(hit):
            row["count_before"] = int(out.iat[int(hit[0]), col])
            row["created"] = False
            row["norm_in_file"] = float(edges["norm"].to_numpy()[hit[0]])
            out.iat[int(hit[0]), col] = count
        else:
            row["count_before"] = 0
            row["created"] = True
            row["norm_in_file"] = None
            new.append({"pre": pre_id, "post": post_id, "count": count,
                        "norm": count / post_in, "post_count": post_in,
                        "pre_count": pre_out})
        row["norm_of_set_count"] = count / post_in
        rows.append(row)
    if new:
        out = pd.concat([out, pd.DataFrame(new).astype(
            {c: edges[c].dtype for c in edges.columns})], ignore_index=True)
    receipt = {"entries": rows, "edges_in_file": int(len(edges)),
               "edges_after": int(len(out)),
               "edges_created": int(len(new)),
               "synapses_in_file": int(edges["count"].sum()),
               "synapses_after": int(out["count"].sum()),
               "norm_read_by_constructor": False}
    return out, receipt


def rows_sha16(meta, rows):
    """First 16 hex of SHA-256 over the rows' BANC ids sorted as integers and
    comma-joined, the digest rule of design 302c1bfb's sets."""
    ids = sorted(meta["banc_888_id"].astype(np.int64).to_numpy()[
        np.asarray(rows, dtype=np.int64)].tolist())
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()[:16]


def evaluate(net_cache, meta, edges, th, seed=0, dur_ms=4000.0, capture=None,
             playback=None, record_rows=None, trace_out=None, run_kwargs=None,
             force_spikes=None, trace_bin_ms=5.0, rig=None, playback_gate_a_flex=None,
             labellar_stimulus=None, tarsal_stimulus=None, claw_rate_out=None):
    """labellar_stimulus (design 0bec278a): a presentation schedule for rows of
    th["labellar_rest"]'s encoder (labellar_stimulus_plan); refused without
    the key. Absent: bit-identical. Never a theta key.
    tarsal_stimulus (design a146ad7f, Build B2): a list of such schedules for
    rows of th["gust_rest"]'s leg taste set, each validated as one
    labellar_stimulus is, the rows distinct across schedules; inside a
    contact a scheduled row fires at its bin's rate, outside it follows the
    gust_rest rule. Refused without gust_rest. Absent: bit-identical. Never
    a theta key.
    rig (design 554a54fc): None builds FlatGroundWorld as always; a
    gpu.ball.world.rig() argument builds the ball or air tether instead, and
    every root read below goes through one accessor whose floor branch is the
    free joint's own arrays. Never a theta key.
    run_kwargs (design 02e57559): extra CNS.run arguments, for walking with
    prescribed receptor tapes (force_spikes) and native recorders
    (native_record); a key evaluate already passes raises. The run's native
    outputs and release log land in LAST_EXTRAS["native_run"].
    force_spikes (design fa5adfc8): {banc_888_id: [CNS steps]}, a static
    schedule of spikes forced through the engine, resolved to rows after the
    network is built; it refuses to compose with run_kwargs["force_spikes"].
    Its receipt, and the ttm_twitch path's (gpu/escape/ttm_jump/path.py),
    land in LAST_EXTRAS. trace_bin_ms: the recorded trace's bin, 5.0 ms
    unless a caller asks for a finer one. Both absent: bit-identical.
    capture: optional list; if given, (xyz, quat) appended per control
    tick. THE one sanctioned way to get trajectories -- side scripts that
    hand-copy this drive path go stale every time a mechanism lands
    (battery_test and direction_test2 already did; see 2026-08-13).
    record_rows/trace_out (2026-08-17, capture-only, default off =
    bit-identical): extra neuron rows to record; the run's binned spike
    trace (rows_sorted, trace) is appended to trace_out. Readout only --
    the record set never feeds back into the simulation.
    claw_rate_out (design bf165b63, Build 3; capture-only, default None =
    bit-identical): an empty dict, filled with the claw rows (claw_flex then
    claw_ext, engine rows), their group and DOF, and per drive tick the time
    and the delivered rate on those rows, float32, read at the point the
    held preparation reads the delivered vector. Never a theta key."""
    _refuse_unknown_theta_keys(th)
    if claw_rate_out is not None and not (isinstance(claw_rate_out, dict) and not claw_rate_out):
        raise ValueError("claw_rate_out must be an empty dict")
    # Design 0bec278a: the key gate counts "labellar_stimulus" as read, because
    # LAST_EXTRAS quotes it, so a stimulus put in the theta is refused here
    # rather than ignored.
    if "labellar_stimulus" in th:
        raise ValueError("labellar_stimulus is an evaluate argument, never a theta key")
    if "tarsal_stimulus" in th:
        raise ValueError("tarsal_stimulus is an evaluate argument, never a theta key")
    # Design 394ad83c: the recorded-site rows a theta's explicit gap pairs
    # name, appended before anything reads meta, as the design's own walkers
    # append them (_with_site_rows). None named = meta untouched.
    meta = _with_site_rows(meta, th)
    # Design 543c8da2 Build 1: per-row label corrections, applied before any
    # selector reads meta (_apply_meta_patch). Absent = meta untouched.
    meta, _meta_patch_receipt = _apply_meta_patch(meta, th.get("meta_patch"))
    # Design fa5adfc8: the TTMn twitch path, off unless the theta names it.
    _twitch_keys = {_k: th[_k] for _k in (
        "ttm_twitch", "ttm_twitch_peak_uNmm", "ttm_twitch_lag_ms",
        "ttm_twitch_release_ms", "ttm_twitch_ctr_limit_rad",
        "ttm_twitch_fv_omega_rad_s", "ttm_twitch_arm") if _k in th}
    _twitch_cfg = None
    if _twitch_keys:
        from gpu.escape.ttm_jump import path as _ttm_path
        _twitch_cfg = _ttm_path.config(
            _twitch_keys, float(th.get("esc_ttm", 0.0) or 0.0))
    if force_spikes is not None and "force_spikes" in (run_kwargs or {}):
        raise ValueError("force_spikes is either evaluate's schedule or a "
                         "run_kwargs tape, not both")
    sign_vector = _motor_sign_vector(
        th.get("signs"), th.get("thc_yaw_moment_arm_orientation"),
        th.get("thc_roll_moment_arm_orientation"))
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import (
        LocomotionAction, apply_locomotion_action, make_locomotion_fly)
    # size-principle mechanism (th["mn_gains"]="size"): measured per-MN
    # force gains in M plus the measured MN input-resistance gradient in
    # the network. Default off; both halves bit-identical when absent.
    use_size = th.get("mn_gains") in ("size", "azevedo_label")
    size_g = None
    # Design 171d6f1a B2 (study cfd71a09): th["size_pools"] names the pools the
    # size rule ranks in. Absent or "banc" is BANC's labels, the code that was
    # here. "transferred" is (muscle, leg) over the live rows of the mapping
    # th["mn_transfer"] gives (B1), built once here, before every consumer of
    # size_factors, and passed to each of them. Absent, nothing below changes.
    size_pools = th.get("size_pools")
    size_pool_map = None                              # the transferred mapping, built when the key is present
    _sp_kw = {}                                       # size_factors' pools, empty unless "transferred"
    if size_pools is not None:
        if size_pools not in ("banc", "transferred"):
            raise ValueError("th['size_pools'] must be absent, 'banc' or 'transferred'")
        if th.get("mn_gains") != "size":
            raise ValueError("th['size_pools'] needs mn_gains 'size'")
        import motormap as _MM_SP
        size_pool_map = _MM_SP.build_map(meta)
        if th.get("mn_transfer") is not None:
            size_pool_map = _apply_mn_transfer(
                size_pool_map, _MM_SP.to_matrix(size_pool_map)[1], th["mn_transfer"])[0]
        if size_pools == "transferred":
            _sp_kw = {"pools": size_pool_map}
    # graded transmission (AGENTS directive 2026-08-17, default off =
    # bit-identical): th["graded"] = "map" enables graded release for
    # every core row whose cell_type appears in graded_map.json's
    # "enabled" section (membership is citation-gated there).
    graded_rows = graded_rows_for(th.get("graded"), meta)
    # per-type transmitter map (judge, 2026-09-03; default off =
    # bit-identical): th["transmitter_map"] = "map" applies every
    # citation-gated entry in transmitter_map.json in cns._resolve_nt.
    transmitter_map = transmitter_map_for(th.get("transmitter_map"))
    # cell_type_patch (judge, 2026-09-05; default off = bit-identical):
    # th["cell_type_patch"] = "patch" applies every citation-gated per-row
    # identity correction in cell_type_patch.json before transmitter resolution.
    cell_type_patch = cell_type_patch_for(th.get("cell_type_patch"))
    # nt_patch_rows (design 1818633f, study 203d56a6; absent = no patch =
    # bit-identical): an inline map banc_888_id -> {cell_type,
    # neurotransmitter, citation}, resolved into the SAME cell_type_patch
    # local as the judge-owned file above and into nothing else. A lane
    # correcting one row against one citation cannot use
    # th["cell_type_patch"], because that value loads the whole judge-owned
    # cell_type_patch.json, whose one enabled entry is the CT1 row carried
    # with "DO NOT ENABLE FOR VISION WORK UNTIL CT1 IS COMPARTMENTALISED";
    # the arm would then carry two changes. It resolves HERE, above the
    # cache key, because the key already carries cell_type_patch: applied to
    # p after the key is computed, two arms in one process would share one
    # cached network while every receipt named its own arm.
    _nt_patch_rows = th.get("nt_patch_rows")
    _hemilineage_nt_audit = None
    _hemilineage_nt = th.get("hemilineage_nt")
    if _hemilineage_nt is not None:
        if _nt_patch_rows or cell_type_patch:
            raise ValueError("hemilineage_nt cannot be combined with another row patch")
        from gpu.transmitter.selector import resolve as _resolve_hemilineage_nt
        _nt_patch_rows, _hemilineage_nt_audit = _resolve_hemilineage_nt(
            meta, _hemilineage_nt, transmitter_map)
    if _nt_patch_rows:
        if cell_type_patch:
            raise ValueError(
                "nt_patch_rows and cell_type_patch both set: one theta, two "
                "sources for cns.SimParams.cell_type_patch")
        _meta_ids = meta["banc_888_id"].astype(str).to_numpy()
        _meta_types = meta["cell_type"].astype(object).to_numpy()
        _meta_ver = meta["neurotransmitter_verified"].to_numpy()
        _nt_tokens = cns.SimParams().nt_sign
        _patch = {}
        for _rid, _spec in _nt_patch_rows.items():
            _hit = np.flatnonzero(_meta_ids == str(_rid))
            if len(_hit) != 1:
                raise ValueError(
                    f"nt_patch_rows: row {_rid} matches {len(_hit)} meta rows, "
                    "expected exactly 1")
            if not isinstance(_spec, dict):
                raise ValueError(f"nt_patch_rows[{_rid}] must be a mapping")
            _ct, _tok = _spec.get("cell_type"), _spec.get("neurotransmitter")
            # The key patches a transmitter and may not relabel a cell: the
            # stated type is the row's own, read back from meta, so a typo in
            # the id cannot silently rename a different cell's type. An
            # untyped row (design 6f704792 Build 3) states "cell_type": null,
            # accepted only where meta's cell_type is null as well; cns then
            # patches its transmitter and leaves the type null.
            _row_ct = _meta_types[_hit[0]]
            if _ct is None and "cell_type" in _spec and pd.isna(_row_ct):
                pass
            elif not _ct or str(_ct) != str(_row_ct):
                raise ValueError(
                    f"nt_patch_rows[{_rid}] states cell_type {_ct!r}; the row "
                    f"carries {_meta_types[_hit[0]]!r}")
            if not _tok or _tok not in _nt_tokens:
                raise ValueError(
                    f"nt_patch_rows[{_rid}] neurotransmitter {_tok!r} is not a "
                    "transmitter token in nt_sign")
            if not _spec.get("citation"):
                raise ValueError(f"nt_patch_rows[{_rid}] has no citation")
            # cns writes the patch into neurotransmitter_predicted only, and
            # prefer_verified takes the verified column where it is not null,
            # so a patch on a verified row would be discarded in silence.
            _ver = _meta_ver[_hit[0]]
            if not (_ver is None or (isinstance(_ver, float) and np.isnan(_ver))
                    or str(_ver) in ("", "nan", "None")):
                raise ValueError(
                    f"nt_patch_rows[{_rid}] targets a row whose "
                    f"neurotransmitter_verified is {_ver!r}; prefer_verified "
                    "would discard the patch")
            _patch[str(_rid)] = {"cell_type": None if _ct is None else str(_ct),
                                 "neurotransmitter": str(_tok)}
        cell_type_patch = _patch
    # balance_scale_override (design 1818633f; absent = recompute =
    # bit-identical): a repository-relative path to a committed .npy of
    # length N, with balance_scale_sha256 beside it, handed to
    # cns.SimParams.balance_scale_override so a sign flip is measured without
    # the homeostatic step re-absorbing it. The hash is checked here rather
    # than trusted, and it is what the cache key carries below.
    balance_scale_override = None
    balance_scale_sha256 = th.get("balance_scale_sha256")
    if th.get("balance_scale_override"):
        import hashlib as _hl_bs
        _bs_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            str(th["balance_scale_override"]))
        if not balance_scale_sha256:
            raise ValueError(
                "balance_scale_override needs balance_scale_sha256 beside it")
        _bs_bytes = open(_bs_path, "rb").read()
        _bs_have = _hl_bs.sha256(_bs_bytes).hexdigest()
        if _bs_have != str(balance_scale_sha256):
            raise ValueError(
                f"balance_scale_override {th['balance_scale_override']} hashes "
                f"{_bs_have}, theta names {balance_scale_sha256}")
        balance_scale_override = np.load(_bs_path, allow_pickle=False)
        # Design b10daa48 B2: one float64 per meta row, refused otherwise.
        # cns checks the length against the built network but casts the
        # dtype in silence, so a float32 file would run as a rounded vector.
        if (balance_scale_override.dtype != np.float64
                or balance_scale_override.shape != (len(meta),)):
            raise ValueError(
                f"balance_scale_override {th['balance_scale_override']} is "
                f"{balance_scale_override.dtype} of shape "
                f"{balance_scale_override.shape}; the build needs float64 of "
                f"shape ({len(meta)},)")
    graded_add_rows = ()
    graded_add_receipt = None
    if th.get("graded_add"):
        if th.get("graded_gain_add") is None:
            raise ValueError("graded_add requires graded_gain_add")
        _base = set(graded_rows)
        # Design e2b1a20d B4: a list of selectors is unioned in order, each
        # limited to rows outside the base set and the earlier selectors; a
        # selector that names no row is refused in either form.
        _sels = th["graded_add"]
        _sels = [_sels] if isinstance(_sels, str) else list(_sels)
        _seen, _added, graded_add_receipt = set(_base), [], []
        for _sel in _sels:
            _named = graded_rows_for(_sel, meta)
            if not _named:
                raise ValueError(f"graded_add selector {_sel!r} names no row")
            _new = [r for r in _named if r not in _seen]
            _seen.update(_new)
            _added.extend(_new)
            graded_add_receipt.append({
                "selector": _sel, "rows_named": len(_named),
                "rows_added": len(_new),
                "added_ids_sha256": sorted_ids_sha256(meta, _new)})
        graded_add_rows = tuple(_added)
        graded_rows = tuple(sorted(_base | set(graded_add_rows)))
    # Design 336cacb4 Build 2 (study efc222c7): the graded addition as a
    # frozen id list with its hash, so a twin or recruitment set that no
    # selector names can join the graded set at graded_gain_add. It shares the
    # per-row gain path graded_add uses; an id already graded is refused
    # rather than dropped, so the list is the addition. Absent = bit-identical.
    # Design 707773e7 Build 1 (study 0fc789d7): the id list composes with
    # graded_add. The graded set is the union of the map, the selectors and
    # the ids; graded_gain_add reaches both additions; an id the map or a
    # selector already grades is refused below. Without graded_add the rows
    # and the receipt are the ones this block built before.
    graded_add_ids_sha256 = None
    graded_add_ids_rows = ()
    if th.get("graded_add_ids") is not None:
        if th.get("graded_gain_add") is None:
            raise ValueError("graded_add_ids requires graded_gain_add")
        _add_ids = [str(x) for x in th["graded_add_ids"]]
        if not _add_ids or len(set(_add_ids)) != len(_add_ids):
            raise ValueError(
                "graded_add_ids must be a non-empty list of distinct ids")
        graded_add_ids_sha256 = hashlib.sha256("\n".join(
            sorted(_add_ids, key=int)).encode()).hexdigest()
        if graded_add_ids_sha256 != th.get("graded_add_ids_sha256"):
            raise ValueError(
                f"graded_add_ids hash {graded_add_ids_sha256} differs from "
                f"graded_add_ids_sha256 {th.get('graded_add_ids_sha256')!r}")
        _add_pos = pd.Series(
            np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
        _add_missing = [x for x in _add_ids if x not in _add_pos.index]
        if _add_missing:
            raise ValueError(
                f"graded_add_ids names {len(_add_missing)} ids absent from "
                f"meta, first {_add_missing[0]!r}")
        _add_rows = sorted(int(_add_pos.loc[x]) for x in _add_ids)
        _already = sorted(set(_add_rows) & set(graded_rows))
        if _already:
            raise ValueError(
                f"graded_add_ids names {len(_already)} rows already graded, "
                f"first {str(meta['banc_888_id'].iloc[_already[0]])!r}")
        graded_add_ids_rows = tuple(_add_rows)
        graded_add_rows = tuple(graded_add_rows) + graded_add_ids_rows
        graded_rows = tuple(sorted(set(graded_rows) | set(graded_add_ids_rows)))
    # Design 707773e7 Build 2: per-set rest and release curve, read against the
    # graded set as it stands after the map, the selectors and the ids.
    row_set_params = None
    if th.get("row_set_params") is not None:
        row_set_params = _row_set_params_parse(
            th["row_set_params"], meta, graded_rows)
    # Design e2b1a20d B1 to B3: release gain, release half-activation and
    # inhibitory reversal by exact BANC cell_type, each absent unless set.
    # Design 0af6d1c5 Build 2: the release slope the same way, in the scalar
    # graded_slope's bound (1.8 to 9.0), graded rows only.
    _type_rows = {"graded_gain_type": {}, "graded_v50_type": {},
                  "graded_slope_type": {}, "e_inh_type_mv": {}}
    _type_bounds = {"graded_gain_type": (0.0, np.inf),
                    "graded_v50_type": (-80.0, -20.0),
                    "graded_slope_type": (1.8, 9.0),
                    "e_inh_type_mv": (-90.0, -40.0)}
    if any(th.get(_k) is not None for _k in _type_rows):
        _ct_exact = meta["cell_type"].fillna("").astype(str).to_numpy()
        _graded_set = set(graded_rows)
        for _k, _out in _type_rows.items():
            _spec = th.get(_k)
            if _spec is None:
                continue
            if not isinstance(_spec, dict) or not _spec:
                raise ValueError(f"{_k} must be a non-empty {{cell_type: value}} object")
            if _k != "e_inh_type_mv" and not graded_rows:
                raise ValueError(f"{_k} requires a graded set")
            _lo, _hi = _type_bounds[_k]
            for _t, _v in _spec.items():
                _v = float(_v)
                _rows_t = tuple(int(r) for r in np.flatnonzero(_ct_exact == str(_t)))
                if not _rows_t:
                    raise ValueError(f"{_k}: cell_type {_t!r} names no row")
                if (not np.isfinite(_v)
                        or (_k == "graded_gain_type" and _v <= _lo)
                        or (_k != "graded_gain_type" and not _lo <= _v <= _hi)):
                    raise ValueError(f"{_k}[{_t!r}] = {_v} is outside its range")
                if _k != "e_inh_type_mv" and not _graded_set.issuperset(_rows_t):
                    raise ValueError(f"{_k}: a {_t} row is outside the graded set")
                _out[str(_t)] = (_rows_t, _v)
        if _type_rows["graded_gain_type"] and th.get("graded_13b_gain") is not None:
            raise ValueError("graded_gain_type does not combine with graded_13b_gain")
        if _type_rows["graded_v50_type"] and th.get("graded_parameter_mode"):
            raise ValueError("graded_v50_type does not combine with graded_parameter_mode")
        if _type_rows["graded_slope_type"] and th.get("graded_parameter_mode"):
            raise ValueError("graded_slope_type does not combine with graded_parameter_mode")
        if _type_rows["e_inh_type_mv"]:
            if not th.get("cbased"):
                raise ValueError("e_inh_type_mv requires conductance-based synapses")
            if th.get("inhibitory_reversal_mode"):
                raise ValueError("e_inh_type_mv does not combine with "
                                 "inhibitory_reversal_mode")
    # Design e2b1a20d B5: acetylcholine transferred onto S by its FAFB label.
    _nt_transfer = th.get("nt_transfer_rows")
    _nt_transfer_ids = None
    if _nt_transfer is not None:
        if (not isinstance(_nt_transfer, dict)
                or set(_nt_transfer) != {"ids_sha256", "transmitter",
                                         "label_source", "citation"}):
            raise ValueError("nt_transfer_rows needs exactly ids_sha256, "
                             "transmitter, label_source and citation")
        if (_nt_transfer["ids_sha256"] != S5585_SHA256
                or _nt_transfer["label_source"] != "fafb_alignment_cell_type"
                or _nt_transfer["transmitter"] != "acetylcholine"
                or not str(_nt_transfer["citation"]).strip()):
            raise ValueError("nt_transfer_rows names a list, label or "
                             "transmitter this tree does not carry")
        _nt_transfer_ids = list(alignment_s5585(meta)[1]["ids"])
    _graded_edge_equilibrium_mode = th.get(
        "vision_hs_dna02_conductance_normalization")
    graded_edge_equilibrium_ids = ()
    graded_edge_equilibrium_plan = None
    if _graded_edge_equilibrium_mode:
        if (_graded_edge_equilibrium_mode
                != "saturation_population_equilibrium"):
            raise ValueError(
                "unknown vision_hs_dna02_conductance_normalization")
        if not (th.get("cbased") and th.get("shunt")):
            raise ValueError(
                "HS-DNa02 conductance normalization requires true shunting")
        if int(th.get("graded_every", 0)) < 1:
            raise ValueError(
                "HS-DNa02 conductance normalization requires graded release")
        if float(th.get("graded_phasic", 0.0) or 0.0) != 0.0:
            raise ValueError(
                "HS-DNa02 conductance normalization requires tonic graded "
                "release")
        if (th.get("vision_cmd") != "mano2023_hs_dna02_relay_v0"
                or th.get("vision_relay_code") != "hs_graded_opponent"
                or th.get("vision_input_source") != _RENDERED_VISUAL_INPUT
                or th.get("vision_hs_signed_drive") is not True):
            raise ValueError(
                "HS-DNa02 conductance normalization requires signed rendered "
                "graded-HS vision")
        hs_mask = (
            meta["cell_type"].fillna("").astype(str).isin(
                _OPTOMOTOR_HS_TYPES)
            & meta["super_class"].fillna("").astype(str).eq(
                "visual_projection"))
        hs_rows_for_normalization = set(
            np.flatnonzero(hs_mask.to_numpy()).astype(int).tolist())
        if (len(hs_rows_for_normalization) != 6
                or not hs_rows_for_normalization.issubset(set(graded_rows))):
            raise ValueError(
                "HS-DNa02 conductance normalization needs six graded HS rows")
        (graded_edge_equilibrium_ids,
         graded_edge_equilibrium_plan) = graded_hs_dna02_direct_edge_plan(
             meta, edges)
    # EL1 (2026-08-19): gap junctions from ANNOTATED data. MANC flags
    # 225 traced neurons electrical/putative-electrical (incl. the
    # canonical DNp01->TTMn giant-fiber synapse). th["gap"]="annotated"
    # wires a pair for every EM edge whose BOTH endpoints carry the
    # flag (tier-3 pair inference over tier-1 per-neuron annotation;
    # honest limits: per-neuron not per-contact, and datasets without
    # the flag column contribute nothing). Default absent = OFF =
    # bit-identical. g_gap from th["g_gap"].
    gap_pairs = ()
    gap_pair_weights = ()
    gap_pair_rule_receipt = ()
    if th.get("gap") == "sham":
        # An INTENTIONALLY inert gap arm. The mechanism panel (2026-08-19)
        # pre-registered exactly this -- "the female has NO electrical
        # column yet, so her gap arms are structurally inert and serve as
        # sham controls" -- and it was a good pipeline control. The defect
        # was never that the female arms were shams; it was that a
        # deliberate sham and an accidental one produced identical code,
        # identical labels and identical silence. Declaring it makes the
        # distinction machine-readable.
        pass
    elif th.get("gap") == "explicit_rules":
        if float(th.get("g_gap", 0.0)) != 0.0:
            raise ValueError("explicit gap rules cannot use scalar g_gap")
        gap_pairs, gap_pair_weights, gap_pair_rule_receipt = (
            _explicit_gap_pair_rules(meta, th.get("gap_pair_rules")))
    elif th.get("gap") == "explicit_pairs":
        # Directed junctions named by BANC id; scalar g_gap scales them, as
        # on the annotated path. See _explicit_gap_pair_ids for why the
        # rules path cannot express this topology.
        gap_pairs = _explicit_gap_pair_ids(meta, th.get("gap_pair_ids"))
        if "gap_pair_weights" in th:
            weights = np.asarray(th["gap_pair_weights"], dtype=np.float64)
            if (weights.shape != (len(gap_pairs),) or not np.isfinite(weights).all()
                    or np.any(weights < 0) or np.any(weights > np.finfo(np.float32).max)):
                raise ValueError("gap_pair_weights needs one finite nonnegative value per pair")
            if float(th.get("g_gap", 0.0)) != 0:
                raise ValueError("gap_pair_weights and g_gap are mutually exclusive")
            gap_pair_weights = tuple(float(x) for x in weights)
    elif th.get("gap") == "annotated" and "electrical" not in meta:
        # AUDIT 2026-08-19, CORRECTED. This used to be part of the `and`
        # below, so a dataset with no `electrical` column wired ZERO gap
        # junctions and reported success. 80 stored female runs are
        # labelled as gap-junction conditions and are bit-identical to
        # their |off twins, verified 80/80
        # (results-mech-panel-female.json 20/20; results-co1-female.json
        # 60/60).
        #
        # BUT THOSE 80 ARE NOT A CONTAMINATION, and an earlier version of
        # this comment said they were. CONSTRAINTS.md pre-registered them:
        # "the female has NO electrical column yet, so her gap arms are
        # structurally inert and serve as sham controls until manc_match
        # label transfer lands", and the result was reported as "female
        # arms are exact shams as predicted". The experiment was run
        # correctly and reported honestly.
        #
        # What was wrong is narrower and still worth fixing: NOTHING IN
        # THE CODE distinguished that deliberate sham from an accidental
        # one. The same silence would have covered a dataset swap, a
        # renamed column, or a future run by someone who had not read the
        # registration. Declare it with gap="sham"; anything else is an
        # error.
        raise ValueError(
            "th['gap']='annotated' but this meta has no 'electrical' column "
            "-- it would wire ZERO gap junctions and look like it worked. "
            "Only manc-as-banc-meta-guessed.feather carries the flag.")
    if th.get("gap") == "annotated" and "electrical" in meta:
        _el = meta["electrical"].fillna(False).to_numpy()
        _ids = meta["banc_888_id"].to_numpy()
        _row = {i: r for r, i in enumerate(_ids)}
        _seen = set()
        for _p, _q in zip(edges["pre"].to_numpy(), edges["post"].to_numpy()):
            _rp, _rq = _row.get(_p), _row.get(_q)
            if _rp is None or _rq is None or _rp == _rq:
                continue
            if _el[_rp] and _el[_rq]:
                _k = (min(_rp, _rq), max(_rp, _rq))
                if _k not in _seen:
                    _seen.add(_k)
        gap_pairs = tuple(sorted(_seen))
    # F27: hypothesised premotor-inhibition gain. Rows = the leg
    # neuromere's inhibitory premotor hemilineages (13A, 21A) by the
    # meta's own hemilineage column. Default 1.0 = bit-identical.
    inh_gain = float(th.get("inh_gain", 1.0))
    inh_rows = ()
    if inh_gain != 1.0:
        _hl = meta["hemilineage"].astype(str) if "hemilineage" in meta \
            else meta["cell_type"].astype(str)
        _m = _hl.str.contains("13A", na=False) | _hl.str.contains("21A",
                                                                  na=False)
        inh_rows = tuple(int(r) for r in np.flatnonzero(_m.to_numpy()))
    inh_13a_21a_gain = None
    if th.get("inh_13a_21a_ratio") is not None:
        inh_13a_21a_gain = premotor_13a_21a_gain_vector(
            meta, th["inh_13a_21a_ratio"])
    # AUDIT 2026-08-19. The cache key used to be built from PARAMETERS
    # ONLY -- the two arguments the network is actually built from, the
    # neuron table and the edge list, appeared nowhere in it. Hand one
    # cache two different brains and the second silently got the first
    # one's network: measured, real connectome 0.918466 mm, shuffled
    # connectome through the same cache 0.918466 mm bit-identical, no
    # rebuild; with a fresh cache the shuffled brain gives 0.847465.
    # Six scripts had each defused this by hand, one of them with the
    # comment "the cache key inside evaluate does not know the
    # connectome, so give each arm its own dict". eval_job.py is what
    # happens when the seventh author misses -- and eval_job drives the
    # NULL PROGRAM, the experiment that asks whether the connectome
    # matters at all. Identity, not content: shape plus the first and
    # last ids and edge endpoints, which is O(1) and distinguishes a
    # shuffle from an original.
    def _brain_id(m, e):
        return (len(m), len(e),
                str(m["banc_888_id"].iat[0]), str(m["banc_888_id"].iat[-1]),
                str(e["pre"].iat[0]), str(e["post"].iat[0]),
                str(e["pre"].iat[-1]), str(e["post"].iat[-1]),
                float(e["count"].sum()))
    _serotonin_keys = {"serotonin_tau_ms", "serotonin_gain_mv_per_hz"}
    # design b1746728, Builds 1, 1b and 2: optional keys of the state, each
    # absent = the R135 path, its cache key and its receipt
    _serotonin_opt = ("serotonin_gain_bound_mv_per_hz", "serotonin_init_hz",
                      "serotonin_target_set")
    _serotonin_present = any(key_ in th for key_ in _serotonin_keys)
    serotonin_source_rows = serotonin_target_rows = ()
    serotonin_tau_ms = serotonin_gain_mv_per_hz = None
    serotonin_gain_bound = serotonin_target_set = None
    serotonin_init_hz = 0.0
    if not _serotonin_present and any(key_ in th for key_ in _serotonin_opt):
        raise ValueError(
            "serotonin_gain_bound_mv_per_hz, serotonin_init_hz and "
            "serotonin_target_set need serotonin_tau_ms and serotonin_gain_mv_per_hz")
    if _serotonin_present:
        if not _serotonin_keys.issubset(th):
            raise ValueError(
                "serotonin_tau_ms and serotonin_gain_mv_per_hz must be set together")
        serotonin_tau_ms = float(th["serotonin_tau_ms"])
        serotonin_gain_mv_per_hz = float(th["serotonin_gain_mv_per_hz"])
        if (not np.isfinite(serotonin_tau_ms)
                or not 100.0 <= serotonin_tau_ms <= 30000.0):
            raise ValueError(
                "serotonin_tau_ms must be in the mapped 100-30000 ms prior")
        if th.get("serotonin_gain_bound_mv_per_hz") is None:
            if (not np.isfinite(serotonin_gain_mv_per_hz)
                    or not (serotonin_gain_mv_per_hz == 0.0
                            or 0.001 <= serotonin_gain_mv_per_hz <= 0.5)):
                raise ValueError(
                    "serotonin_gain_mv_per_hz must be zero or in the mapped 0.001-0.5 prior")
        else:
            # Build 1b: a design whose drive ladder needs G = D/s_dark above
            # the R135 map's 0.5 declares its own bound, and the receipt
            # carries it
            _gb = th["serotonin_gain_bound_mv_per_hz"]
            if (not isinstance(_gb, (list, tuple)) or len(_gb) != 2
                    or not all(isinstance(x, (int, float))
                               and not isinstance(x, bool) for x in _gb)):
                raise ValueError("serotonin_gain_bound_mv_per_hz is [lo, hi] in mV/Hz")
            serotonin_gain_bound = (float(_gb[0]), float(_gb[1]))
            _lo, _hi = serotonin_gain_bound
            if not (np.isfinite(_lo) and np.isfinite(_hi) and 0.0 < _lo <= _hi):
                raise ValueError(
                    "serotonin_gain_bound_mv_per_hz needs finite 0 < lo <= hi")
            if (not np.isfinite(serotonin_gain_mv_per_hz)
                    or not (serotonin_gain_mv_per_hz == 0.0
                            or _lo <= serotonin_gain_mv_per_hz <= _hi)):
                raise ValueError(
                    f"serotonin_gain_mv_per_hz must be zero or in the declared "
                    f"{_lo}-{_hi} bound")
        if th.get("serotonin_init_hz") is not None:
            serotonin_init_hz = float(th["serotonin_init_hz"])
            if not np.isfinite(serotonin_init_hz) or serotonin_init_hz < 0.0:
                raise ValueError("serotonin_init_hz must be finite and non-negative")
        serotonin_source_rows, serotonin_target_rows = (
            serotonin_vnc_5ht7_rows(meta))
        if th.get("serotonin_target_set") is not None:
            serotonin_target_rows, serotonin_target_set = (
                serotonin_target_set_rows(meta, th["serotonin_target_set"]))
    # Design 8b6e0441, Build 2: the octopamine offset, octopamine_offset_mv
    # (mV, negative lowers excitability) added into the membrane equation's
    # extra term on every row of octopamine_target_set every step. Build 1:
    # the octopamine state, octopamine_source_set, octopamine_tau_ms and
    # octopamine_gain_mv_per_hz together (octopamine_init_hz and
    # octopamine_gain_bound_mv_per_hz optional), onto the same target set.
    # The target set comes with the offset, the state or both; absent =
    # bit-identical.
    octopamine_offset_mv = octopamine_target_rows = octopamine_target_set = None
    _os_keys = ("octopamine_source_set", "octopamine_tau_ms", "octopamine_gain_mv_per_hz")
    _os_opt = ("octopamine_init_hz", "octopamine_gain_bound_mv_per_hz")
    _os_present = any(th.get(k_) is not None for k_ in _os_keys)
    octopamine_source_rows = octopamine_source_set = octopamine_gain_bound = None
    octopamine_tau_ms = octopamine_gain_mv_per_hz = None
    octopamine_init_hz = 0.0
    if not _os_present and any(th.get(k_) is not None for k_ in _os_opt):
        raise ValueError("octopamine_init_hz and octopamine_gain_bound_mv_per_hz need "
                         "the octopamine state's keys")
    if _os_present and not all(th.get(k_) is not None for k_ in _os_keys):
        raise ValueError("octopamine_source_set, octopamine_tau_ms and "
                         "octopamine_gain_mv_per_hz must be set together")
    if th.get("octopamine_offset_mv") is not None and th.get("octopamine_target_set") is None:
        raise ValueError("octopamine_offset_mv needs octopamine_target_set")
    if th.get("octopamine_target_set") is not None and not (
            _os_present or th.get("octopamine_offset_mv") is not None):
        raise ValueError("octopamine_target_set needs octopamine_offset_mv or the octopamine state")
    if _os_present and th.get("octopamine_target_set") is None:
        raise ValueError("the octopamine state needs octopamine_target_set")
    if th.get("octopamine_target_set") is not None:
        octopamine_target_rows, octopamine_target_set = octopamine_set_rows(
            meta, th["octopamine_target_set"], "octopamine_target_set")
    if th.get("octopamine_offset_mv") is not None:
        if isinstance(th["octopamine_offset_mv"], bool):
            raise ValueError("octopamine_offset_mv is a number in mV")
        octopamine_offset_mv = float(th["octopamine_offset_mv"])
        if not np.isfinite(octopamine_offset_mv):
            raise ValueError("octopamine_offset_mv must be finite")
    if _os_present:
        octopamine_source_rows, octopamine_source_set = octopamine_set_rows(
            meta, th["octopamine_source_set"], "octopamine_source_set")
        for k_ in ("octopamine_tau_ms", "octopamine_gain_mv_per_hz", "octopamine_init_hz"):
            if isinstance(th.get(k_), bool):
                raise ValueError(f"{k_} is a number")
        octopamine_tau_ms = float(th["octopamine_tau_ms"])
        if not np.isfinite(octopamine_tau_ms) or not 100.0 <= octopamine_tau_ms <= 300000.0:
            raise ValueError("octopamine_tau_ms must be in 100-300000 ms")
        octopamine_gain_mv_per_hz = float(th["octopamine_gain_mv_per_hz"])
        _lo, _hi = 0.001, 0.5
        if th.get("octopamine_gain_bound_mv_per_hz") is not None:
            # a design whose pairing gain Δr/s_I is above 0.5 declares its
            # own bound, and the receipt carries it (as serotonin's Build 1b)
            _gb = th["octopamine_gain_bound_mv_per_hz"]
            if (not isinstance(_gb, (list, tuple)) or len(_gb) != 2
                    or not all(isinstance(x, (int, float))
                               and not isinstance(x, bool) for x in _gb)):
                raise ValueError("octopamine_gain_bound_mv_per_hz is [lo, hi] in mV/Hz")
            _lo, _hi = float(_gb[0]), float(_gb[1])
            if not (np.isfinite(_lo) and np.isfinite(_hi) and 0.0 < _lo <= _hi):
                raise ValueError("octopamine_gain_bound_mv_per_hz needs finite 0 < lo <= hi")
            octopamine_gain_bound = (_lo, _hi)
        if (not np.isfinite(octopamine_gain_mv_per_hz)
                or not (octopamine_gain_mv_per_hz == 0.0
                        or _lo <= octopamine_gain_mv_per_hz <= _hi)):
            raise ValueError(f"octopamine_gain_mv_per_hz must be zero or in {_lo}-{_hi}")
        if th.get("octopamine_init_hz") is not None:
            octopamine_init_hz = float(th["octopamine_init_hz"])
            if not np.isfinite(octopamine_init_hz) or octopamine_init_hz < 0.0:
                raise ValueError("octopamine_init_hz must be finite and non-negative")
    # LAYER 1 receptor seam (lane A, 2026-09-02): a list of class entries,
    # validated here so a malformed table fails before any run. Absent =
    # off, and the cache key is untouched for every historical theta.
    receptor_classes = th.get("receptor_classes") or ()
    if receptor_classes:
        if not isinstance(receptor_classes, (list, tuple)):
            raise ValueError("receptor_classes must be a list of entries")
        for _entry in receptor_classes:
            if not isinstance(_entry, dict) or "post_cell_class" not in _entry:
                raise ValueError(
                    "each receptor class needs post_cell_class (and optionally "
                    "pre_nt, pre_nt_exclusive, sign, gain)")
        receptor_classes = tuple(dict(_entry) for _entry in receptor_classes)
    _presynaptic_inhibition_mode = th.get("presynaptic_inhibition_mode")
    presynaptic_inhibition_rows = ()
    presynaptic_inhibition_source_rows = None
    _presynaptic_inhibition_gain = th.get("presynaptic_inhibition_gain")
    _presynaptic_inhibition_tau = th.get("presynaptic_inhibition_tau_ms")
    # design 7e4173ba, arm 1b's twin: the hemilineage that feeds the shunt's
    # state; absent is 09A, the body's own source, and the network unchanged
    _presynaptic_inhibition_source_hl = th.get("presynaptic_inhibition_source_hemilineage")
    if (_presynaptic_inhibition_source_hl is not None
            and _presynaptic_inhibition_mode != "hook_shunt_9a"):
        raise ValueError(
            "presynaptic_inhibition_source_hemilineage belongs to 'hook_shunt_9a'")
    if _presynaptic_inhibition_mode in ("hook_shunt", "hook_shunt_9a"):
        # the production encoder's membership rather than a parallel selector
        presynaptic_inhibition_rows = tuple(
            int(r) for r in hook_rows_encoded(meta, th))
        if not (th.get("cbased") and th.get("shunt")):
            raise ValueError(
                "hook presynaptic shunt requires true conductance shunting")
        if _presynaptic_inhibition_gain is not None and not (
                np.isfinite(float(_presynaptic_inhibition_gain))
                and float(_presynaptic_inhibition_gain) >= 0.0):
            raise ValueError("presynaptic_inhibition_gain must be finite and >= 0")
        if _presynaptic_inhibition_mode == "hook_shunt_9a":
            # the hook row's factor reads a state fed by hemilineage 09A
            # alone; cns keeps the 09A rows with an inhibitory sign
            _src_hl = ("09A" if _presynaptic_inhibition_source_hl is None
                       else str(_presynaptic_inhibition_source_hl))
            presynaptic_inhibition_source_rows = tuple(int(r) for r in np.flatnonzero(
                meta["hemilineage"].to_numpy() == _src_hl))
            if _presynaptic_inhibition_source_hl is not None and not presynaptic_inhibition_source_rows:
                raise ValueError(f"hemilineage {_src_hl!r} has no rows")
            if _presynaptic_inhibition_tau is not None and not (
                    np.isfinite(float(_presynaptic_inhibition_tau))
                    and float(_presynaptic_inhibition_tau) > 0.0):
                raise ValueError("presynaptic_inhibition_tau_ms must be finite and > 0")
        elif _presynaptic_inhibition_tau is not None:
            raise ValueError(
                "presynaptic_inhibition_tau_ms belongs to 'hook_shunt_9a'")
    elif _presynaptic_inhibition_mode not in (None, ""):
        raise ValueError(
            "unknown presynaptic_inhibition_mode: "
            f"{_presynaptic_inhibition_mode!r}")
    elif _presynaptic_inhibition_gain is not None or _presynaptic_inhibition_tau is not None:
        raise ValueError(
            "presynaptic_inhibition_gain and _tau_ms need a presynaptic_inhibition_mode")
    _spike_reg_mode = th.get("spike_reg_mode")
    spike_reg_per_neuron = None
    if _spike_reg_mode == "afferent_only":
        spike_reg_per_neuron = afferent_regularity_vector(
            meta, th.get("spike_reg", 1.0))
    elif _spike_reg_mode not in (None, ""):
        raise ValueError(f"unknown spike_reg_mode: {_spike_reg_mode!r}")
    _feco_13b_analog_mode = th.get("feco_13b_analog_mode")
    feco_13b_analog = None
    if _feco_13b_analog_mode:
        if _feco_13b_analog_mode not in (
                "structured_13b", "permuted_13b"):
            raise ValueError("unknown feco_13b_analog_mode")
        if th.get("graded") != "19a006_13ba":
            raise ValueError(
                "FeCO-13B analog replacement requires graded=19a006_13ba")
        feco_13b_analog = feco_13b_analog_plan(
            meta, edges, _feco_13b_analog_mode,
            th["feco_13b_analog_permutation_seed"])
    _graded_13b_command_mode = th.get("graded_13b_command_mode")
    graded_13b_command = None
    if _graded_13b_command_mode:
        if th.get("graded") != "19a006_13ba":
            raise ValueError(
                "graded 13B command requires graded=19a006_13ba")
        _g13_level = float(th["graded_13b_command_level"])
        _g13_start = float(th["graded_13b_command_start_ms"])
        _g13_stop = float(th["graded_13b_command_stop_ms"])
        if (not np.isfinite(_g13_level) or not 0.0 <= _g13_level <= 1.0
                or not np.isfinite(_g13_start)
                or not np.isfinite(_g13_stop)
                or not 0.0 <= _g13_start < _g13_stop):
            raise ValueError("invalid graded 13B command level/window")
        _g13_rows = graded_13b_command_rows(
            meta, _graded_13b_command_mode,
            th["graded_13b_command_permutation_seed"])
        graded_13b_command = {
            "mode": _graded_13b_command_mode,
            "rows": np.asarray(_g13_rows, dtype=np.int64),
            "level": _g13_level,
            "start_ms": _g13_start,
            "stop_ms": _g13_stop,
        }
    # Design 336cacb4 Build 1 (study efc222c7): the command selector as a
    # frozen id list with its hash, reusing the level and window keys. The
    # rows join the drive through the same dict and _pack path as the mode's
    # rows. The list is checked in place of the mode's 18/2 assertion: every
    # id in meta, distinct, and inside the built graded set, because a
    # non-graded row would run as a 1 Hz Poisson no-op. Absent = bit-identical.
    graded_command_ids_sha256 = None
    if th.get("graded_command_ids") is not None:
        if _graded_13b_command_mode:
            raise ValueError(
                "graded_command_ids and graded_13b_command_mode are exclusive")
        _gc_ids = [str(x) for x in th["graded_command_ids"]]
        if not _gc_ids or len(set(_gc_ids)) != len(_gc_ids):
            raise ValueError(
                "graded_command_ids must be a non-empty list of distinct ids")
        graded_command_ids_sha256 = hashlib.sha256("\n".join(
            sorted(_gc_ids, key=int)).encode()).hexdigest()
        if graded_command_ids_sha256 != th.get("graded_command_ids_sha256"):
            raise ValueError(
                f"graded_command_ids hash {graded_command_ids_sha256} differs "
                f"from graded_command_ids_sha256 "
                f"{th.get('graded_command_ids_sha256')!r}")
        _gc_level = float(th["graded_13b_command_level"])
        _gc_start = float(th["graded_13b_command_start_ms"])
        _gc_stop = float(th["graded_13b_command_stop_ms"])
        if (not np.isfinite(_gc_level) or not 0.0 <= _gc_level <= 1.0
                or not np.isfinite(_gc_start)
                or not np.isfinite(_gc_stop)
                or not 0.0 <= _gc_start < _gc_stop):
            raise ValueError("invalid graded command level/window")
        _gc_pos = pd.Series(
            np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
        _gc_missing = [x for x in _gc_ids if x not in _gc_pos.index]
        if _gc_missing:
            raise ValueError(
                f"graded_command_ids names {len(_gc_missing)} ids absent "
                f"from meta, first {_gc_missing[0]!r}")
        _gc_rows = np.sort(np.asarray(
            [int(_gc_pos.loc[x]) for x in _gc_ids], dtype=np.int64))
        _gc_outside = np.setdiff1d(
            _gc_rows, np.asarray(graded_rows, dtype=np.int64))
        if _gc_outside.size:
            raise ValueError(
                f"graded_command_ids names {_gc_outside.size} rows outside "
                f"the built graded set, first "
                f"{str(meta['banc_888_id'].iloc[int(_gc_outside[0])])!r}")
        graded_13b_command = {
            "mode": "ids",
            "rows": _gc_rows,
            "level": _gc_level,
            "start_ms": _gc_start,
            "stop_ms": _gc_stop,
        }
    # Design 336cacb4 Build 3: a stimulus-side multiplier m on the commanded
    # rows' graded release while their command is on,
    #   rel_i = G_i * [1 + (m - 1) * chi(t) * c_i] * f(v_i),
    # applied by replacing those rows' device gain at the window that opens
    # the command and restoring it at the window that closes it
    # (cns_metal.run_metal's graded_gain_window, CUDA only). Delivered input
    # stays graded_every * (W_g @ rel). Absent = bit-identical.
    graded_gain_window = None
    if th.get("graded_command_release_gain") is not None:
        _gm = float(th["graded_command_release_gain"])
        if not np.isfinite(_gm) or _gm <= 0.0:
            raise ValueError(
                "graded_command_release_gain must be finite and > 0")
        if graded_13b_command is None:
            raise ValueError(
                "graded_command_release_gain requires a graded command "
                "selector")
        graded_gain_window = {
            "rows": graded_13b_command["rows"],
            "factor": _gm,
            "start_ms": graded_13b_command["start_ms"],
            "stop_ms": graded_13b_command["stop_ms"],
        }
    _graded_13b_gain = th.get("graded_13b_gain")
    if _graded_13b_gain is not None:
        _graded_13b_gain = float(_graded_13b_gain)
        if (not np.isfinite(_graded_13b_gain)
                or _graded_13b_gain <= 0.0
                or th.get("graded") != "19a006_13ba"
                or th.get("graded_add")):
            raise ValueError(
                "graded_13b_gain requires a positive 19a006_13ba-only set")
        _graded_13b_gain_mode = th.get(
            "graded_13b_gain_mode", "structured_13b")
        if _graded_13b_gain_mode not in ("structured_13b", "permuted_13b"):
            raise ValueError("unknown graded_13b_gain_mode")
    _adapt_a = theta_adapt_a(th)
    key = (_brain_id(meta, edges),
           round(th["adapt_b"], 3),
           th.get("mn_gains") if use_size else "",
           round(inh_gain, 3), len(inh_rows),
           len(gap_pairs), round(float(th.get("g_gap", 0.0)), 4),
           bool(th.get("cbased")), bool(th.get("shunt")),
           round(float(th.get("w_syn_scale", 1.0)), 4),
           ("unset" if th.get("bal_target") is None
            else round(float(th["bal_target"]), 4)),
           round(float(th.get("bal_spread", 0.0) or 0.0), 5),
           round(float(th.get("bal_by_type", 0.0) or 0.0), 5),
           tuple(sorted(th["bal_exempt"])) if th.get("bal_exempt") else (),
           round(float(th.get("std_U", 0.0)), 5),
           round(float(th.get("spike_reg", 1.0)), 2),
           round(float(th.get("std_tau_rec", 400.0)), 1),
           round(float(th.get("graded_gain", -1.0)), 4),
           round(float(th.get("graded_phasic", 0.0)), 4),
           round(float(th.get("graded_v50", -48.0)), 3),
           round(float(th.get("graded_slope", 2.5)), 3),
           int(th.get("graded_every", 1) or 1),
           th.get("mn_vreset", ""),
           round(float(th.get("mn_tonic_slow_hz", 0.0) or 0.0), 3),
           round(float(th.get("delay", 1.8)), 2),
           round(float(th.get("tau_w", 100.0)), 1),
           th.get("graded", ""), globals().get("_WSYN_OVERRIDE"),
           globals().get("_BAL_OVERRIDE"))
    if _adapt_a not in (None, 0.0):
        key = key + ("adapt_a", _adapt_a)
    if _meta_patch_receipt is not None:     # _brain_id cannot see a label change
        key = key + ("meta_patch", _meta_patch_receipt["sha256"][:16])
    if th.get("mn_size_flat"):
        key = key + ("mn_size_flat", th["mn_size_flat"])
    if _sp_kw:
        key = key + ("size_pools", _size_pool_sha(size_pool_map))
    _pg_by_id_key = th.get("presyn_gain_by_id")
    if _pg_by_id_key:
        import hashlib as _hl
        key = key + ("presyn_gain_by_id", _hl.sha256(json.dumps(
            {str(k): float(v) for k, v in _pg_by_id_key.items()},
            sort_keys=True).encode()).hexdigest()[:16])
    # ASSIGNMENT ARMS, default-off (design 43eb3960, study 8d6eb80a). Both keys
    # change the built network, so both enter the cache key: without them arm A
    # and arm C would share one cached object and the ladder would compare a body
    # with itself. Absent = key untouched = the same object as before.
    _rfac_by_id_key = th.get("mn_rfac_by_id")
    if _rfac_by_id_key:
        import hashlib as _hl
        key = key + ("mn_rfac_by_id", _hl.sha256(json.dumps(
            {str(k): float(v) for k, v in _rfac_by_id_key.items()},
            sort_keys=True).encode()).hexdigest()[:16])
    _vrest_by_id_key = th.get("mn_vrest_by_id")       # study 7f1b4a1e B1, as above
    if _vrest_by_id_key:
        import hashlib as _hl
        key = key + ("mn_vrest_by_id", _hl.sha256(json.dumps(
            {str(k): (None if v is None else float(v)) for k, v in _vrest_by_id_key.items()},
            sort_keys=True).encode()).hexdigest()[:16])
    _reset_off_key = th.get("mn_reset_offset_by_id")   # study 9e0afa22 Build 1, as above
    if _reset_off_key and isinstance(_reset_off_key, dict):
        import hashlib as _hl
        key = key + ("mn_reset_offset_by_id", _hl.sha256(json.dumps(
            {str(k): (float(v) if type(v) in (int, float) else repr(v))
             for k, v in _reset_off_key.items()},
            sort_keys=True).encode()).hexdigest()[:16])
    _adapt_b_zero_key = th.get("mn_adapt_b_zero_ids")
    if _adapt_b_zero_key:
        import hashlib as _hl
        key = key + ("mn_adapt_b_zero_ids", _hl.sha256(json.dumps(
            sorted(str(i) for i in _adapt_b_zero_key)).encode()).hexdigest()[:16])
    _commissural_gain = th.get("commissural_gain")
    _commissural_sign = th.get("commissural_sign") or "all"
    if th.get("desc_lr_normalise"):
        key = key + ("desc_lr_normalise", True)
    if _commissural_gain is not None and float(_commissural_gain) != 1.0:
        key = key + ("commissural_gain", round(float(_commissural_gain), 6),
                     str(_commissural_sign))
    _intersegmental_gain = th.get("intersegmental_gain")
    _intersegmental_sign = th.get("intersegmental_sign") or "all"
    if _intersegmental_gain is not None and float(_intersegmental_gain) != 1.0:
        key = key + ("intersegmental_gain",
                     round(float(_intersegmental_gain), 6),
                     str(_intersegmental_sign))
    _intersegmental_gain_exc = th.get("intersegmental_gain_exc")
    _intersegmental_gain_inh = th.get("intersegmental_gain_inh")
    for _k, _v in (("intersegmental_gain_exc", _intersegmental_gain_exc),
                   ("intersegmental_gain_inh", _intersegmental_gain_inh)):
        if _v is not None and float(_v) != 1.0:
            key = key + (_k, round(float(_v), 6))
    # CD1-TF row-pair seams: ids resolved to rows here, the key carries a
    # digest of the ids and gains. Absent = () = untouched key.
    _row_pair_seams = th.get("row_pair_seams") or ()
    row_pair_seam_rows = ()
    if _row_pair_seams:
        import hashlib as _hl
        _id2row = {str(_b): _r for _r, _b in enumerate(
            meta["banc_888_id"].astype(str).to_numpy())}
        _resolved = []
        for _sm in _row_pair_seams:
            if not isinstance(_sm, dict) or not {"pre_ids", "post_ids", "gain"} <= set(_sm):
                raise ValueError("each row_pair_seams entry needs pre_ids, post_ids, gain")
            _pr = tuple(sorted(_id2row[str(_i)] for _i in _sm["pre_ids"]))
            _po = tuple(sorted(_id2row[str(_i)] for _i in _sm["post_ids"]))
            _resolved.append((_pr, _po, float(_sm["gain"])))
        row_pair_seam_rows = tuple(_resolved)
        key = key + ("row_pair_seams", _hl.sha256(
            repr(row_pair_seam_rows).encode()).hexdigest()[:16])
    # Design 0c25fab3: {"pre_ids", "post_ids", "gain"} on the existing
    # positive final weights, applied by cns.CNS after the frozen balance.
    # Canonical BANC ID strings resolve to rows here and a missing or repeated
    # ID raises; the key carries a digest of the rows and the exact gain.
    # Design efe88957 Build 5: a non-empty list of such blocks instead, applied
    # in order to every existing edge between each block's sets, sign kept
    # (cns.CNS._post_balance_edge_gain_blocks), so an inhibitory edge takes the
    # gain too; the dict form and its key are unchanged. Absent = untouched key.
    _pbeg = th.get("post_balance_edge_gain")
    post_balance_edge_gain = ()
    post_balance_edge_gain_blocks = ()
    if _pbeg is not None:
        import hashlib as _hl
        _id2row_pb = {str(_b): _r for _r, _b in enumerate(
            meta["banc_888_id"].astype(str).to_numpy())}

        def _pb_block(_blk, _where):
            if not isinstance(_blk, dict) or set(_blk) != {"pre_ids", "post_ids", "gain"}:
                raise ValueError(f"{_where} needs exactly pre_ids, post_ids, gain")
            _pb_rows = []
            for _nm in ("pre_ids", "post_ids"):
                _ids = [str(_i) for _i in _blk[_nm]]
                if (not _ids or len(set(_ids)) != len(_ids)
                        or any(not (_i.isascii() and _i.isdecimal()
                                    and _i == str(int(_i))) for _i in _ids)):
                    raise ValueError(f"{_where} {_nm} must be unique canonical IDs")
                _missing = [_i for _i in _ids if _i not in _id2row_pb]
                if _missing:
                    raise ValueError(f"{_where} {_nm} absent: {_missing[:3]}")
                _pb_rows.append(tuple(sorted(_id2row_pb[_i] for _i in _ids)))
            _pb_gain = float(_blk["gain"])
            if not np.isfinite(_pb_gain) or _pb_gain <= 0.0:
                raise ValueError(f"{_where} gain must be finite and > 0")
            return (_pb_rows[0], _pb_rows[1], _pb_gain)

        if isinstance(_pbeg, list):
            if not _pbeg:
                raise ValueError("post_balance_edge_gain as a list needs at least one block")
            post_balance_edge_gain_blocks = tuple(
                _pb_block(_b, f"post_balance_edge_gain[{_k}]") for _k, _b in enumerate(_pbeg))
            key = key + ("post_balance_edge_gain_blocks", _hl.sha256(
                repr(tuple((_a, _b, _g.hex()) for _a, _b, _g in post_balance_edge_gain_blocks)
                     ).encode()).hexdigest()[:16])
        else:
            post_balance_edge_gain = _pb_block(_pbeg, "post_balance_edge_gain")
            _pr, _po, _pg = post_balance_edge_gain
            key = key + ("post_balance_edge_gain", _hl.sha256(
                repr((_pr, _po, _pg.hex())).encode()).hexdigest()[:16])
    # Design 478fe148, Build B1: [{"pre_id", "post_id", "count", "label",
    # "citation"}, ...] sets each pair's synapse count in the table cns.CNS
    # builds its weights from, creating the edge where BANC has none, so the
    # sign, w_syn_scale, the balance step and the seams apply to it as to any
    # edge. Ids are canonical and present, pairs distinct, counts integers of
    # at least 1. The key carries a digest of the entries. Absent = untouched
    # key and table.
    edge_count_set = None
    _ecs = th.get("edge_count_set")
    if _ecs is not None:
        import hashlib as _hl
        if not isinstance(_ecs, list) or not _ecs:
            raise ValueError("edge_count_set must be a non-empty list")
        _ecs_ids = set(meta["banc_888_id"].astype(str).to_numpy())
        _ecs_rows, _ecs_pairs = [], set()
        for _e in _ecs:
            if (not isinstance(_e, dict) or set(_e) != {
                    "pre_id", "post_id", "count", "label", "citation"}):
                raise ValueError("each edge_count_set entry needs exactly "
                                 "pre_id, post_id, count, label, citation")
            _ep, _eq = _e["pre_id"], _e["post_id"]
            for _i in (_ep, _eq):
                if (not isinstance(_i, str) or not (_i.isascii() and _i.isdecimal()
                                                    and _i == str(int(_i)))):
                    raise ValueError(f"edge_count_set id {_i!r} is not a canonical id string")
                if _i not in _ecs_ids:
                    raise ValueError(f"edge_count_set id {_i} absent from meta")
            if _ep == _eq or (_ep, _eq) in _ecs_pairs:
                raise ValueError("edge_count_set pairs must be distinct and not self-edges")
            _ecs_pairs.add((_ep, _eq))
            _c = _e["count"]
            if isinstance(_c, bool) or not isinstance(_c, int) or _c < 1:
                raise ValueError("edge_count_set count must be an integer >= 1")
            for _f in ("label", "citation"):
                if not isinstance(_e[_f], str) or not _e[_f]:
                    raise ValueError(f"edge_count_set {_f} must be a non-empty string")
            _ecs_rows.append((_ep, _eq, _c, _e["label"], _e["citation"]))
        edge_count_set = tuple(_ecs_rows)
        key = key + ("edge_count_set", _hl.sha256(
            repr(edge_count_set).encode()).hexdigest()[:16])
    # Design 7b0ad6b2: {"pairs": [[pre_id, post_id], ...], "delay_ms",
    # "tau_ms", "gain"}, the separately delayed exponential states cns.CNS
    # builds after every constructor transform. IDs resolve to rows in the
    # stated pair order, a missing or non-canonical ID raises, and null
    # delay_ms or tau_ms inherits the ordinary value. The key carries a digest
    # of the rows and the exact values. Absent = untouched key.
    # Design 727bccb8 (H02) adds optional "beta" in [0, 1] and "factor_from",
    # a permutation of the pair indices in the stated order (needs beta);
    # with neither, the tuple and the key are H01's.
    _ppk = th.get("psi_pair_kernel")
    psi_pair_kernel = ()
    if _ppk is not None:
        import hashlib as _hl
        _pk_base = {"pairs", "delay_ms", "tau_ms", "gain"}
        if (not isinstance(_ppk, dict) or not _pk_base <= set(_ppk)
                or not set(_ppk) <= _pk_base | {"beta", "factor_from"}
                or ("factor_from" in _ppk and "beta" not in _ppk)):
            raise ValueError("psi_pair_kernel needs exactly pairs, delay_ms, tau_ms, gain "
                             "and optionally beta, then factor_from")
        _id2row_pk = {str(_b): _r for _r, _b in enumerate(
            meta["banc_888_id"].astype(str).to_numpy())}
        _pk_rows = []
        for _pair in _ppk["pairs"]:
            _ids = [str(_i) for _i in _pair]
            if (len(_ids) != 2 or any(not (_i.isascii() and _i.isdecimal()
                                           and _i == str(int(_i))) for _i in _ids)):
                raise ValueError("psi_pair_kernel pairs must be [pre_id, post_id] canonical IDs")
            _missing = [_i for _i in _ids if _i not in _id2row_pk]
            if _missing:
                raise ValueError(f"psi_pair_kernel pair ID absent: {_missing[:2]}")
            _pk_rows.append((_id2row_pk[_ids[0]], _id2row_pk[_ids[1]]))
        if not _pk_rows or len(set(_pk_rows)) != len(_pk_rows):
            raise ValueError("psi_pair_kernel pairs must be nonempty and distinct")
        _pk_vals = []
        for _nm in ("delay_ms", "tau_ms", "gain"):
            _v = _ppk[_nm]
            if _v is None and _nm != "gain":
                _pk_vals.append(None)
                continue
            _v = float(_v)
            if not np.isfinite(_v) or _v <= 0.0:
                raise ValueError(f"psi_pair_kernel {_nm} must be finite and > 0")
            _pk_vals.append(_v)
        psi_pair_kernel = (tuple(_pk_rows), *_pk_vals)
        _pk_digest = (tuple(_pk_rows), *(None if _v is None else _v.hex() for _v in _pk_vals))
        if "beta" in _ppk:
            _beta = float(_ppk["beta"])
            if not np.isfinite(_beta) or _beta < 0.0 or _beta > 1.0:
                raise ValueError("psi_pair_kernel beta must be in [0, 1]")
            _ff = _ppk.get("factor_from")
            if _ff is not None:
                if (not isinstance(_ff, list) or any(type(_x) is not int for _x in _ff)
                        or sorted(_ff) != list(range(len(_pk_rows)))):
                    raise ValueError("psi_pair_kernel factor_from must be a permutation "
                                     "of the pair indices")
                _ff = tuple(_ff)
            psi_pair_kernel = psi_pair_kernel + (_beta, _ff)
            _pk_digest = _pk_digest + (_beta.hex(), _ff)
        key = key + ("psi_pair_kernel", _hl.sha256(repr(_pk_digest).encode()).hexdigest()[:16])
    # Design e05c0eef B1: {"form": "dead_time", "tau_rec_ms"} or {"form":
    # "depletion", "tau_rec_ms", "u"}, the recovery of every pair of
    # psi_pair_kernel, which it needs (cns.psi_pair_recovery_values). The key
    # carries the exact values. Absent = the pre-B1 channel and key.
    _ppr = th.get("psi_pair_recovery")
    psi_pair_recovery = ()
    if _ppr is not None:
        if not psi_pair_kernel:
            raise ValueError("psi_pair_recovery needs psi_pair_kernel")
        _pr_form = _ppr.get("form") if isinstance(_ppr, dict) else None
        if _pr_form not in ("dead_time", "depletion") or set(_ppr) != (
                {"form", "tau_rec_ms"} | ({"u"} if _pr_form == "depletion" else set())):
            raise ValueError("psi_pair_recovery needs exactly form and tau_rec_ms, and u "
                             "with form depletion")
        _pr_tau = float(_ppr["tau_rec_ms"])
        _pr_u = float(_ppr["u"]) if _pr_form == "depletion" else None
        psi_pair_recovery = (_pr_form, _pr_tau, _pr_u)
        cns.psi_pair_recovery_values(psi_pair_recovery, 0.1)
        key = key + ("psi_pair_recovery", _pr_form, _pr_tau.hex(),
                     None if _pr_u is None else _pr_u.hex())
    # Design 81427d47 (H03): {"pre_ids", "post_ids", "mode", "k", "eta",
    # "tau_close_ms"} and optionally "interrupt", the occupancy (mode "T") or
    # linear (mode "L", eta null) state that cns.CNS builds in place of every
    # existing entry from the pre rows onto the post rows. IDs resolve to
    # sorted rows and a missing or non-canonical ID raises; the key carries a
    # digest of the rows and the exact values. Absent = untouched key.
    _socc = th.get("shared_occupancy")
    shared_occupancy = ()
    if _socc is not None:
        import hashlib as _hl
        _so_base = {"pre_ids", "post_ids", "mode", "k", "eta", "tau_close_ms"}
        if (not isinstance(_socc, dict) or not _so_base <= set(_socc)
                or not set(_socc) <= _so_base | {"interrupt"}):
            raise ValueError("shared_occupancy needs exactly pre_ids, post_ids, mode, k, eta, "
                             "tau_close_ms and optionally interrupt")
        _id2row_so = {str(_b): _r for _r, _b in enumerate(
            meta["banc_888_id"].astype(str).to_numpy())}
        _so_rows = []
        for _nm in ("pre_ids", "post_ids"):
            _ids = [str(_i) for _i in _socc[_nm]]
            if (not _ids or len(set(_ids)) != len(_ids)
                    or any(not (_i.isascii() and _i.isdecimal()
                                and _i == str(int(_i))) for _i in _ids)):
                raise ValueError(f"shared_occupancy {_nm} must be unique canonical IDs")
            _missing = [_i for _i in _ids if _i not in _id2row_so]
            if _missing:
                raise ValueError(f"shared_occupancy {_nm} absent: {_missing[:3]}")
            _so_rows.append(tuple(sorted(_id2row_so[_i] for _i in _ids)))
        _so_mode = _socc["mode"]
        if _so_mode not in ("T", "L"):
            raise ValueError("shared_occupancy mode is 'T' or 'L'")
        _so_k, _so_tau = float(_socc["k"]), float(_socc["tau_close_ms"])
        if _so_mode == "T":
            _so_eta = float(_socc["eta"])
            if not np.isfinite(_so_eta) or _so_eta <= 0.0:
                raise ValueError("shared_occupancy eta must be finite and > 0 for mode 'T'")
        elif _socc["eta"] is not None:
            raise ValueError("shared_occupancy mode 'L' takes eta null")
        else:
            _so_eta = None
        if not (np.isfinite(_so_k) and _so_k > 0.0 and np.isfinite(_so_tau) and _so_tau > 0.0):
            raise ValueError("shared_occupancy k and tau_close_ms must be finite and > 0")
        _so_int = _socc.get("interrupt", False)
        if type(_so_int) is not bool:
            raise ValueError("shared_occupancy interrupt is true or false")
        shared_occupancy = (_so_rows[0], _so_rows[1], _so_mode, _so_k, _so_eta, _so_tau, _so_int)
        key = key + ("shared_occupancy", _hl.sha256(repr((
            _so_rows[0], _so_rows[1], _so_mode, _so_k.hex(),
            None if _so_eta is None else _so_eta.hex(), _so_tau.hex(), _so_int
        )).encode()).hexdigest()[:16])
    # Design 0e0587be, Build 3: a per-row release resource. Each entry is
    # {"ids", "U", "tau_rec_ms"}; ids resolve through banc_888_id and a
    # missing id raises; a row named twice raises. The key carries a digest
    # of the sorted rows and their values. Absent = untouched key.
    _release_resource = th.get("release_resource") or ()
    release_resource_rows = None
    if _release_resource:
        import hashlib as _hl
        _id2row_rr = {str(_b): _r for _r, _b in enumerate(
            meta["banc_888_id"].astype(str).to_numpy())}
        _rr_map = {}
        for _e in _release_resource:
            if not isinstance(_e, dict) or set(_e) != {"ids", "U", "tau_rec_ms"}:
                raise ValueError("each release_resource entry needs exactly "
                                 "ids, U and tau_rec_ms")
            for _i in _e["ids"]:
                if str(_i) not in _id2row_rr:
                    raise KeyError(f"release_resource id {_i} is not a "
                                   "banc_888_id")
                _r = _id2row_rr[str(_i)]
                if _r in _rr_map:
                    raise ValueError(f"release_resource names row {_r} twice")
                _rr_map[_r] = (float(_e["U"]), float(_e["tau_rec_ms"]))
        _rr_sorted = sorted(_rr_map)
        release_resource_rows = (
            np.asarray(_rr_sorted, dtype=np.int64),
            np.asarray([_rr_map[_r][0] for _r in _rr_sorted], dtype=np.float64),
            np.asarray([_rr_map[_r][1] for _r in _rr_sorted], dtype=np.float64))
        key = key + ("release_resource", _hl.sha256(repr(
            [(int(_r),) + _rr_map[_r] for _r in _rr_sorted]).encode()
            ).hexdigest()[:16])
    # Design 02e57559, Build 1: the emitted resource reaches only these target
    # rows; every other output of a resource row delivers its full weight.
    # A list of banc_888_id strings; a missing or repeated id raises, and the
    # key needs release_resource. The key carries a digest of the sorted rows.
    # Absent = every target, the existing path.
    _rr_targets = th.get("release_resource_targets")
    release_resource_target_rows = None
    if _rr_targets is not None:
        import hashlib as _hl
        if release_resource_rows is None:
            raise ValueError("release_resource_targets needs release_resource")
        _ids = [str(_i) for _i in _rr_targets]
        if not _ids or len(set(_ids)) != len(_ids):
            raise ValueError("release_resource_targets must be unique ids")
        _id2row_rt = {str(_b): _r for _r, _b in enumerate(
            meta["banc_888_id"].astype(str).to_numpy())}
        _missing = [_i for _i in _ids if _i not in _id2row_rt]
        if _missing:
            raise KeyError(f"release_resource_targets absent: {_missing[:3]}")
        release_resource_target_rows = np.asarray(
            sorted(_id2row_rt[_i] for _i in _ids), dtype=np.int64)
        key = key + ("release_resource_targets", _hl.sha256(repr(
            [int(_r) for _r in release_resource_target_rows]).encode()
            ).hexdigest()[:16])
    # Design fbcd0f61, Build B6: the adaptation state w at t = 0 on the listed
    # rows, {banc_888_id: w0_mv}; absent = zeros, bit-identical. A missing id,
    # a repeated row or a value that is not finite raises, and the engines
    # refuse it when adaptation is off. The key carries a digest of the sorted
    # (row, float32 value) pairs.
    _w_init = th.get("adapt_w_init_by_id")
    adapt_w_init = None
    if _w_init is not None:
        import hashlib as _hl
        if not isinstance(_w_init, dict) or not _w_init:
            raise ValueError("adapt_w_init_by_id takes a non-empty {banc_888_id: mV} map")
        _id2row_wi = {str(_b): _r for _r, _b in enumerate(
            meta["banc_888_id"].astype(str).to_numpy())}
        _missing = [str(_i) for _i in _w_init if str(_i) not in _id2row_wi]
        if _missing:
            raise KeyError(f"adapt_w_init_by_id absent: {_missing[:3]}")
        _wi = sorted((_id2row_wi[str(_i)], float(_v)) for _i, _v in _w_init.items())
        if len({_r for _r, _ in _wi}) != len(_wi):
            raise ValueError("adapt_w_init_by_id names a row twice")
        if not all(np.isfinite(_v) for _, _v in _wi):
            raise ValueError("adapt_w_init_by_id values must be finite")
        adapt_w_init = (np.asarray([_r for _r, _ in _wi], dtype=np.int64),
                        np.asarray([_v for _, _v in _wi], dtype=np.float32))
        key = key + ("adapt_w_init_by_id", _hl.sha256(repr(
            [(int(_r), float(_v)) for _r, _v in zip(*adapt_w_init)]).encode()
            ).hexdigest()[:16])
    # Design fb9d4a58, Builds 1 to 3: the left APL row's GABA sign and the
    # KC-to-KC edges. Absent = untouched key.
    _left_apl_gaba = th.get("left_apl_gaba")
    if _left_apl_gaba is not None and _left_apl_gaba is not True:
        raise ValueError("left_apl_gaba takes true or absence")
    if _left_apl_gaba:
        key = key + ("left_apl_gaba",)
    _kc_kc_ach = th.get("kc_kc_ach")
    _kc_kc_calyx_sha = th.get("kc_kc_calyx_table_sha256")
    kc_kc_ids = ()
    kc_kc_calyx = None
    if _kc_kc_ach is not None:
        if _kc_kc_ach not in ("off", "calyx_kept"):
            raise ValueError('kc_kc_ach takes "off", "calyx_kept" or absence')
        _sel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "gpu", "mushroom", "selector-4904.json")
        with open(_sel_path) as _fh:
            kc_kc_ids = tuple(str(_x) for _x in
                              json.load(_fh)["population_ids"]["kenyon_cells"])
        _meta_ids = set(meta["banc_888_id"].astype(str))
        if (len(kc_kc_ids) != 4552 or len(set(kc_kc_ids)) != 4552
                or not set(kc_kc_ids) <= _meta_ids):
            raise ValueError("kc_kc_ach needs the selector's 4,552 KC ids, all in meta")
        key = key + ("kc_kc_ach", str(_kc_kc_ach))
        if _kc_kc_ach == "calyx_kept":
            if not (isinstance(_kc_kc_calyx_sha, str) and len(_kc_kc_calyx_sha) == 64):
                raise ValueError("kc_kc_ach calyx_kept needs kc_kc_calyx_table_sha256")
            _cp = os.path.join(os.path.expanduser("~"), "uploadinglab-store", "objects",
                               _kc_kc_calyx_sha)
            with open(_cp, "rb") as _fh:
                _raw = _fh.read()
            if hashlib.sha256(_raw).hexdigest() != _kc_kc_calyx_sha:
                raise ValueError(f"{_cp} does not hash to kc_kc_calyx_table_sha256")
            _lines = _raw.decode().splitlines()
            if _lines[0] != "pre,post,n,n_ca":
                raise ValueError(f"the calyx table's header is {_lines[0]!r}")
            kc_kc_calyx = (_kc_kc_calyx_sha, tuple(
                (_a, _b, int(_n), int(_c)) for _a, _b, _n, _c in
                (_ln.split(",") for _ln in _lines[1:])))
            key = key + ("kc_kc_calyx", _kc_kc_calyx_sha[:16])
        elif _kc_kc_calyx_sha is not None:
            raise ValueError("kc_kc_calyx_table_sha256 needs kc_kc_ach calyx_kept")
    elif _kc_kc_calyx_sha is not None:
        raise ValueError("kc_kc_calyx_table_sha256 needs kc_kc_ach calyx_kept")
    _type_out_gain = th.get("type_out_gain") or {}
    _type_out_gain_id = th.get("type_out_gain_id") or {}
    type_out_gain_vec = None
    type_out_gain_audit = None
    if _type_out_gain or _type_out_gain_id:
        if _type_out_gain and (
                not isinstance(_type_out_gain, dict)
                or not all(isinstance(k, str) for k in _type_out_gain)):
            raise ValueError("type_out_gain must be a map of cell type to gain")
        if _type_out_gain_id and (
                not isinstance(_type_out_gain_id, dict)
                or not all(isinstance(k, str) for k in _type_out_gain_id)):
            raise ValueError("type_out_gain_id must be a map of BANC id to gain")
        import hashlib as _hl_tog
        type_out_gain_vec, type_out_gain_audit = type_out_gain_vector(
            meta, _type_out_gain, _type_out_gain_id)
        key = key + ("type_out_gain", _hl_tog.sha256(json.dumps(
            {"type": {str(k): float(v) for k, v in _type_out_gain.items()},
             "id": {str(k): float(v) for k, v in _type_out_gain_id.items()}},
            sort_keys=True).encode()).hexdigest()[:16])
    # Design 2163ba7d, Build 3: ported from fc0d6e9d0 (origin/team-henri), the
    # key's shape and its cache entry unchanged; the hold is this design's field.
    _crossing_tw = th.get("crossing_tau_w_mult")
    if _crossing_tw is not None:
        # checked here as well as at the build, because a cached net skips the build
        if not isinstance(_crossing_tw, dict) or set(_crossing_tw) != {"mult", "rows"}:
            raise ValueError("crossing_tau_w_mult must be {mult, rows}")
        import hashlib as _hl_ctw
        key = key + ("crossing_tau_w_mult", _hl_ctw.sha256(json.dumps(
            {"mult": float(_crossing_tw["mult"]),
             "rows": sorted(str(_r) for _r in _crossing_tw["rows"])},
            sort_keys=True).encode()).hexdigest()[:16])
    if th.get("crossing_tau_w_hold_b_tau") is not None:
        key = key + ("crossing_tau_w_hold_b_tau",
                     repr(th["crossing_tau_w_hold_b_tau"]))
    if receptor_classes:
        key = key + ("receptor_classes", tuple(
            (str(_e.get("pre_nt", "*")), str(_e["post_cell_class"]),
             float(_e.get("sign", 1.0)), round(float(_e.get("gain", 1.0)), 6))
            + (("pre_nt_exclusive",) if _e.get("pre_nt_exclusive") is True else ())
            for _e in receptor_classes))
    if _serotonin_present:
        # Extend only for an explicitly enabled map. Historical thetas retain
        # the exact old cache key and path.
        key = key + (
            "serotonin_5ht7_08b",
            round(serotonin_tau_ms, 6),
            round(serotonin_gain_mv_per_hz, 9))
        # design b1746728: the start and the target set change the run, the
        # bound validates only; each extends the key only when set
        if th.get("serotonin_init_hz") is not None:
            key = key + ("serotonin_init_hz", repr(serotonin_init_hz))
        if serotonin_target_set is not None:
            key = key + ("serotonin_target_set",) + serotonin_target_set
    if octopamine_offset_mv is not None:   # design 8b6e0441, Build 2
        key = key + ("octopamine_offset_mv", repr(octopamine_offset_mv),
                     "octopamine_target_set") + octopamine_target_set
    if _os_present:   # design 8b6e0441, Build 1; the bound validates only
        key = key + ("octopamine_state", repr(octopamine_tau_ms),
                     repr(octopamine_gain_mv_per_hz), repr(octopamine_init_hz),
                     "octopamine_source_set") + octopamine_source_set + (
                     "octopamine_target_set",) + octopamine_target_set
    if _presynaptic_inhibition_mode:
        key = key + ("presynaptic_inhibition",
                     _presynaptic_inhibition_mode,
                     len(presynaptic_inhibition_rows))
        if _presynaptic_inhibition_gain is not None:
            key = key + ("presynaptic_inhibition_gain",
                         float(_presynaptic_inhibition_gain))
        if presynaptic_inhibition_source_rows is not None:
            key = key + ("presynaptic_inhibition_source_rows",
                         len(presynaptic_inhibition_source_rows),
                         None if _presynaptic_inhibition_tau is None
                         else float(_presynaptic_inhibition_tau))
        if _presynaptic_inhibition_source_hl is not None:
            key = key + ("presynaptic_inhibition_source_hemilineage",
                         str(_presynaptic_inhibition_source_hl))
    if _spike_reg_mode:
        key = key + ("spike_reg_mode", _spike_reg_mode)
    if _feco_13b_analog_mode:
        key = key + (
            "feco_13b_analog", _feco_13b_analog_mode,
            int(th["feco_13b_analog_permutation_seed"]),
            feco_13b_analog["edge_pairs"],
            feco_13b_analog["chemical_synapses_replaced"])
    if _graded_13b_command_mode:
        key = key + (
            "graded_13b_command", _graded_13b_command_mode,
            int(th["graded_13b_command_permutation_seed"]))
    if graded_command_ids_sha256 is not None:
        key = key + ("graded_command_ids", graded_command_ids_sha256)
    if _graded_13b_gain is not None:
        key = key + (
            "graded_13b_gain", round(_graded_13b_gain, 9),
            _graded_13b_gain_mode,
            int(th.get("graded_13b_gain_permutation_seed", 0)))
    new_gap_keys = ("gap_law", "gap_reverse_ratio", "gap_substep_factor", "gap_current_tap")
    if any(k in th for k in new_gap_keys):
        if not gap_pairs or not gap_pair_weights:
            raise ValueError("new gap controls require pairs and per-pair weights")
        if th.get("gap_law") not in ("conserving", "one_way"):
            raise ValueError("gap_law must be conserving or one_way")
        if "gap_rectify" in th or float(th.get("gap_spikelet", 0)) != 0:
            raise ValueError("gap_law is incompatible with gap_rectify or nonzero spikelet")
        rho = th.get("gap_reverse_ratio")
        if th["gap_law"] == "conserving":
            if rho is None or not np.isfinite(rho) or not 0 <= rho <= 1:
                raise ValueError("conserving gap_law requires reverse ratio in [0, 1]")
        elif "gap_reverse_ratio" in th:
            raise ValueError("one_way law cannot use gap_reverse_ratio")
        factor = th.get("gap_substep_factor", 1)
        tap = th.get("gap_current_tap", False)
        if type(factor) is not int or factor < 1 or type(tap) is not bool:
            raise ValueError("gap_substep_factor must be a positive integer and gap_current_tap boolean")
        key += ("gap_law", th["gap_law"], rho, factor, tap,
                tuple(gap_pairs), tuple(gap_pair_weights))
    if gap_pair_weights:
        key = key + ("gap_pair_weights", tuple(gap_pairs),
                     tuple(float(value) for value in gap_pair_weights))
    # Optional extension only: absent theta keeps the historical cache key
    # exactly unchanged.
    if th.get("cbased_integrator"):
        key = key + ("cbased_integrator", th["cbased_integrator"])
    # Engine backend (compute seat, 2026-09-02): absent = numpy = the
    # historical key and the bit-identical path.
    if th.get("backend"):
        key = key + ("backend", th["backend"])
    # Homology-filled right-T2 Fast (judge 18:58). Absent = named-only
    # overlay, historical key unchanged.
    if th.get("azevedo_rt2_fast"):
        key = key + ("azevedo_rt2_fast", th["azevedo_rt2_fast"])
    # RH-only tonic (judge 02:21). Absent = all slow-class rows,
    # historical key unchanged.
    if th.get("mn_tonic_scope"):
        key = key + ("mn_tonic_scope", th["mn_tonic_scope"])
    if th.get("camp_leg_gains") is not None:
        key = key + (
            "camp_leg_gains",
            tuple(round(float(x), 4) for x in th["camp_leg_gains"]))
    if th.get("rh_fti_mid_offset_deg") is not None:
        key = key + (
            "rh_fti_mid_offset_deg",
            round(float(th["rh_fti_mid_offset_deg"]), 3))
    if th.get("graded_add"):
        key = key + ("graded_add", th["graded_add"] if isinstance(
                         th["graded_add"], str) else tuple(th["graded_add"]),
                     round(float(th["graded_gain_add"]), 6))
    for _k, _rows_by_type in _type_rows.items():
        if _rows_by_type:
            key = key + (_k, tuple(sorted(
                (_t, _v) for _t, (_r, _v) in _rows_by_type.items())))
    if _nt_transfer_ids is not None:
        key = key + ("nt_transfer_rows", S5585_SHA256,
                     str(_nt_transfer["transmitter"]))
    if graded_add_ids_sha256 is not None:
        key = key + ("graded_add_ids", graded_add_ids_sha256,
                     float(th["graded_gain_add"]))
    if row_set_params is not None:
        key = key + ("row_set_params", tuple(
            (s["ids_sha256"],) + tuple((f, s[f]) for f in _ROW_SET_VALUES if f in s)
            for s in row_set_params))
    if _graded_edge_equilibrium_mode:
        key = key + (
            "graded_edge_equilibrium",
            _graded_edge_equilibrium_mode,
            graded_edge_equilibrium_ids,
        )
    # delay_mode extends the key ONLY when set, so every pre-directive
    # theta's key (and cache hit pattern) is byte-identical to before.
    if th.get("delay_delivery"):
        # a cached network carries the p it was built with, so a bucket-built
        # net would silently serve a run that asked for the queue
        key = key + ("deliv", th["delay_delivery"])
    if th.get("graded_delay_delivery"):
        key = key + ("graded_deliv", th["graded_delay_delivery"])
    if th.get("delay_mode"):
        key = key + (th["delay_mode"],
                     tuple(sorted((str(k), round(float(v), 6)) for k, v in
                                  (th.get("delay_type_ms") or {}).items())),
                     round(float(th.get("delay_mean_ms", th.get("delay", 1.8))), 3),
                     round(float(th.get("delay_base_ms", 0.8)), 3),
                     round(float(th.get("delay_velocity_um_ms", 300.0)), 1),
                     round(float(th.get("delay_cap_ms", 20.0)), 2),
                     int(th.get("delay_groups", 8)))
    # 2026-09-07 (lane A, CD2-ER; folded by the judge): delay_region_scale is
    # read by the build path below and was absent from this key, so two thetas
    # differing only in it collided in the cache and the second silently ran
    # the first's network, 31,688 ventral-nerve-cord rows keeping the first
    # arm's delays. Extended only when set, so every theta without the key
    # keeps a byte-identical key, and a changed key cannot change an output: a
    # fresh process starts empty, so the key decides only whether a hit occurs.
    if th.get("delay_region_scale"):
        key = key + ("delay_region_scale", tuple(sorted(
            (str(_rk), round(float(_rv), 6))
            for _rk, _rv in th["delay_region_scale"].items())))
    # delay_class_scale (design 983f1c29 Build 6) is read by the build path
    # and extends the key only when set, as delay_region_scale does.
    if th.get("delay_class_scale"):
        key = key + ("delay_class_scale", json.dumps(
            th["delay_class_scale"], sort_keys=True))
    # delay_row_ms (design 9bdd809a Build 1) sets named rows' delays, so a
    # cached network built without it carries the wrong vector; it extends
    # the key only when set, by the sha256 of its sorted-items JSON.
    if th.get("delay_row_ms"):
        key = key + ("delay_row_ms", hashlib.sha256(json.dumps(
            {str(_rk): _rv for _rk, _rv in th["delay_row_ms"].items()},
            sort_keys=True, default=repr).encode()).hexdigest())
    # delay_type_id (design 6f704792 Build 4) names rows, which no other key
    # term carries, so it extends the key only when set.
    if th.get("delay_type_id"):
        key = key + ("delay_type_id", tuple(sorted(
            (str(_rk), str(_rv)) for _rk, _rv in th["delay_type_id"].items())))
    # graded_v50 / graded_slope were theta-set and engine-ignored
    # (80254). Extend the key only when set so absent thetas stay
    # byte-identical.
    if th.get("graded_v50") is not None:
        key = key + ("gv50", round(float(th["graded_v50"]), 3))
    if th.get("graded_slope") is not None:
        key = key + ("gslope", round(float(th["graded_slope"]), 3))
    # Design 0af6d1c5 Build 2: the per-type slope is part of the built
    # network; extended only when set.
    if th.get("graded_slope_type"):
        key = key + ("gslope_type", tuple(sorted(
            (str(_k), round(float(_v), 6)) for _k, _v in th["graded_slope_type"].items())))
    if th.get("coincidence_gain"):
        key = key + (
            "coincidence", round(float(th["coincidence_gain"]), 6),
            round(float(th.get("coincidence_lag_ms", 60.0)), 3),
            tuple(sorted(map(str, th.get("coincidence_target_types") or ()))),
            tuple(sorted(map(str, th.get("coincidence_fast_types") or ()))),
            tuple(sorted(map(str, th.get("coincidence_slow_types") or ()))),
            round(float(th.get("coincidence_tau_ms", 5.0)), 3))
    # Design efe88957 Build 2: the seam's list form, keyed by its exact JSON.
    if th.get("coincidence_triples") is not None:
        key = key + ("coincidence_triples", hashlib.sha256(json.dumps(
            th["coincidence_triples"], sort_keys=True).encode()).hexdigest())
    if th.get("threshold_mode"):
        key = key + (
            "threshold", th["threshold_mode"],
            round(float(th.get("threshold_mean_mv", -45.0)), 3),
            round(float(th.get("threshold_span_mv", 4.0)), 3),
            tuple(sorted((str(k), round(float(v), 6)) for k, v in
                         (th.get("threshold_type_mv") or {}).items())))
    if th.get("v_rest_mode"):
        key = key + (
            "v_rest", th["v_rest_mode"],
            tuple(sorted((str(k), round(float(v), 6)) for k, v in
                         (th.get("v_rest_type_mv") or {}).items())))
    # H08 (design fa5adfc8): the reset rows are part of the built network.
    if th.get("v_rest_type_reset_follows") is not None:
        key = key + ("v_rest_type_reset_follows",
                     repr(th["v_rest_type_reset_follows"]))
    if th.get("membrane_mode"):
        key = key + ("membrane", th["membrane_mode"],
                     int(th.get("membrane_seed", -1)))
    if th.get("adaptation_parameter_mode"):
        key = key + (
            "adaptation_parameters", th["adaptation_parameter_mode"])
    # Design 2d12a3b3 (B1): the named rows' constants change the built network.
    if th.get("mn_membrane_by_id") is not None:
        import hashlib as _hl
        key = key + ("mn_membrane_by_id", _hl.sha256(json.dumps(
            th["mn_membrane_by_id"], sort_keys=True, default=str).encode()).hexdigest())
    # Design 6d0042d5 (Build 1): the groups' law changes the built network.
    if th.get("adapt_law_rows") is not None:
        key = key + ("adapt_law_rows", hashlib.sha256(json.dumps(
            th["adapt_law_rows"], sort_keys=True, default=str).encode()).hexdigest())
    # Design d27f82df B1: the named rows' input factors change the built network.
    if th.get("r_in_total_by_id") is not None:
        key = key + ("r_in_total_by_id", hashlib.sha256(json.dumps(
            th["r_in_total_by_id"], sort_keys=True, default=str).encode()).hexdigest())
    for _k27 in ("row_membrane_by_id", "holddown_by_id"):   # design d27f82df B2, B3
        if th.get(_k27) is not None:
            key = key + (_k27, hashlib.sha256(json.dumps(
                th[_k27], sort_keys=True, default=str).encode()).hexdigest())
    if th.get("mn_exact_by_id") is not None:   # design 83177ced Build 1
        _mx = th["mn_exact_by_id"]   # str keys: an id under two key types reaches the refusal, not sort's TypeError
        key = key + ("mn_exact_by_id", hashlib.sha256(json.dumps(
            {str(k): v for k, v in _mx.items()} if isinstance(_mx, dict) else _mx,
            sort_keys=True, default=str).encode()).hexdigest())
    if th.get("graded_parameter_mode"):
        key = key + ("graded_parameters", th["graded_parameter_mode"])
        if th["graded_parameter_mode"] == "hemilineage_19a_13b":
            key = key + (
                round(float(th.get("graded_hemilineage_v50_delta_mv", 0.0)),
                      6),
                round(float(th.get("graded_hemilineage_slope_ratio", 1.0)),
                      6),
                (None if th.get("graded_hemilineage_permutation_seed") is None
                 else int(th["graded_hemilineage_permutation_seed"])))
    if th.get("graded_premotor_v_rest_mode"):
        _gvr_values = th.get("graded_premotor_v_rest_values_mv")
        _gvr_population = th.get(
            "graded_premotor_v_rest_population", "19a006_13ba")
        if _gvr_values is None:
            key = key + (
                "graded_premotor_v_rest",
                th["graded_premotor_v_rest_mode"],
                _gvr_population,
                round(float(th["graded_premotor_v_rest_mv"]), 6),
                int(th["graded_premotor_v_rest_permutation_seed"]))
        else:
            key = key + (
                "graded_premotor_v_rest_empirical",
                th["graded_premotor_v_rest_mode"],
                _gvr_population,
                tuple(round(float(value), 6) for value in _gvr_values),
                int(th["graded_premotor_v_rest_assignment_seed"]),
                int(th["graded_premotor_v_rest_permutation_seed"]))
    if th.get("typed_12b_v_rest_mode"):
        key = key + (
            "typed_12b_v_rest",
            th["typed_12b_v_rest_mode"],
            round(float(th["typed_12b_v_rest_mv"]), 6))
    if th.get("typed_21a_inhibitory_gain_mode"):
        key = key + (
            "typed_21a_inhibitory_gain",
            th["typed_21a_inhibitory_gain_mode"],
            round(float(th["typed_21a_inhibitory_gain"]), 6))
    if th.get("typed_19b_excitatory_gain_mode"):
        key = key + (
            "typed_19b_excitatory_gain",
            th["typed_19b_excitatory_gain_mode"],
            round(float(th["typed_19b_excitatory_gain"]), 6))
    if th.get("typed_21a_excitatory_gain_mode"):
        key = key + (
            "typed_21a_excitatory_gain",
            th["typed_21a_excitatory_gain_mode"],
            round(float(th["typed_21a_excitatory_gain"]), 6))
    if th.get("inh_13a_21a_ratio") is not None:
        key = key + ("inh_13a_21a_ratio",
                     round(float(th["inh_13a_21a_ratio"]), 6))
    if th.get("synaptic_decay_mode"):
        key = key + ("synaptic_decay", th["synaptic_decay_mode"],
                     int(th.get("synaptic_decay_seed", -1)))
    if th.get("inhibitory_reversal_mode"):
        key = key + (
            "inhibitory_reversal", th["inhibitory_reversal_mode"],
            int(th.get("inhibitory_reversal_seed", -1)),
            int(th.get("inhibitory_reversal_permutation_seed", -1)))
    if th.get("split_synaptic_decay_mode"):
        key = key + (
            "split_synaptic_decay", th["split_synaptic_decay_mode"],
            round(float(th.get("tau_syn_exc_ms", 1.1)), 3),
            round(float(th.get("tau_syn_inh_ms", 5.2)), 3))
    if transmitter_map:
        key = key + ("transmitter_map", tuple(sorted(transmitter_map.items())))
    if cell_type_patch:
        key = key + ("cell_type_patch", tuple(sorted(
            (k, v["cell_type"], v.get("neurotransmitter")) for k, v in cell_type_patch.items())))
    if _hemilineage_nt_audit is not None:
        key += ("hemilineage_nt", _hemilineage_nt_audit["rule_sha256"],
                _hemilineage_nt_audit["resolved_mask_sha256"])
    if balance_scale_override is not None:
        # In the key because it changes the built network: G-f and G-b differ
        # only here, and without this line they hash to one key, the second
        # arm takes the first arm's cached net, and every probe of the key
        # passes by cache reuse. Same reason as the row_pair_seams entries.
        key = key + ("balance_scale_override", str(balance_scale_sha256))
    if th.get("silence_rows") is not None:
        key = key + ("silence_rows",
                     tuple(sorted(str(x) for x in th["silence_rows"])))
    if th.get("silence_graded_rows") is not None:
        key = key + ("silence_graded_rows",
                     tuple(sorted(str(x) for x in th["silence_graded_rows"])))
    if th.get("adapt_zero_ids") is not None:
        key = key + ("adapt_zero_ids",
                     tuple(sorted(str(x) for x in th["adapt_zero_ids"])))
    if th.get("pulse_exc_ids") is not None:
        # Design 2163ba7d, Build 1: the id list, its hash and every pulse key
        key = key + ("pulse_exc", str(th.get("pulse_exc_ids_sha256")),
                     tuple(sorted(str(x) for x in th["pulse_exc_ids"])),
                     tuple(repr(_v) for _v in _pulse_exc_values(th).values()))
    if th.get("pulse_exc_sets") is not None:
        # Design 568d05a8, Build B5: every set's hash, ids and g, and the shared pulse keys
        _pes, _pes_v = _pulse_exc_sets(th)
        key = key + ("pulse_exc_sets",
                     tuple((s["ids_sha256"], tuple(sorted(s["ids"], key=int)), repr(s["g"]))
                           for s in _pes),
                     tuple(repr(_v) for _v in _pes_v.values()))
    if th.get("slow_inh_out") is not None:
        # Design 2163ba7d, Build 2: the ids and both values set p before the build
        _si_key = _slow_inh_spec(th)
        key = key + ("slow_inh_out", _si_key["ids_sha256"], repr(_si_key["frac"]),
                     repr(_si_key["tau_ms"]))
    if th.get("cmd_presyn_gain") is not None:
        key = key + (
            "cmd_presyn_gain",
            str(th.get("cmd", "")),
            round(float(th["cmd_presyn_gain"]), 6))
    # Design afc8fd03 B1: A4's three wing power keys set p before the build, so
    # two thetas that differ only in them must not share a cached network
    # (team-henri adds the same lines at 8089-8092). `is not None`, never
    # truthiness: false is a valid reset_follows_rest. Absent = untouched key.
    for _wp_key in ("wing_power_mn_rest_mv", "wing_power_mn_reset_follows_rest",
                    "wing_power_mn_r_in_factor"):
        if th.get(_wp_key) is not None:
            key = key + (_wp_key, repr(th[_wp_key]))
    # Design adcf11c7: the input-resistance law. The key carries the realized
    # override's hash; the override needs the size rule's rows and gm needs
    # w_syn_scale over wm1's, so every composition that rewrites either raises.
    _r_in_law = th.get("r_in_law")
    _r_in_law_ov = None
    if _r_in_law is not None:
        if not use_size:
            raise ValueError("r_in_law needs mn_gains 'size' or 'azevedo_label'")
        for _k in ("mn_rfac_by_id", "mn_size_flat", "feco_13b_analog_mode"):
            # Design d033eefd B2: a map whose every value is 1.0 composes; the
            # override is computed on the rfac it writes (rfac_by_id below).
            if th.get(_k) and not (_k == "mn_rfac_by_id" and _r_in_law_rfac_map(th[_k])):
                raise ValueError(f"r_in_law cannot compose with {_k}")
        if globals().get("_WSYN_OVERRIDE") is not None:
            raise ValueError("r_in_law cannot compose with _WSYN_OVERRIDE")
        # Design 80b3da76 B2: the giant-fibre rows the theta's r_in_total_by_id
        # names are written by the law (port 3); any other id refuses at the
        # apply step.
        _gf_total = {str(b): f for b, f in (th.get("r_in_total_by_id") or {}).items()
                     if str(b) in _R_IN_LAW_GF} or None
        _r_in_law_ov, _r_in_law_sha, _r_in_law_audit = r_in_law_override(
            meta, edges, _r_in_law, **_sp_kw, rfac_by_id=th.get("mn_rfac_by_id") or None,
            gf_total_by_id=_gf_total)
        # Design d033eefd B1: base_wss, when the law carries it, replaces wm1's
        # w_syn_scale as the value gm is taken over.
        _base_wss = (float(_r_in_law["base_wss"]) if "base_wss" in _r_in_law
                     else _R_IN_LAW_WSS)
        _gm_wss = _base_wss * _r_in_law_audit["gm"]
        if abs(float(th.get("w_syn_scale") or 0.0) - _gm_wss) > 2 * np.spacing(_gm_wss):
            raise ValueError(
                f"r_in_law: w_syn_scale {th.get('w_syn_scale')!r} is not "
                f"{_base_wss!r} x gm = {_gm_wss!r}")
        key = key + ("r_in_law", _r_in_law_sha)
    # Standing sweep S3 (gpu-5, 2026-09-29): the excitatory and inhibitory
    # reversal potentials as theta keys, so a sweep point on either can be
    # retained into the build body; gpu/retain_standing_sweep.py refuses an
    # arm made through SimParams. Absent = SimParams' 0 and -70 mV, untouched.
    _e_rev = {}
    for _k in ("e_exc_mv", "e_inh_mv"):
        if th.get(_k) is not None:
            _e_rev[_k] = float(th[_k])
            if not np.isfinite(_e_rev[_k]):
                raise ValueError(f"{_k} must be a finite reversal in mV")
            key = key + (_k, _e_rev[_k])
    if "e_inh_mv" in _e_rev and th.get("inhibitory_reversal_mode"):
        raise ValueError("e_inh_mv does not combine with inhibitory_reversal_mode")
    # Standing sweep S4 (gpu-2, 2026-09-29): Shiu's five scalar membrane and
    # synapse constants as theta keys, for the same reason. v_rest_mv sets
    # v_reset with v_rest, since Shiu's neuron resets to rest. Absent =
    # SimParams' -52, -45, 20, 5 and 2.2, untouched. Per-row overlays built
    # on the scalar (the size rest, mn_membrane_by_id, threshold_type_mv)
    # still apply over it; the modes that overwrite it refuse the key.
    _membrane = {}
    for _k, _lo, _hi in (("v_rest_mv", -90.0, -30.0), ("v_threshold_mv", -70.0, -20.0),
                         ("tau_mem_ms", 1.0, 100.0), ("tau_syn_ms", 0.5, 50.0),
                         ("t_refrac_ms", 0.1, 20.0)):
        if th.get(_k) is not None:
            _membrane[_k] = float(th[_k])
            if not _lo <= _membrane[_k] <= _hi:
                raise ValueError(f"{_k} must be finite in [{_lo:g}, {_hi:g}]")
            key = key + (_k, _membrane[_k])
    if _membrane and th.get("membrane_mode"):
        raise ValueError(f"{sorted(_membrane)} do not combine with membrane_mode")
    if "v_threshold_mv" in _membrane and th.get("threshold_mode") in (
            "volume_rank", "mcns_active_proxy"):
        raise ValueError("v_threshold_mv does not combine with threshold_mode "
                         f"{th['threshold_mode']!r}")
    if "tau_syn_ms" in _membrane and (th.get("synaptic_decay_mode")
                                      or th.get("split_synaptic_decay_mode")):
        raise ValueError("tau_syn_ms does not combine with a synaptic decay mode")
    # Standing sweep S19 (gpu-7, 2026-10-07): exponential IF's slope factor
    # as a theta key, so a delta_T arm is a theta arm a retention can write
    # into the build body. Absent = SimParams' 0, the leaky integrate-and-fire
    # neuron, untouched; params.py bounds it at 0 to 3 mV, and v_peak stays
    # SimParams' -30 mV. The native engines refuse delta_T above 0.
    _delta_T = None
    if th.get("delta_T_mv") is not None:
        _delta_T = float(th["delta_T_mv"])
        if not 0.0 <= _delta_T <= 3.0:
            raise ValueError("delta_T_mv must be finite in [0, 3]")
        key = key + ("delta_T_mv", _delta_T)
    if key not in net_cache:
        p = cns.SimParams()
        p.balance_target, p.w_syn = 1.10, 0.65
        if "e_exc_mv" in _e_rev:
            p.e_exc = _e_rev["e_exc_mv"]
        if "e_inh_mv" in _e_rev:
            p.e_inh = _e_rev["e_inh_mv"]
        if "v_rest_mv" in _membrane:
            p.v_rest = p.v_reset = _membrane["v_rest_mv"]
        if "v_threshold_mv" in _membrane:
            p.v_threshold = _membrane["v_threshold_mv"]
        if "tau_mem_ms" in _membrane:
            p.tau_mem = _membrane["tau_mem_ms"]
        if "tau_syn_ms" in _membrane:
            p.tau_syn = _membrane["tau_syn_ms"]
        if "t_refrac_ms" in _membrane:
            p.t_refrac = _membrane["t_refrac_ms"]
        if _delta_T is not None:
            p.delta_T = _delta_T
        # M-CAL (2026-08-18): module-level override for cross-dataset
        # gain calibration; None = untouched (bit-identical)
        if globals().get("_WSYN_OVERRIDE") is not None:
            p.w_syn = float(_WSYN_OVERRIDE)
        # map_theta layer 2: a model change (e.g. true shunting) scales
        # the effective synaptic gain. Carried on the theta so a mapped
        # parameter set is self-contained. Absent = 1.0 = untouched.
        if th.get("w_syn_scale"):
            p.w_syn = p.w_syn * float(th["w_syn_scale"])
        if globals().get("_BAL_OVERRIDE") is not None:
            p.balance_target = float(_BAL_OVERRIDE)
        if receptor_classes:
            p.receptor_classes = receptor_classes
        if _commissural_gain is not None:
            p.commissural_gain = float(_commissural_gain)
            p.commissural_sign = str(_commissural_sign)
        if _intersegmental_gain is not None:
            p.intersegmental_gain = float(_intersegmental_gain)
            p.intersegmental_sign = str(_intersegmental_sign)
        if _intersegmental_gain_exc is not None:
            p.intersegmental_gain_exc = float(_intersegmental_gain_exc)
        if _intersegmental_gain_inh is not None:
            p.intersegmental_gain_inh = float(_intersegmental_gain_inh)
        if th.get("desc_lr_normalise"):
            p.desc_lr_normalise = True
        if row_pair_seam_rows:
            p.row_pair_seams = row_pair_seam_rows
        if post_balance_edge_gain:
            p.post_balance_edge_gain = post_balance_edge_gain
        if post_balance_edge_gain_blocks:
            p.post_balance_edge_gain_blocks = post_balance_edge_gain_blocks
        if psi_pair_kernel:
            p.psi_pair_kernel = psi_pair_kernel
        if psi_pair_recovery:
            p.psi_pair_recovery = psi_pair_recovery
        if shared_occupancy:
            p.shared_occupancy = shared_occupancy
        if release_resource_rows is not None:
            (p.release_resource_rows, p.release_resource_U,
             p.release_resource_tau_rec_ms) = release_resource_rows
        if release_resource_target_rows is not None:
            p.release_resource_target_rows = release_resource_target_rows
        if adapt_w_init is not None:
            p.adapt_w_init_rows, p.adapt_w_init_mv = adapt_w_init
        if _left_apl_gaba:
            p.left_apl_gaba = True
        if _kc_kc_ach is not None:
            p.kc_kc_ach = str(_kc_kc_ach)
            p.kc_kc_ids = kc_kc_ids
            p.kc_kc_calyx = kc_kc_calyx
        if type_out_gain_vec is not None:
            p.type_out_gain = type_out_gain_vec
        if _serotonin_present:
            p.serotonin_source_rows = serotonin_source_rows
            p.serotonin_target_rows = serotonin_target_rows
            p.serotonin_tau_ms = serotonin_tau_ms
            p.serotonin_gain_mv_per_hz = serotonin_gain_mv_per_hz
            p.serotonin_init_hz = serotonin_init_hz
        if octopamine_target_rows is not None:   # design 8b6e0441, Builds 1 and 2
            p.octopamine_target_rows = octopamine_target_rows
        if octopamine_offset_mv is not None:   # design 8b6e0441, Build 2
            p.octopamine_offset_mv = octopamine_offset_mv
        if _os_present:   # design 8b6e0441, Build 1
            p.octopamine_source_rows = octopamine_source_rows
            p.octopamine_tau_ms = octopamine_tau_ms
            p.octopamine_gain_mv_per_hz = octopamine_gain_mv_per_hz
            p.octopamine_init_hz = octopamine_init_hz
        if _presynaptic_inhibition_mode:
            p.presynaptic_inhibition_rows = presynaptic_inhibition_rows
            p.presynaptic_inhibition_gain = _presynaptic_inhibition_gain
            p.presynaptic_inhibition_source_rows = presynaptic_inhibition_source_rows
            p.presynaptic_inhibition_tau_ms = _presynaptic_inhibition_tau
        if spike_reg_per_neuron is not None:
            p.spike_reg_per_neuron = spike_reg_per_neuron
        # CB6 / map_theta layer 2: under TRUE shunting the E/I balance
        # that mattered is a CURRENT balance, not a weight balance --
        # inhibition drives on ~17 mV where excitation drives on ~53.
        # Carried on the theta so a mapped set is self-contained.
        # bal_target=0 must be reachable (Lane A's NEEDS-ROBIN blocker,
        # judge fix 2026-08-24): the truthiness guard silently discarded
        # an explicit 0, so the balance mechanism could never be turned
        # OFF via theta. Absent = preset default = bit-identical; the
        # cache key above distinguishes "unset" from explicit values.
        if th.get("bal_target") is not None:
            p.balance_target = float(th["bal_target"])
        # Per-neuron spread of the balance target (cns.SimParams.balance_spread).
        # Absent or 0 = every target equal = bit-identical.
        if th.get("bal_spread") is not None:
            p.balance_spread = float(th["bal_spread"])
        # Lane E's R360/R362 scoped balance, ported from lane C's verbatim port
        # (c2893b72b) for the sight-reaches-the-legs stitch: exempt named rows
        # from the balance rule. Absent = nothing exempt = bit-identical.
        if th.get("bal_exempt"):
            # Each entry is "<column>:<value>", because the column matters:
            # R362 found that `super_class == "optic_lobe_intrinsic"` misses
            # 23,831 optic-lobe rows BANC did not super-class, which is 6.7% of
            # T4/T5's own input. `region:optic_lobe` is the repo's canonical
            # definition and the one AGENTS.md counts at 108,764.
            _ex = th["bal_exempt"]
            if isinstance(_ex, str):
                _ex = [_ex]
            _mask = np.zeros(len(meta), dtype=bool)
            for _spec in _ex:
                if ":" not in str(_spec):
                    raise ValueError(
                        f"bal_exempt entry {_spec!r} must be "
                        "'<column>:<value>', e.g. 'region:optic_lobe'")
                _col, _val = str(_spec).split(":", 1)
                if _col not in meta.columns:
                    raise ValueError(
                        f"bal_exempt names column {_col!r}, absent from meta")
                _hit = (meta[_col].fillna("").astype(str).to_numpy() == _val)
                if not _hit.any():
                    raise ValueError(
                        f"bal_exempt {_spec!r} matches no row; refusing to "
                        "run an empty exemption")
                _mask |= _hit
            p.balance_exempt = _mask
        if th.get("bal_by_type"):
            p.balance_by_type = float(th["bal_by_type"])
            p.balance_type_log_ratio = balance_type_log_ratio(meta, edges)
        if graded_rows:
            p.graded_rows = graded_rows
        if transmitter_map:
            p.transmitter_map = dict(transmitter_map)
        if cell_type_patch:
            p.cell_type_patch = {k: dict(v) for k, v in cell_type_patch.items()}
        if _nt_transfer_ids is not None:
            p.nt_transfer_rows = {i: str(_nt_transfer["transmitter"])
                                  for i in _nt_transfer_ids}
        if balance_scale_override is not None:
            p.balance_scale_override = balance_scale_override
        if inh_rows:
            p.premotor_inh_rows = inh_rows
            p.premotor_inh_gain = inh_gain
        if inh_13a_21a_gain is not None:
            p.premotor_inh_gain_per_neuron = inh_13a_21a_gain
            p.premotor_inh_parameter_mode = "hemilineage_13a_21a_contrast"
            p.premotor_inh_13a_21a_ratio = float(
                th["inh_13a_21a_ratio"])
        _typed_21a_mode = th.get("typed_21a_inhibitory_gain_mode")
        if _typed_21a_mode:
            if inh_13a_21a_gain is not None:
                raise ValueError(
                    "typed 21A gain cannot compose with 13A/21A contrast")
            _typed_21a_gain, _typed_21a_rows = (
                typed_21a_inhibitory_gain_vector(
                    meta, _typed_21a_mode,
                    th["typed_21a_inhibitory_gain"]))
            p.premotor_inh_gain_per_neuron = _typed_21a_gain
            p.premotor_inh_parameter_mode = "typed_21a_segment"
            p.typed_21a_inhibitory_gain_mode = _typed_21a_mode
            p.typed_21a_inhibitory_gain_rows = len(_typed_21a_rows)
            p.typed_21a_inhibitory_gain = float(
                th["typed_21a_inhibitory_gain"])
        _typed_19b_mode = th.get("typed_19b_excitatory_gain_mode")
        if _typed_19b_mode:
            _typed_19b_gain, _typed_19b_rows = (
                typed_19b_excitatory_gain_vector(
                    meta, _typed_19b_mode,
                    th["typed_19b_excitatory_gain"]))
            p.presyn_gain_per_neuron = _typed_19b_gain
            p.typed_19b_excitatory_gain_mode = _typed_19b_mode
            p.typed_19b_excitatory_gain_rows = len(_typed_19b_rows)
            p.typed_19b_excitatory_gain_ids = list(_typed_19b_rows)
            p.typed_19b_excitatory_gain = float(
                th["typed_19b_excitatory_gain"])
        _typed_21a_exc_mode = th.get("typed_21a_excitatory_gain_mode")
        if _typed_21a_exc_mode:
            if _typed_19b_mode:
                raise ValueError("typed 21A and 19B gains cannot compose")
            _typed_21a_exc_gain, _typed_21a_exc_rows = (
                typed_21a_excitatory_gain_vector(
                    meta, _typed_21a_exc_mode,
                    th["typed_21a_excitatory_gain"]))
            p.presyn_gain_per_neuron = _typed_21a_exc_gain
            p.typed_21a_excitatory_gain_mode = _typed_21a_exc_mode
            p.typed_21a_excitatory_gain_rows = len(_typed_21a_exc_rows)
            p.typed_21a_excitatory_gain_ids = list(_typed_21a_exc_rows)
            p.typed_21a_excitatory_gain = float(
                th["typed_21a_excitatory_gain"])
        if th.get("cmd_presyn_gain") is not None:
            _cmd_gain, _cmd_gain_ids = cmd_presyn_gain_vector(
                meta, th.get("cmd"), th["cmd_presyn_gain"])
            if p.presyn_gain_per_neuron is not None:
                p.presyn_gain_per_neuron = (
                    np.asarray(p.presyn_gain_per_neuron, dtype=np.float64)
                    * _cmd_gain)
            else:
                p.presyn_gain_per_neuron = _cmd_gain
            p.cmd_presyn_gain = float(th["cmd_presyn_gain"])
            p.cmd_presyn_gain_ids = list(_cmd_gain_ids)
        _pg_by_id = th.get("presyn_gain_by_id")
        if _pg_by_id:
            _pg_vec, _pg_rows = presyn_gain_by_id_vector(meta, _pg_by_id)
            if p.presyn_gain_per_neuron is not None:
                p.presyn_gain_per_neuron = (
                    np.asarray(p.presyn_gain_per_neuron, dtype=np.float64)
                    * _pg_vec)
            else:
                p.presyn_gain_per_neuron = _pg_vec
        # CB1 (2026-08-19): conductance-based synapses -- reversal
        # potentials instead of current injection. Real synapses are
        # conductances; this is the correct physics, and it BOUNDS
        # inhibition at e_inh (measured without it: 10% of neurons
        # below -70 mV, worst -169.8 mV). Default off = bit-identical.
        if th.get("cbased"):
            p.conductance_based = True
            # CB3: th["shunt"]=True gives TRUE conductance synapses --
            # E and I tracked separately so they shunt rather than
            # cancel. Absent = the older bounded-but-not-divisive form.
            if th.get("shunt"):
                p.split_conductances = True
        _split_tau_mode = th.get("split_synaptic_decay_mode")
        if _split_tau_mode:
            if _split_tau_mode != "drosophila_fast_receptors":
                raise ValueError("unknown split_synaptic_decay_mode")
            if not (th.get("cbased") and th.get("shunt")):
                raise ValueError(
                    "split synaptic decay requires true shunting")
            p.tau_syn_exc = float(th.get("tau_syn_exc_ms", 1.1))
            p.tau_syn_inh = float(th.get("tau_syn_inh_ms", 5.2))
            p.syn_normalised = True
            p.split_synaptic_decay_mode = _split_tau_mode
        _irmode = th.get("inhibitory_reversal_mode")
        if _irmode == "type_hemilineage":
            if not (th.get("cbased") and th.get("shunt")):
                raise ValueError(
                    "heterogeneous inhibitory reversal requires true shunting")
            p.e_inh = type_hemilineage_inhibitory_reversal_vector(
                meta, th.get("inhibitory_reversal_seed"),
                th.get("inhibitory_reversal_permutation_seed"))
            p.inhibitory_reversal_mode = _irmode
            p.inhibitory_reversal_permutation_seed = th.get(
                "inhibitory_reversal_permutation_seed")
            p.inhibitory_reversal_group_count = int(np.unique(
                neuron_type_hemilineage_groups(meta)).size)
        elif _irmode not in (None, ""):
            raise ValueError(
                f"unknown inhibitory_reversal_mode: {_irmode!r}")
        if _type_rows["e_inh_type_mv"]:
            # B3: wm1's scalar reversal on every row, the named types' own
            # on theirs. Construction never reads e_inh, so this is the array
            # the engine receives, as gpu/optic/lamina_light.py's post-build
            # R80 vector is. The scalar is the theta's e_inh_mv when it has
            # one, else -70 mV.
            if np.ndim(p.e_inh) != 0 or float(p.e_inh) != _e_rev.get("e_inh_mv", -70.0):
                raise ValueError("e_inh_type_mv expects the theta's scalar reversal "
                                 "(e_inh_mv, else -70 mV) on the other rows")
            _einh = np.full(len(meta), float(p.e_inh), dtype=np.float32)
            for _rows_t, _v in _type_rows["e_inh_type_mv"].values():
                _einh[list(_rows_t)] = np.float32(_v)
            p.e_inh = _einh
        _cb_integrator = th.get("cbased_integrator")
        if _cb_integrator:
            if not th.get("cbased"):
                raise ValueError(
                    "cbased_integrator requires th['cbased']=True")
            if _cb_integrator not in ("euler", "analytic"):
                raise ValueError(
                    "cbased_integrator must be 'euler' or 'analytic', got "
                    f"{_cb_integrator!r}")
            p.conductance_integrator = _cb_integrator
        # CB2: graded_gain must be re-settable, because conductances
        # bound inhibition at e_inh and therefore change the effective
        # strength of any graded inhibitory population.
        if th.get("graded_gain") is not None:
            p.graded_gain = float(th["graded_gain"])
        if _graded_13b_gain is not None:
            _gvec, _rows13 = graded_13b_gain_vector(
                meta, p.graded_gain, _graded_13b_gain,
                _graded_13b_gain_mode,
                th.get("graded_13b_gain_permutation_seed", 0))
            p.graded_gain = _gvec
            p.graded_13b_gain = _graded_13b_gain
            p.graded_13b_gain_rows = int(len(_rows13))
            p.graded_13b_gain_mode = _graded_13b_gain_mode
        if _type_rows["graded_gain_type"]:
            # B1: one float64 vector, the dtype lamina_light passes: the
            # scalar gain, graded_gain_add on the added rows, then each named
            # type's gain on its rows.
            _gvec = np.full(len(meta), float(p.graded_gain), dtype=np.float64)
            if graded_add_rows:
                _gvec[list(graded_add_rows)] = float(th["graded_gain_add"])
            for _rows_t, _v in _type_rows["graded_gain_type"].values():
                _gvec[list(_rows_t)] = _v
            p.graded_gain = _gvec
        elif graded_add_rows:
            # Per-row gain: base rows keep the scalar above, added rows get
            # their own. The engine takes an (N,) vector; rows that are not
            # graded never read it.
            _gvec = np.full(len(meta), float(p.graded_gain), dtype=np.float32)
            _gvec[list(graded_add_rows)] = float(th["graded_gain_add"])
            p.graded_gain = _gvec
        if th.get("graded_phasic"):
            p.graded_phasic = float(th["graded_phasic"])
        # The MEASURED-STRUCTURE DIRECTIVE makes graded a SEARCH over four
        # dials, and two of them had no theta pass-through, so the search it
        # mandates could not be run: graded_v50 and graded_slope were fixed
        # universal constants reachable only by editing cns.py. graded_every
        # is the third: it decimates the release update, scaling the injected
        # charge so the delivered total is unchanged, and it is what makes a
        # graded arm affordable to search at all -- graded release costs 54 s
        # of a 68 s evaluation at every-tick updates.
        if th.get("graded_v50") is not None:
            p.graded_v50 = float(th["graded_v50"])
        if _type_rows["graded_v50_type"]:
            # B2: the scalar half-activation on every row, the named types'
            # own on theirs; the driver indexes it by graded_idx.
            _v50 = np.full(len(meta), float(p.graded_v50), dtype=np.float64)
            for _rows_t, _v in _type_rows["graded_v50_type"].values():
                _v50[list(_rows_t)] = _v
            p.graded_v50 = _v50
        if th.get("graded_slope") is not None:
            _gs = float(th["graded_slope"])
            if _gs <= 0.0:
                raise ValueError("graded_slope must be positive")
            p.graded_slope = _gs
        if _type_rows["graded_slope_type"]:
            # Build 2 of design 0af6d1c5: the scalar slope on every row, the
            # named types' own on theirs; the engines pack it by graded_idx.
            _gsv = np.full(len(meta), float(p.graded_slope), dtype=np.float64)
            for _rows_t, _v in _type_rows["graded_slope_type"].values():
                _gsv[list(_rows_t)] = _v
            p.graded_slope = _gsv
        if row_set_params is not None:
            _row_set_curve_apply(row_set_params, len(meta), p)
        if th.get("graded_every") is not None:
            _ge = int(th["graded_every"])
            if _ge < 1:
                raise ValueError("graded_every must be at least 1")
            p.graded_every = _ge
        if gap_pairs and float(th.get("g_gap", 0.0)) > 0:
            p.gap_pairs = gap_pairs
            p.g_gap = float(th.get("g_gap", 0.0))
        if gap_pairs and gap_pair_weights:
            p.gap_pairs = gap_pairs
            p.gap_pair_weights = gap_pair_weights
        for name in new_gap_keys:
            if name in th:
                setattr(p, name, th[name])
        # gap_spikelet / gap_delay_ms (longevity lane, 2026-09-12): the
        # spikelet and its conduction time, cns.SimParams fields since
        # 2026-08-10 and today, reachable from theta only here. Both absent =
        # bit-identical; both need gap pairs (explicit rules above).
        if th.get("gap_spikelet") is not None:
            if not gap_pairs:
                raise ValueError("gap_spikelet requires gap pairs")
            p.gap_spikelet = float(th["gap_spikelet"])
        if th.get("gap_delay_ms") is not None:
            if not gap_pairs:
                raise ValueError("gap_delay_ms requires gap pairs")
            p.gap_delay_ms = float(th["gap_delay_ms"])
        # gap_ap_mv / gap_ap_ms (longevity lane, 2026-09-12): the presynaptic
        # action potential through the junction conductance, cns.SimParams
        # fields of the same day. Absent = bit-identical; needs gap pairs.
        if th.get("gap_ap_mv") is not None:
            if not gap_pairs:
                raise ValueError("gap_ap_mv requires gap pairs")
            p.gap_ap_mv = float(th["gap_ap_mv"])
            if th.get("gap_ap_ms") is not None:
                p.gap_ap_ms = float(th["gap_ap_ms"])
        # gap_rectify (longevity lane, 2026-09-12): one-way junctions, pair
        # (pre, post) couples post only (Phelan 2008). Absent = bit-identical.
        if th.get("gap_rectify") is not None:
            if not gap_pairs:
                raise ValueError("gap_rectify requires gap pairs")
            p.gap_rectify = bool(th["gap_rectify"])
            if p.gap_rectify and th.get("gap_pair_rules"):
                # the rules path stores each pair as (min, max) of the row
                # indices, which loses the direction a rectifying junction
                # needs; the rule's FIRST cell type is the presynaptic side
                # (measured 2026-09-12: TTMn spikes drove the GF in the loop)
                _ct = meta["cell_type"].astype(str)
                _pre_rows = set()
                for _rule in th["gap_pair_rules"]:
                    _pre_rows.update(int(r) for r in np.flatnonzero(_ct.eq(_rule["cell_types"][0]).to_numpy()))
                _oriented = []
                for _a, _b in p.gap_pairs:
                    if int(_b) in _pre_rows and int(_a) not in _pre_rows:
                        _a, _b = _b, _a
                    _oriented.append((int(_a), int(_b)))
                p.gap_pairs = tuple(_oriented)
        # SHORT-TERM SYNAPTIC DEPRESSION (2026-08-19). Implemented in
        # cns.py since the beginning, defaulted off, and never wired into
        # the drive path at all -- so no search has ever been able to
        # reach it. Its only prior measurement was in a BODILESS network
        # against a spontaneous-activity criterion, where it looked
        # catastrophic (persistence 0.84 -> 0.0). In the closed loop the
        # sign of the effect REVERSES: without a body, depression removes
        # the self-sustaining activity that "persistence" scores; with a
        # body and load feedback, the same brake is what stops the fly
        # falling over. std_U = 0 is off and bit-identical.
        # N1: afferent spike regularity (registered 2026-08-21).
        # Absent/1.0 = Poisson = bit-identical.
        if th.get("spike_reg"):
            p.spike_reg = float(th["spike_reg"])
        if th.get("backend"):
            p.backend = str(th["backend"])
        if th.get("std_U"):
            p.std_U = float(th["std_U"])
            p.std_tau_rec = float(th.get("std_tau_rec", 400.0))
        p.delay = float(th.get("delay", 1.8))   # sweepable; default = the verified OP
        # MEASURED-STRUCTURE DIRECTIVE (Robin, 2026-09-01): per-neuron
        # presynaptic delay from measured cable length. Absent = the scalar
        # delay above, bit-identical (probe-gated).
        _dmode = th.get("delay_mode")
        _mcns_structure = None
        if _dmode == "length":
            # canonical: length-proportional, mean anchored at the verified
            # scalar (lane B semantics, judge-canonicalised 2026-09-01)
            p.delay_per_neuron = length_delay_vector(
                meta, float(th.get("delay_mean_ms", th.get("delay", 1.8))),
                float(th.get("delay_cap_ms", 20.0)),
                int(th.get("delay_groups", 8)))
        elif _dmode == "length_bv":
            p.delay_per_neuron = length_bv_delay_vector(
                meta, float(th.get("delay_base_ms", 0.8)),
                float(th.get("delay_velocity_um_ms", 300.0)),
                float(th.get("delay_cap_ms", 20.0)),
                int(th.get("delay_groups", 8)))
        elif _dmode == "by_type":
            _tmap = th.get("delay_type_ms")
            if (not isinstance(_tmap, dict) or not _tmap
                    or not all(isinstance(k, str) for k in _tmap)
                    or not all(type(v) in (int, float) and np.isfinite(v)
                               and 0.1 <= float(v) <= 100.0
                               for v in _tmap.values())):
                raise ValueError(
                    "delay_type_ms must be a non-empty map of cell type to "
                    "a finite delay in [0.1, 100] ms")
            p.delay_per_neuron = type_delay_vector(
                meta, float(th.get("delay_mean_ms", th.get("delay", 1.8))),
                _tmap, float(th.get("delay_cap_ms", 100.0)))
        elif _dmode == "length_by_type":
            # the length vector as the base, listed types overridden; the
            # length cap keeps its 20 ms default and the overrides are
            # bounded by the by_type map's own [0.1, 100] check
            _tmap = th.get("delay_type_ms")
            if (not isinstance(_tmap, dict) or not _tmap
                    or not all(isinstance(k, str) for k in _tmap)
                    or not all(type(v) in (int, float) and np.isfinite(v)
                               and 0.1 <= float(v) <= 100.0
                               for v in _tmap.values())):
                raise ValueError(
                    "delay_type_ms must be a non-empty map of cell type to "
                    "a finite delay in [0.1, 100] ms")
            p.delay_per_neuron = type_delay_vector(
                meta, length_delay_vector(
                    meta, float(th.get("delay_mean_ms", th.get("delay", 1.8))),
                    float(th.get("delay_cap_ms", 20.0)),
                    int(th.get("delay_groups", 8))),
                _tmap, 100.0)
            # delay_type_id (design 6f704792 Build 4; absent = bit-identical):
            # {banc_888_id: cell_type} gives a named row the delay its stated
            # type carries in delay_type_ms, for a row the type map cannot
            # reach because meta types it as something else. Applied before
            # delay_region_scale, which then scales it as it scales every row.
            _tid = th.get("delay_type_id")
            if _tid:
                if not isinstance(_tid, dict):
                    raise ValueError("delay_type_id must map banc_888_id to cell type")
                _ids_d = meta["banc_888_id"].astype(str).to_numpy()
                for _rid, _rtype in _tid.items():
                    _hit = np.flatnonzero(_ids_d == str(_rid))
                    if len(_hit) != 1:
                        raise ValueError(
                            f"delay_type_id: row {_rid} matches {len(_hit)} meta "
                            "rows, expected exactly 1")
                    if not isinstance(_rtype, str) or _rtype not in _tmap:
                        raise ValueError(
                            f"delay_type_id[{_rid}] names {_rtype!r}, which "
                            "delay_type_ms does not carry")
                    p.delay_per_neuron[_hit[0]] = float(_tmap[_rtype])
        elif _dmode == "mcns_active_proxy":
            _mcns_structure = _load_mcns_active_structure(meta)
            p.delay_per_neuron = _mcns_structure[0]
            p.structure_active_mask = _mcns_structure[2]
        elif _dmode not in (None, ""):
            raise ValueError(f"unknown delay_mode: {_dmode!r}")
        if th.get("delay_type_id") and _dmode != "length_by_type":
            raise ValueError("delay_type_id is read under delay_mode "
                             f"length_by_type only, not {_dmode!r}")
        # Delivery mechanism for whichever vector the mode above built.
        # CURRENT.md (2026-09-01 16:48) directs delay work to
        # delay_delivery="queue" with delay_groups=0 for full-resolution
        # measured cable lengths, and the field had no theta pass-through, so
        # that instruction could not be followed from a theta. Absent leaves
        # the canonical bucket path untouched.
        _ddel = th.get("delay_delivery")
        if _ddel:
            if _ddel not in ("bucket", "queue"):
                raise ValueError(
                    f"delay_delivery must be 'bucket' or 'queue', got "
                    f"{_ddel!r}")
            if p.delay_per_neuron is None:
                raise ValueError(
                    "delay_delivery needs a delay_mode; without a per-neuron "
                    "vector the scalar path is used and the flag would be a "
                    "claim the run does not carry")
            p.delay_delivery = _ddel
        # Per-region delay scale (lane A, CD2-CG 2026-09-06). The delay law
        # `period = 22.08 x mean_delay + 33.41` (CD2-CB, R^2 1.00000) says the
        # dominant loop is about 22 synapses deep against a measured leg reflex
        # arc of three to five, so which region carries that depth is the
        # question the global scale cannot answer. Absent = untouched =
        # bit-identical; scaling only, never a shift, so no value can cross the
        # 30 ms Metal reproducibility bound that the vector already respects.
        _rscale = th.get("delay_region_scale")
        if _rscale:
            if p.delay_per_neuron is None:
                raise ValueError(
                    "delay_region_scale needs a delay_mode; the scalar path has "
                    "no per-neuron vector to scale and the key would be a claim "
                    "the run does not carry")
            if not isinstance(_rscale, dict) or not _rscale:
                raise ValueError("delay_region_scale must be a non-empty map")
            _regions = meta["region"].to_numpy()
            _known = set(pd.unique(_regions[pd.notna(_regions)]))
            for _rk, _rv in _rscale.items():
                if _rk not in _known:
                    raise ValueError(
                        f"delay_region_scale key {_rk!r} is not a BANC region; "
                        f"known regions are {sorted(_known)}")
                if type(_rv) not in (int, float) or not np.isfinite(_rv) or _rv <= 0:
                    raise ValueError(
                        f"delay_region_scale[{_rk!r}] must be a finite positive "
                        f"factor, got {_rv!r}")
                _m = (_regions == _rk)
                p.delay_per_neuron = np.where(
                    _m, p.delay_per_neuron * float(_rv), p.delay_per_neuron)
            p.delay_per_neuron = np.clip(p.delay_per_neuron, 0.1, None)
        # delay_class_scale (design 983f1c29 Build 6, amendment 384d4988 item
        # 6; absent = untouched = bit-identical): {cell_class: s} multiplies
        # the class's rows, or {"rows": [banc_888_id, ...], "scale": s} the
        # listed rows, after the delay law above (renormalisation, type map,
        # region scale and their clips), so every other row keeps its value
        # to the bit. The scaled rows are clipped to [0.1, delay_cap_ms].
        _dcs_receipt = None
        _cscale = th.get("delay_class_scale")
        if _cscale:
            if p.delay_per_neuron is None:
                raise ValueError(
                    "delay_class_scale needs a delay_mode; the scalar path has "
                    "no per-neuron vector to scale")
            if not isinstance(_cscale, dict):
                raise ValueError("delay_class_scale must be a map")
            if "rows" in _cscale:
                if set(_cscale) != {"rows", "scale"}:
                    raise ValueError(
                        "delay_class_scale's row form is {rows, scale} only")
                _ids_c = meta["banc_888_id"].astype(str).to_numpy()
                _pos_c = {v: i for i, v in enumerate(_ids_c)}
                _want = [str(x) for x in _cscale["rows"]]
                if not _want or len(set(_want)) != len(_want):
                    raise ValueError(
                        "delay_class_scale rows must be non-empty and unique")
                _miss = [x for x in _want if x not in _pos_c]
                if _miss:
                    raise ValueError(
                        f"delay_class_scale names {len(_miss)} ids absent "
                        f"from meta, first {_miss[0]!r}")
                _groups_c = [("rows", np.asarray(sorted(_pos_c[x] for x in _want),
                                                 dtype=np.int64),
                              _cscale["scale"])]
            else:
                _cls = meta["cell_class"].to_numpy()
                _groups_c = []
                for _ck, _cv in _cscale.items():
                    _rows_c = np.flatnonzero(_cls == _ck)
                    if not len(_rows_c):
                        raise ValueError(
                            f"delay_class_scale class {_ck!r} matches no row")
                    _groups_c.append((str(_ck), _rows_c, _cv))
            _cap_c = float(th.get("delay_cap_ms", 20.0))
            _before_c = np.array(p.delay_per_neuron, dtype=np.float64, copy=True)
            _scaled_c = np.zeros(len(_before_c), dtype=bool)
            _dcs_receipt = {"cap_ms": _cap_c, "groups": {}}
            for _cname, _rows_c, _cv in _groups_c:
                if (type(_cv) not in (int, float) or not np.isfinite(_cv)
                        or _cv <= 0):
                    raise ValueError(
                        f"delay_class_scale[{_cname!r}] must be a finite "
                        f"positive factor, got {_cv!r}")
                if _scaled_c[_rows_c].any():
                    raise ValueError(
                        f"delay_class_scale[{_cname!r}] scales a row twice")
                _new_c = np.clip(_before_c[_rows_c] * float(_cv), 0.1, _cap_c)
                p.delay_per_neuron = np.array(p.delay_per_neuron,
                                              dtype=np.float64, copy=True)
                p.delay_per_neuron[_rows_c] = _new_c
                _scaled_c[_rows_c] = True
                _dcs_receipt["groups"][_cname] = {
                    "scale": float(_cv), "rows": int(len(_rows_c)),
                    "median_before_ms": float(np.median(_before_c[_rows_c])),
                    "median_after_ms": float(np.median(_new_c)),
                    "clipped": int(np.sum(_new_c != _before_c[_rows_c]
                                          * float(_cv))),
                    "rows_sha16": hashlib.sha256(",".join(
                        meta["banc_888_id"].astype(str).to_numpy()[_rows_c]
                    ).encode()).hexdigest()[:16]}
            _dcs_receipt["off_class_identical"] = bool(np.array_equal(
                p.delay_per_neuron[~_scaled_c], _before_c[~_scaled_c]))
            _dcs_receipt["off_class_sha256"] = hashlib.sha256(
                np.ascontiguousarray(p.delay_per_neuron[~_scaled_c],
                                     dtype=np.float64).tobytes()).hexdigest()
            _dcs_receipt["before_off_class_sha256"] = hashlib.sha256(
                np.ascontiguousarray(_before_c[~_scaled_c],
                                     dtype=np.float64).tobytes()).hexdigest()
        # delay_row_ms (design 9bdd809a Build 1; absent or empty = untouched =
        # bit-identical): {banc_888_id: ms} gives each listed row that delay,
        # after every transform above (law, type map, delay_type_id, region
        # scale and its clip, delay_class_scale), so every other row keeps its
        # value to the bit and nothing is restored. Listed values are clipped
        # to [0.1, delay_cap_ms] as delay_class_scale's rows are.
        _drm_receipt = None
        _drm = th.get("delay_row_ms")
        if _drm:
            if p.delay_per_neuron is None:
                raise ValueError(
                    "delay_row_ms needs a delay_mode; the scalar path has no "
                    "per-neuron vector to set")
            if not isinstance(_drm, dict):
                raise ValueError("delay_row_ms must map banc_888_id to ms")
            _want_r = [str(k) for k in _drm]
            if len(set(_want_r)) != len(_want_r):
                raise ValueError("delay_row_ms names an id twice")
            _ids_r = meta["banc_888_id"].astype(str).to_numpy()
            _pos_r, _dup_r = {}, set()
            for _i, _v in enumerate(_ids_r):
                if _v in _pos_r:
                    _dup_r.add(_v)
                _pos_r[_v] = _i
            _rows_r, _vals_r = [], []
            for _k, _v in _drm.items():
                _sk = str(_k)
                if _sk not in _pos_r or _sk in _dup_r:
                    raise ValueError(
                        f"delay_row_ms: row {_sk} matches "
                        f"{int(np.sum(_ids_r == _sk))} meta rows, expected "
                        "exactly 1")
                if type(_v) not in (int, float) or not np.isfinite(_v) or _v <= 0:
                    raise ValueError(
                        f"delay_row_ms[{_sk}] must be a finite positive delay "
                        f"in ms, got {_v!r}")
                _rows_r.append(_pos_r[_sk])
                _vals_r.append(float(_v))
            _rows_r = np.asarray(_rows_r, dtype=np.int64)
            _cap_r = float(th.get("delay_cap_ms", 20.0))
            _asked_r = np.asarray(_vals_r, dtype=np.float64)
            _new_r = np.clip(_asked_r, 0.1, _cap_r)
            _before_r = np.array(p.delay_per_neuron, dtype=np.float64, copy=True)
            p.delay_per_neuron = _before_r.copy()
            p.delay_per_neuron[_rows_r] = _new_r
            _off_r = np.ones(len(_before_r), dtype=bool)
            _off_r[_rows_r] = False
            _slots_r = np.maximum(
                1, np.rint(p.delay_per_neuron / float(p.dt))).astype(np.int64)
            _drm_receipt = {
                "rows": int(len(_rows_r)),
                "ids_sha256": hashlib.sha256(
                    ",".join(sorted(_want_r)).encode()).hexdigest(),
                "median_before_ms": float(np.median(_before_r[_rows_r])),
                "median_after_ms": float(np.median(_new_r)),
                "clipped": int(np.sum(_new_r != _asked_r)),
                "cap_ms": _cap_r,
                "off_rows_identical": bool(np.array_equal(
                    p.delay_per_neuron[_off_r], _before_r[_off_r])),
                "off_rows_sha256": hashlib.sha256(np.ascontiguousarray(
                    p.delay_per_neuron[_off_r], dtype=np.float64
                ).tobytes()).hexdigest(),
                "before_off_rows_sha256": hashlib.sha256(np.ascontiguousarray(
                    _before_r[_off_r], dtype=np.float64).tobytes()).hexdigest(),
                "n_rows": int(len(_before_r)),
                "vector_sha256": hashlib.sha256(np.ascontiguousarray(
                    p.delay_per_neuron, dtype=np.float64).tobytes()).hexdigest(),
                "dt_ms": float(p.dt),
                "slots_sha256": hashlib.sha256(
                    np.ascontiguousarray(_slots_r).tobytes()).hexdigest()}
        _gddel = th.get("graded_delay_delivery")
        if _gddel:
            if _gddel not in ("bucket", "queue"):
                raise ValueError(
                    "graded_delay_delivery must be 'bucket' or 'queue', got "
                    f"{_gddel!r}")
            if p.delay_per_neuron is None:
                raise ValueError(
                    "graded_delay_delivery needs a per-neuron delay mode")
            p.graded_delay_delivery = _gddel
        # coincidence nonlinearity (lane C, 2026-09-04): designated target
        # cell types get an extra current proportional to the product of a
        # fast and a delayed source pool. Absent or zero gain = off =
        # bit-identical.
        _cgain = th.get("coincidence_gain")
        _ctrip = th.get("coincidence_triples")
        if _cgain and _ctrip is not None:
            raise ValueError(
                "coincidence_gain and coincidence_triples are two forms of one "
                "seam; give one")
        if _cgain or _ctrip is not None:
            _ct = meta["cell_type"].astype("string")
            if "fafb_alignment_cell_type" in meta.columns:
                _ct = _ct.where(_ct.notna(),
                                meta["fafb_alignment_cell_type"].astype("string"))
            _lab = np.array([x if isinstance(x, str) else ""
                             for x in _ct.to_numpy()], dtype=object)
        if _ctrip is not None:
            p.coincidence_triples = _coincidence_triples_parse(
                _ctrip, _lab, meta, graded_rows)
        if _cgain:

            def _rows_for(key):
                names = th.get(key)
                if not isinstance(names, (list, tuple)) or not names:
                    raise ValueError(f"{key} must be a non-empty list")
                rows = np.flatnonzero(np.isin(_lab, list(names)))
                if not rows.size:
                    raise ValueError(f"{key} matched no rows in this core")
                return rows

            p.coincidence_targets = _rows_for("coincidence_target_types")
            p.coincidence_fast_rows = _rows_for("coincidence_fast_types")
            p.coincidence_slow_rows = _rows_for("coincidence_slow_types")
            p.coincidence_gain = float(_cgain)
            _clag = th.get("coincidence_lag_ms", 60.0)
            if (type(_clag) not in (int, float) or not np.isfinite(_clag)
                    or not 0.0 < float(_clag) <= 500.0):
                raise ValueError(
                    "coincidence_lag_ms must be finite in (0, 500]")
            p.coincidence_lag_ms = float(_clag)
            _ctau = th.get("coincidence_tau_ms", 5.0)
            if (type(_ctau) not in (int, float) or not np.isfinite(_ctau)
                    or not 0.0 < float(_ctau) <= 200.0):
                raise ValueError(
                    "coincidence_tau_ms must be finite in (0, 200]")
            p.coincidence_tau_ms = float(_ctau)
        _tmode = th.get("threshold_mode")
        if _tmode == "volume_rank":
            p.v_threshold = volume_rank_threshold_vector(
                meta,
                float(th.get("threshold_mean_mv", -45.0)),
                float(th.get("threshold_span_mv", 4.0)))
        elif _tmode in ("by_type", "by_exact_type"):
            _thmap = th.get("threshold_type_mv")
            if (not isinstance(_thmap, dict) or not _thmap
                    or not all(isinstance(k, str) for k in _thmap)
                    or not all(type(v) in (int, float) and np.isfinite(v)
                               and -70.0 <= float(v) <= -20.0
                               for v in _thmap.values())):
                raise ValueError(
                    "threshold_type_mv must be a non-empty map of cell type "
                    "to a finite threshold in [-70, -20] mV")
            p.v_threshold, p.threshold_type_audit = type_threshold_vector(
                meta, p.v_threshold, _thmap,
                exact=(_tmode == "by_exact_type"))
        elif _tmode == "mcns_active_proxy":
            if _mcns_structure is None:
                _mcns_structure = _load_mcns_active_structure(meta)
            p.structure_active_mask = _mcns_structure[2]
            p.v_threshold = _mcns_structure[1]
        elif _tmode not in (None, ""):
            raise ValueError(f"unknown threshold_mode: {_tmode!r}")
        p.r_in_alpha = 0.10
        if th["adapt_b"] > 0:
            p.adapt_b = th["adapt_b"]
            p.tau_w = float(th.get("tau_w", 100.0))   # sweepable; 100 = the historical default
        if _adapt_a is not None:
            p.adapt_a = _adapt_a
            p.tau_w = float(th.get("tau_w", 100.0))
        # ASSIGNMENT ARM D, default-off (design 43eb3960, study 8d6eb80a):
        # mn_adapt_b_zero_ids removes the spike-triggered adaptation increment on
        # named rows and leaves adapt_a and tau_w alone, so the arm asks about the
        # spike-triggered coefficient and not about the whole adaptation current.
        # p.adapt_b becomes a per-row float array; cns.py:3005-3006 and
        # cns_metal.py:484 both already accept one. Absent = untouched = the scalar
        # that was here. It sits outside the `use_size` block, one line later than
        # design 43eb3960's prose places it, because adapt_b has nothing to do with
        # the size rule and a key that silently does nothing when `mn_gains` is
        # absent is the inert element AGENTS.md forbids. For every theta in this
        # design `use_size` is true, so the two placements are the same arithmetic.
        _adapt_b_zero = th.get("mn_adapt_b_zero_ids")
        if _adapt_b_zero is not None:
            if not isinstance(_adapt_b_zero, (list, tuple)) or not _adapt_b_zero:
                raise ValueError(
                    "mn_adapt_b_zero_ids must be a non-empty list of banc_888_id")
            _ab_scalar = th.get("adapt_b")
            if _ab_scalar is None or not float(_ab_scalar) > 0.0:
                raise ValueError(
                    "mn_adapt_b_zero_ids requires adapt_b > 0: zeroing a coefficient "
                    "that is absent or already zero is an inert arm that would pass "
                    "silently")
            _ab_scalar = float(_ab_scalar)
            _ab_ids = list(meta["banc_888_id"].astype(str))
            _ab_pos = {b: i for i, b in enumerate(_ab_ids)}
            _ab = np.full(len(_ab_ids), _ab_scalar, dtype=np.float64)
            _ab_rows = []
            for _b in _adapt_b_zero:
                _i = _ab_pos.get(str(_b))
                if _i is None:
                    raise ValueError(
                        f"mn_adapt_b_zero_ids id {_b} is not a row of meta")
                _ab[_i] = 0.0
                _ab_rows.append(int(_i))
            if len(set(_ab_rows)) != len(_ab_rows):
                raise ValueError("mn_adapt_b_zero_ids names a row twice")
            p.adapt_b = _ab
            p.mn_adapt_b_zero_audit = {
                "n_requested": len(_adapt_b_zero),
                "n_zeroed": int(np.count_nonzero(_ab == 0.0)),
                "rows": sorted(_ab_rows),
                "ids": [str(b) for b in _adapt_b_zero],
                "scalar_elsewhere": _ab_scalar,
                "n_at_scalar": int(np.count_nonzero(_ab == _ab_scalar)),
                "length": int(len(_ab)),
            }
        if use_size:
            _, rfac, vrest = size_factors(meta, edges, **_sp_kw)
            # LESION ARMS, default-off (design 98153175, study bbaa404e): "r"
            # removes the size resistance factor and "v" the size rest gradient,
            # so a ladder can ask which of the two count-dependent terms carries
            # the recruitment order. Absent, this is the code that was here.
            _flat = th.get("mn_size_flat")
            if _flat is not None:
                if _flat not in ("r", "v"):
                    raise ValueError("th['mn_size_flat'] must be absent, 'r' or 'v'")
                if _flat == "v" and th.get("mn_vreset"):
                    raise ValueError("mn_size_flat='v' cannot compose with mn_vreset")
                if _flat == "r":
                    rfac = {i: 1.0 for i in rfac}
                else:
                    vrest = {}
            # ASSIGNMENT ARMS A AND P, default-off (design 43eb3960, study
            # 8d6eb80a): the size rule fixes the within-pool resistance span at
            # 4.6667 by construction (:5827 raises 700/150 to k/(n-1)), so a ladder
            # cannot widen or narrow the set; what it can move is which member
            # carries which value. mn_rfac_by_id overwrites named entries of rfac
            # before it becomes p.r_in_override, which is the shape size_factors
            # already returns. An id absent from rfac raises rather than adding a
            # row, so a mistyped id cannot silently become a new override.
            # Absent = untouched = the code that was here.
            _rfac_by_id = th.get("mn_rfac_by_id")
            if _rfac_by_id is not None:
                if _flat is not None:
                    raise ValueError(
                        "mn_rfac_by_id cannot compose with mn_size_flat: both write "
                        "the same vector")
                if not isinstance(_rfac_by_id, dict) or not _rfac_by_id:
                    raise ValueError(
                        "mn_rfac_by_id must be a non-empty map of banc_888_id to a "
                        "finite positive resistance factor")
                _rf_written, _rf_changed = [], []
                for _b, _v in _rfac_by_id.items():
                    _bid = str(_b)
                    if _bid not in rfac:
                        raise ValueError(
                            f"mn_rfac_by_id id {_bid} is not a size-override row, so "
                            f"it would add a row rather than reassign one")
                    if (type(_v) not in (int, float) or not np.isfinite(float(_v))
                            or float(_v) <= 0.0):
                        raise ValueError(
                            f"mn_rfac_by_id[{_bid}] must be finite and strictly "
                            f"positive, not {_v!r}")
                    if float(_v) != float(rfac[_bid]):
                        _rf_changed.append(_bid)
                    rfac[_bid] = float(_v)
                    _rf_written.append(_bid)
                if len(set(_rf_written)) != len(_rf_written):
                    raise ValueError("mn_rfac_by_id names an id twice")
                p.mn_rfac_by_id_audit = {
                    "n_written": len(_rf_written),
                    "n_changed": len(_rf_changed),
                    "written_ids": _rf_written,
                    "changed_ids": _rf_changed,
                    "n_override_entries": len(rfac),
                }
            # SINGLETON REMEDY, default-off (study 7f1b4a1e, decision 7b6efe4a
            # Build B1): th["mn_vrest_by_id"] overwrites or deletes named entries
            # of the size rule's rest before it becomes p.v_rest_override, as
            # mn_rfac_by_id does for rfac. A value in [-80, -30] mV overwrites the
            # entry; null deletes it, so the row rests at p.v_rest, which is what
            # the rule gives a pool of one. An id the rule did not rest raises, so
            # a mistyped id cannot add a row. mn_vrest_by_id_readback checks the
            # built rest before the walk and reads it back from the engine after.
            # Absent = untouched = the code that was here.
            _vrest_by_id = th.get("mn_vrest_by_id")
            if _vrest_by_id is not None:
                if not isinstance(_vrest_by_id, dict) or not _vrest_by_id:
                    raise ValueError(
                        "mn_vrest_by_id must be a non-empty map of banc_888_id to a "
                        "resting potential in [-80, -30] mV or null")
                _vr_rule = {}
                for _b, _v in _vrest_by_id.items():
                    _bid = str(_b)
                    if _bid in _vr_rule:
                        raise ValueError("mn_vrest_by_id names an id twice")
                    if _bid not in vrest:
                        raise ValueError(
                            f"mn_vrest_by_id id {_bid} is not a row the size rule "
                            f"rests, so it would add a row rather than reassign one")
                    if _v is not None and (
                            type(_v) not in (int, float) or not np.isfinite(float(_v))
                            or not -80.0 <= float(_v) <= -30.0):
                        raise ValueError(
                            f"mn_vrest_by_id[{_bid}] must be null or finite in "
                            f"[-80, -30] mV, not {_v!r}")
                    _vr_rule[_bid] = float(vrest[_bid])
                    if _v is None:
                        del vrest[_bid]
                    else:
                        vrest[_bid] = float(_v)
                p.mn_vrest_by_id_audit = {
                    "requested_mV": {str(_b): (None if _v is None else float(_v))
                                     for _b, _v in _vrest_by_id.items()},
                    "size_rule_mV": _vr_rule,
                    "n_deleted": sum(_v is None for _v in _vrest_by_id.values()),
                    "n_override_entries": len(vrest),
                }
            if _r_in_law_ov is not None:
                if (float(p.r_in_alpha), float(p.r_in_max)) != (0.10, 8.0):
                    raise ValueError("r_in_law's override was computed at r_in_alpha "
                                     "0.10 and r_in_max 8.0")
                for _b, _v in rfac.items():
                    if _r_in_law_ov.get(str(_b)) != float(
                            np.float32(float(_v) / _r_in_law_audit["gm"])):
                        raise ValueError(f"r_in_law: size-rule row {_b} moved")
                rfac = dict(_r_in_law_ov)
                p.r_in_law_audit = _r_in_law_audit
            p.r_in_override = rfac
            p.v_rest_override = vrest
            # SHAPE REPAIR, default-off (CD1-NI, 2026-09-02): the size override
            # gives 378 motor neurons a per-class v_rest of -68 to -48 while
            # v_reset stays a global -52.0, so a slow motor neuron resets BELOW
            # its own resting potential and a fast one resets 16 mV ABOVE it.
            # Shiu et al.'s inherited parameterisation sets v_reset = v_rest,
            # and the size override silently broke that identity for exactly
            # the cells it was meant to differentiate. th["mn_vreset"]="rest"
            # restores it per neuron. This adds NO free parameter: the value is
            # taken from v_rest, which is already set. Absent = untouched.
            if th.get("mn_vreset"):
                if th["mn_vreset"] != "rest":
                    raise ValueError(
                        "th['mn_vreset'] must be absent or 'rest'")
                _ids = list(meta["banc_888_id"])
                _vr = np.full(len(_ids), float(p.v_reset), dtype=np.float64)
                _pos = {b: i for i, b in enumerate(_ids)}
                _n = 0
                for _b, _v in vrest.items():
                    _i = _pos.get(_b)
                    if _i is not None:
                        _vr[_i] = float(_v)
                        _n += 1
                if _n != len(vrest):
                    raise ValueError(
                        f"mn_vreset covered {_n} of {len(vrest)} override rows")
                p.v_reset = _vr
            # STUDY 9e0afa22 BUILD 1, default-off (design 7c57b886): each named
            # motor row resets at its own rest plus an offset, the rest being
            # vrest after mn_vrest_by_id or the shared v_rest where the rule
            # rests none. Absent = untouched = the code that was here.
            if th.get("mn_reset_offset_by_id") is not None:
                mn_reset_offset_by_id_apply(th, meta, p, vrest)
        if th.get("mn_reset_offset_by_id") is not None and not use_size:
            raise ValueError(
                "mn_reset_offset_by_id needs mn_gains 'size': the offset is "
                "read at the size rule's mn_vreset seam")
        _vrmode = th.get("v_rest_mode")
        # H08 (design fa5adfc8, study 3820bed5), default-off:
        # th["v_rest_type_reset_follows"] = true sets v_reset to the exact-type
        # rest on exactly the rows that rest is written to. Without it a TTMn
        # moved to -68 mV resets to the global -52, 16 mV above its own rest,
        # and mn_vreset "rest" would move all 378 size-rule resets instead.
        # Absent = untouched = bit-identical.
        _vr_follows = th.get("v_rest_type_reset_follows")
        if _vr_follows is not None:
            if _vr_follows is not True:
                raise ValueError(
                    "v_rest_type_reset_follows must be true or absent")
            if _vrmode != "by_exact_type":
                raise ValueError(
                    "v_rest_type_reset_follows requires v_rest_mode "
                    "'by_exact_type'")
            if th.get("mn_vreset"):
                raise ValueError(
                    "v_rest_type_reset_follows cannot compose with mn_vreset")
            if th.get("mn_size_flat") == "v":
                raise ValueError(
                    "v_rest_type_reset_follows cannot compose with "
                    "mn_size_flat 'v'")
        if _vrmode == "by_exact_type":
            if th.get("membrane_mode"):
                raise ValueError(
                    "exact-type v_rest cannot compose with membrane_mode")
            _vrmap = th.get("v_rest_type_mv")
            if (not isinstance(_vrmap, dict) or not _vrmap
                    or not all(isinstance(k, str) for k in _vrmap)
                    or not all(type(v) in (int, float) and np.isfinite(v)
                               and -80.0 <= float(v) <= -30.0
                               for v in _vrmap.values())):
                raise ValueError(
                    "v_rest_type_mv must be a non-empty map of cell type "
                    "to a finite resting potential in [-80, -30] mV")
            _vr_overrides, p.v_rest_type_audit = (
                exact_type_vrest_overrides(meta, _vrmap))
            _prior_vrest = dict(p.v_rest_override or {})
            _merged_vrest = dict(_prior_vrest)
            _merged_vrest.update(_vr_overrides)
            p.v_rest_override = _merged_vrest
            if _vr_follows:
                _vrf_ids = meta["banc_888_id"].astype(str).to_numpy()
                _vrf_pos = {b: i for i, b in enumerate(_vrf_ids)}
                _vrf_types = meta["cell_type"].astype("string").fillna(
                    "").to_numpy()
                _vrf_rest0 = np.broadcast_to(
                    np.asarray(p.v_rest, dtype=np.float64), (len(meta),))
                _vrf_reset = np.broadcast_to(
                    np.asarray(p.v_reset, dtype=np.float64),
                    (len(meta),)).astype(np.float64).copy()
                _vrf_rows = []
                for _b, _v in _vr_overrides.items():
                    _i = _vrf_pos[str(_b)]
                    _prior = _prior_vrest.get(_b, _prior_vrest.get(str(_b)))
                    _vrf_rows.append({
                        "banc_888_id": str(_b),
                        "row_index": int(_i),
                        "cell_type": str(_vrf_types[_i]),
                        "v_rest_before_mv": float(
                            _vrf_rest0[_i] if _prior is None else _prior),
                        "v_rest_after_mv": float(_v),
                        "v_reset_before_mv": float(_vrf_reset[_i]),
                        "v_reset_after_mv": float(_v),
                    })
                    _vrf_reset[_i] = float(_v)
                p.v_reset = _vrf_reset
                p.v_rest_type_reset_audit = {
                    "key": "v_rest_type_reset_follows",
                    "rows_changed": len(_vrf_rows),
                    "rows": _vrf_rows,
                }
        elif _vrmode not in (None, ""):
            raise ValueError(f"unknown v_rest_mode: {_vrmode!r}")
        if row_set_params is not None:
            _row_set_rest_apply(row_set_params, meta, p)
        # FL5co (lane A, 2026-09-13 18:20; judge 2026-09-14 13:09: "admitted
        # as nominee keys on A's nomination"). The 24 wing power motor
        # neurons (DLM1-4, DLM5, DVM1a-c, DVM2a-b, DVM3a-b; BANC exact cell
        # types, 12 per side) rest near -60 mV (Hurkey et al. 2023, PMID
        # 37225999, measured on the DLM; the DVM is ASSUMED equal, registry
        # cns.dlm_mn_rest_mv) where the engine's default and the size rule
        # leave them at -52. th["wing_power_mn_rest_mv"] sets v_rest on
        # exactly those 24 rows, applied after every other v_rest path so it
        # is the final word on them. th["wing_power_mn_reset_follows_rest"]
        # = true also sets their v_reset to the same value: Shiu et al.'s
        # v_reset = v_rest identity, which the bare override breaks by 8 mV
        # (the override-alone variant is the one that fell in FL5co).
        # Absent = untouched = bit-identical.
        _wp_rest = th.get("wing_power_mn_rest_mv")
        if _wp_rest is not None:
            if (type(_wp_rest) not in (int, float) or not np.isfinite(_wp_rest)
                    or not -80.0 <= float(_wp_rest) <= -30.0):
                raise ValueError(
                    "wing_power_mn_rest_mv must be finite in [-80, -30] mV")
            _wp_reset = th.get("wing_power_mn_reset_follows_rest", False)
            if _wp_reset not in (True, False):
                raise ValueError(
                    "wing_power_mn_reset_follows_rest must be true or false")
            _wp_types = ("DLM1-4", "DLM5", "DVM1a-c", "DVM2a-b", "DVM3a-b")
            _wp_rows = np.flatnonzero(
                meta["cell_type"].astype(str).isin(_wp_types).to_numpy())
            if len(_wp_rows) != 24:
                raise ValueError(
                    f"wing power pool is {len(_wp_rows)} rows, expected 24")
            _wp_ids = [str(b) for b in meta["banc_888_id"].to_numpy()[_wp_rows]]
            _merged_vrest = dict(p.v_rest_override or {})
            for _b in _wp_ids:
                _merged_vrest[_b] = float(_wp_rest)
            p.v_rest_override = _merged_vrest
            if _wp_reset:
                _vr = np.broadcast_to(
                    np.asarray(p.v_reset, dtype=np.float64),
                    (len(meta),)).astype(np.float64).copy()
                _vr[_wp_rows] = float(_wp_rest)
                p.v_reset = _vr
        # A4's third key, ported from team-henri 97d7a92a5 walk_search.py
        # 9036-9065 (study d705a4b1, design b1127dc4) by design afc8fd03 B1
        # (study 9a1ef96a, gpu-4, 2026-09-29). The same 24 wing power motor
        # neurons' input resistance as a class transfer of Azevedo 2020's fast
        # profile: th["wing_power_mn_r_in_factor"] enters their banc ids into
        # r_in_override, which cns.py folds into every incoming weight after
        # the balance rule. cns.py skips an id it cannot find without a word,
        # so the 24 are matched against meta here, and a row the size rule
        # already covers is refused rather than overwritten. Absent = untouched
        # = bit-identical.
        _wp_rin = th.get("wing_power_mn_r_in_factor")
        if _wp_rin is not None:
            if (type(_wp_rin) not in (int, float) or not np.isfinite(_wp_rin)
                    or float(_wp_rin) <= 0.0):
                raise ValueError(
                    "wing_power_mn_r_in_factor must be finite and > 0")
            _wp_types = ("DLM1-4", "DLM5", "DVM1a-c", "DVM2a-b", "DVM3a-b")
            _wp_rows = np.flatnonzero(
                meta["cell_type"].astype(str).isin(_wp_types).to_numpy())
            _wp_ids = [str(b) for b in meta["banc_888_id"].to_numpy()[_wp_rows]]
            _meta_ids = set(meta["banc_888_id"].astype(str))
            if len(_wp_rows) != 24 or len(set(_wp_ids) & _meta_ids) != 24:
                raise ValueError(
                    f"wing power pool matched {len(_wp_rows)} rows, expected 24")
            _merged_rin = dict(p.r_in_override or {})
            _clash = sorted(b for b in _wp_ids if b in _merged_rin)
            if _clash:
                raise ValueError(
                    f"wing_power_mn_r_in_factor: {len(_clash)} rows already "
                    f"carry an r_in_override entry: {_clash[:3]}")
            for _b in _wp_ids:
                _merged_rin[_b] = float(_wp_rin)
            p.r_in_override = _merged_rin
        # Design d27f82df B1: after the last write of p.r_in_override.
        if th.get("r_in_total_by_id") is not None:
            r_in_total_by_id_apply(p, meta, th["r_in_total_by_id"],
                                   r_in_law=th.get("r_in_law"))
        _gvr_mode = th.get("graded_premotor_v_rest_mode")
        if _gvr_mode:
            if th.get("membrane_mode"):
                raise ValueError(
                    "graded premotor v_rest cannot compose with membrane_mode")
            _gvr_population = th.get(
                "graded_premotor_v_rest_population", "19a006_13ba")
            if _gvr_population not in (
                    "19a006_13ba", "19a006_13ba_disinhibition"):
                raise ValueError("unknown graded premotor v_rest population")
            if ("graded_premotor_v_rest_population" in th
                    and th.get("graded") != _gvr_population):
                raise ValueError(
                    "graded premotor v_rest population must match graded set")
            _premotor_rows = graded_rows_for(_gvr_population, meta)
            _gvr_values = th.get("graded_premotor_v_rest_values_mv")
            if _gvr_values is None:
                _gvr_overrides, _gvr_rows = graded_premotor_vrest_overrides(
                    meta, _premotor_rows, _gvr_mode,
                    th["graded_premotor_v_rest_mv"],
                    th["graded_premotor_v_rest_permutation_seed"])
            else:
                _gvr_overrides, _gvr_rows = (
                    graded_premotor_empirical_vrest_overrides(
                        meta, _premotor_rows, _gvr_mode, _gvr_values,
                        th["graded_premotor_v_rest_assignment_seed"],
                        th["graded_premotor_v_rest_permutation_seed"]))
            _merged_vrest = dict(p.v_rest_override or {})
            _merged_vrest.update(_gvr_overrides)
            p.v_rest_override = _merged_vrest
            p.graded_premotor_v_rest_mode = _gvr_mode
            p.graded_premotor_v_rest_population = _gvr_population
            p.graded_premotor_v_rest_rows = len(_gvr_rows)
            if _gvr_values is None:
                p.graded_premotor_v_rest_mv = float(
                    th["graded_premotor_v_rest_mv"])
                p.graded_premotor_v_rest_values_mv = None
                p.graded_premotor_v_rest_unique_values = 1
            else:
                p.graded_premotor_v_rest_mv = None
                p.graded_premotor_v_rest_values_mv = [
                    float(value) for value in _gvr_values]
                p.graded_premotor_v_rest_unique_values = len(
                    np.unique(np.asarray(_gvr_values, dtype=np.float64)))
        _typed_12b_mode = th.get("typed_12b_v_rest_mode")
        if _typed_12b_mode:
            if th.get("membrane_mode"):
                raise ValueError(
                    "typed 12B v_rest cannot compose with membrane_mode")
            _typed_12b = typed_12b_vrest_overrides(
                meta, _typed_12b_mode, th["typed_12b_v_rest_mv"])
            _merged_vrest = dict(p.v_rest_override or {})
            _merged_vrest.update(_typed_12b)
            p.v_rest_override = _merged_vrest
            # Reset follows rest for the same two cells.  This keeps the
            # firing margin coherent instead of introducing a universal reset.
            _v_reset = np.broadcast_to(
                np.asarray(p.v_reset, dtype=np.float64), (len(meta),)).copy()
            _positions = pd.Series(
                np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
            for _body_id, _value_mv in _typed_12b.items():
                _v_reset[int(_positions.loc[_body_id])] = _value_mv
            p.v_reset = _v_reset
            p.typed_12b_v_rest_mode = _typed_12b_mode
            p.typed_12b_v_rest_rows = len(_typed_12b)
            p.typed_12b_v_rest_mv = float(th["typed_12b_v_rest_mv"])
        _mmode = th.get("membrane_mode")
        if _mmode == "class_draw":
            _rest, _tau, _refrac, _threshold = class_draw_membrane_vectors(
                meta, p.v_threshold, th.get("membrane_seed"),
                p.v_rest_override)
            p.v_rest = _rest
            p.v_reset = _rest.copy()
            p.tau_mem = _tau
            p.t_refrac = _refrac
            p.v_threshold = _threshold
            # Overrides have been folded into the vector so reset and firing
            # margin stay consistent for the measured motor-size rows.
            p.v_rest_override = None
            p.membrane_mode = "class_draw"
            p.membrane_seed = int(th["membrane_seed"])
            p.membrane_class_count = int(
                np.unique(membrane_annotation_groups(meta)).size)
        elif _mmode not in (None, ""):
            raise ValueError(f"unknown membrane_mode: {_mmode!r}")
        _apmode = th.get("adaptation_parameter_mode")
        if _apmode == "membrane_coupled":
            if _mmode != "class_draw":
                raise ValueError(
                    "membrane-coupled adaptation requires class_draw")
            p.adapt_b, p.tau_w = membrane_coupled_adaptation_vectors(
                p.v_rest, p.v_threshold, p.tau_mem,
                float(th["adapt_b"]), float(th.get("tau_w", 100.0)))
            p.adaptation_parameter_mode = _apmode
        elif _apmode not in (None, ""):
            raise ValueError(
                f"unknown adaptation_parameter_mode: {_apmode!r}")
        # Design 2163ba7d, Build 3, ported from Henri's fc0d6e9d0: the scalar
        # tau_w becomes a float32 vector, tau_w on every row and tau_w x mult on
        # the named rows. After every tau_w path, and it refuses a theta that
        # already builds a vector, so the multiplier has one base to scale.
        # Both engines take a per-row tau_w (cns.py dt_over_tauw; the CUDA
        # all-same fast path keeps mult 1.0 on the scalar route). Absent =
        # bit-identical.
        _ct = th.get("crossing_tau_w_mult")
        _ct_idx = None
        if _ct is not None:
            if not isinstance(_ct, dict) or set(_ct) != {"mult", "rows"}:
                raise ValueError(
                    "crossing_tau_w_mult must be {mult, rows}")
            _ct_m = _ct["mult"]
            if (type(_ct_m) not in (int, float) or not np.isfinite(_ct_m)
                    or not float(_ct_m) > 0.0):
                raise ValueError(
                    "crossing_tau_w_mult mult must be finite and positive")
            _ct_rows = [str(_r) for _r in _ct["rows"]]
            if not _ct_rows or len(set(_ct_rows)) != len(_ct_rows):
                raise ValueError(
                    "crossing_tau_w_mult rows must be a non-empty list of distinct ids")
            if np.ndim(p.tau_w) != 0:
                raise ValueError(
                    "crossing_tau_w_mult needs a scalar tau_w; this theta already "
                    "builds a per-neuron vector")
            if not float(p.tau_w) > 0.0:
                raise ValueError("crossing_tau_w_mult needs tau_w > 0")
            _ct_pos = pd.Series(
                np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
            _ct_missing = [x for x in _ct_rows if x not in _ct_pos.index]
            if _ct_missing:
                raise ValueError(
                    f"crossing_tau_w_mult names {len(_ct_missing)} ids absent "
                    f"from meta, first {_ct_missing[0]!r}")
            _ct_idx = _ct_pos.loc[_ct_rows].to_numpy()
            _tw = np.full(len(meta), float(p.tau_w), dtype=np.float32)
            _tw[_ct_idx] = np.float32(float(p.tau_w) * float(_ct_m))
            p.crossing_tau_w_scalar = float(p.tau_w)
            p.tau_w = _tw
            p.crossing_tau_w_mult = float(_ct_m)
            p.crossing_tau_w_rows = len(_ct_rows)
            p.crossing_tau_w_row_index = np.sort(_ct_idx).astype(np.int64)
        _ct_hold = th.get("crossing_tau_w_hold_b_tau")
        if _ct_hold is not None and (_ct_hold is not True or _ct is None):
            raise ValueError(
                "crossing_tau_w_hold_b_tau is true or absent, and needs "
                "crossing_tau_w_mult")
        _gpmode = th.get("graded_parameter_mode")
        if _gpmode == "membrane_range_10_90":
            if not graded_rows:
                raise ValueError(
                    "membrane-normalized graded parameters require graded rows")
            if _mmode != "class_draw":
                raise ValueError(
                    "membrane-normalized graded parameters require class_draw")
            p.graded_v50, p.graded_slope = membrane_range_graded_vectors(
                p.v_rest, p.v_threshold)
            p.graded_parameter_mode = _gpmode
        elif _gpmode == "hemilineage_19a_13b":
            if not graded_rows:
                raise ValueError(
                    "hemilineage graded parameters require graded rows")
            p.graded_v50, p.graded_slope = (
                hemilineage_graded_parameter_vectors(
                    meta, graded_rows, p.graded_v50, p.graded_slope,
                    th.get("graded_hemilineage_v50_delta_mv", 0.0),
                    th.get("graded_hemilineage_slope_ratio", 1.0),
                    th.get("graded_hemilineage_permutation_seed")))
            p.graded_parameter_mode = _gpmode
            p.graded_hemilineage_permutation_seed = th.get(
                "graded_hemilineage_permutation_seed")
        elif _gpmode not in (None, ""):
            raise ValueError(f"unknown graded_parameter_mode: {_gpmode!r}")
        _sdmode = th.get("synaptic_decay_mode")
        if _sdmode == "class_draw_area_normalized":
            p.tau_syn = class_draw_synaptic_decay_vector(
                meta, th.get("synaptic_decay_seed"))
            p.syn_normalised = True
            p.synaptic_decay_mode = _sdmode
            p.synaptic_decay_seed = int(th["synaptic_decay_seed"])
            p.synaptic_decay_class_count = int(
                np.unique(membrane_annotation_groups(meta)).size)
        elif _sdmode not in (None, ""):
            raise ValueError(f"unknown synaptic_decay_mode: {_sdmode!r}")
        if _graded_edge_equilibrium_mode:
            p.graded_edge_equilibrium_ids = graded_edge_equilibrium_ids
        # Judge seam, 2026-09-03 (lane C's 14:05 request): silence a named
        # row set through the one drive path, the way lesion_test.py does it,
        # by raising those rows' threshold to a value the membrane never
        # reaches (1e9 mV). The rows still receive and integrate; they emit
        # nothing. Absent = bit-identical. Every id must be a BANC id in
        # meta, and none may be a graded row: a graded row releases from its
        # membrane potential with no threshold gate, so raising its threshold
        # would silence nothing and the run would report a lesion it did not
        # make. Applied after every other threshold mode so it is the last
        # word on those rows.
        if th.get("silence_rows") is not None:
            _silence_ids = [str(x) for x in th["silence_rows"]]
            if not _silence_ids or len(set(_silence_ids)) != len(_silence_ids):
                raise ValueError(
                    "silence_rows must be a non-empty list of distinct ids")
            _silence_pos = pd.Series(
                np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
            _missing = [x for x in _silence_ids if x not in _silence_pos.index]
            if _missing:
                raise ValueError(
                    f"silence_rows names {len(_missing)} ids absent from "
                    f"meta, first {_missing[0]!r}")
            _silence_rows = np.asarray(
                [int(_silence_pos.loc[x]) for x in _silence_ids], dtype=int)
            _graded_hit = np.intersect1d(_silence_rows, np.asarray(
                graded_rows, dtype=int))
            if _graded_hit.size:
                raise ValueError(
                    f"silence_rows includes {_graded_hit.size} graded rows, "
                    "which a threshold raise cannot silence")
            # design 302c1bfb, Build 10: with silence_rows_window_ms the
            # engine raises the threshold inside the window only, so the
            # built network keeps every row's own threshold
            if th.get("silence_rows_window_ms") is None:
                _thr = np.broadcast_to(
                    np.asarray(p.v_threshold, dtype=np.float64),
                    (len(meta),)).copy()
                _thr[_silence_rows] = 1e9
                p.v_threshold = _thr
        # Design efe88957 Build 10 (H7): silence named GRADED rows, which
        # silence_rows refuses. Each row's per-row graded gain is set to 0, so
        # the engines' graded gather (delivered charge = gain x release x |w|
        # on every outgoing edge) delivers nothing from it on any tick, while
        # its membrane receives and integrates as before. Both engines take a
        # finite non-negative (N,) gain vector (cns.py, cns_metal.py), so no
        # engine change. Every id must be a BANC id in meta and a graded row.
        # The vector keeps the built gain's dtype, so every other row's gain
        # carries the bits it had. Applied after every graded gain mode, so it
        # is the last word on those rows' gain. Absent = bit-identical.
        if th.get("silence_graded_rows") is not None:
            _sg_ids = [str(x) for x in th["silence_graded_rows"]]
            if not _sg_ids or len(set(_sg_ids)) != len(_sg_ids):
                raise ValueError(
                    "silence_graded_rows must be a non-empty list of distinct ids")
            _sg_all = meta["banc_888_id"].astype(str).to_numpy()
            _sg_hits = [np.flatnonzero(_sg_all == x) for x in _sg_ids]
            _missing = [x for x, h in zip(_sg_ids, _sg_hits) if h.size != 1]
            if _missing:
                raise ValueError(
                    f"silence_graded_rows names {len(_missing)} ids not on "
                    f"exactly one meta row, first {_missing[0]!r}")
            _sg_rows = np.asarray([int(h[0]) for h in _sg_hits], dtype=int)
            _sg_spiking = np.setdiff1d(_sg_rows, np.asarray(graded_rows, dtype=int))
            if _sg_spiking.size:
                raise ValueError(
                    f"silence_graded_rows names {_sg_spiking.size} rows outside "
                    "the graded set; silence_rows silences a spiking row")
            _sg_gain = np.asarray(p.graded_gain)
            _sg_vec = np.array(np.broadcast_to(_sg_gain, (len(meta),)),
                               dtype=_sg_gain.dtype)
            _sg_before = [float(x) for x in _sg_vec[_sg_rows]]
            _sg_vec[_sg_rows] = 0
            p.graded_gain = _sg_vec
            p.silence_graded_rows_audit = {
                "ids": _sg_ids, "rows": [int(r) for r in _sg_rows],
                "gain_before": _sg_before, "gain_after": 0.0,
                "dtype": str(_sg_vec.dtype)}
        # Design 1 revision 5bff1280 for study 0124fdfc, 2026-09-20: turn
        # adaptation off on a named row set and nowhere else, so an arm can ask
        # whether adaptation is what holds a response down. Scalar adapt_a and
        # adapt_b expand into N-vectors, zero on those rows and unchanged
        # elsewhere; cns_metal.py:347-349 and 483-484 take vectors through _vec
        # and use_adapt stays true, so no engine change. Absent = bit-identical.
        # Applied after every other adaptation mode, including the
        # membrane-coupled vectors above, so it is the last word on those rows.
        if th.get("adapt_zero_ids") is not None:
            _az_ids = [str(x) for x in th["adapt_zero_ids"]]
            if not _az_ids or len(set(_az_ids)) != len(_az_ids):
                raise ValueError(
                    "adapt_zero_ids must be a non-empty list of distinct ids")
            _az_pos = pd.Series(
                np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
            _az_missing = [x for x in _az_ids if x not in _az_pos.index]
            if _az_missing:
                raise ValueError(
                    f"adapt_zero_ids names {len(_az_missing)} ids absent from "
                    f"meta, first {_az_missing[0]!r}")
            _az_rows = np.asarray(
                [int(_az_pos.loc[x]) for x in _az_ids], dtype=int)
            for _az_name in ("adapt_a", "adapt_b"):
                _az_vec = np.broadcast_to(
                    np.asarray(getattr(p, _az_name, 0.0), dtype=np.float64),
                    (len(meta),)).copy()
                _az_vec[_az_rows] = 0.0
                setattr(p, _az_name, _az_vec)
        # Design 2d12a3b3 (B1), default-off: mn_membrane_by_id gives named leg
        # motor neurons their own membrane constant and adaptation,
        # {banc_888_id: {"tau_mem_ms", "adapt_a", "adapt_b_mv", "tau_w_ms"}}.
        # tau_mem, adapt_a, adapt_b and tau_w each become a float64 vector over
        # every row at the value it has here and the named rows are written;
        # the driver casts a scalar and a vector through one float32 path, so
        # naming a row at its own values changes no engine array. v_rest is not
        # touched, so the conductance scale keeps its value. It refuses to
        # compose with the other adaptation and membrane modes rather than
        # decide an order between them. Absent = bit-identical.
        _mmb = th.get("mn_membrane_by_id")
        if _mmb is not None:
            for _clash in ("adapt_zero_ids", "mn_adapt_b_zero_ids",
                           "membrane_mode", "adaptation_parameter_mode",
                           "crossing_tau_w_mult"):
                if th.get(_clash) not in (None, ""):
                    raise ValueError(
                        f"mn_membrane_by_id cannot compose with {_clash}")
            if not isinstance(_mmb, dict) or not _mmb:
                raise ValueError(
                    "mn_membrane_by_id must be a non-empty map of banc_888_id "
                    "to {tau_mem_ms, adapt_a, adapt_b_mv, tau_w_ms}")
            import motormap as _MMb
            _mmb_ok = {str(x) for x in _MMb.to_matrix(_MMb.build_map(meta))[1]}
            _mmb_pos = {b: i for i, b in enumerate(meta["banc_888_id"].astype(str))}
            _mmb_fields = {"tau_mem_ms": ("tau_mem", 1.0, 100.0),
                           "adapt_a": ("adapt_a", 0.0, 10.0),
                           "adapt_b_mv": ("adapt_b", 0.0, 20.0),
                           "tau_w_ms": ("tau_w", 1.0, 5000.0)}
            _mmb_vec = {
                _attr: np.broadcast_to(np.asarray(
                    getattr(p, _attr), dtype=np.float64), (len(meta),)).copy()
                for _attr, _lo, _hi in _mmb_fields.values()}
            _mmb_rows = []
            for _b, _spec in _mmb.items():
                _bid = str(_b)
                if _bid not in _mmb_ok:
                    raise ValueError(
                        f"mn_membrane_by_id id {_bid} is not one of the motor "
                        f"map's {len(_mmb_ok)} to_matrix ids")
                if not isinstance(_spec, dict) or set(_spec) != set(_mmb_fields):
                    raise ValueError(
                        f"mn_membrane_by_id[{_bid}] must carry exactly "
                        f"{sorted(_mmb_fields)}")
                _row = _mmb_pos[_bid]
                for _f, (_attr, _lo, _hi) in _mmb_fields.items():
                    _x = _spec[_f]
                    if (type(_x) not in (int, float) or not np.isfinite(float(_x))
                            or not _lo <= float(_x) <= _hi):
                        raise ValueError(
                            f"mn_membrane_by_id[{_bid}][{_f}] must be finite and in "
                            f"[{_lo}, {_hi}], not {_x!r}")
                    _mmb_vec[_attr][_row] = float(_x)
                _mmb_rows.append(_row)
            if len(set(_mmb_rows)) != len(_mmb_rows):
                raise ValueError("mn_membrane_by_id names a row twice")
            for _attr, _v in _mmb_vec.items():
                setattr(p, _attr, _v)
            _mmb_sel = np.asarray(sorted(_mmb_rows), dtype=np.int64)
            p.mn_membrane_by_id_audit = {
                "n_rows": len(_mmb_rows),
                "min": {_f: float(_mmb_vec[_a][_mmb_sel].min())
                        for _f, (_a, _l, _h) in _mmb_fields.items()},
                "max": {_f: float(_mmb_vec[_a][_mmb_sel].max())
                        for _f, (_a, _l, _h) in _mmb_fields.items()},
                "sha256": {_a: hashlib.sha256(np.ascontiguousarray(
                    _v, dtype=np.float64).tobytes()).hexdigest()
                    for _a, _v in _mmb_vec.items()},
            }
        # Design 2163ba7d, Build 3's hold: adapt_b divided by the same mult on
        # the same rows, so b x tau_w, the area of each spike's adaptation
        # current, is unchanged on every dosed row. After adapt_zero_ids, so it
        # divides whatever adapt_b every other path left (a zeroed row stays
        # zero), and it refuses a hold that would divide nothing.
        if _ct_hold:
            _ctb = np.broadcast_to(
                np.asarray(p.adapt_b, dtype=np.float64), (len(meta),)).copy()
            if not np.any(_ctb[_ct_idx] != 0.0):
                raise ValueError(
                    "crossing_tau_w_hold_b_tau: adapt_b is zero on every dosed "
                    "row, so the hold would change nothing")
            _ctb[_ct_idx] = _ctb[_ct_idx] / float(_ct["mult"])
            p.adapt_b = _ctb
            p.crossing_tau_w_hold_b_tau = True
        # Design 6d0042d5, Build 1, default-off: adapt_law_rows sets the
        # adaptation law per row (adapt_law_rows_vectors), after every other
        # adaptation path so it writes over what they left and composes only
        # where they allow it. Present, it always sets p.adapt_a_below, so a
        # group at the scalars runs the engines' split path (Build 2), which
        # runs bit-identical to the unsplit law. Absent = bit-identical.
        if th.get("adapt_law_rows") is not None:
            p.adapt_law_rows_audit = adapt_law_rows_vectors(th, meta, p)
        # Design d27f82df B2 and B3, default-off: named rows' own membrane,
        # threshold, adaptation and exact update, after every other path
        # (row_membrane_by_id_apply), then their hold-down conductance
        # (holddown_by_id_apply). Absent = bit-identical.
        if th.get("row_membrane_by_id") is not None:
            row_membrane_by_id_apply(th, meta, p, graded_rows)
        # Design 83177ced Build 1, default-off: named motor rows join the exact
        # update (mn_exact_by_id_apply), after row_membrane_by_id_apply, which
        # sets p.row_exact_mask afresh, and before the hold-down, which needs
        # its rows on it. Absent = bit-identical.
        if th.get("mn_exact_by_id") is not None:
            mn_exact_by_id_apply(th, meta, p)
        if th.get("holddown_by_id") is not None:
            holddown_by_id_apply(th, meta, p)
        # Design 2163ba7d, Build 1 (piece 2 of study 60c2e7f7, CsChrimson
        # set): a light pulse train drives the 87a3ad35 opsin conductance on
        # the listed rows, g_max relative to leak (not through gscale), with
        # the excitatory reversal e_exc, opening and closing by the one-step
        # first-order updates. The term g_max s (e_exc - v) is appended last
        # in both integrators, the value adding g_max s to ge would give.
        # u is 1 when start <= t < stop and (t - start) mod (on + off) < on,
        # every edge a whole number of steps (cns.opsin_arrays raises
        # otherwise). A graded listed row is allowed and flagged in the
        # receipt. Absent = bit-identical.
        if th.get("pulse_exc_sets") is not None and th.get("pulse_exc_ids") is not None:
            raise ValueError("pulse_exc_sets replaces pulse_exc_ids; give one form")
        if th.get("pulse_exc_ids") is not None:
            _pe_ids = [str(x) for x in th["pulse_exc_ids"]]
            if not p.conductance_based:
                raise ValueError("pulse_exc_ids needs the conductance law (cbased)")
            if not _pe_ids or len(set(_pe_ids)) != len(_pe_ids):
                raise ValueError(
                    "pulse_exc_ids must be a non-empty list of distinct ids")
            _pe_sha = hashlib.sha256(
                "\n".join(sorted(_pe_ids, key=int)).encode()).hexdigest()
            if th.get("pulse_exc_ids_sha256") != _pe_sha:
                raise ValueError(
                    f"pulse_exc_ids hash {_pe_sha} differs from "
                    f"pulse_exc_ids_sha256 {th.get('pulse_exc_ids_sha256')!r}")
            _pe_pos = pd.Series(
                np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
            _pe_missing = [x for x in _pe_ids if x not in _pe_pos.index]
            if _pe_missing:
                raise ValueError(
                    f"pulse_exc_ids names {len(_pe_missing)} ids absent from "
                    f"meta, first {_pe_missing[0]!r}")
            if np.ndim(p.e_exc) != 0:
                raise ValueError("pulse_exc needs a scalar e_exc")
            _pe_rows = np.sort(np.asarray(
                [int(_pe_pos.loc[x]) for x in _pe_ids], dtype=np.int64))
            _pe = _pulse_exc_values(th)
            p.opsin_rows = (_pe_rows,)
            p.opsin_e_rev_mv = (float(p.e_exc),)
            p.opsin_g_max = (_pe["pulse_exc_g"],)
            p.opsin_tau_on_ms = (_pe["pulse_tau_open_ms"],)
            p.opsin_tau_off_ms = (_pe["pulse_tau_close_ms"],)
            p.opsin_windows_ms = ((),)
            p.opsin_train_ms = ((_pe["pulse_start_ms"], _pe["pulse_stop_ms"],
                                 _pe["pulse_on_ms"], _pe["pulse_off_ms"]),)
        # Design 568d05a8, Build B5: pulse_exc_sets = [{ids, ids_sha256, g}],
        # one opsin set per entry in list order, each at its own g_max, all
        # with e_exc, the shared pulse train and taus; the slots follow set
        # order and cns.opsin_arrays refuses a row in two sets. Absent =
        # bit-identical.
        elif th.get("pulse_exc_sets") is not None:
            _pes, _pes_v = _pulse_exc_sets(th)
            if not p.conductance_based:
                raise ValueError("pulse_exc_sets needs the conductance law (cbased)")
            if np.ndim(p.e_exc) != 0:
                raise ValueError("pulse_exc needs a scalar e_exc")
            _pe_pos = pd.Series(
                np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
            _pes_rows = []
            for _j, _s in enumerate(_pes):
                _pe_missing = [x for x in _s["ids"] if x not in _pe_pos.index]
                if _pe_missing:
                    raise ValueError(
                        f"pulse_exc_sets[{_j}] names {len(_pe_missing)} ids absent "
                        f"from meta, first {_pe_missing[0]!r}")
                _pes_rows.append(np.sort(np.asarray(
                    [int(_pe_pos.loc[x]) for x in _s["ids"]], dtype=np.int64)))
            _n = len(_pes)
            p.opsin_rows = tuple(_pes_rows)
            p.opsin_e_rev_mv = (float(p.e_exc),) * _n
            p.opsin_g_max = tuple(_s["g"] for _s in _pes)
            p.opsin_tau_on_ms = (_pes_v["pulse_tau_open_ms"],) * _n
            p.opsin_tau_off_ms = (_pes_v["pulse_tau_close_ms"],) * _n
            p.opsin_windows_ms = ((),) * _n
            p.opsin_train_ms = ((_pes_v["pulse_start_ms"], _pes_v["pulse_stop_ms"],
                                 _pes_v["pulse_on_ms"], _pes_v["pulse_off_ms"]),) * _n
        # Design 2163ba7d, Build 2 (H02): slow_inh_out = {ids, frac, tau_ms},
        # a slow inhibitory pool on the listed rows' output (cns.SimParams
        # slow_inh_*): of each inhibitory increment a listed row's spike
        # delivers, frac goes to the target's slow conductance, decaying with
        # tau_ms at the fast pool's reversal, scaled so the conductance-time
        # per spike is unchanged. Each front-leg motor neuron's slow
        # conductance is recorded per 5 ms bin. The conductance law with split
        # pools only; cns.slow_inh_arrays validates the rest at the run's
        # start. CUDA and NumPy; Metal refuses it. Absent = bit-identical.
        if th.get("slow_inh_out") is not None:
            _si = _slow_inh_spec(th)
            if not (p.conductance_based and p.split_conductances):
                raise ValueError("slow_inh_out needs cbased and shunt (split pools)")
            _si_pos = pd.Series(
                np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
            _si_missing = [x for x in _si["ids"] if x not in _si_pos.index]
            if _si_missing:
                raise ValueError(
                    f"slow_inh_out names {len(_si_missing)} ids absent from "
                    f"meta, first {_si_missing[0]!r}")
            p.slow_inh_rows = np.sort(np.asarray(
                [int(_si_pos.loc[x]) for x in _si["ids"]], dtype=np.int64))
            p.slow_inh_frac = _si["frac"]
            p.slow_inh_tau_ms = _si["tau_ms"]
            p.slow_inh_record_rows = _front_leg_mn_rows(meta)
            p.slow_inh_record_bin_ms = 5.0
        if len(net_cache) >= 3:
            net_cache.pop(next(iter(net_cache)))
        _network_edges = edges
        if feco_13b_analog is not None:
            _keep_edges = np.ones(len(edges), dtype=bool)
            _keep_edges[feco_13b_analog["drop_edge_positions"]] = False
            _network_edges = edges.iloc[_keep_edges]
            p.feco_13b_analog_mode = _feco_13b_analog_mode
            p.feco_13b_analog_edge_pairs = feco_13b_analog["edge_pairs"]
            p.feco_13b_analog_chemical_synapses_replaced = (
                feco_13b_analog["chemical_synapses_replaced"])
            p.feco_13b_analog_target_rows = (
                feco_13b_analog["driven_target_rows"])
        if th.get("mn_vrest_by_id") is not None:     # study 7f1b4a1e B1, before the walk
            mn_vrest_by_id_readback(p, meta["banc_888_id"].to_numpy())
        if th.get("mn_reset_offset_by_id") is not None:     # study 9e0afa22 Build 1, before the walk
            mn_reset_offset_by_id_check(p, meta["banc_888_id"].to_numpy())
        _ecs_receipt = None
        if edge_count_set is not None:     # design 478fe148 B1, last table edit
            _network_edges, _ecs_receipt = apply_edge_count_set(
                _network_edges, edge_count_set)
        net_cache[key] = cns.CNS(meta, _network_edges, p)
        net_cache[key].delay_class_scale_receipt = _dcs_receipt
        net_cache[key].delay_row_ms_receipt = _drm_receipt
        if _ecs_receipt is not None:
            _ecs_receipt["edges_kept_by_constructor"] = int(
                len(_network_edges) - net_cache[key].n_edges_dropped)
            net_cache[key].edge_count_set_receipt = _ecs_receipt
    net = net_cache[key]
    # Design fa5adfc8: evaluate's static spike schedule, resolved to rows in
    # the engine's {round(step x dt, 3): rows} form. Empty is absent.
    _forced_ms, _forced_rows, _forced_receipt = None, {}, None
    if force_spikes:
        _fdt = float(net.p.dt)
        _fsteps = int(round(dur_ms / _fdt))
        _id_row = {str(_b): _r for _r, _b in enumerate(net.ids)}
        if len({str(_b) for _b in force_spikes}) != len(force_spikes):
            raise ValueError("force_spikes names one id twice")
        _resolved = []
        for _b, _ks in force_spikes.items():
            if str(_b) not in _id_row:
                raise ValueError(f"force_spikes names {_b}, not a network id")
            _ks = list(_ks)
            if any(isinstance(_k, bool) or not isinstance(_k, (int, np.integer))
                   for _k in _ks):
                raise ValueError(f"force_spikes steps of {_b} must be integers")
            _ks = sorted(int(_k) for _k in _ks)
            if len(set(_ks)) != len(_ks) or any(not 0 <= _k < _fsteps for _k in _ks):
                raise ValueError(f"force_spikes steps of {_b} repeat or leave "
                                 f"[0, {_fsteps})")
            if _ks:
                _forced_rows[_id_row[str(_b)]] = _ks
                _resolved.append({"banc_888_id": str(_b), "row": _id_row[str(_b)],
                                  "steps": _ks,
                                  "times_ms": [round(_k * _fdt, 3) for _k in _ks]})
        if _forced_rows:
            _forced_ms = {}
            for _r, _ks in sorted(_forced_rows.items()):
                for _k in _ks:
                    _forced_ms.setdefault(round(_k * _fdt, 3), []).append(_r)
            _forced_receipt = {
                "schema": "fa5adfc8-force-spikes-v1", "resolved": _resolved,
                "engine_form": {repr(_ms): _rs for _ms, _rs in sorted(_forced_ms.items())},
                "dt_ms": _fdt, "run_steps": _fsteps,
                "first_forced_step": min(min(_ks) for _ks in _forced_rows.values())}
    if _graded_edge_equilibrium_mode:
        _audit = net.graded_edge_equilibrium_audit
        if (graded_edge_equilibrium_plan is None
                or _audit.get("mode") != _graded_edge_equilibrium_mode
                or _audit.get("requested_pair_count") != 6
                or _audit.get("selected_raw_row_count") != 6
                or _audit.get("selected_total_count") != 58
                or sorted(
                    (row["pre_banc_888_id"], row["post_banc_888_id"])
                    for row in _audit.get("pairs", []))
                != sorted(graded_edge_equilibrium_ids)):
            raise ValueError("HS-DNa02 conductance normalization audit drift")
    ro = Readout(net)
    # CD1 (2026-08-20): th["camp_src"] resolves the campaniform double
    # drive. Absent/None = both channels, bit-identical. "load" silences
    # the deprecated activation stand-in (Proprioceptors emits rest rate
    # for those rows); "load_matched" additionally rescales camp_gain by
    # th["camp_match_factor"] -- a REQUIRED, pre-measured number when
    # that mode is set, so the matching is a declaration code can check.
    _camp_src = th.get("camp_src")
    if _camp_src not in (None, "load", "load_matched", "load_rise"):
        raise ValueError(f"th['camp_src'] must be absent, 'load', "
                         f"'load_matched' or 'load_rise', got {_camp_src!r}")
    if _camp_src == "load_matched" and "camp_match_factor" not in th:
        raise ValueError("camp_src='load_matched' requires "
                         "th['camp_match_factor'] (measured, frozen)")
    # Design 6cc949c1, Build B1: "load_rise" is "load_matched"'s channel (the
    # activation stand-in at rest rate, camp_gain x camp_match_factor) under
    # CampRiseLaw; its four constants are validated where the tick is known.
    if _camp_src == "load_rise" and "camp_match_factor" not in th:
        raise ValueError("camp_src='load_rise' requires "
                         "th['camp_match_factor'] (measured, frozen)")
    # CD1-KL: th["hair_dof"] absent/None = BIT-IDENTICAL.
    _hair_dof = th.get("hair_dof")
    if _hair_dof not in (None, "spread"):
        raise ValueError(f"th['hair_dof'] must be absent or 'spread', "
                         f"got {_hair_dof!r}")
    _feco_polarity_mode = th.get("feco_polarity_mode")
    if _feco_polarity_mode not in (None, "banc_type", "banc_type_hook"):
        raise ValueError(
            "th['feco_polarity_mode'] must be absent, 'banc_type' or 'banc_type_hook', "
            f"got {_feco_polarity_mode!r}")
    # R271/R299 (lane E, 2026-09-04), ported by the judge 2026-09-05 from lane
    # D's verbatim port: absent = BIT-IDENTICAL; "exact_type" routes the 71 leg
    # sensory rows whose exact BANC type names a modality other than the one
    # their subclass gave them. Validated here, applied in body.Proprioceptors.
    _feco_modality_mode = th.get("feco_modality_mode")
    if _feco_modality_mode not in (None, "banc_type", "exact_type"):
        raise ValueError(
            "th['feco_modality_mode'] must be absent, 'banc_type' or "
            f"'exact_type', got {_feco_modality_mode!r}")
    _proprioceptor_phasic_mode = th.get("proprioceptor_phasic_mode")
    if _proprioceptor_phasic_mode not in (None, "claw_tonic") and not (
            isinstance(_proprioceptor_phasic_mode, str)
            and _proprioceptor_phasic_mode.startswith("tonic:")):
        raise ValueError(
            "th['proprioceptor_phasic_mode'] must be absent, 'claw_tonic' "
            f"or 'tonic:<pops>' (CD1-SM), got {_proprioceptor_phasic_mode!r}")
    prop = B.Proprioceptors(meta, net, phasic=float(th.get("phasic", 0.0)),
                            claw_polarity=th["claw"],
                            hook_polarity=th["hook"],
                            camp_src=("rest" if _camp_src else "act"),
                            hair_dof=_hair_dof,
                            feco_polarity_mode=_feco_polarity_mode,
                            feco_modality_mode=_feco_modality_mode,
                            proprioceptor_phasic_mode=(
                                _proprioceptor_phasic_mode),
                            # CD1-SK: absent = V_SCALE = bit-identical
                            v_scale_hook=th.get("feco_v_scale_hook"),
                            v_scale_club=th.get("feco_v_scale_club"),
                            # 302c1bfb Build 12: absent = R_REST = bit-identical
                            r_rest_hook=th.get("feco_r_rest_hook"),
                            # 4cbc7232 Build B4: absent = R_REST = bit-identical
                            r_rest_hair=th.get("feco_r_rest_hair"),
                            # CD1-SN: absent = 1.0 = bit-identical
                            claw_gain=th.get("feco_claw_gain"),
                            # CD1-TN: absent = linear = bit-identical
                            claw_code=th.get("feco_claw_code"),
                            claw_flex_frac=th.get("feco_claw_flex_frac"),
                            reflex_map=th.get("feco_reflex_map"),
                            claw_width=th.get("feco_claw_width"),
                            # 2284c922 B1: feco_claw_code "dallmann_u";
                            # absent = bit-identical
                            claw_u_hz=th.get("feco_claw_u_hz"),
                            claw_u_offset_deg=th.get("feco_claw_u_offset_deg"),
                            claw_u_coeffs=th.get("feco_claw_u_coeffs"),
                            claw_u_angle=th.get("feco_claw_u_angle"),
                            # bf165b63 Build 2: absent = bit-identical
                            claw_u_range=th.get("feco_claw_u_range"))
    # measured joint ranges (Haustein 2024) opt-in via th["ranges"]
    body_joint_bias = _resolve_body_joint_bias(th)
    body_joint_tau_act = _resolve_body_joint_tau_act(th)
    body_tau_deact = _resolve_tau_deact(th)
    body_joint_drive_scale = _resolve_body_joint_drive_scale(th)
    body_joint_drive_offset = _resolve_body_joint_drive_offset(th)
    body_joint_f0 = _resolve_body_joint_f0(th)
    # CD2-AV (lane A, 2026-09-05; ported by the judge under the probe): the ROM
    # table is a measured WALKING range used as a joint LIMIT, and the limit is a
    # gain on everything the joint does, so a theta may override it per joint and
    # leg. Absent = bit-identical.
    if th.get("rom_override") is not None and th.get("ranges") != "measured":
        raise ValueError("th['rom_override'] requires th['ranges'] == 'measured'; "
                         "without it the joint limits never come from the ROM table")
    bod = B.Body(tau_joint_ms=th["tau_joint"], drive_scale=th["drive_scale"],
                 tau_act_ms=(B.TAU_ACT_MS if body_joint_tau_act is None
                             else body_joint_tau_act),
                 ranges=(B.measured_ranges(rom_override=th.get("rom_override"))
                         if th.get("ranges") == "measured" else None),
                 f0_hz=(th.get("f0_hz") if body_joint_f0 is None
                        else body_joint_f0),
                 zeta=float(th.get("zeta", 0.8)),
                 exact_filters=(th.get("filt") == "exact"),
                 joint_bias_rad=body_joint_bias,
                 drive_scale_vec=body_joint_drive_scale,
                 joint_integrator=th.get("joint_int", "euler"),
                 drive_offset_vec=body_joint_drive_offset,
                 tau_deact_ms=body_tau_deact)
    # gap.abdomen_actuation step 2 (judge, 2026-09-03): th["abd_gain"] > 0
    # builds the abdomen adapter -- ten actuated abdomen DOFs on the fly,
    # BANC's abdominal motor neurons appended to the feedback rows, a
    # second first-order Body driving them (see abdomen.py). Absent or
    # 0 = not built = bit-identical.
    abd = None
    if float(th.get("abd_gain", 0.0) or 0.0) > 0.0:
        import abdomen as AB
        abd = AB.AbdomenAdapter(th, net)
    # design 3285beba: the air rig's "proboscis" key builds the proboscis
    # adapter -- two actuated pitch joints after the abdomen's ten, BANC's
    # 59 proboscis, pharynx and salivary motor rows after the abdomen's
    # rows (gpu/proboscis/adapter.py). Absent = not built = unchanged.
    prob = None
    if rig is not None and rig.get("proboscis") is not None:
        if abd is None or rig.get("kind") != "air":
            raise ValueError("the proboscis needs the abdomen adapter and the air rig")
        from gpu.proboscis.adapter import ProboscisAdapter
        prob = ProboscisAdapter(rig["proboscis"], net)
    # D 03:11: RH FTi rest 110 vs LH 102 deg. Shift the RH FTi
    # working-range centre. Absent = bit-identical.
    if th.get("rh_fti_mid_offset_deg") is not None:
        import motormap as _MM_RHF
        _off = float(th["rh_fti_mid_offset_deg"])
        if not np.isfinite(_off):
            raise ValueError("rh_fti_mid_offset_deg must be finite")
        _j = _MM_RHF.DOF_INDEX["rh_FTi_pitch"]
        _d = np.radians(_off)
        bod.lo[_j] += _d
        bod.hi[_j] += _d
        bod.mid[_j] += _d
        bod.reset()
    # secondary moment arms (motormap's hypothesised entries, gain 0 at
    # baseline) opt-in via th["g_sec"] > 0 -- wakes the dormant ThC
    # antagonists the battery saw as frozen/asymmetric DOFs
    g_sec = float(th.get("g_sec", 0.0))
    mn_transfer = th.get("mn_transfer")
    mn_transferred = None                             # receipt: cells re-labelled, None when absent
    muscle_gain = th.get("muscle_gain")
    muscle_gain_receipt = None                        # absent = no consumer path
    # CD2-AA (lane A, 2026-09-05; ported by the judge 2026-09-05 21:5x under the
    # probe, once a body stood 12 of 12 with the corrected sign: CD2-AJ). Absent
    # is bit-identical; {} is the full map, not absent.
    thc_map = th.get("thc_map")
    thc_map_moved = None                              # receipt: live entries the map moved, None when absent
    mapping_eff = ro.ro.mapping
    if (g_sec > 0 or use_size or mn_transfer is not None
            or thc_map is not None or muscle_gain is not None):
        import motormap as MM2
        M_raw = ro.ro.M.copy()
        if mn_transfer is not None:
            # CD2-AB (lane A, 2026-09-05; ported by the judge the same day
            # under the probe): the six middle-leg cells onto the muscle both
            # male reconstructions name, before every other motor-matrix
            # step, so the secondary arms and size gains see the transferred
            # rows. Absent = the copy above, bit-identical.
            mapping_eff, M_raw, mn_transferred = _apply_mn_transfer(ro.ro.mapping, ro.ro.mn_ids, mn_transfer)
        if size_pool_map is not None and _size_pool_sha(mapping_eff) != _size_pool_sha(size_pool_map):
            raise ValueError("size_pools: the pools' mapping is not the motor matrix's")
        if thc_map is not None:
            # thorax-coxa muscles onto the hinges of their named actions, after
            # any transfer and before the secondary arms and size gains.
            M_raw, thc_map_moved = _apply_thc_map(M_raw, mapping_eff, ro.ro.mn_ids, thc_map, sign_vector)
        if g_sec > 0:
            hyp = mapping_eff
            hyp = hyp[hyp["hypothesised"] & hyp["dof"].notna()]
            # default-off scope: absent = all hyp arms (bit-identical
            # to the g_sec=1.0 one-shot). A list wakes only those DOFs.
            if th.get("g_sec_dofs"):
                hyp = hyp[hyp["dof"].isin(list(th["g_sec_dofs"]))]
            H = np.zeros_like(ro.ro.M)
            row_of = {i: k for k, i in enumerate(ro.ro.mn_ids)}
            for r_ in hyp.itertuples(index=False):
                if r_.banc_888_id in row_of:
                    H[row_of[r_.banc_888_id], MM2.DOF_INDEX[r_.dof]] += (
                        r_.sign * MM2.SIGN_FLIP[r_.dof])
            M_raw += g_sec * H
        if use_size:
            size_g, _, _ = size_factors(meta, edges, **_sp_kw)
            gvec = np.array([size_g.get(i, 1.0) for i in ro.ro.mn_ids])
            # CD1-QE (lane A, 2026-09-03): searchable size-gain span. The
            # 770x is Azevedo's per-spike twitch span; a first-order
            # linear muscle has no tetanic summation, so the slow MNs a
            # pool recruits first produce nothing at that span (hind
            # levator: gain-1.0 cells at 0 Hz, pool at 70 Hz, activation
            # 0.001). gvec is 770**(-f); this makes it span**(-f) with
            # every 1.0 (singletons, overlays) unchanged. Absent =
            # bit-identical; overlays below still set their pools to 1.0.
            if th.get("size_span") is not None:
                _span = float(th["size_span"])
                if not _span > 1.0:
                    raise ValueError("size_span must exceed 1")
                gvec = gvec ** (np.log(_span) / np.log(770.0))
            # default-off: BANC-named FETi get force-class gain 1.0.
            # Synapse-count rank assigned 0.0013 to the hind-left FETi
            # that fires 34 Hz. Absent = bit-identical size path.
            if th.get("feti_force"):
                gvec = feti_force_overlay(
                    gvec, ro.ro.mn_ids, meta, scope=th.get("feti_scope"))
            if th.get("roll_force"):
                gvec = roll_mn_force_overlay(gvec, ro.ro.mn_ids, meta)
            if th.get("reductor_force"):
                gvec = reductor_force_overlay(gvec, ro.ro.mn_ids, meta)
            if th.get("flexor_force"):
                gvec = flexor_force_overlay(gvec, ro.ro.mn_ids, meta)
            if th.get("seti_force"):
                gvec = seti_force_overlay(gvec, ro.ro.mn_ids, meta)
            if th.get("tita_force"):
                gvec = tita_force_overlay(gvec, ro.ro.mn_ids, meta)
            if th.get("ctr_ext_force"):
                gvec = ctr_ext_force_overlay(gvec, ro.ro.mn_ids, meta)
            if th.get("mn_gains") == "azevedo_label":
                extra = ()
                skip = ()
                mode = th.get("azevedo_rt2_fast")
                if mode == "homology":
                    extra = (azevedo_rt2_fast_homology_id(meta),)
                elif mode == "leave_size":
                    skip = (("right", "T2"),)
                gvec = azevedo_label_overlay(
                    gvec, ro.ro.mn_ids, meta,
                    extra_fast_ids=extra, skip_legs=skip)
            M_raw = M_raw * gvec[:, None]
        M, muscle_gain_receipt = finalize_motor_matrix(
            M_raw, mapping_eff, ro.ro.mn_ids, muscle_gain)
    else:
        M = ro.M / np.maximum(np.abs(ro.M).sum(axis=0, keepdims=True), 1.0)
    # Design fcb48834 Build 7 (gpu, 2026-10-02): musc_class_gain scales each
    # recorded row of the finalised map by its class's gain, the cap's
    # constant-gain twin z = g_class y. It acts after the frozen denominator,
    # as muscle_gain does: a gain inside gvec is renormalised away on every
    # column whose sum of |M_raw| exceeds 1 (finalize_motor_matrix). Classes
    # by lanes/E/run_recruitment_reread_r368.classes_from under the input-
    # synapse rank. Absent or all 1.0 = bit-identical (probe P6).
    musc_class_gain = _resolve_musc_class_gain(th)
    musc_class_gain_receipt = None
    if musc_class_gain is not None:
        if not use_size:
            raise ValueError("musc_class_gain requires mn_gains 'size' or 'azevedo_label'")
        from gpu.bob.forms import unit_classes as _bob_unit_classes
        _cg_cls, _cg_missing = _bob_unit_classes(ro.ro.mn_ids, meta, edges)
        _cg = np.array([musc_class_gain[c] for c in _cg_cls], dtype=float)
        M = M * _cg[:, None]
        musc_class_gain_receipt = {
            "gain": dict(musc_class_gain),
            "rows_by_class": {c: int((_cg_cls == c).sum()) for c in ("fast", "mid", "slow")},
            "rows_unlisted_as_mid": len(_cg_missing),
            "classes_sha256": hashlib.sha256("\n".join(
                f"{i}:{c}" for i, c in zip(ro.ro.mn_ids, _cg_cls)).encode()).hexdigest()}
    # CD1-MK: apply the measured signs to the final map, after secondary
    # moment arms and per-MN gains. Applying them only inside JointReadout
    # would leave r23's hypothesised FTi rows at the old +1 pin.
    M = _apply_motor_sign_vector(M, sign_vector)
    # Study 4b4112bf: split only the registered surviving reductor entries.
    # Preserve the original map's stiffness eligibility separately.
    reductor_ids = th.get("reductor_split_ids")
    reductor_mask = reductor_receipt = None
    if reductor_ids is not None:
        if float(th.get("k_gain", 0.0)) <= 0 or th.get("k_mode") != "load":
            raise ValueError("reductor_split_ids requires wm1's load stiffness path")
        from gpu.fall_controls import split_reductors
        M, reductor_mask, reductor_receipt = split_reductors(M, ro.ro.mn_ids, reductor_ids)
    # channel_pool_map (judge, 2026-09-05, gap.fti_capture_channel_labels):
    # the capture's a_ext and a_flex channels are the POSITIVE and NEGATIVE
    # halves of this signed map, so their names follow the DOF's direction
    # and not the muscle. At the six FTi joints under signs "measured" the
    # sign vector is -1, which puts the tibia EXTENSOR pool in the channel
    # named a_flex and the flexor pool in a_ext; on 2026-09-05 a lane read
    # those names at face value and a knee verdict had to be inverted. This
    # receipt states, per DOF, which muscles actually feed each channel, so
    # no reader has to infer it. Read-only: it touches no array the run uses.
    _cpm = {}
    try:
        MM_DOF_INDEX_FOR_RECEIPT = _dof_index_for_receipt()
        _mp_eff = mapping_eff
        _mus = dict(zip(_mp_eff["banc_888_id"].astype(str), _mp_eff["muscle"].astype(str)))
        _rows = [str(i) for i in ro.ro.mn_ids]
        for _dof, _j in MM_DOF_INDEX_FOR_RECEIPT.items():
            _col = M[:, _j]
            _pos = sorted({_mus.get(_rows[_k], "?") for _k in np.flatnonzero(_col > 0)})
            _neg = sorted({_mus.get(_rows[_k], "?") for _k in np.flatnonzero(_col < 0)})
            if _pos or _neg:
                _cpm[_dof] = {"a_ext": _pos, "a_flex": _neg}
    except Exception as _e:                       # a receipt never fails a run
        _cpm = {"error": str(_e)[:200]}
    # two-channel muscle split (MUSCLE-SPLIT.md): opt-in via th["k_gain"] > 0.
    # Default off; the else-branches below are the original code, untouched.
    use_split = float(th.get("k_gain", 0.0)) > 0.0
    if use_split:
        import muscle_split as MS
        M_flex, M_ext, both_mask = MS.split_map(M)
        if reductor_mask is not None:
            both_mask = reductor_mask
    motor_sequence = _motor_sequence_config(th, use_split)
    measured_step = _measured_step_config(th, use_split, bod)
    measured_step_contact_gain = _measured_step_contact_gain_config(
        th, measured_step)
    measured_step_hs_outer_contact_gain = (
        _measured_step_hs_outer_contact_gain_config(th, measured_step))
    measured_step_hs_conservative_push_pull = (
        _measured_step_hs_conservative_push_pull_config(th, measured_step))
    measured_step_hs_saturation_aware_push_pull = (
        _measured_step_hs_saturation_aware_push_pull_config(
            th, measured_step))
    body_drive_leg_scale = _resolve_body_drive_leg_scale(th)
    if playback is not None and body_drive_leg_scale is not None:
        raise ValueError("body_drive_leg_gains is unavailable in playback mode")
    neural_motor_scale = _resolve_neural_motor_scale(th)
    measured_step_neural_upstream_mix = (
        _measured_step_neural_upstream_mix_config(
            th, measured_step, use_split, neural_motor_scale))
    measured_step_neural_sign_gate = (
        _measured_step_neural_sign_gate_config(
            th, measured_step, use_split, neural_motor_scale,
            measured_step_neural_upstream_mix))
    neural_motor_right_class_dof_substitution = (
        _resolve_neural_motor_right_class_dof_substitution(th))
    # Lane D measured that live motor-neuron drive added on top of the full
    # external gait spares the late arm and the gait and breaks the left turn
    # and the stop. This gates that drive by the decoded visual class, so the
    # neurons can own the joints where they do no harm. Absent = ungated and
    # bit-identical; a list of two gives class 0 and class 1 their own scale,
    # and ticks with no class use the ungated value.
    neural_motor_scale_by_class = th.get("neural_motor_scale_by_class")
    if neural_motor_scale_by_class is not None:
        if (not isinstance(neural_motor_scale_by_class, list)
                or len(neural_motor_scale_by_class) != 2
                or any(type(item) not in (int, float)
                       or not np.isfinite(item)
                       or not 0.0 <= float(item) <= 1.0
                       for item in neural_motor_scale_by_class)):
            raise ValueError(
                "neural_motor_scale_by_class must be two finite scales in "
                "[0, 1], ordered by decoded class")
        neural_motor_scale_by_class = tuple(
            float(item) for item in neural_motor_scale_by_class)
        if playback is not None:
            raise ValueError(
                "neural_motor_scale_by_class is unavailable in playback mode")
    # Row 29 needs front-leg excursion without moving the static window: every
    # dose of the template-level gain that reaches the range loses static,
    # because that gain scales the drive during the static segment too. These
    # gains apply per tick and ONLY while the image is moving, so the static
    # segment keeps the selection's amplitude. Absent = bit-identical.
    moving_leg_gains = th.get("measured_step_leg_gains_moving")
    if moving_leg_gains is not None:
        if (not isinstance(moving_leg_gains, list)
                or len(moving_leg_gains) != 6
                or any(type(item) not in (int, float)
                       or not np.isfinite(item) or float(item) < 0.0
                       for item in moving_leg_gains)):
            raise ValueError(
                "measured_step_leg_gains_moving must be six finite "
                "nonnegative numbers, ordered lf lm lh rf rm rh")
        import motormap as _MM_MG
        _by_leg = {leg: float(val) for leg, val
                   in zip(_MM_MG.LEGS, moving_leg_gains)}
        moving_leg_scale = np.asarray(
            [_by_leg[name.split("_", 1)[0]] for name in _MM_MG.DOF_NAMES],
            dtype=float)
    else:
        moving_leg_scale = None
    moving_gain_ticks = np.zeros(2, dtype=np.int64)
    neural_motor_gate_ticks = np.zeros(3, dtype=np.int64)
    visual_phase_brake_strength_sum = 0.0
    visual_phase_brake_strength_max = 0.0
    visual_phase_strength = 1.0
    if playback is not None and neural_motor_scale != 1.0:
        raise ValueError("neural_motor_scale is unavailable in playback mode")
    if playback is not None and measured_step_neural_upstream_mix is not None:
        raise ValueError("upstream neural mix is unavailable in playback mode")
    if neural_motor_right_class_dof_substitution is not None:
        if playback is not None or measured_step is None or not use_split:
            raise ValueError(
                "neural motor DOF substitution needs split measured-step mode")
        if neural_motor_scale != 1.0 or neural_motor_scale_by_class is not None:
            raise ValueError(
                "neural motor DOF substitution needs full ungated raw readout")
    if playback is not None and measured_step_contact_gain is not None:
        raise ValueError(
            "measured-step contact gain is unavailable in playback mode")
    if (playback is not None
            and measured_step_hs_outer_contact_gain is not None):
        raise ValueError(
            "HS-outer contact gain is unavailable in playback mode")
    if (playback is not None
            and measured_step_hs_conservative_push_pull is not None):
        raise ValueError(
            "conservative push-pull is unavailable in playback mode")
    if (playback is not None
            and measured_step_hs_saturation_aware_push_pull is not None):
        raise ValueError(
            "saturation-aware push-pull is unavailable in playback mode")
    cmd = net.select(cell_type=th["cmd"])
    # C-A14 / C-V97: the descending command carried one scalar rate across
    # every row `cmd` selects, so no setting of cmd_hz could express a turn
    # (the 2026-09-01 measured-structure clause: a quantity that varies per
    # element in the animal gets an element scope). The GATING SIGNAL is
    # `visual_relay_side`, the relay's own contralateral decode of the live
    # HS readout, not the stimulus label: it follows what the HS cells do
    # inside the tick. The named side's command rows get cmd_hz * (1 + g)
    # and the other side's cmd_hz * (1 - g); with no decode both stay at
    # cmd_hz. Default 0.0 is bit-identical.
    vision_cmd_side_gain = th.get("vision_cmd_side_gain", 0.0)
    if (type(vision_cmd_side_gain) not in (int, float)
            or not np.isfinite(vision_cmd_side_gain)
            or not 0.0 <= float(vision_cmd_side_gain) <= 1.0):
        raise ValueError("vision_cmd_side_gain must be finite in [0, 1]")
    vision_cmd_side_gain = float(vision_cmd_side_gain)
    vision_cmd_side_swap = th.get("vision_cmd_side_swap", False)
    if not isinstance(vision_cmd_side_swap, bool):
        raise ValueError("vision_cmd_side_swap must be Boolean")
    # default off and bit-identical: with it False the gain is scaled by
    # exactly 1.0, which is what every run before this seam did
    vision_cmd_side_magnitude = th.get("vision_cmd_side_magnitude", False)
    if not isinstance(vision_cmd_side_magnitude, bool):
        raise ValueError("vision_cmd_side_magnitude must be Boolean")
    if vision_cmd_side_magnitude and vision_cmd_side_gain == 0.0 and (
            th.get("vision_cmd2_side_gain", 0.0) == 0.0):
        raise ValueError(
            "vision_cmd_side_magnitude requires a non-zero side gain")
    cmd_side = None
    if vision_cmd_side_gain != 0.0:
        if not th.get("vision_cmd"):
            raise ValueError(
                "vision_cmd_side_gain requires an active visual relay")
        _cs = meta.iloc[cmd]["side"].astype(str).to_numpy()
        if not np.isin(_cs, ("left", "right")).all():
            raise ValueError(
                "vision_cmd_side_gain requires every command row to carry "
                "a left or right side")
        cmd_side = np.where(_cs == "left", 0, 1).astype(np.int8)
        if not (np.any(cmd_side == 0) and np.any(cmd_side == 1)):
            raise ValueError(
                "vision_cmd_side_gain requires command rows on both sides")
    # second descending command (th["cmd2"], default off = bit-identical):
    # swap1 (2026-08-16) showed command IDENTITY selects the behaviour
    # (DNg100 -> steps, MDN -> stands, same theta). Real descending control
    # is population-level, so co-activation is a hypothesized parameter --
    # a constant rate into a second named DN type, exactly like cmd.
    cmd2_ids = th.get("cmd2_banc_ids")
    if cmd2_ids is not None:
        if th.get("cmd2"):
            raise ValueError("cmd2 and cmd2_banc_ids are mutually exclusive")
        if (not isinstance(cmd2_ids, list) or not cmd2_ids
                or any(not isinstance(i, str) or not i.isascii()
                       or not i.isdecimal() or i != str(int(i))
                       for i in cmd2_ids)
                or len(set(cmd2_ids)) != len(cmd2_ids)):
            raise ValueError("cmd2_banc_ids must be unique exact-decimal strings")
        cmd2_pos = {str(identifier): row
                    for row, identifier in enumerate(net.ids)}
        if any(identifier not in cmd2_pos for identifier in cmd2_ids):
            raise ValueError("cmd2_banc_ids contains an ID absent from this network")
        cmd2 = np.asarray([cmd2_pos[identifier] for identifier in cmd2_ids],
                          dtype=np.int64)
    else:
        cmd2 = (net.select(cell_type=th["cmd2"]) if th.get("cmd2")
                else cmd[:0])
    cmd2_hz = float(th.get("cmd2_hz", 0.0))
    # Design 478fe148, Build B3: a third forced group, delivered exactly as
    # cmd2_banc_ids is and drawn right after cmd2 (`_pack` appends it before
    # anything else, and every call ends its parts with cmd and cmd2).
    # th["cmd3"] names a cell type instead; the two are mutually exclusive.
    # No row may also be a cmd or cmd2 row. Absent = no group = bit-identical.
    cmd3_ids = th.get("cmd3_banc_ids")
    if cmd3_ids is not None:
        if th.get("cmd3"):
            raise ValueError("cmd3 and cmd3_banc_ids are mutually exclusive")
        if (not isinstance(cmd3_ids, list) or not cmd3_ids
                or any(not isinstance(i, str) or not i.isascii()
                       or not i.isdecimal() or i != str(int(i))
                       for i in cmd3_ids)
                or len(set(cmd3_ids)) != len(cmd3_ids)):
            raise ValueError("cmd3_banc_ids must be unique exact-decimal strings")
        cmd3_pos = {str(identifier): row
                    for row, identifier in enumerate(net.ids)}
        if any(identifier not in cmd3_pos for identifier in cmd3_ids):
            raise ValueError("cmd3_banc_ids contains an ID absent from this network")
        cmd3 = np.asarray([cmd3_pos[identifier] for identifier in cmd3_ids],
                          dtype=np.int64)
    else:
        cmd3 = (net.select(cell_type=th["cmd3"]) if th.get("cmd3")
                else cmd[:0])
        if th.get("cmd3") and not len(cmd3):
            raise ValueError("cmd3 names a cell type with no rows")
    cmd3_hz = float(th.get("cmd3_hz", 0.0))
    if len(cmd3):
        if np.intersect1d(cmd3, np.concatenate([cmd, cmd2])).size:
            raise ValueError("cmd3 rows overlap the cmd or cmd2 rows")
        if not np.isfinite(cmd3_hz) or cmd3_hz < 0.0:
            raise ValueError("cmd3_hz must be finite and >= 0")
    elif th.get("cmd3_hz") is not None:
        raise ValueError("cmd3_hz needs cmd3 or cmd3_banc_ids")
    # The same element scope on the SECOND command, so a body can keep its
    # step-dose command ungated and give the turn to a direction command.
    vision_cmd2_side_gain = th.get("vision_cmd2_side_gain", 0.0)
    if (type(vision_cmd2_side_gain) not in (int, float)
            or not np.isfinite(vision_cmd2_side_gain)
            or not 0.0 <= float(vision_cmd2_side_gain) <= 1.0):
        raise ValueError("vision_cmd2_side_gain must be finite in [0, 1]")
    vision_cmd2_side_gain = float(vision_cmd2_side_gain)
    vision_cmd2_side_swap = th.get("vision_cmd2_side_swap", False)
    if not isinstance(vision_cmd2_side_swap, bool):
        raise ValueError("vision_cmd2_side_swap must be Boolean")
    cmd2_side = None
    if vision_cmd2_side_gain != 0.0:
        if not th.get("vision_cmd"):
            raise ValueError(
                "vision_cmd2_side_gain requires an active visual relay")
        if not len(cmd2):
            raise ValueError(
                "vision_cmd2_side_gain requires a second command")
        _c2s = meta.iloc[cmd2]["side"].astype(str).to_numpy()
        if not np.isin(_c2s, ("left", "right")).all():
            raise ValueError(
                "vision_cmd2_side_gain requires every second-command row to "
                "carry a left or right side")
        cmd2_side = np.where(_c2s == "left", 0, 1).astype(np.int8)
        if not (np.any(cmd2_side == 0) and np.any(cmd2_side == 1)):
            raise ValueError(
                "vision_cmd2_side_gain requires second-command rows on both "
                "sides")

    steering_trim = _resolve_steering_trim(th)
    if playback is not None and steering_trim != 0.0:
        raise ValueError("steering_trim is unavailable in playback mode")
    visual_bridge = _resolve_optomotor_bridge(th, meta, net)
    if (measured_step_contact_gain is not None
            and measured_step_contact_gain["gain_by_visual_class"] is not None
            and (visual_bridge is None
                 or visual_bridge["relay_code"] not in
                 _HS_CLASS_RELAY_CODES)):
        raise ValueError(
            "visual-class contact gain requires hs_centroid_v1")
    if (measured_step_hs_outer_contact_gain is not None
            and (visual_bridge is None
                 or visual_bridge["relay_code"] not in
                 _HS_CLASS_RELAY_CODES)):
        raise ValueError(
            "measured_step_hs_outer_contact_gain requires hs_centroid_v1")
    if (measured_step_hs_conservative_push_pull is not None
            and (visual_bridge is None
                 or visual_bridge["relay_code"] not in
                 _HS_CLASS_RELAY_CODES)):
        raise ValueError(
            "measured_step_hs_conservative_push_pull_gain requires "
            "hs_centroid_v1")
    if (measured_step_hs_saturation_aware_push_pull is not None
            and (visual_bridge is None
                 or visual_bridge["relay_code"] not in
                 _HS_CLASS_RELAY_CODES)):
        raise ValueError(
            "measured_step_hs_saturation_aware_push_pull_gain requires "
            "hs_centroid_v1")
    visual_image_motion = None
    visual_flyvis_t4t5_tape = None
    visual_carrier_tape = None
    visual_carrier_receipt = None
    if visual_bridge is not None:
        if visual_bridge["input_source"] == _RENDERED_VISUAL_INPUT:
            visual_image_motion = _new_rendered_motion_state()
        elif (visual_bridge["input_source"]
              == _FLYVIS_HS_TAPE_VISUAL_INPUT):
            visual_image_motion = _new_flyvis_side_tape_state(visual_bridge)
        elif (visual_bridge["input_source"]
              == _FLYVIS_BANC_OPPONENT_T4T5_VISUAL_INPUT):
            visual_flyvis_t4t5_tape = (
                _new_flyvis_banc_t4t5_rate_tape_state(visual_bridge))
        elif (visual_bridge["input_source"]
              == _RENDERED_CARRIER_VISUAL_INPUT):
            _tick_ms = float(th.get("tick_ms", 5.0))
            _ticks = int(round(float(dur_ms) / _tick_ms)) + 1
            visual_carrier_tape = _carrier_drive_tape(
                None, visual_bridge["input_rows"], visual_bridge["carrier_u"],
                visual_bridge["carrier_on"], _ticks, _tick_ms,
                visual_bridge["carrier_spec"])
            visual_carrier_receipt = {
                "cells": int(len(visual_bridge["input_rows"])),
                "on_cells": int(visual_bridge["carrier_on"].sum()),
                "off_cells": int((~visual_bridge["carrier_on"]).sum()),
                "ticks": int(_ticks),
                "max_rate_hz": float(visual_carrier_tape.max()),
                "mean_rate_hz_in_window": float(
                    visual_carrier_tape[visual_carrier_tape > 0].mean())
                if (visual_carrier_tape > 0).any() else 0.0,
                "nonzero_ticks": int(
                    (visual_carrier_tape.sum(axis=1) > 0).sum()),
                "spec": dict(visual_bridge["carrier_spec"]),
            }
    visual_image_rates = None
    visual_image_side = None
    visual_teacher_phase = _visual_teacher_side_phase_config(
        th, measured_step, use_split)
    if visual_bridge is not None:
        occupied = np.concatenate([cmd, cmd2, cmd3])
        if np.intersect1d(visual_bridge["input_rows"], occupied).size:
            raise ValueError("visual input rows overlap another command input")
        if playback is not None and visual_bridge["return_scale"] is not None:
            raise ValueError("vision_return_scale is unavailable in playback mode")
        if (visual_bridge["teacher_steering_gain"] != 0.0
                and measured_step is None):
            raise ValueError(
                "vision_teacher_steering_gain requires an active measured step")
        if (playback is not None
                and visual_bridge["teacher_steering_gain"] != 0.0):
            raise ValueError(
                "vision_teacher_steering_gain is unavailable in playback mode")
        if (visual_teacher_phase is not None
                and visual_bridge["relay_code"] not in
                _HS_CLASS_RELAY_CODES):
            raise ValueError(
                "visual side-phase control requires hs_centroid_v1")
        if playback is not None and visual_teacher_phase is not None:
            raise ValueError(
                "vision_teacher_phase_asymmetry is unavailable in playback mode")
        if th.get("vision_phase_from_detector") and (
                visual_bridge.get("image_motion_hz", 0.0) <= 0.0):
            raise ValueError(
                "vision_phase_from_detector requires rendered panorama motion")
    elif visual_teacher_phase is not None:
        raise ValueError(
            "vision_teacher_phase_asymmetry requires a visual bridge")
    visual_teacher_hs_thc_yaw_phase = (
        _visual_teacher_hs_thc_yaw_phase_config(
            th, measured_step, use_split, visual_teacher_phase,
            visual_bridge))
    if (visual_teacher_hs_thc_yaw_phase is not None
            and playback is not None):
        raise ValueError("HS ThC-yaw phase is unavailable in playback mode")
    side_drive_dofs = None
    if (steering_trim != 0.0
            or (visual_bridge is not None
                and (visual_bridge["return_scale"] is not None
                     or visual_bridge["dna02_measured_step_return_gate"]
                     or visual_bridge["teacher_steering_gain"] != 0.0))):
        import motormap as _MM_VIS
        side_drive_dofs = {
            0: {leg: np.asarray([
                i for i, name in enumerate(_MM_VIS.DOF_NAMES)
                if name.startswith(leg + "_")
            ], dtype=np.int64) for leg in ("lf", "lm", "lh")},
            1: {leg: np.asarray([
                i for i, name in enumerate(_MM_VIS.DOF_NAMES)
                if name.startswith(leg + "_")
            ], dtype=np.int64) for leg in ("rf", "rm", "rh")},
        }
        if any(len(indices) != 7 for side in side_drive_dofs.values()
               for indices in side.values()):
            raise ValueError("side drive map did not resolve seven DOFs per leg")
    visual_return_dofs = side_drive_dofs
    visual_teacher_dofs = side_drive_dofs
    if (side_drive_dofs is not None and visual_bridge is not None
            and visual_bridge["teacher_steering_legs"] == "front_middle"):
        visual_teacher_dofs = {
            side: {leg: dofs for leg, dofs in legs.items()
                   if leg in (("lf", "lm") if side == 0 else ("rf", "rm"))}
            for side, legs in side_drive_dofs.items()
        }
        if any(len(legs) != 2 for legs in visual_teacher_dofs.values()):
            raise ValueError("front-middle steering did not resolve two legs per side")
    camp = camp_by_leg(meta, net)
    # M3 (2026-08-18): hypothesised dense load sensing. th["camp_extra"]
    # = extra sensors per leg (0 = off, bit-identical). th["camp_match"]
    # rescales camp_gain by n_before/n_after so TOTAL campaniform drive
    # is unchanged -- without it, "more sensors" and "more input" are
    # confounded, and this project has already measured that total
    # activity alone changes everything (AGENTS.md tuning curve).
    _camp_gain = float(th["camp_gain"])
    if _camp_src in ("load_matched", "load_rise"):
        # CD1 substitution arm: the load channel inherits the silenced
        # activation channel's mean effective drive via the frozen,
        # step-0-measured factor.
        _camp_gain = _camp_gain * float(th["camp_match_factor"])
    if int(th.get("camp_extra", 0)):
        _extra = camp_extra_by_leg(meta, int(th["camp_extra"]))
        _n0 = sum(len(v) for v in camp.values())
        camp = {leg: np.unique(np.concatenate([camp.get(leg, np.array([], int)),
                                               _extra.get(leg, np.array([], int))]))
                if len(_extra.get(leg, [])) else camp.get(leg, np.array([], int))
                for leg in camp}
        _n1 = sum(len(v) for v in camp.values())
        if th.get("camp_match") and _n1 > 0:
            _camp_gain = _camp_gain * max(_n0, 1) / _n1
    _camp_leg_gains = None
    if th.get("camp_leg_gains") is not None:
        import motormap as _MM_CAMP
        _clg = np.asarray(th["camp_leg_gains"], dtype=float).reshape(-1)
        if _clg.shape != (6,):
            raise ValueError(
                "camp_leg_gains must be six nonnegative numbers")
        if not np.isfinite(_clg).all() or np.any(_clg < 0.0):
            raise ValueError(
                "camp_leg_gains must be finite and nonnegative")
        _camp_leg_gains = {
            leg: float(val) for leg, val in zip(_MM_CAMP.LEGS, _clg)}
    camp_all = np.concatenate([v for v in camp.values() if len(v)])
    # Design 6cc949c1, with Build B2 (note 423b75cd): silence_rows raises a
    # threshold, and a drive event fires its row whatever the threshold, so a
    # row the proprioceptor rates (prop.all_rows, drive_fn's idx) or the load
    # pool (camp_all) feeds would be reported silenced while it fires. Refused
    # as graded rows are; drive_silence is the key for those rows.
    if th.get("silence_rows") is not None:
        _sr_pos = pd.Series(
            np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
        _sr_rows = np.asarray([int(_sr_pos.loc[str(x)])
                               for x in th["silence_rows"]], dtype=int)
        _sr_driven = np.intersect1d(
            _sr_rows, np.concatenate([prop.all_rows, camp_all]))
        if _sr_driven.size:
            raise ValueError(
                f"silence_rows includes {_sr_driven.size} rows in the drive "
                "set, which a threshold raise cannot silence; use drive_silence")
    # Design 983f1c29 Build 5: th["touchdown_pulse"], absent = BIT-IDENTICAL,
    # and hz 0 changes no rate. A list of {leg, ids, ids_sha256, hz, dur_ms,
    # delay_ms, mode, after_ms, holdoff_ms}, one entry per pulsed leg. Contact
    # is design b22d659e's B4 rule read online: the leg's tip z (the capture's
    # element 5, min over the leg's geoms) less its running 5th percentile from
    # 0 ms, into the air above 0.15 mm and onto the ground below 0.05 mm, the
    # state starting on the ground. A touchdown (air to ground at tick k0) is
    # accepted at k0 + 3, after four ticks on the ground, unless it falls
    # within holdoff_ms of the last accepted one. `hz` is added to the listed
    # rows' law rate for dur_ms from delay_ms after the accepted tick: "first"
    # pulses the first accepted touchdown at or after after_ms, "every" each
    # one. The rows must be in the leg's load list, since the pulse rides on
    # that list's rate (the pulse-plus-level form).
    _td_pulse = None
    if th.get("touchdown_pulse") is not None:
        _tdp_raw = th["touchdown_pulse"]
        if isinstance(_tdp_raw, dict):
            _tdp_raw = [_tdp_raw]
        if not isinstance(_tdp_raw, list) or not _tdp_raw:
            raise ValueError("touchdown_pulse must be a dict or a non-empty list of dicts")
        if th.get("camp_load") == "per_segment":
            raise ValueError("touchdown_pulse rides on the leg-level load law; "
                             "camp_load='per_segment' has none")
        if (th.get("prop_tonic_hz") is not None
                and th.get("prop_tonic_scope", "all") in ("all", "camp")):
            raise ValueError("touchdown_pulse does not combine with a tonic "
                             "rate that overwrites the load list's rates")
        _tdp_row = {str(b): r for r, b in enumerate(net.ids)}
        _tdp_keys = {"leg", "ids", "ids_sha256", "hz", "dur_ms", "delay_ms",
                     "mode", "after_ms", "holdoff_ms"}
        _td_pulse = []
        for _e in _tdp_raw:
            if not isinstance(_e, dict) or set(_e) != _tdp_keys:
                raise ValueError(f"touchdown_pulse entries carry exactly {sorted(_tdp_keys)}")
            _leg = str(_e["leg"])
            if _leg not in camp or not len(camp[_leg]):
                raise ValueError(f"touchdown_pulse: leg {_leg!r} has no load rows")
            if any(_d["leg"] == _leg for _d in _td_pulse):
                raise ValueError(f"touchdown_pulse: leg {_leg!r} appears twice")
            _ids = [str(i) for i in _e["ids"]]
            if not _ids or len(set(_ids)) != len(_ids):
                raise ValueError("touchdown_pulse: ids must be unique and non-empty")
            _sha = hashlib.sha256(json.dumps(
                sorted(_ids), separators=(",", ":")).encode()).hexdigest()
            if _sha != str(_e["ids_sha256"]):
                raise ValueError(f"touchdown_pulse: ids hash {_sha} is not "
                                 f"ids_sha256 {_e['ids_sha256']}")
            _missing = [i for i in _ids if i not in _tdp_row]
            if _missing:
                raise ValueError(f"touchdown_pulse: ids not in the network: {_missing[:5]}")
            _rows = np.array([_tdp_row[i] for i in _ids], dtype=np.int64)
            _pos = np.flatnonzero(np.isin(camp[_leg], _rows))
            if len(_pos) != len(_rows):
                raise ValueError(f"touchdown_pulse: {len(_rows) - len(_pos)} of "
                                 f"{len(_rows)} rows are not in {_leg}'s load list")
            _num = {}
            for _k, _lo in (("hz", 0.0), ("dur_ms", 0.0), ("delay_ms", 0.0),
                            ("after_ms", 0.0), ("holdoff_ms", 0.0)):
                _num[_k] = float(_e[_k])
                if not np.isfinite(_num[_k]) or _num[_k] < _lo:
                    raise ValueError(f"touchdown_pulse: {_k} must be finite and >= {_lo:g}")
            if _num["dur_ms"] <= 0.0:
                raise ValueError("touchdown_pulse: dur_ms must be positive")
            if _e["mode"] not in ("first", "every"):
                raise ValueError("touchdown_pulse: mode is 'first' or 'every'")
            _td_pulse.append({"leg": _leg, "ids": _ids, "ids_sha256": _sha,
                              "rows": _rows, "pos": _pos, "mode": _e["mode"], **_num,
                              # trigger state and log
                              "geoms": None, "hist": [], "air": False,
                              "pending": None, "last_acc_k0": None,
                              "raw": [], "accepted": [],
                              "unconfirmed": [], "held_off": [], "pulses": [],
                              "pulse_ticks": 0, "pulse_law_hz": []})
    # Design e9b68973 Build B1: th["bristle_pulse"], absent = BIT-IDENTICAL.
    # {"rows": [ids], "rows_sha256", "pulses": [...]}. `rows` are bristle_neuron
    # rows outside every leg's load list; they join the drive once, right
    # after cmd2 and cmd3 in _pack, at 0 Hz wherever no pulse is active, so
    # arms carrying the same rows share one vector layout and one afferent
    # draw until a pulse fires. A pulse is {"name", "ids", "ids_sha256",
    # "trigger", "schedule"} plus "t_ms" for trigger "time", or "leg", "mode",
    # "holdoff_ms", "after_ms", "until_lift" for trigger "touchdown". Its ids
    # are a subset of rows; its schedule is contiguous [a_ms, b_ms, hz] pieces
    # from the trigger, every edge on a control tick, so the rate is constant
    # per tick. "time" starts the schedule at t_ms. "touchdown" starts it at
    # the tick touchdown_pulse's tip rule accepts a touchdown on `leg` (the
    # same rule, read whatever the load list holds; "first" or "every",
    # holdoff_ms, after_ms), and until_lift ends it at that leg's next air
    # crossing. Overlapping schedules on one row add their rates.
    _br_pulse = None
    if th.get("bristle_pulse") is not None:
        _bpr = th["bristle_pulse"]
        if not isinstance(_bpr, dict) or set(_bpr) != {"rows", "rows_sha256", "pulses"}:
            raise ValueError("bristle_pulse carries exactly rows, rows_sha256 and pulses")
        if th.get("bristle_rest"):
            raise ValueError("bristle_pulse and bristle_rest are exclusive")
        if (th.get("prop_tonic_hz") is not None
                and th.get("prop_tonic_scope", "all") in ("all", "camp")):
            raise ValueError("bristle_pulse does not combine with a tonic "
                             "rate that overwrites the load list's rates")

        def _bp_sha(_ids):
            return hashlib.sha256(json.dumps(
                sorted(_ids), separators=(",", ":")).encode()).hexdigest()

        _bp_tick = float(th.get("tick_ms", 5.0))

        def _bp_ticks(_ms, _what):
            _q = float(_ms) / _bp_tick
            if not np.isfinite(_q) or _q < 0 or abs(_q - round(_q)) > 1e-9:
                raise ValueError(f"bristle_pulse: {_what} {_ms} ms is not on a "
                                 f"{_bp_tick:g} ms control tick")
            return int(round(_q))

        _bp_row = {str(b): r for r, b in enumerate(net.ids)}
        _bp_ids = [str(i) for i in _bpr["rows"]]
        if not _bp_ids or len(set(_bp_ids)) != len(_bp_ids):
            raise ValueError("bristle_pulse: rows must be unique and non-empty")
        if _bp_sha(_bp_ids) != str(_bpr["rows_sha256"]):
            raise ValueError(f"bristle_pulse: rows hash {_bp_sha(_bp_ids)} is not "
                             f"rows_sha256 {_bpr['rows_sha256']}")
        _missing = [i for i in _bp_ids if i not in _bp_row]
        if _missing:
            raise ValueError(f"bristle_pulse: rows not in the network: {_missing[:5]}")
        _bp_rows = np.array([_bp_row[i] for i in _bp_ids], dtype=np.int64)
        _bp_cc = meta["cell_class"].fillna("").astype(str).to_numpy()
        _bad = [i for i, r in zip(_bp_ids, _bp_rows) if _bp_cc[r] != "bristle_neuron"]
        if _bad:
            raise ValueError(f"bristle_pulse: rows that are not bristle_neuron: {_bad[:5]}")
        if np.isin(_bp_rows, camp_all).any():
            raise ValueError("bristle_pulse: rows in a leg's load list")
        _bp_at = {i: k for k, i in enumerate(_bp_ids)}
        _br_pulse = {"ids": _bp_ids, "rows_sha256": _bp_sha(_bp_ids), "rows": _bp_rows,
                     "rate": np.zeros(len(_bp_rows)), "pulses": []}
        for _e in _bpr["pulses"] if isinstance(_bpr["pulses"], list) else [None]:
            _trig = _e.get("trigger") if isinstance(_e, dict) else None
            _keys = {"name", "ids", "ids_sha256", "trigger", "schedule"} | (
                {"t_ms"} if _trig == "time" else
                {"leg", "mode", "holdoff_ms", "after_ms", "until_lift"})
            if _trig not in ("time", "touchdown") or set(_e) != _keys:
                raise ValueError("bristle_pulse: a pulse carries name, ids, ids_sha256, "
                                 "trigger, schedule and t_ms (time) or leg, mode, "
                                 "holdoff_ms, after_ms, until_lift (touchdown)")
            _nm = str(_e["name"])
            if any(_d["name"] == _nm for _d in _br_pulse["pulses"]):
                raise ValueError(f"bristle_pulse: pulse {_nm!r} appears twice")
            _ids = [str(i) for i in _e["ids"]]
            if not _ids or len(set(_ids)) != len(_ids):
                raise ValueError(f"bristle_pulse {_nm}: ids must be unique and non-empty")
            if _bp_sha(_ids) != str(_e["ids_sha256"]):
                raise ValueError(f"bristle_pulse {_nm}: ids hash {_bp_sha(_ids)} is not "
                                 f"ids_sha256 {_e['ids_sha256']}")
            _out = [i for i in _ids if i not in _bp_at]
            if _out:
                raise ValueError(f"bristle_pulse {_nm}: ids not in rows: {_out[:5]}")
            _sch = []
            for _j, _piece in enumerate(_e["schedule"] or []):
                _a, _b, _hz = (float(x) for x in _piece)
                if not np.isfinite(_hz) or _hz < 0:
                    raise ValueError(f"bristle_pulse {_nm}: hz must be finite and >= 0")
                _ka, _kb = _bp_ticks(_a, "edge"), _bp_ticks(_b, "edge")
                if _kb <= _ka or (_sch and _ka != _sch[-1][1]) or (not _sch and _ka != 0):
                    raise ValueError(f"bristle_pulse {_nm}: the schedule is contiguous "
                                     "pieces from 0 ms, each of positive length")
                _sch.append((_ka, _kb, _hz))
            if not _sch:
                raise ValueError(f"bristle_pulse {_nm}: the schedule is empty")
            _d = {"name": _nm, "trigger": _trig, "ids": _ids, "ids_sha256": _bp_sha(_ids),
                  "pos": np.array([_bp_at[i] for i in _ids], dtype=np.int64),
                  "schedule": [list(map(float, p_)) for p_ in _e["schedule"]],
                  "sched": _sch, "len": _sch[-1][1],
                  "hz_of": np.concatenate([np.full(_kb - _ka, _hz) for _ka, _kb, _hz in _sch]),
                  "runs": []}
            if _trig == "time":
                _d["t_ms"] = float(_e["t_ms"])
                _d["start_tick"] = _bp_ticks(_d["t_ms"], "t_ms")
            else:
                _leg = str(_e["leg"])
                if _leg not in camp:
                    raise ValueError(f"bristle_pulse {_nm}: no leg {_leg!r}")
                if _e["mode"] not in ("first", "every"):
                    raise ValueError(f"bristle_pulse {_nm}: mode is 'first' or 'every'")
                if not isinstance(_e["until_lift"], bool):
                    raise ValueError(f"bristle_pulse {_nm}: until_lift is true or false")
                for _k in ("holdoff_ms", "after_ms"):
                    if not np.isfinite(float(_e[_k])) or float(_e[_k]) < 0:
                        raise ValueError(f"bristle_pulse {_nm}: {_k} must be finite and >= 0")
                _d.update({"leg": _leg, "mode": _e["mode"], "until_lift": _e["until_lift"],
                           "holdoff_ms": float(_e["holdoff_ms"]),
                           "after_ms": float(_e["after_ms"]),
                           # the tip rule's state and log, as touchdown_pulse's
                           "geoms": None, "hist": [], "air": False,
                           "pending": None, "last_acc_k0": None, "raw": [],
                           "accepted": [], "unconfirmed": [], "held_off": [],
                           "lifts": []})
            _br_pulse["pulses"].append(_d)
        if not _br_pulse["pulses"]:
            raise ValueError("bristle_pulse: pulses is a non-empty list")
    # hair-plate afferents (th["hp_gain"] > 0, default off): BANC names
    # 233 per-leg hair_plate_neuron cells (88.7k core synapses) that this
    # model never drove. Hair plates are joint-extreme detectors -- the
    # swing-licensing sensor class of the stick-insect coordination rules
    # (the measured R2 gap, CONSTRAINTS.md 2026-08-14). Transduction only:
    # rate = hp_gain * rectified proximal-joint extremeness past
    # hp_thresh, polarity searched (hp_pol) since which end deflects the
    # hairs is unmeasured here. What the signal DOES is the connectome's.
    hp_gain = float(th.get("hp_gain", 0.0))
    if hp_gain > 0:
        import motormap as MM3
        cc_ = meta["cell_class"].astype(str)
        bp_ = meta["body_part_sensory"].astype(str)
        sd_ = meta["side"].astype(str).str.upper().str[:1]
        dof_ix = {n: i for i, n in enumerate(MM3.DOF_NAMES)}
        hp_cells, hp_dofs = {}, {}
        for (part, s_), leg in LEG_KEYS.items():
            m_ = ((cc_ == "hair_plate_neuron") & (bp_ == part)
                  & (sd_ == s_)).to_numpy()
            cells = np.flatnonzero(m_)
            if len(cells):
                hp_cells[leg] = cells
                hp_dofs[leg] = [dof_ix[f"{leg}_ThC_pitch"],
                                dof_ix[f"{leg}_CTr_pitch"]]
        hp_all = np.concatenate(list(hp_cells.values()))
        hp_thresh = float(th.get("hp_thresh", 0.7))
        hp_pol = float(th.get("hp_pol", 1))
    # No-inert bristles (default off = bit-identical). BANC names 3,014
    # leg bristle_neuron cells that this rig never drove. Educated guess:
    # silent in swing, R_REST=5 Hz when that leg's contact load exceeds
    # the existing 1.0 N contact gate. Not a searched gain.
    bristle_rest = bool(th.get("bristle_rest"))
    bristle_rest_hz = th.get("bristle_rest_hz", 5.0)
    if (type(bristle_rest_hz) not in (int, float)
            or not np.isfinite(bristle_rest_hz)
            or not 0.0 <= bristle_rest_hz <= 5.0):
        raise ValueError("bristle_rest_hz must be finite in [0, 5] Hz")
    if "bristle_rest_hz" in th and not bristle_rest:
        raise ValueError("bristle_rest_hz requires bristle_rest=true")
    # R677 proposal: source-confirmed tactile membership for MaleCNS.
    # Absence preserves the original selector and input order exactly.
    bristle_source_mode = th.get("bristle_source_mode")
    if bristle_source_mode not in (None, "native_tactile"):
        raise ValueError("bristle_source_mode must be absent or native_tactile")
    if bristle_source_mode is not None and not bristle_rest:
        raise ValueError("bristle_source_mode requires bristle_rest=true")
    bristle_by_leg = {}
    if bristle_rest:
        from body import NERVE_LEG as _NERVE_LEG
        _cc = meta["cell_class"].fillna("").astype(str)
        _bp = meta["body_part_sensory"].fillna("").astype(str)
        _nv = meta["nerve"].fillna("").astype(str)
        _is = (
            _bp.isin(["front_leg", "middle_leg", "hind_leg"])
            & (_cc == "bristle_neuron")
        )
        if bristle_source_mode == "native_tactile":
            if "mcns_class_raw" not in meta:
                raise ValueError("native_tactile requires MaleCNS source classes")
            _is &= meta["mcns_class_raw"].fillna("").eq("mechanosensory_tactile")
            if not _is.any():
                raise ValueError("native_tactile matched no leg bristle rows")
        for _row in np.flatnonzero(_is.to_numpy()):
            _name = _nv.iat[int(_row)]
            _leg = None
            for (_sd, _seg), _code in _NERVE_LEG.items():
                if _name.startswith(_sd) and _seg in _name:
                    _leg = _code
                    break
            if _leg is None:
                continue
            bristle_by_leg.setdefault(_leg, []).append(int(_row))
        bristle_by_leg = {
            _leg: np.asarray(_rows, dtype=np.int64)
            for _leg, _rows in bristle_by_leg.items()
        }
    # No-inert gustatory (default off = bit-identical). BANC names
    # 772 exact-leg taste_bristle_gustatory_neuron cells this rig
    # never drove. Same frozen contact gate and R_REST as bristles.
    # Allowed only because the female already has BOTH. Not a rate.
    # Design a146ad7f B1: a dict is gust_rest_plan's per-leg rate table.
    gust_rest = bool(th.get("gust_rest"))
    gust_by_leg = {}
    gust_rates_by_leg, gust_load_n, gust_receipt = None, 1.0, None
    if isinstance(th.get("gust_rest"), dict):
        gust_by_leg, gust_rates_by_leg, gust_load_n, gust_receipt = gust_rest_plan(meta, th["gust_rest"])
    elif gust_rest:
        from body import NERVE_LEG as _GNERVE
        _gcc = meta["cell_class"].fillna("").astype(str)
        _gbp = meta["body_part_sensory"].fillna("").astype(str)
        _gnv = meta["nerve"].fillna("").astype(str)
        _gis = (
            _gbp.isin(["front_leg", "middle_leg", "hind_leg"])
            & (_gcc == "taste_bristle_gustatory_neuron")
        )
        for _row in np.flatnonzero(_gis.to_numpy()):
            _name = _gnv.iat[int(_row)]
            _leg = None
            for (_sd, _seg), _code in _GNERVE.items():
                if _name.startswith(_sd) and _seg in _name:
                    _leg = _code
                    break
            if _leg is None:
                continue
            gust_by_leg.setdefault(_leg, []).append(int(_row))
        gust_by_leg = {
            _leg: np.asarray(_rows, dtype=np.int64)
            for _leg, _rows in gust_by_leg.items()
        }
    md_rest = bool(th.get("md_rest"))
    md_by_leg = {}
    if md_rest:
        from body import NERVE_LEG as _MDNERVE
        _mcc = meta["cell_class"].fillna("").astype(str)
        _mbp = meta["body_part_sensory"].fillna("").astype(str)
        _mnv = meta["nerve"].fillna("").astype(str)
        _mis = (
            _mbp.isin(["front_leg", "middle_leg", "hind_leg"])
            & (_mcc == "multidendritic_neuron")
        )
        for _row in np.flatnonzero(_mis.to_numpy()):
            _name = _mnv.iat[int(_row)]
            _leg = None
            for (_sd, _seg), _code in _MDNERVE.items():
                if _name.startswith(_sd) and _seg in _name:
                    _leg = _code
                    break
            if _leg is None:
                continue
            md_by_leg.setdefault(_leg, []).append(int(_row))
        md_by_leg = {
            _leg: np.asarray(_rows, dtype=np.int64)
            for _leg, _rows in md_by_leg.items()
        }
    # No-inert haltere campaniforms (default off = bit-identical).
    # BANC names 337 CS on the organ. attitude1 queued them as the
    # righting channel; this rig has never driven them. Educated
    # guess, not the wingbeat phase code: 5 Hz (same frozen R_REST
    # as the bristle channel) while |body roll| exceeds 16 deg, the
    # low end of the tolerated band attitude1 measured on settled
    # walkers. Not a searched gain or threshold.
    haltere_rest = bool(th.get("haltere_rest"))
    halt_rows = None
    if haltere_rest:
        _hbp = meta["body_part_sensory"].fillna("").astype(str)
        _hcc = meta["cell_class"].fillna("").astype(str)
        halt_rows = np.flatnonzero(
            (_hbp == "haltere") & (_hcc == "campaniform_sensillum_neuron")
        ).astype(np.int64)
    drop_hp_rows, drop_hp_dofs = (dropped_hp_channel(meta)
                                  if th.get("hp_dropped") else (None, None))
    cohp8_rows, cohp8_dofs = (cohp8_channel(meta)
                              if th.get("cohp8") else (None, None))
    # Build 5 of design 2163ba7d: `lag_ticks` feeds the law the angles that many body ticks earlier through a
    # ring (the ring's first reading fills it); a code without the key reads the current angles directly.
    hp_code = hp_row_code_channel(meta, th["hp_row_code"]) if th.get("hp_row_code") else None
    if hp_code is not None:
        hp_code["lag"] = th["hp_row_code"].get("lag_ticks")
        hp_code.update(ring=[], p_now_tick=[], p_tick=[], rate_tick=[])
    cohp8_stats = {
        "ticks": 0,
        "rate_sum_hz": 0.0,
        "above_rest_cell_ticks": 0,
        "max_rate_hz": 0.0,
    }
    thorax_cs_rows = None
    if th.get("thorax_cs"):
        _tbp = meta["body_part_sensory"].fillna("").astype(str)
        _tcc = meta["cell_class"].fillna("").astype(str)
        thorax_cs_rows = np.flatnonzero(
            (_tbp == "thorax") & (_tcc == "campaniform_sensillum_neuron")
        ).astype(np.int64)
    wheeler_rows = None
    if th.get("wheeler_rest"):
        _wbp = meta["body_part_sensory"].fillna("").astype(str)
        _wcc = meta["cell_class"].fillna("").astype(str)
        wheeler_rows = np.flatnonzero(
            (_wbp == "wheelers_organ")
            & (_wcc == "chordotonal_organ_neuron")
        ).astype(np.int64)
    neck_rows = None
    if th.get("neck_rest"):
        _nbp = meta["body_part_sensory"].fillna("").astype(str)
        _ncc = meta["cell_class"].fillna("").astype(str)
        neck_rows = np.flatnonzero(
            (_nbp == "neck") & (_ncc == "chordotonal_organ_neuron")
        ).astype(np.int64)
    abd_left = abd_right = None
    if th.get("abd_rest"):
        abd_left, abd_right = abdomen_bristle_channel(meta)
    thorax_br_rows = None
    if th.get("thorax_br_rest"):
        thorax_br_rows = thorax_bristle_channel(meta)
    orphan_rows = orphan_dofs = None
    if th.get("orphan_club"):
        orphan_rows, orphan_dofs = orphan_club_channel(meta)
    wing_cs_rows = None
    if th.get("wing_cs_rollvel"):
        wing_cs_rows = wing_base_cs_channel(meta)
    # Design 1e8f1a64, Build B1: th["jo_rest"]'s organ rows at fixed resting
    # rates, appended last in _pack. Absent = bit-identical.
    jo_rest_rows = jo_rest_rates = jo_rest_receipt = None
    if th.get("jo_rest") is not None:
        jo_rest_rows, jo_rest_rates, jo_rest_receipt = jo_rest_plan(meta, th["jo_rest"])
    # Design 0bec278a: th["labellar_rest"]'s rows at their resting rates from
    # tick 0, appended after jo_rest in _pack. Absent = bit-identical.
    labellar_rows = labellar_rates = labellar_receipt = labellar_mask = None
    if th.get("labellar_rest") is not None:
        labellar_rows, labellar_rates, labellar_receipt = labellar_rest_plan(
            meta, th["labellar_rest"])
        labellar_mask = np.zeros(len(meta), dtype=bool)
        labellar_mask[labellar_rows] = True
    elif labellar_stimulus is not None:
        raise ValueError("labellar_stimulus needs the theta key labellar_rest")
    # adh_gain/mu ported from lane-a's rig 2026-08-24 (judge): both are
    # ledgered mechanisms (CD1-N adhesion dial, CD1-Q pair_friction) that
    # render_split already honors; a battery instrument that silently
    # drops them mis-measures any theta carrying them (found when r23's
    # adh_gain 31.4 ran as 40 in a judge battery). Absent keys keep the
    # factory values = bit-identical.
    _make_fly = make_locomotion_fly if abd is None else AB.make_fly
    if prob is not None:
        _make_fly = lambda **_kw: AB.make_fly(proboscis=True, **_kw)
    if float(th.get("esc_ttm", 0.0) or 0.0) > 0.0:
        # Enabled TTM owner resolves frozen inv/ CTr hinge names.
        fly = _make_fly(name="inv",
                        add_adhesion=True,
                        adhesion_gain=float(th.get("adh_gain", 40.0)))
    else:
        fly = _make_fly(name=f"s{int(time.time()*10)%100000}",
                        add_adhesion=True,
                        adhesion_gain=float(th.get("adh_gain", 40.0)))
    if rig is None:
        world = FlatGroundWorld()
        world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
        sim = Simulation(world)
    else:
        from gpu.ball import world as _rig_world
        world, sim = _rig_world.build(fly, rig)
    # the root accessor (design 554a54fc): on the floor the free joint's own
    # arrays, sliced at each read exactly as before; on a tether the root
    # mocap pose, zero velocity, acceleration and generalized force. The
    # floor kind (design 20aedfac) has the free root, so it reads the floor's.
    if rig is None or rig.get("kind") == "floor" or rig.get("kind") == "arena":
        _rq = lambda: sim.mj_data.qpos
        _rv = lambda: sim.mj_data.qvel
        _ra = lambda: sim.mj_data.qacc
        _rf = lambda name: getattr(sim.mj_data, name)
    else:
        _rz6 = np.zeros(6)
        _rq = lambda: sim.root_qpos7()
        _rv = _ra = lambda: _rz6
        _rf = lambda name: _rz6
    sim.reset()
    if th.get("mu") is not None:
        _mu = float(th["mu"])
        _mdl = sim.mj_model
        for _p in range(_mdl.npair):
            _n1 = (_mdl.geom(int(_mdl.pair_geom1[_p])).name or "").lower()
            _n2 = (_mdl.geom(int(_mdl.pair_geom2[_p])).name or "").lower()
            if "tarsus" in _n1 or "tarsus" in _n2:
                _mdl.pair_friction[_p, 0] = _mu
                _mdl.pair_friction[_p, 1] = _mu
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=(bod.theta if abd is None else abd.joint_angles(bod.theta)
                      if prob is None else prob.joint_angles(abd.joint_angles(bod.theta))),
        adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()
    body_geoms = (body_contact_geoms(sim.mj_model)
                  if thorax_br_rows is not None else None)
    # cocon_scale: the pilot measured k saturated (mean 97 of [45,105]) --
    # tanh(cocon/drive_scale) pins near 1 during activity, so modulation needs
    # its own scale. Searched, per the parameter rule.
    stiff = (MS.StiffnessController(sim, fly,
                                    (both_mask if abd is None
                                     else abd.pad_mask(both_mask) if prob is None
                                     else prob.pad_mask(abd.pad_mask(both_mask))),
                                    th.get("k_min", 45.0), th["k_gain"],
                                    th["drive_scale"]
                                    * th.get("cocon_scale", 1.0))
             if use_split else None)
    # k_mode "load" = load-closed per-leg stiffness (round 5); default "cocon"
    k_load = use_split and th.get("k_mode", "cocon") == "load"
    if k_load:
        import motormap as MM
        stiff.set_leg_map(MM.DOF_NAMES, th.get("F_ref_k", 35.0))
    # per-leg adhesion release (th["adh"]="flex", default off): the model
    # has always glued all six tarsi down every tick, and the lift probe
    # measured the cost -- mid/hind feet NEVER leave the ground (max
    # clearance 0.01-0.02 mm) while sweeping up to 63 deg of joint range.
    # In the fly, releasing the substrate is an ACTIVE motor act
    # (retractor unguis disengages claws/pulvilli before swing), so the
    # gate consumes the leg's own flexor-channel activation: adhesion off
    # while mean a_flex over the leg's DOFs exceeds adh_thresh *
    # drive_scale. Which leg releases when stays the network's decision.
    # pitch_only: default-off roll/yaw drive lesion. Computed once.
    po_idx = pitch_only_idx() if th.get("pitch_only") else None
    adh_flex = th.get("adh") == "flex"
    adh_release_ratio = float(th.get("adh_release_ratio", 1.0))
    adh_dwell_ms = float(th.get("adh_dwell_ms", 0.0))
    adh_tau_ms = float(th.get("adh_tau_ms", 0.0))
    adh_state = None
    if (adh_release_ratio, adh_dwell_ms, adh_tau_ms) != (1.0, 0.0, 0.0):
        if not adh_flex or th.get("window") or measured_step is not None:
            raise ValueError("fall-study adhesion controls require the unoverridden flex gate")
        from gpu.fall_controls import AdhesionState
        adh_state = AdhesionState(release_ratio=adh_release_ratio,
                                  dwell_ms=adh_dwell_ms, tau_ms=adh_tau_ms)
    # th["adh"]="ctr_flex" (lane C, C-N389, default off): the same gate reading
    # the LEG LIFTER'S own channel instead of the mean over the leg's seven
    # DOFs.  Measured on ten standing captures of the champion base against the
    # judge's own stance episodes: the mean over seven carries no stance
    # information at all (AUC 0.5014, gripping 0.2627 of stance and 0.2637 of
    # swing, C-N387), while the CTr_pitch a_flex channel reads 0.615 to 0.779
    # on the four legs that step and separates them at Youden J 0.29 against
    # 0.03 -- gripping 0.72 of stance where the shipped gate grips 0.23.  That
    # channel is fed by the trochanter flexor, the muscle that lifts the leg,
    # so this is the gate's own docstring above taken literally: the foot lets
    # go when the lifter fires, which is the active motor act the animal makes
    # before swing.  lm and rm invert (0.415, 0.475) and are the two legs
    # already recorded as never commanded a knee sweep (C-N377).
    adh_ctr_flex = th.get("adh") == "ctr_flex"
    if adh_ctr_flex and not use_split:
        raise ValueError(
            "th['adh']='ctr_flex' requires the split muscle path: the\n"
            "gate reads bod.a_flex at one DOF, which only the split path writes")
    adh_motor_balance = _resolve_motor_balance_adhesion(th, use_split)
    adh_leg_threshold_scales = _resolve_adhesion_leg_threshold_scales(th)
    if adh_leg_threshold_scales is not None and not (adh_flex or adh_ctr_flex):
        raise ValueError(
            "adh_leg_threshold_scales requires th['adh'] in ('flex', 'ctr_flex')")
    adh_knee_legs = _resolve_adhesion_knee_legs(th)
    if adh_knee_legs is not None and not adh_flex:
        raise ValueError("adh_knee_legs requires th['adh']='flex'")
    if adh_knee_legs is not None and adh_state is not None:
        raise ValueError("fall-study adhesion controls do not include the knee override")
    if adh_flex or adh_ctr_flex or adh_motor_balance:
        import motormap as MM5
        adh_legs = ["lf", "lm", "lh", "rf", "rm", "rh"]  # get_legs_order()
        adh_dofs = [np.array([i for i, nm in enumerate(MM5.DOF_NAMES)
                              if nm.startswith(leg_ + "_")])
                    for leg_ in adh_legs]
    if adh_knee_legs is not None:
        adh_knee_dof = [int(d[[MM5.DOF_NAMES[i].endswith("_FTi_pitch")
                               for i in d]][0]) for d in adh_dofs]
    if adh_ctr_flex:
        adh_ctr_dof = [int(d[[MM5.DOF_NAMES[i].endswith("_CTr_pitch")
                              for i in d]][0]) for d in adh_dofs]
    if adh_flex or adh_ctr_flex:
        adh_lvl = float(th.get("adh_thresh", 1.0)) * th["drive_scale"]
        adh_lvls = (np.full(6, adh_lvl, dtype=float)
                    if adh_leg_threshold_scales is None else
                    adh_lvl * adh_leg_threshold_scales)
    # Design fcb48834 Build 3 (gpu, 2026-10-02): in playback the flex gate may
    # read a recorded (T, 42) a_flex series in place of bod.a_flex, so a notch
    # on the program does not reach the gate (critique ff9c49cf). Row k is read
    # at the callback of t = k tick, the row the capture appended there.
    # None = bod.a_flex, bit-identical.
    if playback_gate_a_flex is not None:
        if playback is None or not adh_flex or adh_knee_legs is not None:
            raise ValueError("playback_gate_a_flex needs playback and the plain flex gate")
        playback_gate_a_flex = np.asarray(playback_gate_a_flex, dtype=float)
        if (playback_gate_a_flex.shape != (len(playback), bod.n)
                or not np.isfinite(playback_gate_a_flex).all()):
            raise ValueError("playback_gate_a_flex must be a finite (len(playback), 42) array")
    prev_loads = {}
    g2l = leg_of_geom(sim.mj_model)
    # CD1-KH: absent/None -> _g2s unused and nothing changes.
    _camp_load = th.get("camp_load")
    if _camp_load not in (None, "per_segment"):
        raise ValueError(f"th['camp_load'] must be absent or "
                         f"'per_segment', got {_camp_load!r}")
    _g2s = seg_of_geom(sim.mj_model) if _camp_load == "per_segment" else None
    # Design 304e8cbe (2026-09-19): th["camp_adapt"] absent or 0.0 = the
    # original expression = BIT-IDENTICAL. Non-zero adds a rectified
    # UNLOADING component: a first-order low pass x of the same clipped
    # load u, and the term fires only while x exceeds u, which is what a
    # force DECREASE looks like. Zill 2022 (10.1152/jn.00285.2022) finds
    # distal campaniform receptors that encode force decrease and fire
    # maximally at complete unloading, in stick insect and cockroach; the
    # expression here is an ASSUMED prototype and inherits no coefficient
    # from it, since those authors state their own model's parameters are
    # undetermined for their data. The level term u sits OUTSIDE the
    # rectifier, so no gain collapses the rate to a function of x alone.
    _camp_adapt = float(th.get("camp_adapt", 0.0) or 0.0)
    _camp_adapt_tau_ms = th.get("camp_adapt_tau_ms")
    _camp_adapt_k = None
    if _camp_adapt:
        if _camp_load == "per_segment":
            raise ValueError(
                "th['camp_adapt'] cannot be used with camp_load='per_segment': "
                "that branch carries three contact forces per leg and the "
                "unloading filter is defined on one load per leg")
        if _camp_adapt_tau_ms is None or float(_camp_adapt_tau_ms) <= 0.0:
            raise ValueError(
                "th['camp_adapt_tau_ms'] is required and must be positive "
                f"when camp_adapt is non-zero, got {_camp_adapt_tau_ms!r}")
        _camp_adapt_tau_ms = float(_camp_adapt_tau_ms)
        # k is set from the tick below, where the tick itself is resolved.
    # the filter state, one scalar per leg, in the units of u. A NEW dict:
    # prev_loads holds the CURRENT loads at the injection site.
    _camp_x = {}
    # capture-only, elements 26 to 28, legs_sorted order. The capture row is
    # appended BEFORE the injection site runs, and `loads` is final at the
    # append (walk_search.py, `loads = contact_loads(...)` above it), so the
    # three arrays are appended EMPTY and filled in place by the injection a
    # few lines later, within the same tick. Writing a buffer at the injection
    # and copying it at the next append would put tick t-1's rate in the row
    # that carries tick t's load, a 2.5 ms skew a reader cannot see.
    _camp_row = None
    # foot-tip height instrumentation, capture-only (2026-08-14: Robin's
    # video catch -- load oscillation is NOT lift; only tip z can tell a
    # stepping foot from one bending in place under always-on adhesion)
    tip_geoms = None
    if capture is not None:
        tip_geoms = {}
        for g_, leg_ in g2l.items():
            tip_geoms.setdefault(leg_, []).append(g_)
    # posture instrumentation, capture-only (2026-08-18: Robin's video
    # catch -- the standing gate is blind to WHICH segments carry the
    # fly; coxa/femur/tibia and the body all have their own ground
    # pairs, so a fly resting on its shins at altitude passes). The
    # battery scores "stands on its feet" from this flag.
    nonfoot_geoms = None
    if capture is not None:
        import mujoco as _mj
        nonfoot_geoms = set()
        for g_ in range(sim.mj_model.ngeom):
            nm = (_mj.mj_id2name(sim.mj_model, _mj.mjtObj.mjOBJ_GEOM, g_)
                  or "").lower()
            if "tarsus" in nm:
                continue
            if any(s_ in nm for s_ in ("coxa", "trochanterfemur", "tibia",
                                       "thorax", "abdomen", "head")):
                nonfoot_geoms.add(g_)
    # control tick (th["tick_ms"], default 5.0 = the historical value).
    # cscale keeps drive magnitude tick-invariant: the network hands
    # drive_fn spike COUNTS PER WINDOW, so a shorter window would halve
    # the drive and silently redefine drive_scale.
    tick = float(th.get("tick_ms", 5.0))
    if _camp_adapt:
        # b = 1: the exact zero-order-hold solution of tau dx/dt = u - x at
        # a fixed tick. ASSUMED; the source's power-law exponent lives in a
        # different paper (10.1088/1748-3190/ac1ced) and is not reproduced.
        _camp_adapt_k = float(np.exp(-tick / _camp_adapt_tau_ms))
    cscale = 5.0 / tick
    # Design 6cc949c1, Build B1: the lagged law's constants and one
    # CampRiseLaw per leg, its input the level law's (the net load under
    # camp_load_net_adhesion, the gross load otherwise). Absent = no law.
    _camp_rise = None
    if _camp_src == "load_rise":
        if _camp_load == "per_segment" or _camp_adapt:
            raise ValueError("camp_src='load_rise' reads one load per leg "
                             "and carries no unloading filter")
        _crk = {}
        for _key in ("camp_rise_gain", "camp_rise_latency_ms",
                     "camp_fall_floor", "camp_fall_gate", "F_ref"):
            _v = th.get(_key)
            if (isinstance(_v, bool) or not isinstance(_v, (int, float))
                    or not np.isfinite(_v)):
                raise ValueError(f"camp_src='load_rise' requires th[{_key!r}] "
                                 f"as a finite number, got {_v!r}")
            _crk[_key] = float(_v)
        if _crk["camp_rise_gain"] < 0.0 or _crk["camp_fall_gate"] < 0.0:
            raise ValueError("camp_rise_gain and camp_fall_gate must be >= 0")
        if not 0.0 <= _crk["camp_fall_floor"] <= 1.0:
            raise ValueError("camp_fall_floor must lie in [0, 1]")
        if not _crk["F_ref"] > 0.0:
            raise ValueError("camp_src='load_rise' requires F_ref > 0")
        _cr_lag = int(round(_crk["camp_rise_latency_ms"] / tick))
        if _cr_lag < 1 or abs(_cr_lag * tick - _crk["camp_rise_latency_ms"]) > 1e-9:
            raise ValueError("camp_rise_latency_ms must be a whole number of "
                             f"ticks of {tick} ms, at least one")

        def _cr_s(leg):
            return 1.0 if _camp_leg_gains is None else _camp_leg_gains[leg]
        _camp_rise = {
            "lag_ticks": _cr_lag, "constants": _crk,
            "law": {leg: CampRiseLaw(_cr_lag, _camp_gain, _crk["F_ref"],
                                     _crk["camp_rise_gain"],
                                     _crk["camp_fall_floor"],
                                     _crk["camp_fall_gate"], tick, _cr_s(leg))
                    for leg in camp},
            "gated": {leg: 0 for leg in camp}, "rising": {leg: 0 for leg in camp},
            "rate_sum_hz": {leg: 0.0 for leg in camp}, "ticks": 0}
    # Design 6cc949c1, Build B2: th["drive_silence"] = {"ids": [...],
    # "window_ms": [t_on, t_off]} sets every drive position of the named rows,
    # in the proprioceptor rates (idx) and in the load pool (camp_all), to 0 Hz
    # on the drive ticks t_on <= t < t_off, after prop_tonic_scope and the
    # touchdown pulse; the rows and the window also go to the engine's silence
    # window (cns._silence_window_steps), so a synaptic spike is stopped too.
    # Zeroing a rate keeps every random draw. Absent = BIT-IDENTICAL.
    _drive_silence = None
    if th.get("drive_silence") is not None:
        _ds = th["drive_silence"]
        if not isinstance(_ds, dict) or set(_ds) != {"ids", "window_ms"}:
            raise ValueError('drive_silence is {"ids": [...], "window_ms": '
                             '[t_on, t_off]}')
        if th.get("silence_rows_window_ms") is not None:
            raise ValueError("drive_silence and silence_rows_window_ms both "
                             "set the engine's one silence window")
        _ds_ids = [str(x) for x in _ds["ids"]]
        if not _ds_ids or len(set(_ds_ids)) != len(_ds_ids):
            raise ValueError("drive_silence ids must be a non-empty list of "
                             "distinct ids")
        _ds_pos = pd.Series(
            np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
        _ds_missing = [x for x in _ds_ids if x not in _ds_pos.index]
        if _ds_missing:
            raise ValueError(f"drive_silence names {len(_ds_missing)} ids "
                             f"absent from meta, first {_ds_missing[0]!r}")
        _ds_rows = np.asarray([int(_ds_pos.loc[x]) for x in _ds_ids],
                              dtype=np.int64)
        _ds_w = _ds["window_ms"]
        if (not isinstance(_ds_w, (list, tuple)) or len(_ds_w) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not np.isfinite(x) for x in _ds_w)):
            raise ValueError("drive_silence window_ms is [t_on_ms, t_off_ms]")
        _ds_k = [int(round(float(x) / tick)) for x in _ds_w]
        if (any(abs(k * tick - float(x)) > 1e-9 for k, x in zip(_ds_k, _ds_w))
                or not 0 <= _ds_k[0] < _ds_k[1]):
            raise ValueError("drive_silence window_ms edges must be whole drive "
                             f"ticks of {tick} ms with 0 <= t_on < t_off")
        _ds_in_r = np.array([int(np.count_nonzero(prop.all_rows == x))
                             for x in _ds_rows])
        _ds_in_c = np.array([int(np.count_nonzero(camp_all == x))
                             for x in _ds_rows])
        if np.any(_ds_in_r + _ds_in_c == 0):
            raise ValueError(f"drive_silence names {int(np.sum(_ds_in_r + _ds_in_c == 0))} "
                             "rows with no drive position")
        _drive_silence = {
            "ids": _ds_ids, "rows": _ds_rows,
            "on_ms": float(_ds_w[0]), "off_ms": float(_ds_w[1]),
            "k_on": _ds_k[0], "k_off": _ds_k[1],
            "r_pos": np.flatnonzero(np.isin(prop.all_rows, _ds_rows)),
            "c_pos": np.flatnonzero(np.isin(camp_all, _ds_rows)),
            "positions_per_row": {"r": _ds_in_r.tolist(), "cr": _ds_in_c.tolist()},
            "ticks": 0, "removed_hz": {"r": 0.0, "cr": 0.0}}
    labellar_stim = None
    if labellar_stimulus is not None:
        labellar_stim = labellar_stimulus_plan(
            meta, labellar_rows, labellar_stimulus, tick, dur_ms, force_spikes,
            (run_kwargs or {}).get("force_spikes"))
    tarsal_stim = None
    if tarsal_stimulus is not None:
        if not gust_by_leg:
            raise ValueError("tarsal_stimulus needs the theta key gust_rest")
        if not isinstance(tarsal_stimulus, list) or not tarsal_stimulus:
            raise ValueError("tarsal_stimulus must be a non-empty list of schedules")
        _tar_rows = np.concatenate(list(gust_by_leg.values()))
        tarsal_stim = [labellar_stimulus_plan(
            meta, _tar_rows, _s, tick, dur_ms, force_spikes,
            (run_kwargs or {}).get("force_spikes"), what="tarsal_stimulus")
            for _s in tarsal_stimulus]
        _tar_pos = np.concatenate([_p["pos"] for _p in tarsal_stim])
        if len(np.unique(_tar_pos)) != len(_tar_pos):
            raise ValueError("tarsal_stimulus schedules name the same row twice")
    _ttm_bridge = None
    if float(th.get("esc_ttm", 0.0) or 0.0) > 0.0:
        import importlib.util as _ilu
        import sys as _sys
        _ttm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "lanes", "B", "escape_ttm_evaluate_bridge.py")
        _ttm_spec = _ilu.spec_from_file_location(
            "escape_ttm_evaluate_bridge", _ttm_path)
        _ttm_mod = _ilu.module_from_spec(_ttm_spec)
        _sys.modules["escape_ttm_evaluate_bridge"] = _ttm_mod
        _ttm_spec.loader.exec_module(_ttm_mod)
        _ttm_bridge = _ttm_mod.attach_ttm_evaluate_bridge(
            th, net, ro, sim, tick)
    # CD1-AA (2026-08-21): ACTUATOR FORCE and ACTUAL joint angle,
    # capture-only, APPENDED as elements 11 and 12 -- never inserted, so
    # every existing positional index (c[6]-c[10] in attitude1/cpg_gap/
    # layer1c/step1/regression_walk) is untouched. The 42 position
    # servos carry kv = 0 and a hard forcerange of +-65 (measured,
    # lanes/A/2026-08-22-cd1z-readout.txt); if they saturate, the leg
    # stops tracking the commanded angle c[3] and the body decouples
    # from the command. Readout only: nothing here feeds back into the
    # simulation, and with capture=None not one line of it runs.
    pos_act, pos_qadr = [], []
    # 2026-09-03 (lane D's 09:25 ask, judge-owned): the most posterior
    # abdominal segment's world position, capture element 18, APPENDED.
    # flyscore's posture_variability reads the abdomen-thorax vector and
    # was the one scale-free metric never measured on a connectome body
    # because no capture carried the abdomen. Index into flygym's own
    # body-segment order; readout only, capture-only.
    _body_position_selector = None
    _abd_row = None
    if capture is not None:
        _body_position_selector = CompiledBodyPositionSelector.from_simulation(
            sim, fly)
        _abd_rows = [
            i_ for i_, s_ in enumerate(_body_position_selector.body_names)
            if "abdomen6" in s_
        ]
        _abd_row = _abd_rows[0] if len(_abd_rows) == 1 else None
    if capture is not None:
        import mujoco as _mj3
        for _u in range(sim.mj_model.nu):
            _nm = (_mj3.mj_id2name(sim.mj_model,
                                   _mj3.mjtObj.mjOBJ_ACTUATOR, _u) or "")
            if ("adhesion" in _nm.lower() or "abdomen" in _nm.lower()
                    or (prob is not None and "rostrum" in _nm.lower())):
                continue
            _j = int(sim.mj_model.actuator_trnid[_u, 0])
            if _j < 0:
                continue
            pos_act.append(_u)
            pos_qadr.append(int(sim.mj_model.jnt_qposadr[_j]))

    per = int(round(tick / 1000.0 / sim.timestep))
    # Design fa5adfc8: the twitch path and the jump capture on one physics
    # step counter. With neither on, _ttm_hooks is None and nothing runs.
    _ttm_hooks = None
    if _twitch_cfg is not None or _forced_rows:
        from gpu.escape.ttm_jump import path as _ttm_path
        _hsteps = int(round(dur_ms / float(net.p.dt)))
        if (per != 25 or abs(float(net.p.dt) - 0.1) > 1e-12
                or abs(float(sim.timestep) * 1000.0 - 0.1) > 1e-12
                or _hsteps % per):
            raise ValueError("the twitch path and jump capture need 0.1 ms "
                             "CNS and physics steps, 25 per window, whole windows")
        _id_row_t = {str(_b): _r for _r, _b in enumerate(net.ids)}
        _ttm_twitch_path = (
            _ttm_path.TwitchPath(_twitch_cfg, sim, fly.name,
                                 {_s: _id_row_t[_b] for _b, _s in _ttm_path.TTMN.items()},
                                 _forced_rows, _hsteps, per)
            if _twitch_cfg is not None else None)
        _ttm_hooks = _ttm_path.Hooks(
            _ttm_twitch_path,
            _ttm_path.JumpCapture(sim, fly.name, _forced_receipt["first_forced_step"],
                                  _hsteps, _ttm_twitch_path)
            # the air world has no floor plane at geom 0, which the jump
            # capture needs, so no air run records one (design 3285beba's
            # proboscis runs, and design d27f82df's loom trains in air)
            if _forced_rows and (rig is None or rig.get("kind") != "air")
            else None, per, _hsteps)
    xy0, xy = None, None
    prev_roll = None
    quats, zs = [], []
    tilts = []  # v3.2 posture angle per tick (re-freeze event 2026-08-31)
    legs_sorted = sorted(LEG_KEYS.values())
    # Study b309bcde: an empty window leaves the update path unchanged.
    # The recorder appends read-only columns; existing positions stay fixed.
    from gpu.window_hooks import WindowHook, WindowRecorder
    window_spec = th.get("window")
    window_hook = (WindowHook(window_spec, tick=tick, theta=th, body=bod,
                              simulation=sim, pool_map=_cpm, split=use_split,
                              playback=playback) if window_spec else None)
    # Design fcb48834 Build 1 (gpu, 2026-10-02): the per-unit activation path.
    # One low-pass state per (motor row, DOF) pair with the class-specific form
    # of gpu/bob/forms.py, pooled into a_flex and a_ext and written through
    # Body.step2(a_in=). It replaces the pooled low-pass only, so a modifier
    # that acts on the pooled drive before that low-pass has no per-unit form
    # and is refused. Absent = bit-identical; probe P1 reads the linear form.
    musc_unit = None
    _mu_keys = [k for k in ("musc_unit_form", "musc_unit_law_n0", "musc_unit_facil")
                if th.get(k) is not None]
    if not th.get("musc_unit_path"):
        if _mu_keys:
            raise ValueError(f"{_mu_keys} need musc_unit_path true")
    else:
        _mu_refuse = [name for name, on in (
            ("no two-channel split (k_gain > 0)", not use_split),
            ("playback", playback is not None),
            ("neural_motor_scale != 1", neural_motor_scale != 1.0),
            ("neural_motor_scale_by_class", neural_motor_scale_by_class is not None),
            ("body_drive_leg_gains", body_drive_leg_scale is not None),
            ("pitch_only", bool(th.get("pitch_only"))),
            ("measured_step", measured_step is not None),
            ("motor_sequence", motor_sequence is not None),
            ("steering_trim != 0", steering_trim != 0.0),
            ("a vision gate", visual_bridge is not None),
            ("window", window_hook is not None),
            ("tau_deact_ms (a_in skips the body's low-pass)",
             bod.tau_deact is not None)) if on]
        if _mu_refuse:
            raise ValueError("musc_unit_path refuses: " + ", ".join(_mu_refuse))
        from gpu.bob import forms as _bob_forms
        _mu_form = th.get("musc_unit_form")
        if not isinstance(_mu_form, dict):
            raise ValueError("musc_unit_path needs musc_unit_form {fast, mid, slow}")
        if th.get("musc_unit_law_n0") is not None and "law" not in _mu_form.values():
            raise ValueError("musc_unit_law_n0 is set but no class uses the law form")
        if th.get("musc_unit_facil") is not None and "facil" not in _mu_form.values():
            raise ValueError("musc_unit_facil is set but no class uses the facil form")
        _mu_cls, _mu_missing = _bob_forms.unit_classes(ro.ro.mn_ids, meta, edges)
        musc_unit = _bob_forms.UnitPath(
            M_ext - M_flex, bod._alpha(bod.tau_act, tick), cscale, _mu_cls, _mu_form,
            law_n0=float(th.get("musc_unit_law_n0", _bob_forms.LAW_N0)),
            facil=th.get("musc_unit_facil"), tick_ms=tick,
            unlisted=len(_mu_missing))
    # Design 0beeec74 Build 1 (gpu-7, 2026-10-10): the twitch kernel. Each motor
    # class's pooled drive per DOF side passes through n equal first-order stages
    # (gpu/musc_kernel.py) in place of the body's own low-pass, written through
    # Body.step2(a_in=) as the per-unit path is, so it refuses that path's list and
    # the path itself. Absent = no object, bit-identical; {} = every channel at
    # n = 1 on the DOF's tau, the pooled low-pass to float association (probe P1).
    musc_kernel = None
    if th.get("musc_kernel") is not None:
        import motormap as _MK_MM
        from gpu import musc_kernel as _musc_kernel
        _mk_spec = _musc_kernel.resolve(th, _MK_MM.DOF_INDEX)
        _mk_refuse = [name for name, on in (
            ("no two-channel split (k_gain > 0)", not use_split),
            ("musc_unit_path", musc_unit is not None),
            ("filt not exact (the recursion is the exact zero-order hold)",
             not bod.exact),
            ("playback", playback is not None),
            ("neural_motor_scale != 1", neural_motor_scale != 1.0),
            ("neural_motor_scale_by_class", neural_motor_scale_by_class is not None),
            ("body_drive_leg_gains", body_drive_leg_scale is not None),
            ("pitch_only", bool(th.get("pitch_only"))),
            ("measured_step", measured_step is not None),
            ("motor_sequence", motor_sequence is not None),
            ("steering_trim != 0", steering_trim != 0.0),
            ("a vision gate", visual_bridge is not None),
            ("window", window_hook is not None),
            ("tau_deact_ms (a_in skips the body's low-pass)",
             bod.tau_deact is not None)) if on]
        if _mk_refuse:
            raise ValueError("musc_kernel refuses: " + ", ".join(_mk_refuse))
        from gpu.bob import forms as _bob_forms
        _mk_cls, _mk_missing = _bob_forms.unit_classes(ro.ro.mn_ids, meta, edges)
        musc_kernel = _musc_kernel.MuscKernel(
            M_flex, M_ext, cscale, _mk_cls, bod._alpha(bod.tau_act, tick), _mk_spec,
            tick, _MK_MM.DOF_NAMES, unlisted=len(_mk_missing))
    # Design afc8fd03 B14: the held flight preparation, a key of the air rig
    # argument and never a theta key. Absent, not one line of it runs.
    held = None
    if rig is not None and rig.get("held") is not None:
        if rig["held"].get("design") == "6d0042d5-235f-464f-a4a4-bf937b12698a":
            # Design 6d0042d5 Build 4: run M's front femora held at a right angle,
            # the same hooks on ten entries of the front legs.
            import importlib
            Held = importlib.import_module("gpu.6d0042d5.hold").Hold
        elif rig["held"].get("design") == "bf165b63-28c8-46c1-880a-ed58d66fdc75":
            # Design bf165b63 Build 4: the six FTi_pitch held at one U angle
            # in the reset pose, the claw rows' delivered rates recorded.
            import importlib
            Held = importlib.import_module("gpu.3be5a2e2.hold").Hold
        else:
            from gpu.afc8fd03.held import Held
        held = Held(rig, theta=th, body=bod, proprioceptors=prop,
                    simulation=sim, tick=tick, per=per)
    window_recorder = (WindowRecorder(sim, fly, abd)
                       if capture is not None else None)
    # Design 983f1c29 Build 4 (absent = bit-identical): the load law's input
    # net of adhesion, F_net = max(0, F_contact - F_adhesion) per leg and tick,
    # the adhesion actuator's force counted only while a tarsal or claw geom
    # of the leg is in contact. The geom and actuator tables and the contact
    # rule are WindowRecorder's (capture elements 21 and 22), so the online
    # input equals gpu/kneeload/read_b.py's offline net tick for tick.
    _net_tab, _net_acc = None, None
    if th.get("camp_load_net_adhesion"):
        if th.get("camp_load_net_adhesion") is not True:
            raise ValueError("camp_load_net_adhesion is true or absent")
        if _camp_load == "per_segment":
            raise ValueError("camp_load_net_adhesion has no per_segment form")
        from gpu.window_hooks import CAPTURE_LEGS as _NCL
        if list(_NCL) != list(legs_sorted):
            raise ValueError("camp_load_net_adhesion: leg orders differ")
        _net_tab = (window_recorder if window_recorder is not None
                    else WindowRecorder(sim, fly, abd))
        _net_acc = {"n": np.zeros(6, dtype=np.int64), "gross": np.zeros(6),
                    "adhesion": np.zeros(6), "net": np.zeros(6), "ticks": 0,
                    "window": np.zeros(3)}
    if window_recorder is not None:
        from gpu.fall_controls import position_order
        window_recorder.metadata["position_order"] = position_order(sim.mj_model, pos_act)
    reductor_stiffness = None
    if reductor_receipt is not None and capture is not None:
        from gpu.fall_controls import stiffness_receipt
        reductor_stiffness = stiffness_receipt(
            sim.mj_model, stiff, pos_act,
            reductor_mask if abd is None else abd.pad_mask(reductor_mask)
            if prob is None else prob.pad_mask(abd.pad_mask(reductor_mask)), th)
        reductor_receipt["stiffness"] = reductor_stiffness
    if window_recorder is not None and adh_tau_ms:
        window_recorder.metadata.update(
            adhesion_command_order=list(adh_legs),
            adhesion_onoff_semantics="requested Boolean gate before actuator lag",
            adhesion_command_semantics="continuous applied control in [0, 1]")
    visual_relay_mode = bool(
        visual_bridge is not None
        and visual_bridge["mode"] == _HS_DNA02_RELAY_MODE)
    # C-M86/C-M87: visual_relay_mode does two jobs -- it selects the carrier
    # rate path that drives T4/T5, and it enables the decoder's motor gate.
    # Deliver-only keeps the first and withholds the second, so sight reaches
    # the cord only through the connectome's own edges.
    visual_deliver_only = bool(
        visual_bridge is not None
        and visual_bridge.get("deliver_only", False))
    visual_relay_counts = np.zeros(2, dtype=np.int64)
    visual_relay_stimulus_counts = np.zeros(2, dtype=np.int64)
    visual_relay_gate_ticks = np.zeros(2, dtype=np.int64)
    visual_relay_input_ticks = np.zeros(2, dtype=np.int64)
    visual_relay_negative_input_ticks = np.zeros(2, dtype=np.int64)
    visual_relay_rate_sums = np.zeros(2, dtype=float)
    visual_relay_abs_rate_sums = np.zeros(2, dtype=float)
    visual_input_row_count = (
        6 if visual_bridge is None else len(visual_bridge["input_rows"]))
    visual_relay_command_min = np.full(
        visual_input_row_count, np.inf, dtype=float)
    visual_relay_command_max = np.full(
        visual_input_row_count, -np.inf, dtype=float)
    visual_relay_saturation_samples = np.zeros(
        visual_input_row_count, dtype=np.int64)
    visual_relay_baseline_sum = np.zeros(2, dtype=np.int64)
    visual_relay_baseline_ticks = 0
    visual_relay_hs_baseline_sum = np.zeros(6, dtype=float)
    visual_relay_hs_baseline_ticks = 0
    visual_relay_hs_baseline_mean = np.zeros(6, dtype=float)
    visual_relay_baseline_mean = np.zeros(2, dtype=float)
    visual_relay_first_gate_ms = None
    visual_teacher_first_gate_ms = None
    visual_relay_early_gate_ticks = 0
    visual_relay_side = None
    visual_relay_magnitude = 1.0

    # Design 302c1bfb, Build 11: th["cmd_off_ms"] = t_off sets the command
    # rows' rate to 0 from the first tick at or after t_off. drive_fn writes
    # the tick it is building into cmd_clock. Absent = bit-identical.
    cmd_clock = [0]
    cmd_off_tick = None
    if th.get("cmd_off_ms") is not None:
        _cmd_off = float(th["cmd_off_ms"])
        if not np.isfinite(_cmd_off) or _cmd_off <= 0.0:
            raise ValueError("cmd_off_ms must be finite and > 0")
        cmd_off_tick = int(np.ceil(round(_cmd_off / tick, 9)))

    # Design 478fe148, Build B2: th["cmd_on_ms"] = t_on sets every forced
    # group's rate (cmd, cmd2, cmd3) to 0 before the first tick at or after
    # t_on, the mirror of cmd_off_ms; th["cmd2_off_ms"] and th["cmd3_off_ms"]
    # end their groups from the first tick at or after them, as cmd_off_ms
    # ends cmd. Absent = bit-identical.
    def _gate_tick(key):
        if th.get(key) is None:
            return None
        _v = float(th[key])
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError(f"{key} must be finite and > 0")
        return int(np.ceil(round(_v / tick, 9)))
    cmd_on_tick = _gate_tick("cmd_on_ms")
    cmd2_off_tick = _gate_tick("cmd2_off_ms")
    cmd3_off_tick = _gate_tick("cmd3_off_ms")
    # Design 707773e7 Build 3 (study 0fc789d7): th["cmd2_on_ms"] = t_on sets
    # the second group's rate to 0 before the first tick at or after t_on and
    # leaves cmd and cmd3 as they were, so a cmd2 pulse can sit inside a
    # cmd-driven walk. Refused with cmd_on_ms (which gates every group), with
    # the command OU keys, and unless cmd2_off_ms falls after it. The pulse's
    # first and last nonzero tick are the cmd_gates receipt's. Absent =
    # bit-identical.
    cmd2_on_tick = _gate_tick("cmd2_on_ms")
    if cmd2_on_tick is not None:
        if not len(cmd2):
            raise ValueError("cmd2_on_ms needs a second command")
        if cmd_on_tick is not None:
            raise ValueError("cmd2_on_ms is refused with cmd_on_ms, which gates every group")
        if cmd2_off_tick is None or cmd2_off_tick <= cmd2_on_tick:
            raise ValueError("cmd2_on_ms needs cmd2_off_ms after it")
    if cmd2_off_tick is not None and not len(cmd2):
        raise ValueError("cmd2_off_ms needs a second command")
    if cmd3_off_tick is not None and not len(cmd3):
        raise ValueError("cmd3_off_ms needs a third command")
    for _off in (cmd_off_tick, cmd2_off_tick, cmd3_off_tick):
        if cmd_on_tick is not None and _off is not None and _off <= cmd_on_tick:
            raise ValueError("a command's end must fall after cmd_on_ms")
    # Henri study dabc305b, ported by design 5ec96ce5 Build 1: a stream that
    # changes the command rows' rates only, per control tick. Absent keys
    # leave _cmd_rates() the constant it was. Refused with the vision command
    # seam and with the command gates, which no design has combined with it.
    _cmd_ou_keys = ("cmd_rate_ou_tau_ms", "cmd_rate_ou_cv", "cmd_rate_ou_mode",
                    "cmd_rate_ou_stream", "cmd_rate_ou_rho")
    _cmd_ou = None
    if any(k in th for k in _cmd_ou_keys):
        if cmd_side is not None:
            raise ValueError("command OU cannot use the vision command seam")
        if cmd_on_tick is not None or cmd_off_tick is not None:
            raise ValueError("command OU is not defined with cmd_on_ms or cmd_off_ms")
        if cmd2_on_tick is not None:
            raise ValueError("command OU is not defined with cmd2_on_ms")
        from gpu.command_rate_ou import CommandRateOU
        _cmd_ou = CommandRateOU(
            len(cmd), seed, tick, float(th["cmd_rate_ou_tau_ms"]),
            float(th["cmd_rate_ou_cv"]), th["cmd_rate_ou_mode"],
            th.get("cmd_rate_ou_stream", 20260918), th.get("cmd_rate_ou_rho"))
    # first and last tick on which a group routed through _gated carried a
    # nonzero rate; a group never routed through it keeps [None, None]
    cmd_gate_ticks = {"cmd": [None, None], "cmd2": [None, None],
                      "cmd3": [None, None]}

    def _gated(name, off_tick, rates, on_tick=None):
        if cmd_on_tick is not None and cmd_clock[0] < cmd_on_tick:
            return np.zeros(len(rates))
        if on_tick is not None and cmd_clock[0] < on_tick:
            return np.zeros(len(rates))
        if off_tick is not None and cmd_clock[0] >= off_tick:
            return np.zeros(len(rates))
        if np.any(rates):
            _g = cmd_gate_ticks[name]
            _g[0] = cmd_clock[0] if _g[0] is None else _g[0]
            _g[1] = cmd_clock[0]
        return rates

    def _cmd3_rates():
        """Third-command rates (Build B3), gated by Build B2's keys."""
        return _gated("cmd3", cmd3_off_tick, np.full(len(cmd3), cmd3_hz))

    def _cmd_rates():
        """Command rates for this tick; ungated unless the seam is on."""
        if cmd_on_tick is not None and cmd_clock[0] < cmd_on_tick:
            return np.zeros(len(cmd))
        if cmd_off_tick is not None and cmd_clock[0] >= cmd_off_tick:
            return np.zeros(len(cmd))
        if cmd_on_tick is not None:
            if cmd_side is None:
                return _gated("cmd", None, np.full(len(cmd), th["cmd_hz"]))
            return _gated("cmd", None, _vision_gated_cmd_rates(
                th["cmd_hz"], cmd_side, vision_cmd_side_gain, visual_relay_side,
                vision_cmd_side_swap, visual_relay_magnitude))
        if cmd_side is None:
            if _cmd_ou is not None:
                return _cmd_ou.rates(cmd_clock[0], th["cmd_hz"])
            return np.full(len(cmd), th["cmd_hz"])
        return _vision_gated_cmd_rates(
            th["cmd_hz"], cmd_side, vision_cmd_side_gain, visual_relay_side,
            vision_cmd_side_swap, visual_relay_magnitude)

    def _cmd2_rates():
        """Second-command rates; ungated unless its own seam is on."""
        if cmd_on_tick is not None or cmd2_off_tick is not None:
            return _gated("cmd2", cmd2_off_tick, (
                np.full(len(cmd2), cmd2_hz) if cmd2_side is None
                else _vision_gated_cmd_rates(
                    cmd2_hz, cmd2_side, vision_cmd2_side_gain,
                    visual_relay_side, vision_cmd2_side_swap,
                    visual_relay_magnitude)), cmd2_on_tick)
        if cmd2_side is None:
            return np.full(len(cmd2), cmd2_hz)
        return _vision_gated_cmd_rates(
            cmd2_hz, cmd2_side, vision_cmd2_side_gain, visual_relay_side,
            vision_cmd2_side_swap, visual_relay_magnitude)

    visual_relay_class = None
    visual_delta_mode = bool(
        visual_relay_mode
        and visual_bridge["relay_code"] == "prestim_delta")
    visual_hs_opponent_mode = bool(
        visual_relay_mode
        and visual_bridge["relay_code"] == "hs_opponent")
    visual_hs_centroid_mode = bool(
        visual_relay_mode
        and visual_bridge["relay_code"] == "hs_centroid_v1")
    visual_hs_graded_mode = bool(
        visual_relay_mode
        and visual_bridge["relay_code"] == "hs_graded_opponent")
    visual_direct_hs_clamp_mode = bool(
        visual_hs_graded_mode
        and np.array_equal(
            visual_bridge["input_rows"], visual_bridge["hs_rows"]))
    visual_hs_forward_speed_mode = bool(
        visual_hs_graded_mode
        and visual_bridge["hs_forward_speed_phase_gain"])
    visual_hs_feedback_mode = bool(
        visual_hs_opponent_mode or visual_hs_centroid_mode
        or visual_hs_graded_mode)
    visual_relay_ema = np.zeros(
        6 if (visual_hs_centroid_mode or visual_hs_graded_mode) else 2,
        dtype=float)
    visual_relay_dna02_ema = np.zeros(2, dtype=float)
    visual_relay_hs_counts = np.zeros(6, dtype=(
        float if visual_hs_graded_mode else np.int64))
    visual_relay_hs_stimulus_counts = np.zeros(6, dtype=(
        float if visual_hs_graded_mode else np.int64))
    visual_relay_decoded_class_ticks = np.zeros(2, dtype=np.int64)
    visual_dna02_step_return_ticks = np.zeros(2, dtype=np.int64)
    visual_dna02_step_return_scale_min = np.ones(2, dtype=float)
    visual_relay_raw_hs_class_ticks = np.zeros(2, dtype=np.int64)
    visual_relay_raw_hs_class_ticks_by_epoch = np.zeros(
        (2, 2), dtype=np.int64)
    visual_relay_raw_hs_class_ticks_by_tape_segment = np.zeros(
        (3, 2), dtype=np.int64)
    visual_relay_raw_hs_class_ticks_by_body_window = np.zeros(
        (3, 2), dtype=np.int64)
    visual_relay_raw_hs_detector_agreement_ticks = 0
    visual_relay_raw_hs_detector_mismatch_ticks = 0
    visual_phase_reset_ticks_by_segment = np.zeros(3, dtype=np.int64)
    visual_phase_brake_ticks_by_class = np.zeros(2, dtype=np.int64)
    visual_phase_brake_ticks_by_segment_class = np.zeros(
        (3, 2), dtype=np.int64)
    visual_phase_brake_neutral_reset_ticks = 0
    visual_phase_brake_opposition_violation_ticks = 0
    visual_phase_brake_static_abs_yaw_rate_max = 0.0
    visual_phase_brake_filtered_yaw_rate = 0.0
    visual_phase_brake_filtered_abs_yaw_rate_max = 0.0
    visual_phase_brake_filter_ticks = 0
    visual_phase_brake_filter_changed_ticks = 0
    visual_phase_brake_proportional_envelope_max = 0.0
    visual_phase_brake_proportional_strength_min = 1.0
    visual_phase_brake_proportional_strength_max = 0.0
    visual_phase_brake_proportional_strength_sum = 0.0
    visual_phase_brake_proportional_strength_ticks = 0
    visual_hs_forward_speed_max = 0.0
    visual_hs_forward_speed_sum = 0.0
    visual_hs_forward_speed_samples = 0
    visual_hs_forward_speed_strength_min = 2.0
    visual_hs_forward_speed_strength_max = 0.0
    visual_hs_forward_speed_strength_sum = 0.0
    visual_hs_forward_speed_motion_ticks = 0
    visual_hs_forward_speed_no_motion_ticks = 0
    visual_teacher_gate_ticks = np.zeros(2, dtype=np.int64)
    if visual_delta_mode and not np.isclose(
            round(500.0 / tick) * tick, 500.0, rtol=0.0, atol=1e-9):
        raise ValueError(
            "prestim_delta requires tick_ms to partition 500 ms exactly")
    if sum((feco_13b_analog is not None,
            bool(visual_hs_graded_mode),
            graded_13b_command is not None)) > 1:
        raise ValueError(
            "analog clamp modes cannot share one scalar graded-drive "
            "normalization")
    feco_13b_rate_min = (None if feco_13b_analog is None else
                         np.full(len(feco_13b_analog["drive_rows"]),
                                 np.inf, dtype=float))
    feco_13b_rate_max = (None if feco_13b_analog is None else
                         np.full(len(feco_13b_analog["drive_rows"]),
                                 -np.inf, dtype=float))
    feco_13b_saturation_samples = (None if feco_13b_analog is None else
                                   np.zeros(len(feco_13b_analog["drive_rows"]),
                                            dtype=np.int64))
    graded_13b_command_active_ticks = np.zeros(1, dtype=np.int64)

    # Resolve the tonic-drive target once. The class boundary is the same one
    # mn_drive_probe.py uses, so the body arm and the probe target identically
    # the same cells: the slow end of the size override, v_rest in [-53, -47.9).
    # CD1-QB (lane A, 2026-09-03): th["prop_tonic_hz"] absent = bit-identical.
    # A float > 0 replaces every leg proprioceptor rate that Proprioceptors
    # .rates emits (claw, hook, club, hair plate, activation campaniform)
    # AND the load-driven campaniform vector with that one constant, after
    # the phasic transform so the transform cannot zero it. Entry counts and
    # order are unchanged, so the noise stream is the same. The open-loop
    # arm of the masked-rhythm question (Sapkal 2024; the 2026-08-10 tonic
    # tuning curve): the body keeps running, its feedback is held flat.
    prop_tonic_hz = th.get("prop_tonic_hz")
    if prop_tonic_hz is not None:
        if isinstance(prop_tonic_hz, bool) or not isinstance(
                prop_tonic_hz, (int, float)) or not prop_tonic_hz > 0.0:
            raise ValueError("prop_tonic_hz must be absent or a number > 0")
        prop_tonic_hz = float(prop_tonic_hz)
    # CD1-SG (lane A, 2026-09-04): th["prop_tonic_scope"] absent = "all" =
    # bit-identical. "feco" holds only the Proprioceptors.rates vector flat
    # (claw, hook, club, hair plate, activation campaniform) and leaves the
    # load-driven campaniforms live; "camp" the reverse. Which afferent
    # channel degrades the walk (CD1-SF: both held flat, speed 0.86 to 1.39).
    prop_tonic_scope = th.get("prop_tonic_scope", "all")
    # CD1-SI: one population of the rates vector at a time, by the
    # Proprioceptors.groups keys ("claw" = claw_flex + claw_ext, "hook" =
    # hook_flex + hook_ext, "club", "hair", "actcamp" = the activation
    # campaniforms). The load campaniforms stay live under these five.
    # CD1-SJ: "live_<pop>" holds every rates-vector population flat EXCEPT
    # that one (the load campaniforms stay live). Readout only, every mode:
    # LAST_EXTRAS["prop_rate_mean"] carries the per-population mean of the
    # rates vector as delivered, over the run.
    _PROP_POPS = ("claw", "hook", "club", "hair", "actcamp")
    _PROP_SCOPES = (("all", "feco", "camp", "vel") + _PROP_POPS
                    + tuple("live_" + k for k in _PROP_POPS))
    # Design 4ee8b7f3, Build B4: prop_tonic_scope as a dict {population: Hz}
    # over the five populations, with prop_tonic_hz absent, holds each named
    # population's rows at its own constant rate every tick; unnamed
    # populations and the load campaniforms keep the closed loop. A string
    # or an absent key is bit-identical.
    _prop_tonic_pops = None
    if isinstance(prop_tonic_scope, dict):
        if prop_tonic_hz is not None:
            raise ValueError("prop_tonic_scope as a dict carries its own rates; "
                             "prop_tonic_hz must be absent")
        if not prop_tonic_scope or any(k not in _PROP_POPS for k in prop_tonic_scope):
            raise ValueError("prop_tonic_scope as a dict names one or more of %s" % (_PROP_POPS,))
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not np.isfinite(v) or v < 0.0 for v in prop_tonic_scope.values()):
            raise ValueError("prop_tonic_scope's rates must be finite numbers >= 0")
        _prop_tonic_pops = {k: float(v) for k, v in prop_tonic_scope.items()}
    elif prop_tonic_scope not in _PROP_SCOPES:
        raise ValueError("prop_tonic_scope must be absent or one of %s" % (_PROP_SCOPES,))
    _prop_scope_mask = [None]  # filled on the first tick inside drive_fn
    _prop_rate_acc = [None, 0]  # per-population sums of r, tick count
    mn_tonic_hz = float(th.get("mn_tonic_slow_hz", 0.0) or 0.0)
    mn_tonic_scope = th.get("mn_tonic_scope") or ""
    mn_tonic_rows = None
    if mn_tonic_scope and mn_tonic_hz <= 0.0:
        raise ValueError("mn_tonic_scope requires mn_tonic_slow_hz > 0")
    if mn_tonic_hz > 0.0:
        if not use_size:
            raise ValueError(
                "mn_tonic_slow_hz needs mn_gains='size': without the size "
                "override there is no slow class to drive")
        _, _, _vrest_map = size_factors(meta, edges, **_sp_kw)
        _pos = {b: i for i, b in enumerate(meta["banc_888_id"])}
        mn_tonic_rows = np.asarray(
            [_pos[b] for b, v in _vrest_map.items()
             if b in _pos and -53.0 <= v < -47.9], dtype=np.int64)
        if mn_tonic_scope:
            if mn_tonic_scope != "right_hind":
                raise ValueError(
                    "mn_tonic_scope must be absent or 'right_hind'")
            _sides = meta["side"].astype(str).to_numpy()
            _parts = meta["body_part_effector"].astype(str).to_numpy()
            mn_tonic_rows = np.asarray(
                [r for r in mn_tonic_rows
                 if _sides[r] == "right" and _parts[r] == "hind_leg"],
                dtype=np.int64)
        if not len(mn_tonic_rows):
            raise ValueError("mn_tonic_slow_hz resolved to zero slow-class rows")

    def drive_fn(t, counts):
        nonlocal xy0, xy, prev_loads, prev_roll, visual_relay_ema
        nonlocal _camp_row  # assigned below; a walk with no capture read it before any assignment
        nonlocal visual_relay_side, visual_relay_class
        nonlocal visual_relay_magnitude
        nonlocal visual_image_rates, visual_image_side
        nonlocal visual_relay_baseline_ticks
        nonlocal visual_relay_baseline_mean
        nonlocal visual_teacher_first_gate_ms
        nonlocal visual_relay_raw_hs_detector_agreement_ticks
        nonlocal visual_relay_raw_hs_detector_mismatch_ticks
        nonlocal visual_phase_brake_neutral_reset_ticks
        nonlocal visual_phase_brake_opposition_violation_ticks
        nonlocal visual_phase_brake_static_abs_yaw_rate_max
        nonlocal visual_phase_brake_strength_sum
        nonlocal visual_phase_brake_strength_max
        nonlocal visual_phase_brake_filtered_yaw_rate
        nonlocal visual_phase_brake_filtered_abs_yaw_rate_max
        nonlocal visual_phase_brake_filter_ticks
        nonlocal visual_phase_brake_filter_changed_ticks
        nonlocal visual_phase_brake_proportional_envelope_max
        nonlocal visual_phase_brake_proportional_strength_min
        nonlocal visual_phase_brake_proportional_strength_max
        nonlocal visual_phase_brake_proportional_strength_sum
        nonlocal visual_phase_brake_proportional_strength_ticks
        nonlocal visual_hs_forward_speed_max
        nonlocal visual_hs_forward_speed_sum
        nonlocal visual_hs_forward_speed_samples
        nonlocal visual_hs_forward_speed_strength_min
        nonlocal visual_hs_forward_speed_strength_max
        nonlocal visual_hs_forward_speed_strength_sum
        nonlocal visual_hs_forward_speed_motion_ticks
        nonlocal visual_hs_forward_speed_no_motion_ticks
        nonlocal visual_relay_hs_baseline_ticks
        nonlocal visual_relay_hs_baseline_mean
        cmd_clock[0] = int(round(t / tick))
        prob_counts = None
        if prob is not None:
            prob_counts = counts[len(counts) - prob.n_rows:]
            counts = counts[:len(counts) - prob.n_rows]
        abd_counts = None
        if abd is not None:
            abd_counts = counts[len(counts) - abd.n_rows:]
            counts = counts[:len(counts) - abd.n_rows]
        neural_decoded_class = None
        neural_contact_class = None
        visual_t4t5_diagnostic_side = None
        visual_phase_brake_class = None
        visual_phase_strength = 1.0
        visual_phase_brake_strength = 1.0
        visual_hs_forward_speed_strength = 1.0
        if visual_carrier_tape is not None:
            _i = min(int(t / float(th.get("tick_ms", 5.0))),
                     len(visual_carrier_tape) - 1)
            visual_image_rates = visual_carrier_tape[_i]
            visual_image_side = None
        elif visual_flyvis_t4t5_tape is not None:
            group_rates, visual_t4t5_diagnostic_side = (
                _flyvis_banc_t4t5_rate_tape_rates(
                    t, visual_bridge, visual_flyvis_t4t5_tape))
            # The frozen tape is an input into actual BANC T4/T5 rows, never a
            # parallel controller.  Only the live evidence-graded HS state may
            # gate the body path; this side is retained for diagnostics alone.
            visual_image_side = None
            visual_image_rates = group_rates[
                visual_bridge["input_group_indices"]]
        elif visual_image_motion is not None:
            if (visual_bridge["input_source"]
                    == _FLYVIS_HS_TAPE_VISUAL_INPUT):
                visual_image_rates, visual_image_side = (
                    _flyvis_side_tape_hs_rates(
                        t, visual_bridge, visual_image_motion))
            else:
                visual_image_rates, visual_image_side = (
                    _rendered_motion_hs_rates(
                        t, visual_bridge, visual_image_motion,
                        body_yaw_rad=_body_yaw(_rq()),
                        tick_ms=tick))
        if visual_relay_mode:
            motor_n = len(ro.rows)
            expected_feedback = motor_n + 2 + (
                6 if visual_hs_feedback_mode else 0)
            if counts.shape != (expected_feedback,):
                raise ValueError(
                    "HS-DNa02 relay feedback must contain motor rows plus "
                    "two DNa02 rows and optional six HS rows")
            dna_counts = _integer_spike_feedback(
                counts[motor_n:motor_n + 2], "DNa02")
            hs_counts = (counts[motor_n + 2:]
                         if visual_hs_feedback_mode else None)
            counts = _integer_spike_feedback(counts[:motor_n], "motor")
            visual_relay_counts[:] += dna_counts
            dna02_phase_multiplier = 1.0
            dna02_phase_rate_hz = 0.0
            if visual_bridge["phase_dna02_amplitude"]:
                (visual_relay_dna02_ema[:], dna02_phase_multiplier,
                 dna02_phase_rate_hz) = _dna02_phase_amplitude_update(
                    visual_relay_dna02_ema, dna_counts, tick,
                    visual_bridge["relay_tau_ms"],
                    visual_bridge["dna02_max_rate_hz"])
            if 500.0 < t <= 5500.0:
                visual_relay_stimulus_counts[:] += dna_counts
            if visual_hs_feedback_mode:
                visual_relay_hs_counts[:] += hs_counts
                if 500.0 < t <= 5500.0:
                    visual_relay_hs_stimulus_counts[:] += hs_counts
            if (visual_hs_feedback_mode
                    and visual_bridge["hs_opponent_prestim_baseline"]
                    and 0.0 < t <= 500.0):
                visual_relay_hs_baseline_sum[:] += hs_counts
                visual_relay_hs_baseline_ticks += 1
                visual_relay_hs_baseline_mean = (
                    visual_relay_hs_baseline_sum
                    / float(visual_relay_hs_baseline_ticks))
            if visual_hs_opponent_mode:
                if visual_bridge["hs_opponent_prestim_baseline"]:
                    visual_relay_ema, candidate_side = (
                        _update_visual_hs_opponent_prestim_relay(
                            visual_relay_ema, hs_counts,
                            visual_relay_hs_baseline_mean,
                            visual_bridge["hs_sides"], tick,
                            visual_bridge["relay_tau_ms"]))
                else:
                    visual_relay_ema, candidate_side = (
                        _update_visual_hs_opponent_relay(
                            visual_relay_ema, hs_counts,
                            visual_bridge["hs_sides"], tick,
                            visual_bridge["relay_tau_ms"]))
                decoded_class = None
            elif visual_hs_centroid_mode:
                visual_relay_ema, decoded_class = (
                    _update_visual_hs_centroid_relay(
                        visual_relay_ema, hs_counts, tick,
                        visual_bridge["relay_tau_ms"]))
                neural_decoded_class = decoded_class
                if (visual_bridge["phase_dna02_amplitude_mode"]
                        == "routed"):
                    (dna02_phase_multiplier,
                     dna02_phase_rate_hz) = (
                        _dna02_routed_phase_amplitude(
                            visual_relay_dna02_ema, tick,
                            visual_bridge["dna02_max_rate_hz_by_side"],
                            neural_decoded_class))
                candidate_side = (None if decoded_class is None
                                  else 1 - decoded_class)
            elif visual_hs_graded_mode:
                visual_relay_ema, decoded_class = (
                    _update_visual_hs_graded_opponent(
                        hs_counts, visual_bridge["hs_sides"]))
                neural_decoded_class = decoded_class
                candidate_side = (None if decoded_class is None
                                  else 1 - decoded_class)
            elif visual_delta_mode and 0.0 < t <= 500.0:
                visual_relay_baseline_sum[:] += dna_counts
                visual_relay_baseline_ticks += 1
                decoded_class = None
            else:
                decoded_class = None
            if (not visual_hs_feedback_mode
                    and visual_delta_mode and t == 500.0):
                visual_relay_baseline_mean = (
                    visual_relay_baseline_sum.astype(float)
                    / float(visual_relay_baseline_ticks))
                visual_relay_ema[:] = 0.0
                candidate_side = None
            elif (not visual_hs_feedback_mode
                  and visual_delta_mode and t > 500.0):
                visual_relay_ema, candidate_side = (
                    _update_visual_delta_relay(
                        visual_relay_ema, dna_counts,
                        visual_relay_baseline_mean, tick,
                        visual_bridge["relay_tau_ms"],
                        swap=(visual_bridge["side_map"] == "swapped")))
            elif not visual_hs_feedback_mode and visual_delta_mode:
                candidate_side = None
            elif not visual_hs_feedback_mode:
                visual_relay_ema, candidate_side = _update_visual_relay(
                    visual_relay_ema, dna_counts, tick,
                    visual_bridge["relay_tau_ms"],
                    swap=(visual_bridge["side_map"] == "swapped"))
            if visual_flyvis_t4t5_tape is not None:
                in_stimulus = (
                    bool(float(np.sum(group_rates)) > 0.0)
                    if visual_bridge["flyvis_t4t5_stimulus_from_drive"]
                    else neural_decoded_class is not None)
            elif visual_image_motion is not None:
                in_stimulus = visual_image_side is not None
            else:
                in_stimulus = _optomotor_active_side(
                    t, visual_bridge["stimulus_direction"],
                    visual_bridge["contrast"]) is not None
            if in_stimulus and neural_decoded_class is not None:
                neural_class_index = int(neural_decoded_class)
                neural_contact_class = neural_decoded_class
                visual_relay_raw_hs_class_ticks[neural_class_index] += 1
                epoch = int(float(t) > 3000.0)
                visual_relay_raw_hs_class_ticks_by_epoch[
                    epoch, neural_class_index] += 1
                if visual_flyvis_t4t5_tape is not None:
                    tape_segment = (0 if float(t) < 500.0 else
                                    1 if float(t) < 5500.0 else 2)
                    visual_relay_raw_hs_class_ticks_by_tape_segment[
                        tape_segment, neural_class_index] += 1
                    for body_window, (start_ms, stop_ms) in enumerate((
                            (1000.0, 2000.0),
                            (2750.0, 3500.0),
                            (4250.0, 5250.0))):
                        if start_ms <= float(t) < stop_ms:
                            visual_relay_raw_hs_class_ticks_by_body_window[
                                body_window, neural_class_index] += 1
                diagnostic_side = (
                    visual_t4t5_diagnostic_side
                    if visual_flyvis_t4t5_tape is not None
                    else visual_image_side)
                if diagnostic_side is not None:
                    if neural_class_index == int(diagnostic_side):
                        visual_relay_raw_hs_detector_agreement_ticks += 1
                    else:
                        visual_relay_raw_hs_detector_mismatch_ticks += 1
            if visual_bridge["phase_from_detector"]:
                if visual_image_motion is None:
                    raise ValueError(
                        "vision_phase_from_detector requires rendered panorama")
                decoded_class = visual_image_side
            if visual_bridge["phase_brake_on_no_motion"]:
                raw_yaw_rate = float(
                    visual_image_motion["last_body_yaw_rate_deg_s"])
                yaw_rate = raw_yaw_rate
                if visual_bridge["phase_brake_filter_body_rate"]:
                    visual_phase_brake_filtered_yaw_rate = (
                        _visual_phase_brake_filtered_rate(
                            visual_phase_brake_filtered_yaw_rate,
                            raw_yaw_rate, tick,
                            visual_bridge["relay_tau_ms"]))
                    yaw_rate = visual_phase_brake_filtered_yaw_rate
                    visual_phase_brake_filter_ticks += 1
                brake_static_ms = _motion_stop_static_ms(
                    visual_bridge["image_motion_profile"])
                if (brake_static_ms is not None
                        and brake_static_ms[0] <= float(t)
                        < brake_static_ms[1]):
                    visual_phase_brake_static_abs_yaw_rate_max = max(
                        visual_phase_brake_static_abs_yaw_rate_max,
                        abs(raw_yaw_rate))
                    visual_phase_brake_filtered_abs_yaw_rate_max = max(
                        visual_phase_brake_filtered_abs_yaw_rate_max,
                        abs(yaw_rate))
                if (visual_bridge["phase_brake_proportional_body_rate"]
                        and brake_static_ms is not None
                        and brake_static_ms[0] <= float(t)
                        < brake_static_ms[1]):
                    (visual_phase_brake_class,
                     visual_phase_brake_strength,
                     visual_phase_brake_proportional_envelope_max) = (
                        _visual_phase_brake_proportional_class_strength(
                            yaw_rate,
                            visual_phase_brake_proportional_envelope_max))
                    if visual_phase_brake_class is not None:
                        visual_phase_brake_proportional_strength_min = min(
                            visual_phase_brake_proportional_strength_min,
                            visual_phase_brake_strength)
                        visual_phase_brake_proportional_strength_max = max(
                            visual_phase_brake_proportional_strength_max,
                            visual_phase_brake_strength)
                        visual_phase_brake_proportional_strength_sum += (
                            visual_phase_brake_strength)
                        visual_phase_brake_proportional_strength_ticks += 1
                else:
                    visual_phase_brake_class = _visual_phase_brake_class(
                        t, yaw_rate, True, brake_static_ms,
                        visual_bridge["phase_brake_deadband_deg_s"])
                _cap = visual_bridge["phase_brake_proportional_cap"]
                if (visual_phase_brake_class is not None
                        and _cap is not None):
                    _band = float(
                        visual_bridge["phase_brake_deadband_deg_s"])
                    visual_phase_strength = float(min(
                        _cap, abs(float(yaw_rate)) / _band))
                    visual_phase_brake_strength_sum += visual_phase_strength
                    visual_phase_brake_strength_max = max(
                        visual_phase_brake_strength_max,
                        visual_phase_strength)
                if visual_bridge["phase_brake_filter_body_rate"]:
                    if visual_bridge["phase_brake_proportional_body_rate"]:
                        raw_brake_class = (
                            1 if raw_yaw_rate > 0.0
                            else 0 if raw_yaw_rate < 0.0 else None)
                    else:
                        raw_brake_class = _visual_phase_brake_class(
                            t, raw_yaw_rate, True, brake_static_ms,
                            visual_bridge["phase_brake_deadband_deg_s"])
                    if raw_brake_class != visual_phase_brake_class:
                        visual_phase_brake_filter_changed_ticks += 1
                if visual_phase_brake_class is not None:
                    decoded_class = visual_phase_brake_class
                    visual_phase_brake_ticks_by_class[
                        visual_phase_brake_class] += 1
                    visual_phase_brake_ticks_by_segment_class[
                        1, visual_phase_brake_class] += 1
                    opposes = (
                        (visual_phase_brake_class == 1 and yaw_rate > 0.0)
                        or (visual_phase_brake_class == 0 and yaw_rate < 0.0)
                        if visual_bridge[
                            "phase_brake_proportional_body_rate"]
                        else _visual_phase_brake_opposes(
                            visual_phase_brake_class, yaw_rate,
                            visual_bridge["phase_brake_deadband_deg_s"]))
                    if not opposes:
                        visual_phase_brake_opposition_violation_ticks += 1
            if in_stimulus and decoded_class is not None:
                visual_relay_decoded_class_ticks[decoded_class] += 1
            if visual_hs_forward_speed_mode:
                forward_speed = _body_forward_speed(
                    _rq(), _rv())
                (visual_hs_forward_speed_max,
                 visual_hs_forward_speed_strength) = (
                    _hs_forward_speed_phase_strength(
                        forward_speed, visual_hs_forward_speed_max,
                        visual_image_side is not None))
                visual_hs_forward_speed_sum += forward_speed
                visual_hs_forward_speed_samples += 1
                if visual_image_side is None:
                    # No sensed image motion: neutral controller, while the
                    # gait oscillator itself continues uninterrupted.
                    visual_hs_forward_speed_no_motion_ticks += 1
                else:
                    visual_hs_forward_speed_strength_min = min(
                        visual_hs_forward_speed_strength_min,
                        visual_hs_forward_speed_strength)
                    visual_hs_forward_speed_strength_max = max(
                        visual_hs_forward_speed_strength_max,
                        visual_hs_forward_speed_strength)
                    visual_hs_forward_speed_strength_sum += (
                        visual_hs_forward_speed_strength)
                    visual_hs_forward_speed_motion_ticks += 1
            visual_relay_side = (candidate_side if in_stimulus
                                 and visual_bridge["rate_hz"] > 0.0 else None)
            visual_relay_magnitude = (
                _visual_side_magnitude(visual_relay_ema)
                if (vision_cmd_side_magnitude
                    and np.asarray(visual_relay_ema).shape == (2,))
                else 1.0)
            phase_command_active = bool(
                in_stimulus or visual_phase_brake_class is not None)
            visual_relay_class = (
                decoded_class if phase_command_active
                and visual_bridge["rate_hz"] > 0.0 else None)
            if (visual_teacher_hs_thc_yaw_phase is not None
                    and visual_bridge["phase_dna02_amplitude"]):
                use_amplitude = bool(
                    in_stimulus and neural_decoded_class is not None
                    and decoded_class == neural_decoded_class)
                amplitude = dna02_phase_multiplier if use_amplitude else 1.0
                visual_teacher_hs_thc_yaw_phase["dna02_amplitude"] = amplitude
                if use_amplitude:
                    visual_teacher_hs_thc_yaw_phase[
                        "dna02_amplitude_ticks"] += 1
                    visual_teacher_hs_thc_yaw_phase[
                        "dna02_amplitude_max"] = max(
                            visual_teacher_hs_thc_yaw_phase[
                                "dna02_amplitude_max"], amplitude)
                    visual_teacher_hs_thc_yaw_phase[
                        "dna02_rate_hz_max"] = max(
                            visual_teacher_hs_thc_yaw_phase[
                                "dna02_rate_hz_max"], dna02_phase_rate_hz)
        if (visual_teacher_phase is not None
                and visual_bridge["phase_reset_on_no_motion"]
                and visual_image_side is None
                and (visual_phase_brake_class is None
                     or visual_hs_forward_speed_mode)):
            visual_teacher_phase["phase_offsets"][:] = 0.0
            if 500.0 <= float(t) < 5500.0:
                reset_static_ms = _motion_stop_static_ms(
                    visual_bridge["image_motion_profile"]) or (2250.0, 3750.0)
                segment = (0 if float(t) < reset_static_ms[0] else
                           1 if float(t) < reset_static_ms[1] else 2)
                visual_phase_reset_ticks_by_segment[segment] += 1
                if (visual_bridge["phase_brake_on_no_motion"]
                        and segment == 1):
                    visual_phase_brake_neutral_reset_ticks += 1
        if (measured_step is not None
                and measured_step["front_rom_floor_motion_gate"] is not None):
            measured_step["front_rom_floor_motion_gate"][
                "motion_active"] = neural_contact_class is not None
        phase_step = (_visual_teacher_side_phase_drive(
            t, measured_step, visual_teacher_phase, visual_relay_class,
            prev_loads=prev_loads, hs_load_class=neural_contact_class,
            decoded_strength=(
                visual_hs_forward_speed_strength
                if visual_hs_forward_speed_mode
                else visual_phase_brake_strength
                if visual_phase_brake_class is not None else 1.0),
            decoded_memoryless=bool(
                visual_phase_brake_class is not None
                and visual_bridge[
                    "phase_brake_memoryless_body_rate"]),
            strength=visual_phase_strength)
            if visual_teacher_phase is not None else None)
        if visual_teacher_hs_thc_yaw_phase is not None:
            phase_step = _apply_visual_teacher_hs_thc_yaw_phase(
                phase_step, measured_step,
                visual_teacher_phase["sampled_phases_rad"],
                neural_contact_class,
                visual_teacher_hs_thc_yaw_phase)
        if _ttm_bridge is not None:
            _ttm_bridge.observe_counts(counts)

        def apply_visual_return_gate(*channels):
            nonlocal visual_relay_first_gate_ms
            nonlocal visual_relay_early_gate_ticks
            if visual_relay_mode and not visual_deliver_only:
                # A zero final neural-readout coefficient keeps the visual CNS
                # live but has no motor channel to gate. Decoder state is still
                # recorded; applied motor-gate ticks remain exactly zero.
                if visual_relay_side is None or neural_motor_scale == 0.0:
                    return
                affected = _apply_optomotor_return_gate(
                    t, visual_bridge, visual_return_dofs, prev_loads,
                    *channels, active_side=visual_relay_side)
                if affected:
                    visual_relay_gate_ticks[visual_relay_side] += 1
                    if visual_relay_first_gate_ms is None:
                        visual_relay_first_gate_ms = float(t)
                    if t <= 500.0:
                        visual_relay_early_gate_ticks += 1
                return
            _apply_optomotor_return_gate(
                t, visual_bridge, visual_return_dofs, prev_loads, *channels)

        def apply_visual_teacher_gate(*channels):
            nonlocal visual_teacher_first_gate_ms
            if (not (visual_hs_centroid_mode or visual_hs_graded_mode)
                    or visual_bridge["teacher_steering_gain"] == 0.0
                    or visual_relay_side is None):
                return
            affected = _apply_optomotor_return_gate(
                t, visual_bridge, visual_teacher_dofs, prev_loads,
                *channels, active_side=visual_relay_side,
                scale_override=(
                    1.0 - visual_bridge["teacher_steering_gain"]))
            if affected:
                visual_teacher_gate_ticks[visual_relay_side] += 1
                if visual_teacher_first_gate_ms is None:
                    visual_teacher_first_gate_ms = float(t)
        # playback (REVERSE ladder, layer 1): drive the body from a
        # precomputed (T, 42, 2) flex/ext program instead of the
        # network's MN counts -- the muscle-space walking experiment.
        # None = network-driven, bit-identical.
        if window_hook is not None:
            window_hook.begin(t, bod)
        if playback is not None:
            k_t = min(int(round(t / tick)), len(playback) - 1)
            # CD1-CN (2026-08-22, 30th defect): element 17 read _drv
            # unconditionally, but this branch never assigned it, so
            # ANY capture on the playback path raised UnboundLocalError.
            # The probe runs evaluate WITHOUT capture and is structurally
            # blind to it. The honest value here is the program's own
            # commanded [df, de] -- the drive that actually reaches the
            # muscle model on this path -- not an empty stub.
            if capture is not None:
                _drv = np.stack([
                    np.asarray(playback[k_t, :, 0], dtype=float),
                    np.asarray(playback[k_t, :, 1], dtype=float)])
            bod.step2(playback[k_t, :, 0], playback[k_t, :, 1], tick)
            if use_split:
                if k_load:
                    stiff.apply_load(prev_loads)
                else:
                    stiff.apply(bod.cocon)
        elif use_split:
            c = counts.astype(float) * cscale
            df = M_flex.T @ c
            de = M_ext.T @ c
            tick_motor_scale = neural_motor_scale
            if neural_motor_scale_by_class is not None:
                if neural_contact_class is not None:
                    tick_motor_scale = neural_motor_scale_by_class[
                        int(neural_contact_class)]
                neural_motor_gate_ticks[
                    2 if neural_contact_class is None
                    else int(neural_contact_class)] += 1
            if tick_motor_scale != 1.0:
                df *= tick_motor_scale
                de *= tick_motor_scale
            _apply_side_drive_trim(side_drive_dofs, steering_trim, df, de)
            apply_visual_return_gate(df, de)
            # default-off lesion: zero roll/yaw drive. FETi sitters
            # fail on body roll (50-180 deg) with roll-DOF a ~2.5x
            # pitch. Absent = bit-identical. Not a promoted model.
            if th.get("pitch_only"):
                df[po_idx] = 0.0
                de[po_idx] = 0.0
            if motor_sequence is not None:
                seq_flex, seq_ext = _motor_sequence_drive(t, motor_sequence)
                df = df + seq_flex
                de = de + seq_ext
            if measured_step is not None:
                if phase_step is None:
                    step_flex, step_ext, _ = _measured_step_drive(
                        t, measured_step)
                else:
                    step_flex, step_ext, _ = phase_step
                if measured_step_neural_upstream_mix is not None:
                    step_flex, step_ext = (
                        _apply_measured_step_neural_upstream_mix(
                            df, de, step_flex, step_ext,
                            measured_step_neural_upstream_mix))
                    df = np.zeros_like(df)
                    de = np.zeros_like(de)
                if measured_step_neural_sign_gate is not None:
                    _sign_class_slot = (2 if neural_contact_class is None
                                        else int(neural_contact_class))
                    _sign_eligible = True
                    if (measured_step_neural_sign_gate["route"]
                            == "right_and_detector_zero"):
                        _sign_eligible = bool(
                            neural_contact_class == 1
                            or _visual_detector_zero_in_stimulus(
                                t, visual_image_motion, visual_image_side))
                    elif (measured_step_neural_sign_gate["route"]
                          == "right_only"):
                        _sign_eligible = bool(neural_contact_class == 1)
                    step_flex, step_ext = (
                        _apply_measured_step_neural_sign_gate(
                            df, de, step_flex, step_ext,
                            measured_step_neural_sign_gate,
                            eligible=_sign_eligible,
                            class_slot=_sign_class_slot))
                    df = np.zeros_like(df)
                    de = np.zeros_like(de)
                apply_visual_teacher_gate(step_flex, step_ext)
                _apply_measured_step_contact_gain(
                    step_flex, step_ext, prev_loads,
                    measured_step_contact_gain,
                    visual_class=visual_relay_class)
                _apply_measured_step_hs_outer_contact_gain(
                    step_flex, step_ext, prev_loads,
                    neural_contact_class,
                    measured_step_hs_outer_contact_gain)
                _apply_measured_step_hs_conservative_push_pull(
                    step_flex, step_ext, prev_loads,
                    neural_contact_class,
                    measured_step_hs_conservative_push_pull)
                if (visual_bridge is not None
                        and visual_bridge[
                            "dna02_measured_step_return_gate"]
                        and in_stimulus
                        and neural_decoded_class is not None
                        and dna02_phase_multiplier > 1.0):
                    step_return_scale = 2.0 - dna02_phase_multiplier
                    affected = _apply_optomotor_return_gate(
                        t, visual_bridge, visual_return_dofs, prev_loads,
                        step_flex, step_ext,
                        active_side=int(neural_decoded_class),
                        scale_override=step_return_scale)
                    if affected:
                        side = int(neural_decoded_class)
                        visual_dna02_step_return_ticks[side] += 1
                        visual_dna02_step_return_scale_min[side] = min(
                            visual_dna02_step_return_scale_min[side],
                            step_return_scale)
                _apply_measured_step_hs_saturation_aware_push_pull(
                    step_flex, step_ext, prev_loads,
                    neural_contact_class,
                    measured_step_hs_saturation_aware_push_pull)
                if moving_leg_scale is not None:
                    if visual_phase_brake_class is None:
                        step_flex *= moving_leg_scale
                        step_ext *= moving_leg_scale
                        moving_gain_ticks[0] += 1
                    else:
                        moving_gain_ticks[1] += 1
                if neural_motor_right_class_dof_substitution is not None:
                    class_slot = (2 if neural_contact_class is None
                                  else int(neural_contact_class))
                    neural_motor_right_class_dof_substitution[
                        "ticks_class0_class1_none"][class_slot] += 1
                    eligible_right = bool(
                        in_stimulus and neural_contact_class == 1)
                    eligible_detector_zero = bool(
                        neural_motor_right_class_dof_substitution[
                            "detector_zero"]
                        and _visual_detector_zero_in_stimulus(
                            t, visual_image_motion, visual_image_side))
                    eligible = bool(
                        eligible_right or eligible_detector_zero)
                    eligible_source = (
                        "right" if eligible_right else "detector_zero")
                    df, de = (
                        _apply_neural_motor_right_class_dof_substitution(
                            df, de, step_flex, step_ext, eligible,
                            neural_motor_right_class_dof_substitution,
                            eligible_source=eligible_source))
                else:
                    df = df + step_flex
                    de = de + step_ext
            if body_drive_leg_scale is not None:
                df *= body_drive_leg_scale
                de *= body_drive_leg_scale
            if window_hook is not None:
                window_hook.before_body(bod, df, de)
            if musc_kernel is not None:
                bod.step2(df, de, tick, a_in=musc_kernel.step(counts))
            elif musc_unit is None:
                bod.step2(df, de, tick)
            else:
                bod.step2(df, de, tick, a_in=musc_unit.step(counts))
            # CD1-BP (2026-08-22): the muscle model's INPUT -- the
            # network's per-DOF motor drive before any activation
            # filtering. Captured as element 17 (2, 42), APPENDED.
            # CD1-BO exonerated the joint model and could not separate
            # network from muscle; this does. Readout only.
            if capture is not None:
                _drv = np.stack([np.asarray(df, dtype=float),
                                 np.asarray(de, dtype=float)])
            if k_load:
                stiff.apply_load(prev_loads)
            else:
                stiff.apply(bod.cocon)
        else:
            d = M.T @ (counts.astype(float) * cscale)
            if neural_motor_scale != 1.0:
                d *= neural_motor_scale
            _apply_side_drive_trim(side_drive_dofs, steering_trim, d)
            apply_visual_return_gate(d)
            if capture is not None:
                _drv = np.stack([np.asarray(d, dtype=float),
                                 np.zeros_like(np.asarray(d, dtype=float))])
            if th.get("pitch_only"):
                d[po_idx] = 0.0
            if body_drive_leg_scale is not None:
                d *= body_drive_leg_scale
                if capture is not None:
                    _drv = np.stack([np.asarray(d, dtype=float),
                                     np.zeros_like(np.asarray(d, dtype=float))])
            bod.step(d, tick)
        if window_hook is not None:
            window_hook.after_body(bod)
        if held is not None:
            held.after_body(bod)
        if abd is not None:
            abd.step(abd_counts, cscale, tick)
        if prob is not None:
            prob.step(prob_counts)
        if window_hook is not None:
            window_hook.after_abdomen(abd)
        # settle phase (th["settle_ms"], default 0): hold adhesion ON for
        # the first settle_ms so the fly grips before it walks, as real
        # flies do -- attitude1 measured that 4 of 5 falls happen in the
        # first ~750 ms (settling failures), while settled flies tolerate
        # 37 deg of roll.
        if measured_step is not None and measured_step["adhesion"] is not None:
            if phase_step is None:
                _, _, onoff = _measured_step_drive(t, measured_step)
            else:
                _, _, onoff = phase_step
        elif adh_flex and t >= float(th.get("settle_ms", 0.0)):
            _gate_af = (bod.a_flex if playback_gate_a_flex is None else
                        playback_gate_a_flex[min(int(round(t / tick)),
                                                 len(playback_gate_a_flex) - 1)])
            if adh_state is None:
                onoff = np.array([_gate_af[d].mean() <= adh_lvls[i]
                                  for i, d in enumerate(adh_dofs)])
            else:
                onoff = adh_state.gate(
                    np.array([_gate_af[d].mean() for d in adh_dofs]), adh_lvls, t + tick)
            if adh_knee_legs is not None:
                for i, d in enumerate(adh_dofs):
                    if adh_knee_legs[i]:
                        onoff[i] = bod.a_ext[adh_knee_dof[i]] <= adh_lvls[i]
        elif adh_ctr_flex and t >= float(th.get("settle_ms", 0.0)):
            onoff = np.array([bod.a_flex[adh_ctr_dof[i]] <= adh_lvls[i]
                              for i in range(len(adh_dofs))])
        elif adh_motor_balance and t >= float(th.get("settle_ms", 0.0)):
            onoff = _motor_balance_adhesion_onoff(
                bod.a_flex, bod.a_ext, adh_dofs)
        else:
            onoff = np.ones(6, dtype=bool)
        if window_hook is not None:
            window_hook.adhesion(onoff)
        if held is not None:
            held.adhesion(onoff)
        adhesion_command = adh_state.apply(onoff, tick) if adh_state is not None else onoff
        act = LocomotionAction(joint_angles=(bod.theta if abd is None
                                             else abd.joint_angles(bod.theta)
                                             if prob is None
                                             else prob.joint_angles(abd.joint_angles(bod.theta))),
                               adhesion_onoff=adhesion_command)
        # CD1-AV (2026-08-22): accumulate the per-leg contact IMPULSE
        # across the tick's physics steps, capture-only, element 14
        # APPENDED. CD1-AU found the instantaneous force sample 2.7x
        # larger than the acceleration it produces; impulse/dt is the
        # interval-averaged force that Newton actually compares to an
        # interval-averaged acceleration. Reads mj_contactForce after
        # each step and writes nothing back; with capture=None not one
        # line of this runs.
        if capture is not None:
            import mujoco as _mj4
            _imp = {l_: np.zeros(3) for l_ in legs_sorted}
            _b6i = np.zeros(6)
            _hdt = float(sim.mj_model.opt.timestep)
        if _ttm_hooks is not None:
            _ttm_hooks.window(t)
        for _ in range(per):
            apply_locomotion_action(sim, fly.name, act)
            if _ttm_bridge is not None:
                _ttm_bridge.apply_before_step(sim)
            if _ttm_hooks is not None:
                _ttm_hooks.before_step()
            sim.step()
            if _ttm_hooks is not None:
                _ttm_hooks.after_step()
            if held is not None:
                held.after_step(sim)
            if capture is not None:
                for _c in range(sim.mj_data.ncon):
                    _con = sim.mj_data.contact[_c]
                    _lg2 = g2l.get(_con.geom1) or g2l.get(_con.geom2)
                    if _lg2 in _imp:
                        _mj4.mj_contactForce(sim.mj_model, sim.mj_data,
                                             _c, _b6i)
                        _R2 = np.array(_con.frame,
                                       dtype=float).reshape(3, 3)
                        _imp[_lg2] += (_R2.T @ np.asarray(_b6i[:3],
                                                          dtype=float)) * _hdt
        if held is not None:
            held.after_tick(t, sim)
        loads = contact_loads(sim, g2l)
        prev_loads = loads
        _net_loads, _net_cols = None, ()
        if _net_tab is not None:     # Build 4: the law's input, this tick
            _nflag = np.zeros(6, dtype=bool)
            for _ci in range(sim.mj_data.ncon):
                _cn = sim.mj_data.contact[_ci]
                for _cg in (int(_cn.geom1), int(_cn.geom2)):
                    _cl = _net_tab.tarsal.get(_cg)
                    if _cl is not None:
                        _nflag[_cl] = True
            _ngross = np.array([loads.get(l, 0.0) for l in legs_sorted],
                               dtype=float)
            _nadh = np.where(_nflag, np.asarray(
                sim.mj_data.actuator_force[_net_tab.adhesion_ids],
                dtype=float), 0.0)
            _nnet = np.maximum(0.0, _ngross - _nadh)
            _net_loads = dict(zip(legs_sorted, _nnet.tolist()))
            _net_cols = (_ngross, _nadh, _nnet)
            if t >= 500:
                _net_acc["n"] += _nflag
                _net_acc["gross"] += np.where(_nflag, _ngross, 0.0)
                _net_acc["adhesion"] += _nadh
                _net_acc["net"] += np.where(_nflag, _nnet, 0.0)
                _net_acc["ticks"] += 1
                _net_acc["window"] += (_ngross.sum(), _nadh.sum(), _nnet.sum())
        p_ = np.array(_rq()[:2], dtype=float)
        if t >= 500 and xy0 is None:
            xy0 = p_.copy()
        xy = p_
        quats.append(abs(float(_rq()[3])))
        # v3.2 tilt measure (re-freeze event 2026-08-31): posture-only
        # uprightness. qpos[3:7] is (w,x,y,z); the body-z-vs-world-up
        # angle is acos(1 - 2(x^2 + y^2)), yaw-independent. Additive:
        # nothing downstream changes unless a consumer reads LAST_EXTRAS.
        _qx, _qy = float(_rq()[4]), float(_rq()[5])
        tilts.append(float(np.degrees(np.arccos(
            np.clip(1.0 - 2.0 * (_qx * _qx + _qy * _qy), -1.0, 1.0)))))
        if capture is not None:
            gx = sim.mj_data.geom_xpos
            # 2026-08-18 (Robin's catch #2): a tick-count scores a
            # weightless graze the same as resting body weight on the
            # floor. Record the FORCE too, so the battery can report
            # both. nf_ stays nonzero iff any non-foot contact exists,
            # so the legacy tick measure is unchanged bit for bit.
            import mujoco as _mj2
            nf_, tot_ = 0.0, 0.0
            _b6 = np.zeros(6)
            # CD1-AP (2026-08-22): per-leg contact force VECTOR in world
            # coords, capture-only, element 13 APPENDED. mj_contactForce
            # returns the force in the CONTACT frame; contact.frame is
            # the 3x3 rotation whose rows are that frame's axes, so
            # frame.T @ f maps it to world. Reuses the loop and the
            # mj_contactForce call already running here -- no new physics.
            _legvec = {l_: np.zeros(3) for l_ in legs_sorted}
            for c_ in range(sim.mj_data.ncon):
                con_ = sim.mj_data.contact[c_]
                _mj2.mj_contactForce(sim.mj_model, sim.mj_data, c_, _b6)
                _f = float(np.linalg.norm(_b6[:3]))
                tot_ += _f
                if con_.geom1 in nonfoot_geoms or con_.geom2 in nonfoot_geoms:
                    nf_ += _f
                _lg = g2l.get(con_.geom1) or g2l.get(con_.geom2)
                if _lg in _legvec:
                    _R = np.array(con_.frame, dtype=float).reshape(3, 3)
                    _legvec[_lg] += _R.T @ np.asarray(_b6[:3], dtype=float)
            _camp_row = (np.zeros(6), np.full(6, np.nan), np.full(6, np.nan))
            capture.append((np.array(_rq()[:3], dtype=float),
                            np.array(_rq()[3:7], dtype=float),
                            counts.copy(), bod.theta.copy(),
                            [loads.get(l, 0.0) for l in legs_sorted],
                            [float(min(gx[g][2] for g in
                                       tip_geoms.get(l, [0])))
                             for l in legs_sorted],
                            bod.a_flex.copy(), bod.a_ext.copy(), nf_,
                            tot_,
                            # 2026-08-18: tarsus-tip world xyz per leg
                            # (element 10, APPENDED -- never inserted:
                            # attitude1/cpg_gap/layer1c/step1 index
                            # c[6]/c[7] and regression_walk c[8]/c[9]
                            # positionally). Lowest tip geom per leg,
                            # the same geom the tip-z field uses.
                            # For footfall placement in body coords
                            # (Mendes 2013 AEP/PEP). Capture-only.
                            [[float(v) for v in gx[min(
                                tip_geoms.get(l, [0]),
                                key=lambda g: gx[g][2])]]
                             for l in legs_sorted],
                            # CD1-AA: elements 11 and 12, APPENDED.
                            [float(sim.mj_data.actuator_force[u_])
                             for u_ in pos_act],
                            [float(sim.mj_data.qpos[a_])
                             for a_ in pos_qadr],
                            # CD1-AP: element 13, APPENDED.
                            [[float(x) for x in _legvec[l_]]
                             for l_ in legs_sorted],
                            # CD1-AV: element 14, APPENDED.
                            [[float(x) for x in _imp[l_]]
                             for l_ in legs_sorted],
                            # CD1-AW (2026-08-22): element 15, APPENDED.
                            # The root free joint's translational
                            # generalised forces -- constraint, actuator,
                            # passive, bias. CD1-AV found the contact
                            # impulse accounts for 29% of the body's
                            # horizontal momentum change; MuJoCo knows
                            # what force is actually on the body, so
                            # read all four channels rather than guess.
                            # Readout only.
                            [[float(x) for x in
                              _rf("qfrc_constraint")[:3]],
                             [float(x) for x in
                              _rf("qfrc_actuator")[:3]],
                             [float(x) for x in
                              _rf("qfrc_passive")[:3]],
                             [float(x) for x in
                              _rf("qfrc_bias")[:3]]],
                            # CD1-AX (2026-08-22): element 16, APPENDED.
                            # The simulator's OWN translational
                            # acceleration and velocity. CD1-AW found
                            # every force channel agreeing with each
                            # other and disagreeing with my second
                            # difference of position; this replaces the
                            # suspect quantity. Readout only.
                            [[float(x) for x in _ra()[:3]],
                             [float(x) for x in _rv()[:3]]],
                            # CD1-BP: element 17, APPENDED.
                            _drv.tolist() if capture is not None else [],
                            # 2026-09-03: element 18, APPENDED. World xyz
                            # of c_abdomen6 (mm), for posture_variability.
                            ([float(x) for x in _body_position_selector.read(
                                sim.mj_data.xpos)[_abd_row]]
                             if _abd_row is not None else []),
                            # 2026-09-03: element 19, present ONLY when the
                            # abdomen adapter is built (abd_gain > 0): its
                            # ten joint angles, rad, abdomen.DOF_NAMES order.
                            ([float(x) for x in abd.bod.theta]
                             if abd is not None else []),
                            *window_recorder.read(onoff if adh_tau_ms else act.adhesion_onoff),
                            # Full achieved configuration for rendering the
                            # recorded motion, including passive tarsal joints.
                            np.array(sim.mj_data.qpos, dtype=float),
                            # Element 25 is emitted at EVERY tick, zero-width
                            # when adh_tau_ms is 0, because a conditional
                            # element makes every later position depend on the
                            # theta. capture_arrays reads a zero-width column
                            # as shape (ticks, 0) and does not skip it, the
                            # same idiom element 19 already uses.
                            (adhesion_command.copy() if adh_tau_ms else []),
                            # Design 304e8cbe: elements 26 to 28, APPENDED.
                            # Per-leg clipped load, the unloading filter state
                            # (NaN where no filter runs) and the commanded
                            # campaniform rate, legs_sorted order, six wide at
                            # every tick. A commanded rate, not a delivered
                            # current and not a spike count. Filled in place
                            # by the injection site below, this same tick.
                            *_camp_row,
                            # Design 983f1c29 Build 4: elements 29 to 31,
                            # APPENDED and present ONLY with
                            # camp_load_net_adhesion: per-leg gross load, the
                            # adhesion force subtracted (zero without tarsal
                            # contact) and the net, legs_sorted order.
                            *_net_cols))
            if reductor_stiffness is not None:
                _ctr_ids = reductor_stiffness["ctr_actuator_ids"]
                reductor_stiffness["row_ms"].append(float(t + tick))
                reductor_stiffness["ctr_gainprm_0"].append(
                    sim.mj_model.actuator_gainprm[_ctr_ids, 0].tolist())
                reductor_stiffness["ctr_biasprm_1"].append(
                    sim.mj_model.actuator_biasprm[_ctr_ids, 1].tolist())
        zs.append(float(_rq()[2]))
        idx, r = prop.rates(bod)
        # CD1-KH (2026-08-24): th["camp_load"] absent/None = the original
        # expression = BIT-IDENTICAL. "per_segment" splits each leg's
        # campaniforms into three index-ordered groups and gives each
        # group its OWN segment's contact force instead of the leg sum.
        # The SIGNAL is measured (MuJoCo knows the geom); WHICH
        # campaniform belongs to which segment is a GUESS -- BANC names
        # campaniform cell_type but no leg segment -- and is labelled one
        # wherever this result is quoted.
        def _camp_gain_leg(leg):
            g = _camp_gain
            if _camp_leg_gains is not None:
                g = g * _camp_leg_gains[leg]
            return g

        # Design 983f1c29 Build 5: the touchdown trigger, every tick, on the
        # tip z the capture's element 5 carries at this same call.
        _tdp_on = None
        if _td_pulse is not None:
            _k = int(round(t / tick))
            _tdp_on = {}
            for _d in _td_pulse:
                if _d["geoms"] is None:
                    _d["geoms"] = [g_ for g_, l_ in g2l.items() if l_ == _d["leg"]]
                _z = min(float(sim.mj_data.geom_xpos[g_][2]) for g_ in _d["geoms"])
                _d["hist"].append(_z)
                _clear = _z - float(np.percentile(_d["hist"], 5))
                _was = _d["air"]
                if _was and _clear < 0.05:
                    _d["air"] = False
                elif not _was and _clear > 0.15:
                    _d["air"] = True
                if _d["pending"] is not None and _d["air"]:
                    _d["unconfirmed"].append(_d["pending"])
                    _d["pending"] = None
                if _was and not _d["air"]:
                    _d["raw"].append(_k)
                    if (_d["last_acc_k0"] is not None
                            and (_k - _d["last_acc_k0"]) * tick < _d["holdoff_ms"]):
                        _d["held_off"].append(_k)
                    else:
                        _d["pending"] = _k
                if (_d["pending"] is not None and not _d["air"]
                        and _k - _d["pending"] >= 3):
                    _k0, _d["pending"] = _d["pending"], None
                    _d["last_acc_k0"] = _k0
                    _d["accepted"].append([_k0, _k])
                    if (_k0 * tick >= _d["after_ms"]
                            and (_d["mode"] == "every" or not _d["pulses"])):
                        _on = _k + int(round(_d["delay_ms"] / tick))
                        _d["pulses"].append(
                            [_k0, _k, _on, _on + int(round(_d["dur_ms"] / tick))])
                if any(_p[2] <= _k < _p[3] for _p in _d["pulses"]):
                    _d["pulse_ticks"] += 1
                    _tdp_on[_d["leg"]] = _d

        # Design e9b68973 Build B1: the bristle pulses' triggers and this
        # tick's rates; the rate vector rides in _pack after cmd2 and cmd3.
        if _br_pulse is not None:
            _kb = int(round(t / tick))
            _br_pulse["rate"][:] = 0.0
            for _d in _br_pulse["pulses"]:
                if _d["trigger"] == "time":
                    if _kb == _d["start_tick"]:
                        _d["runs"].append([None, _kb, None, None])
                else:
                    if _d["geoms"] is None:
                        _d["geoms"] = [g_ for g_, l_ in g2l.items() if l_ == _d["leg"]]
                    _z = min(float(sim.mj_data.geom_xpos[g_][2]) for g_ in _d["geoms"])
                    _d["hist"].append(_z)
                    _clear = _z - float(np.percentile(_d["hist"], 5))
                    _was = _d["air"]
                    if _was and _clear < 0.05:
                        _d["air"] = False
                    elif not _was and _clear > 0.15:
                        _d["air"] = True
                    if _d["air"] and not _was:
                        _d["lifts"].append(_kb)
                        if _d["until_lift"]:
                            for _r in _d["runs"]:
                                if _r[2] is None:
                                    _r[2], _r[3] = _kb, "lift"
                    if _d["pending"] is not None and _d["air"]:
                        _d["unconfirmed"].append(_d["pending"])
                        _d["pending"] = None
                    if _was and not _d["air"]:
                        _d["raw"].append(_kb)
                        if (_d["last_acc_k0"] is not None
                                and (_kb - _d["last_acc_k0"]) * tick < _d["holdoff_ms"]):
                            _d["held_off"].append(_kb)
                        else:
                            _d["pending"] = _kb
                    if (_d["pending"] is not None and not _d["air"]
                            and _kb - _d["pending"] >= 3):
                        _k0, _d["pending"] = _d["pending"], None
                        _d["last_acc_k0"] = _k0
                        _d["accepted"].append([_k0, _kb])
                        if (_k0 * tick >= _d["after_ms"]
                                and (_d["mode"] == "every" or not _d["runs"])):
                            _d["runs"].append([_k0, _kb, None, None])
                for _r in _d["runs"]:
                    if _r[2] is not None:
                        continue
                    _s = _kb - _r[1]
                    if _s >= _d["len"]:
                        _r[2], _r[3] = _r[1] + _d["len"], "schedule"
                        continue
                    _br_pulse["rate"][_d["pos"]] += _d["hz_of"][_s]

        if _camp_load == "per_segment":
            _segl = contact_loads_by_segment(sim, _g2s)
            cr = []
            for leg, cells in camp.items():
                if not len(cells):
                    continue
                parts = np.array_split(np.arange(len(cells)), len(SEG_TAGS))
                v = np.empty(len(cells), dtype=float)
                for si, sg in enumerate(SEG_TAGS):
                    v[parts[si]] = _camp_gain_leg(leg) * float(np.clip(
                        _segl[(leg, sg)] / th["F_ref"], 0, 1))
                cr.append(v)
            if _camp_row is not None:
                # per_segment carries three forces per leg and no filter, so
                # the leg-level state and rate are not defined here.
                _camp_row[0][:] = [float(np.clip(
                    loads.get(l, 0.0) / th["F_ref"], 0, 1)) for l in legs_sorted]
                _camp_row = None
        elif _camp_rise is not None:
            # Design 6cc949c1, Build B1: every row of a leg's load pool at the
            # leg's CampRiseLaw rate; the touchdown pulse adds after it as in
            # the level law. Capture elements 26 to 28 carry clip(F(t - n) /
            # F_ref, 0, 1), q signed (this law has no filter state) and the
            # commanded rate.
            cr = []
            _rate_of, _u_of, _q_of = {}, {}, {}
            _cr_in = loads if _net_loads is None else _net_loads
            for leg, cells in camp.items():
                rate, _u_of[leg], _q_of[leg], _g, _ri = (
                    _camp_rise["law"][leg].step(_cr_in[leg]))
                _camp_rise["gated"][leg] += int(_g)
                _camp_rise["rising"][leg] += int(_ri)
                _camp_rise["rate_sum_hz"][leg] += rate
                _rate_of[leg] = rate
                if len(cells):
                    _v = np.full(len(cells), rate)
                    if _tdp_on and leg in _tdp_on:
                        _v[_tdp_on[leg]["pos"]] += _tdp_on[leg]["hz"]
                        _tdp_on[leg]["pulse_law_hz"].append([_k, float(rate)])
                    cr.append(_v)
            _camp_rise["ticks"] += 1
            if _camp_row is not None:
                _camp_row[0][:] = [_u_of.get(l, 0.0) for l in legs_sorted]
                _camp_row[1][:] = [_q_of.get(l, np.nan) for l in legs_sorted]
                _camp_row[2][:] = [_rate_of.get(l, 0.0) for l in legs_sorted]
                _camp_row = None
        else:
            cr = []
            _rate_of = {}
            for leg, cells in camp.items():
                u = float(np.clip((loads[leg] if _net_loads is None
                                   else _net_loads[leg]) / th["F_ref"], 0, 1))
                if _camp_adapt_k is None:
                    rate = _camp_gain_leg(leg) * u
                else:
                    if leg not in _camp_x:
                        _camp_x[leg] = u
                    _camp_x[leg] = _camp_adapt_k * _camp_x[leg] + (1.0 - _camp_adapt_k) * u
                    rate = _camp_gain_leg(leg) * (
                        u + _camp_adapt * max(0.0, _camp_x[leg] - u))
                _rate_of[leg] = rate
                if len(cells):
                    _v = np.full(len(cells), rate)
                    if _tdp_on and leg in _tdp_on:
                        _v[_tdp_on[leg]["pos"]] += _tdp_on[leg]["hz"]
                        _tdp_on[leg]["pulse_law_hz"].append([_k, float(rate)])
                    cr.append(_v)
            if _camp_row is not None:
                _camp_row[0][:] = [float(np.clip(
                    (loads if _net_loads is None else _net_loads).get(l, 0.0)
                    / th["F_ref"], 0, 1)) for l in legs_sorted]
                _camp_row[1][:] = [_camp_x.get(l, np.nan) for l in legs_sorted]
                _camp_row[2][:] = [_rate_of.get(l, 0.0) for l in legs_sorted]
                _camp_row = None
        if prop_tonic_hz is not None:
            if prop_tonic_scope in ("all", "feco"):
                r = np.full(len(r), prop_tonic_hz)
            elif prop_tonic_scope != "camp":
                if _prop_scope_mask[0] is None:
                    _pop = prop_tonic_scope.replace("live_", "")
                    _keys = {"claw": ("claw_flex", "claw_ext"),
                             "hook": ("hook_flex", "hook_ext"),
                             "club": ("club",), "hair": ("hair",),
                             "actcamp": ("campaniform",),
                             # CD1-SK: both velocity populations
                             "vel": ("hook_flex", "hook_ext", "club")}[_pop]
                    _m = np.isin(idx, np.concatenate(
                        [prop.groups[k] for k in _keys]))
                    _prop_scope_mask[0] = ~_m if prop_tonic_scope.startswith("live_") else _m
                r = np.where(_prop_scope_mask[0], prop_tonic_hz, r)
            if prop_tonic_scope in ("all", "camp"):
                cr = [np.full(len(c), prop_tonic_hz) for c in cr]
        if _prop_tonic_pops is not None:
            if _prop_scope_mask[0] is None:
                _prop_scope_mask[0] = {_k: np.isin(idx, np.concatenate(
                    [prop.groups[g] for g in {
                        "claw": ("claw_flex", "claw_ext"),
                        "hook": ("hook_flex", "hook_ext"), "club": ("club",),
                        "hair": ("hair",), "actcamp": ("campaniform",)}[_k]]))
                    for _k in _prop_tonic_pops}
            for _k, _hz in _prop_tonic_pops.items():
                r = np.where(_prop_scope_mask[0][_k], _hz, r)
        if _drive_silence is not None:
            # Design 6cc949c1, Build B2: the last word on the named rows' rates
            if (_drive_silence["k_on"] <= int(round(t / tick))
                    < _drive_silence["k_off"]):
                if not np.array_equal(idx, prop.all_rows):
                    raise RuntimeError("drive_silence: the proprioceptor "
                                       "index is not prop.all_rows")
                _ds_cr = np.concatenate(cr)
                _drive_silence["removed_hz"]["r"] += float(
                    np.sum(r[_drive_silence["r_pos"]]))
                _drive_silence["removed_hz"]["cr"] += float(
                    np.sum(_ds_cr[_drive_silence["c_pos"]]))
                r = np.array(r, copy=True)
                r[_drive_silence["r_pos"]] = 0.0
                _ds_cr[_drive_silence["c_pos"]] = 0.0
                cr = np.split(_ds_cr, np.cumsum([len(c) for c in cr])[:-1])
                _drive_silence["ticks"] += 1
        if _prop_rate_acc[0] is None:
            _prop_rate_acc[0] = {k: [np.isin(idx, np.concatenate(
                [prop.groups[g] for g in gs])), 0.0] for k, gs in (
                ("claw", ("claw_flex", "claw_ext")), ("hook", ("hook_flex", "hook_ext")),
                ("club", ("club",)), ("hair", ("hair",)), ("actcamp", ("campaniform",)))}
        for _k, _mv in _prop_rate_acc[0].items():
            if _mv[0].any():
                _mv[1] += float(r[_mv[0]].mean())
        _prop_rate_acc[1] += 1
        if claw_rate_out is not None:
            # Design bf165b63 Build 3: the delivered rate on the claw rows
            if "rows" not in claw_rate_out:
                _cro = np.concatenate([prop.groups["claw_flex"], prop.groups["claw_ext"]])
                claw_rate_out.update(
                    rows=_cro, t_ms=[], rate_hz=[],
                    group=(["claw_flex"] * len(prop.groups["claw_flex"])
                           + ["claw_ext"] * len(prop.groups["claw_ext"])),
                    dof=np.concatenate([prop.dof_of["claw_flex"], prop.dof_of["claw_ext"]]),
                    # the encoder's own table: the cell_type the membership read
                    cell_type=[str(x) for x in (meta["cell_type"].fillna("").to_numpy()[_cro]
                                                if "cell_type" in meta else [""] * len(_cro))],
                    banc_888_id=[str(x) for x in meta["banc_888_id"].to_numpy()[_cro]],
                    _idx=None, _pos=None)
            _ci = np.asarray(idx)
            if claw_rate_out["_idx"] is None or not np.array_equal(_ci, claw_rate_out["_idx"]):
                _order = np.argsort(_ci, kind="stable")
                _lo = np.searchsorted(_ci[_order], claw_rate_out["rows"], "left")
                _hi = np.searchsorted(_ci[_order], claw_rate_out["rows"], "right")
                if np.any(_hi - _lo != 1):
                    raise RuntimeError("claw_rate_out: a claw row is absent from the "
                                       "delivered index or present twice")
                claw_rate_out["_idx"], claw_rate_out["_pos"] = _ci.copy(), _order[_lo]
            claw_rate_out["t_ms"].append(float(t))
            claw_rate_out["rate_hz"].append(
                np.asarray(r, dtype=np.float64)[claw_rate_out["_pos"]].astype(np.float32))
        if held is not None:
            held.delivered(t, idx, r, cr)

        def _pack(parts_i, parts_r):
            # Design 478fe148, Build B3: every call's parts end with cmd and
            # cmd2, so the third group is drawn right after cmd2, ahead of
            # every later addition. Absent is bit-identical.
            if len(cmd3):
                parts_i = list(parts_i) + [cmd3]
                parts_r = list(parts_r) + [_cmd3_rates()]
            # Design e9b68973 Build B1: the bristle rows, right after cmd2 and
            # cmd3, at this tick's pulse rates (0 Hz between pulses).
            if _br_pulse is not None:
                parts_i = list(parts_i) + [_br_pulse["rows"]]
                parts_r = list(parts_r) + [_br_pulse["rate"].copy()]
            # MISSING TONIC DRIVE (CD1-NP, registry gap owned by Lane A):
            # rows 27 and 28 are species-level conflicts with one recorded
            # cause. Azevedo's MLA control shows the slow motor neuron's ~30 Hz
            # resting rate is set by excitatory synaptic input; this model
            # supplies none to any motor neuron, so the class is silent at rest
            # (row 28) and no recruitment ordering appears (row 27). At the
            # probe level a 32 Hz tonic input to the slow class alone puts the
            # female at 30.00 Hz with fast and intermediate silent, and
            # produces the ordering. This carries that input into the body.
            #
            # Every return path funnels through _pack, so adding it here means
            # no path can silently omit it. Absent or zero is bit-identical.
            if mn_tonic_rows is not None and mn_tonic_hz > 0.0:
                parts_i = list(parts_i) + [mn_tonic_rows]
                parts_r = list(parts_r) + [
                    np.full(len(mn_tonic_rows), mn_tonic_hz)]
            if feco_13b_analog is not None:
                base_idx = np.asarray(parts_i[0], dtype=np.int64)
                base_rate = np.asarray(parts_r[0], dtype=float)
                if base_idx.shape != base_rate.shape:
                    raise ValueError("proprioceptor index/rate shape drift")
                position = {int(row): pos
                            for pos, row in enumerate(base_idx)}
                try:
                    source_rate = np.asarray([
                        base_rate[position[int(row)]]
                        for row in feco_13b_analog["source_rows"]],
                        dtype=float)
                except KeyError as error:
                    raise ValueError(
                        "FeCO-13B analog source absent from proprioceptor "
                        "drive") from error
                analog_rate = np.zeros(
                    len(feco_13b_analog["drive_rows"]), dtype=float)
                np.add.at(
                    analog_rate, feco_13b_analog["target_slots"],
                    source_rate * feco_13b_analog["weights"])
                if not np.all(np.isfinite(analog_rate)):
                    raise FloatingPointError(
                        "non-finite FeCO-13B analog membrane command")
                feco_13b_rate_min[:] = np.minimum(
                    feco_13b_rate_min, analog_rate)
                feco_13b_rate_max[:] = np.maximum(
                    feco_13b_rate_max, analog_rate)
                feco_13b_saturation_samples[:] += (
                    analog_rate >= B.Proprioceptors.R_MAX - 1e-12)
                parts_i = list(parts_i) + [feco_13b_analog["drive_rows"]]
                parts_r = list(parts_r) + [analog_rate]
            if (graded_13b_command is not None
                    and graded_13b_command["start_ms"] <= t
                    < graded_13b_command["stop_ms"]):
                parts_i = list(parts_i) + [graded_13b_command["rows"]]
                parts_r = list(parts_r) + [np.full(
                    len(graded_13b_command["rows"]),
                    graded_13b_command["level"], dtype=float)]
                graded_13b_command_active_ticks[0] += 1
            if visual_bridge is not None and visual_bridge["rate_hz"] > 0.0:
                parts_i = list(parts_i) + [visual_bridge["input_rows"]]
                if visual_relay_mode:
                    if visual_image_motion is not None:
                        if (visual_image_rates is None
                                or np.asarray(visual_image_rates).shape
                                != (len(visual_bridge["input_rows"]),)):
                            raise ValueError(
                                "rendered image rates were not prepared")
                        visual_rates = visual_image_rates
                    elif visual_flyvis_t4t5_tape is not None:
                        if (visual_image_rates is None
                                or np.asarray(visual_image_rates).shape
                                != (len(visual_bridge["input_rows"]),)):
                            raise ValueError(
                                "FlyVis T4/T5 rates were not prepared")
                        visual_rates = visual_image_rates
                    elif visual_carrier_tape is not None:
                        if (visual_image_rates is None
                                or np.asarray(visual_image_rates).shape
                                != (len(visual_bridge["input_rows"]),)):
                            raise ValueError(
                                "carrier rates were not prepared")
                        visual_rates = visual_image_rates
                    else:
                        visual_rates = _optomotor_hs_rates(
                            t,
                            visual_bridge["stimulus_direction"],
                            visual_bridge["contrast"],
                            visual_bridge["rate_hz"],
                            visual_bridge["input_sides"],
                            visual_bridge["hs_baseline_hz"],
                        )
                    for side in (0, 1):
                        side_rates = visual_rates[
                            visual_bridge["input_sides"] == side]
                        if np.any(side_rates > 0.0):
                            visual_relay_input_ticks[side] += 1
                        if np.any(side_rates < 0.0):
                            visual_relay_negative_input_ticks[side] += 1
                        visual_relay_rate_sums[side] += float(side_rates.sum())
                        visual_relay_abs_rate_sums[side] += float(
                            np.abs(side_rates).sum())
                    if visual_direct_hs_clamp_mode:
                        command = np.asarray(
                            visual_rates, dtype=float) / float(
                                visual_bridge["rate_hz"])
                        visual_relay_command_min[:] = np.minimum(
                            visual_relay_command_min, command)
                        visual_relay_command_max[:] = np.maximum(
                            visual_relay_command_max, command)
                        visual_relay_saturation_samples[:] += (
                            np.abs(command) >= 1.0 - 1e-12)
                else:
                    visual_rates = _optomotor_dna02_rates(
                        t,
                        visual_bridge["direction"],
                        visual_bridge["contrast"],
                        visual_bridge["rate_hz"],
                    )
                parts_r = list(parts_r) + [visual_rates]
            if halt_rows is not None and len(halt_rows):
                qw, qx, qy, qz = (float(_rq()[k]) for k in range(3, 7))
                roll = np.degrees(np.arctan2(
                    2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
                hz = 5.0 if abs(roll) > 16.0 else 0.0
                parts_i = list(parts_i) + [halt_rows]
                parts_r = list(parts_r) + [np.full(len(halt_rows), hz)]
            if drop_hp_rows is not None and len(drop_hp_rows):
                parts_i = list(parts_i) + [drop_hp_rows]
                parts_r = list(parts_r) + [dropped_hp_rates(bod, drop_hp_dofs)]
            if hp_code is not None:
                _hp_now = bod.normalised()[hp_code["dofs"]]
                _hp_p = _hp_now
                if hp_code["lag"] is not None:
                    if not hp_code["p_tick"]:
                        hp_code["ring"].extend([_hp_now] * int(hp_code["lag"]))
                    hp_code["ring"].append(_hp_now)
                    _hp_p = hp_code["ring"].pop(0)
                _hp_rates = hp_row_code_rates(_hp_p, hp_code["sides"], hp_code["hinges"])
                hp_code["p_now_tick"].append(_hp_now)
                hp_code["p_tick"].append(_hp_p)
                hp_code["rate_tick"].append(_hp_rates)
                parts_i = list(parts_i) + [hp_code["rows"]]
                parts_r = list(parts_r) + [_hp_rates]
            if cohp8_rows is not None and len(cohp8_rows):
                _cohp8_rates = cohp8_rates(bod, cohp8_dofs)
                cohp8_stats["ticks"] += 1
                cohp8_stats["rate_sum_hz"] += float(_cohp8_rates.sum())
                cohp8_stats["above_rest_cell_ticks"] += int(
                    np.count_nonzero(_cohp8_rates > 5.0))
                cohp8_stats["max_rate_hz"] = max(
                    cohp8_stats["max_rate_hz"], float(_cohp8_rates.max()))
                parts_i = list(parts_i) + [cohp8_rows]
                parts_r = list(parts_r) + [_cohp8_rates]
            if gust_by_leg:
                g_idx, g_r = [], []
                for leg, cells in gust_by_leg.items():
                    g_idx.append(cells)
                    if gust_rates_by_leg is None:
                        hz = 5.0 if float(loads.get(leg, 0.0)) > 1.0 else 0.0
                        g_r.append(np.full(len(cells), hz))
                    elif float(loads.get(leg, 0.0)) > gust_load_n:
                        g_r.append(gust_rates_by_leg[leg])
                    else:
                        g_r.append(np.zeros(len(cells)))
                g_r = np.concatenate(g_r)
                if tarsal_stim is not None:
                    _tar_k = int(round(t / tick))
                    for _tp in tarsal_stim:
                        _tar_tab = _tp["rate_by_tick"]
                        _tar_hz = _tar_tab[_tar_k] if _tar_k < len(_tar_tab) else np.nan
                        if not np.isnan(_tar_hz):
                            g_r[_tp["pos"]] = _tar_hz
                        _tar_s = g_r[_tp["pos"]]
                        _tp["log"].append((
                            _tar_k, float(g_r[_tp["trace_pos"]]),
                            float(_tar_s.min()), float(_tar_s.max())))
                parts_i = list(parts_i) + [np.concatenate(g_idx)]
                parts_r = list(parts_r) + [g_r]
            if thorax_cs_rows is not None and len(thorax_cs_rows):
                planted = sum(float(loads.get(leg, 0.0)) for leg in loads) > 1.0
                parts_i = list(parts_i) + [thorax_cs_rows]
                parts_r = list(parts_r) + [
                    np.full(len(thorax_cs_rows), 5.0 if planted else 0.0)
                ]
            if wheeler_rows is not None and len(wheeler_rows):
                qw, qx, qy, qz = (float(_rq()[k]) for k in range(3, 7))
                roll = np.degrees(np.arctan2(
                    2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
                hz = 5.0 + 120.0 * float(np.clip(abs(roll) / 180.0, 0.0, 1.0))
                parts_i = list(parts_i) + [wheeler_rows]
                parts_r = list(parts_r) + [np.full(len(wheeler_rows), hz)]
            if neck_rows is not None and len(neck_rows):
                qw, qx, qy, qz = (float(_rq()[k]) for k in range(3, 7))
                sinp = 2 * (qw * qy - qz * qx)
                pitch = np.degrees(np.arcsin(float(np.clip(sinp, -1.0, 1.0))))
                hz = 5.0 + 120.0 * float(np.clip(abs(pitch) / 180.0, 0.0, 1.0))
                parts_i = list(parts_i) + [neck_rows]
                parts_r = list(parts_r) + [np.full(len(neck_rows), hz)]
            if md_by_leg:
                m_idx, m_r = [], []
                for leg, cells in md_by_leg.items():
                    hz = 5.0 if float(loads.get(leg, 0.0)) > 1.0 else 0.0
                    m_idx.append(cells)
                    m_r.append(np.full(len(cells), hz))
                parts_i = list(parts_i) + [np.concatenate(m_idx)]
                parts_r = list(parts_r) + [np.concatenate(m_r)]
            if abd_left is not None:
                qw, qx, qy, qz = (float(_rq()[k]) for k in range(3, 7))
                roll = np.degrees(np.arctan2(
                    2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
                hz_l, hz_r = abdomen_bristle_rates(
                    roll, len(abd_left), len(abd_right))
                if len(abd_left):
                    parts_i = list(parts_i) + [abd_left]
                    parts_r = list(parts_r) + [hz_l]
                if len(abd_right):
                    parts_i = list(parts_i) + [abd_right]
                    parts_r = list(parts_r) + [hz_r]
            if thorax_br_rows is not None and len(thorax_br_rows):
                hz = 5.0 if body_contact_force(sim, body_geoms) > 1.0 else 0.0
                parts_i = list(parts_i) + [thorax_br_rows]
                parts_r = list(parts_r) + [np.full(len(thorax_br_rows), hz)]
            if orphan_rows is not None and len(orphan_rows):
                parts_i = list(parts_i) + [orphan_rows]
                parts_r = list(parts_r) + [orphan_club_rates(bod, orphan_dofs)]
            if wing_cs_rows is not None and len(wing_cs_rows):
                nonlocal prev_roll
                roll = body_roll_deg(_rq())
                droll = 0.0 if prev_roll is None else (roll - prev_roll) / (tick / 1000.0)
                prev_roll = roll
                hz = wing_cs_rollvel_rate(droll)
                parts_i = list(parts_i) + [wing_cs_rows]
                parts_r = list(parts_r) + [np.full(len(wing_cs_rows), hz)]
            if jo_rest_rows is not None:
                parts_i = list(parts_i) + [jo_rest_rows]
                parts_r = list(parts_r) + [jo_rest_rates]
            if labellar_rows is not None:
                if labellar_mask[np.concatenate(parts_i)].any():
                    raise ValueError("labellar_rest rows are already in the drive set")
                _lab_r = labellar_rates
                if labellar_stim is not None:
                    _lab_k = int(round(t / tick))
                    _lab_tab = labellar_stim["rate_by_tick"]
                    _lab_hz = _lab_tab[_lab_k] if _lab_k < len(_lab_tab) else np.nan
                    if not np.isnan(_lab_hz):
                        _lab_r = labellar_rates.copy()
                        _lab_r[labellar_stim["pos"]] = _lab_hz
                    _lab_s = _lab_r[labellar_stim["pos"]]
                    labellar_stim["log"].append((
                        _lab_k, float(_lab_r[labellar_stim["trace_pos"]]),
                        float(_lab_s.min()), float(_lab_s.max())))
                parts_i = list(parts_i) + [labellar_rows]
                parts_r = list(parts_r) + [_lab_r]
            return np.concatenate(parts_i), np.concatenate(parts_r)

        if hp_gain > 0:
            npos = bod.normalised()
            hr = []
            for leg, cells in hp_cells.items():
                x = float(np.mean(npos[hp_dofs[leg]]))
                if hp_pol < 0:
                    x = 1.0 - x
                r_hp = hp_gain * float(np.clip(
                    (x - hp_thresh) / max(1e-9, 1.0 - hp_thresh), 0.0, 1.0))
                hr.append(np.full(len(cells), r_hp))
            if bristle_rest:
                br_idx, br_r = [], []
                for leg, cells in bristle_by_leg.items():
                    hz = float(bristle_rest_hz) if float(loads.get(leg, 0.0)) > 1.0 else 0.0
                    br_idx.append(cells)
                    br_r.append(np.full(len(cells), hz))
                return _pack(
                    [idx, camp_all, hp_all, np.concatenate(br_idx), cmd, cmd2],
                    [r, np.concatenate(cr), np.concatenate(hr),
                     np.concatenate(br_r),
                     _cmd_rates(),
                     _cmd2_rates()])
            return _pack(
                [idx, camp_all, hp_all, cmd, cmd2],
                [r, np.concatenate(cr), np.concatenate(hr),
                 _cmd_rates(),
                 _cmd2_rates()])
        if bristle_rest:
            br_idx, br_r = [], []
            for leg, cells in bristle_by_leg.items():
                hz = float(bristle_rest_hz) if float(loads.get(leg, 0.0)) > 1.0 else 0.0
                br_idx.append(cells)
                br_r.append(np.full(len(cells), hz))
            return _pack(
                [idx, camp_all, np.concatenate(br_idx), cmd, cmd2],
                [r, np.concatenate(cr), np.concatenate(br_r),
                 _cmd_rates(),
                 _cmd2_rates()])
        return _pack(
            [idx, camp_all, cmd, cmd2],
            [r, np.concatenate(cr),
             _cmd_rates(),
             _cmd2_rates()])

    if record_rows is None:
        rec = np.sort(ro.rows)
    else:
        rec = np.sort(np.unique(np.concatenate(
            [np.asarray(ro.rows), np.asarray(record_rows)])))
    if visual_hs_feedback_mode:
        feedback_rows = np.concatenate([
            ro.rows, visual_bridge["rows"], visual_bridge["hs_rows"]])
    elif visual_relay_mode:
        feedback_rows = np.concatenate([ro.rows, visual_bridge["rows"]])
    else:
        feedback_rows = ro.rows
    if abd is not None:
        # abdomen motor rows LAST; drive_fn slices them off first thing
        feedback_rows = np.concatenate([feedback_rows, abd.rows])
    if prob is not None:
        # proboscis motor rows after the abdomen's; drive_fn slices them first
        if np.intersect1d(feedback_rows, prob.rows).size:
            raise ValueError("a proboscis row is already a feedback row")
        feedback_rows = np.concatenate([feedback_rows, prob.rows])
    # gap.command_membrane_drive (lane A, 2026-09-02): the descending
    # command is a forced spike today, so nothing in the connectome can
    # oppose it (lane D, DNge129 arm). th["cmd_mode"] = "membrane" delivers
    # each command event to th["cmd"]'s rows as either an absolute
    # th["cmd_drive_mv"] kick or a cell-relative th["cmd_drive_fraction"] of
    # that row's own threshold-minus-effective-rest margin. The cell
    # integrates the kick instead; absent or "spike" = bit-identical. cmd2
    # stays on the forced path in both modes.
    cmd_mode = th.get("cmd_mode", "spike") or "spike"
    if cmd_mode not in ("spike", "membrane"):
        raise ValueError("cmd_mode must be 'spike' or 'membrane'")
    membrane_drive_rows, membrane_drive_mv = None, 0.0
    membrane_drive_fraction = None
    if cmd_mode == "membrane":
        has_mv = "cmd_drive_mv" in th
        has_fraction = "cmd_drive_fraction" in th
        if has_mv == has_fraction:
            raise ValueError(
                "cmd_mode='membrane' requires exactly one of "
                "cmd_drive_mv or cmd_drive_fraction")
        if has_fraction:
            membrane_drive_fraction = float(th["cmd_drive_fraction"])
            if (not np.isfinite(membrane_drive_fraction)
                    or membrane_drive_fraction <= 0.0):
                raise ValueError(
                    "cmd_drive_fraction must be finite and > 0")
        else:
            membrane_drive_mv = float(th["cmd_drive_mv"])
            if (not np.isfinite(membrane_drive_mv)
                    or membrane_drive_mv <= 0.0):
                raise ValueError("cmd_drive_mv must be finite and > 0 mV")
        membrane_drive_rows = np.asarray(cmd, dtype=np.int64)
    elif "cmd_drive_fraction" in th:
        raise ValueError(
            "cmd_drive_fraction requires cmd_mode='membrane'")
    # Design 302c1bfb, Build 9 (H05): the encoder's drawn events on the hook
    # rows reach the membrane as a kick of th["hook_membrane_drive_fraction"]
    # of each row's threshold-minus-effective-rest margin in place of a spike.
    # Claw, club, hair-plate and campaniform rows stay on the spike path.
    # The engine has one membrane drive, so the command stays a spike here.
    # Absent = bit-identical.
    hook_membrane_rows = None
    hook_membrane_fraction = th.get("hook_membrane_drive_fraction")
    if hook_membrane_fraction is not None:
        hook_membrane_fraction = float(hook_membrane_fraction)
        if (not np.isfinite(hook_membrane_fraction)
                or hook_membrane_fraction <= 0.0):
            raise ValueError(
                "hook_membrane_drive_fraction must be finite and > 0")
        if cmd_mode == "membrane":
            raise ValueError(
                "hook_membrane_drive_fraction and cmd_mode='membrane' "
                "share the engine's one membrane drive")
        hook_membrane_rows = np.asarray(
            hook_rows_encoded(meta, th), dtype=np.int64)
        membrane_drive_rows = hook_membrane_rows
        membrane_drive_fraction = hook_membrane_fraction
    # Design 1e20a949, Build C1: th["sez_rest"] kicks its rows' membranes from
    # their own random stream, never the network's. Absent = bit-identical.
    sez_rest = (None if th.get("sez_rest") is None
                else _sez_rest_spec(th, meta, cmd_mode))
    # Design 302c1bfb, Build 10: th["silence_rows_window_ms"] = [t_on, t_off]
    # silences th["silence_rows"] for t_on <= t < t_off only, both edges on
    # drive ticks; the rows keep their own thresholds outside it. It draws
    # no random number. Absent = the whole-run silencing above.
    silence_window = None
    if th.get("silence_rows_window_ms") is not None:
        if th.get("silence_rows") is None:
            raise ValueError("silence_rows_window_ms needs silence_rows")
        _sw_edges = [float(x) for x in th["silence_rows_window_ms"]]
        if len(_sw_edges) != 2:
            raise ValueError("silence_rows_window_ms is [t_on_ms, t_off_ms]")
        _sw_pos = pd.Series(
            np.arange(len(meta)), index=meta["banc_888_id"].astype(str))
        silence_window = {
            "rows": np.asarray([int(_sw_pos.loc[str(x)])
                                for x in th["silence_rows"]], dtype=np.int64),
            "on_ms": _sw_edges[0], "off_ms": _sw_edges[1]}
    if _drive_silence is not None:     # design 6cc949c1, Build B2
        silence_window = {"rows": _drive_silence["rows"],
                          "on_ms": _drive_silence["on_ms"],
                          "off_ms": _drive_silence["off_ms"]}
    # Design fa5adfc8: the trace bin, the static schedule and the TTMn spike
    # tap reach the engine only when asked for.
    from fractions import Fraction as _Fraction
    if (isinstance(trace_bin_ms, bool) or not isinstance(trace_bin_ms, (int, float))
            or not np.isfinite(trace_bin_ms) or trace_bin_ms <= 0
            or (_Fraction(str(float(trace_bin_ms)))
                / _Fraction(str(float(net.p.dt)))).denominator != 1):
        raise ValueError("trace_bin_ms must be a positive whole number of CNS steps")
    _fa5_run = {}
    if silence_window is not None:
        _fa5_run["silence_window"] = silence_window
    if _forced_ms:
        _fa5_run["force_spikes"] = _forced_ms
    if _ttm_hooks is not None and _ttm_hooks.path is not None:
        _fa5_run["spike_tap_rows"] = [_ttm_hooks.path.rows[_s] for _s in _ttm_path.SIDES]
        _fa5_run["spike_tap"] = _ttm_hooks.path.tap
    # design 8b6e0441, Build 2's receipt: the rows the walk forces are the rows
    # drive_fn hands a positive rate on any drive tick, recorded as the walk
    # runs; the wrapper returns drive_fn's own objects, so the walk is unchanged
    _oa_forced = None
    if octopamine_offset_mv is not None and drive_fn is not None:
        _oa_forced = np.zeros(net.N, dtype=bool)
        _oa_inner = drive_fn

        def drive_fn(t, counts):
            idx, rate = _oa_inner(t, counts)
            _oa_forced[np.asarray(idx, dtype=np.int64)[
                np.asarray(rate, dtype=np.float64) > 0.0]] = True
            return idx, rate
    run_out = net.run(dur_ms, seed=seed, record=rec,
                      record_trace_every=float(trace_bin_ms), drive_fn=drive_fn,
                      feedback_rows=feedback_rows, drive_every_ms=tick,
                      membrane_drive_rows=membrane_drive_rows,
                      membrane_drive_mv=membrane_drive_mv,
                      membrane_drive_fraction=membrane_drive_fraction,
                      feedback_graded_release_rows=(
                          visual_bridge["hs_rows"]
                          if visual_hs_graded_mode else None),
                      graded_drive_max_rate_hz=(
                          (visual_bridge["rate_hz"]
                           if visual_direct_hs_clamp_mode else
                           B.Proprioceptors.R_MAX
                           if feco_13b_analog is not None else
                           1.0 if graded_13b_command is not None else None)),
                      graded_drive_signed=(
                          visual_bridge["hs_signed_drive"]
                          if visual_direct_hs_clamp_mode else False),
                      **_fa5_run,
                      **({"graded_gain_window": graded_gain_window}
                         if graded_gain_window is not None else {}),
                      **({"record_opsin": True}
                         if (th.get("pulse_exc_ids") is not None
                             or th.get("pulse_exc_sets") is not None)
                         and _opsin_tap_carried(net.p) else {}),
                      **({"rest_drive": sez_rest[0]}
                         if sez_rest is not None else {}),
                      **(run_kwargs or {}))
    if trace_out is not None:
        trace_out.append((rec, run_out.get("trace")))
    disp = float(np.linalg.norm(xy - xy0)) if xy0 is not None else 0.0
    win = int(round(1000.0 / tick))   # gate windows = last 1 s, tick-invariant
    upright = float(np.mean(quats[-win:])) if quats else 0.0
    LAST_EXTRAS.clear()
    if _meta_patch_receipt is not None:
        LAST_EXTRAS["meta_patch"] = dict(_meta_patch_receipt)
    # Design 983f1c29 Build 3: both campaniform lists' per-leg membership. The
    # load list is camp (camp_by_leg, after camp_extra); the activation list is
    # prop's campaniform group, legs by its DOF names. Reporting only.
    LAST_EXTRAS["camp_lists"] = {
        "load": {leg: int(len(r)) for leg, r in camp.items()},
        "act": {leg: int(sum(prop._dof_name[int(d)][:2] == leg
                             for d in prop.dof_of["campaniform"]))
                for leg in camp},
    }
    if _camp_rise is not None:     # design 6cc949c1, Build B1's receipt
        LAST_EXTRAS["camp_load_rise"] = {
            **_camp_rise["constants"], "lag_ticks": _camp_rise["lag_ticks"],
            "tick_ms": float(tick), "G_hz": float(_camp_gain),
            "input": "net" if _net_tab is not None else "gross",
            "s": {leg: law.s for leg, law in _camp_rise["law"].items()},
            "load_rows": {leg: int(len(c)) for leg, c in camp.items()},
            "ticks": int(_camp_rise["ticks"]),
            "gated_ticks": dict(_camp_rise["gated"]),
            "rising_ticks": dict(_camp_rise["rising"]),
            "rate_sum_hz": {leg: float(v) for leg, v in
                            _camp_rise["rate_sum_hz"].items()}}
    if _drive_silence is not None:     # design 6cc949c1, Build B2's receipt
        # the named rows' recorded spikes inside the window, from the trace of
        # the record rows; None unless every named row is recorded
        _ds_spk, _ds_tr = None, run_out.get("trace")
        _ds_b = [int(round(_drive_silence[k] / float(trace_bin_ms)))
                 for k in ("on_ms", "off_ms")]
        if (_ds_tr is not None and np.isin(_drive_silence["rows"], rec).all()
                and all(abs(b * float(trace_bin_ms) - _drive_silence[k]) < 1e-9
                        for b, k in zip(_ds_b, ("on_ms", "off_ms")))):
            _ds_col = np.searchsorted(rec, _drive_silence["rows"])
            _ds_spk = int(np.asarray(_ds_tr)[_ds_b[0]:_ds_b[1]][:, _ds_col]
                          .sum(dtype=np.int64))
        _ds_eng = dict(run_out["silence_window"])
        LAST_EXTRAS["drive_silence"] = {
            "ids": list(_drive_silence["ids"]),
            "ids_sha16": rows_sha16(meta, _drive_silence["rows"]),
            "rows": [int(x) for x in _drive_silence["rows"]],
            "window_ms": [_drive_silence["on_ms"], _drive_silence["off_ms"]],
            "ticks_zeroed": int(_drive_silence["ticks"]),
            "ticks_in_window": int(_drive_silence["k_off"] - _drive_silence["k_on"]),
            "positions_zeroed": {"r": int(len(_drive_silence["r_pos"])),
                                 "cr": int(len(_drive_silence["c_pos"]))},
            "positions_per_row": _drive_silence["positions_per_row"],
            "removed_rate_sum_hz": dict(_drive_silence["removed_hz"]),
            "engine_silence_window": _ds_eng,
            "recorded_spikes_in_window": _ds_spk,
            "valid": bool(_ds_spk == 0 and _ds_eng["spikes_in_window"] == 0)}
    # Design 983f1c29 Build 5: the trigger's log per pulsed leg, ticks as
    # round(t / tick_ms), which is the capture row. accepted: [k0, k accepted];
    # pulses: [k0, k accepted, first pulsed tick, first tick after the pulse];
    # pulse_law_hz: [tick, the load law's rate under the pulse].
    if _td_pulse is not None:
        LAST_EXTRAS["touchdown_pulse"] = {
            "rule": ("tip z, min over the leg's geoms, less its running 5th "
                     "percentile from 0 ms; air above 0.15 mm, ground below "
                     "0.05 mm, starting on the ground; a touchdown at k0 is "
                     "accepted at k0 + 3 after four ground ticks unless within "
                     "holdoff_ms of the last accepted k0"),
            "tick_ms": float(tick),
            "legs": {d["leg"]: {
                "ids": list(d["ids"]), "ids_sha256": d["ids_sha256"],
                "rows": [int(r) for r in d["rows"]], "mode": d["mode"],
                **{k: float(d[k]) for k in ("hz", "dur_ms", "delay_ms",
                                            "after_ms", "holdoff_ms")},
                "delay_ticks": int(round(d["delay_ms"] / tick)),
                "dur_ticks": int(round(d["dur_ms"] / tick)),
                "raw": [int(k) for k in d["raw"]],
                "accepted": [[int(a), int(b)] for a, b in d["accepted"]],
                "unconfirmed": [int(k) for k in d["unconfirmed"]],
                "held_off": [int(k) for k in d["held_off"]],
                "pulses": [[int(x) for x in p_] for p_ in d["pulses"]],
                "pulse_ticks": int(d["pulse_ticks"]),
                "pulse_law_hz": [[int(k), float(v)] for k, v in d["pulse_law_hz"]],
            } for d in _td_pulse},
        }
    # Design e9b68973 Build B1: each pulse's triggers and runs, ticks as
    # round(t / tick_ms), the capture row. runs: [k0 (None for "time"),
    # first pulsed tick, first tick after, how it ended, expected spikes per
    # row = the schedule's integral over the run]; lifts are the leg's air
    # crossings. Delivered spikes are read from the trace, not from here.
    if _br_pulse is not None:
        _bp_end = int(round(dur_ms / tick))

        def _bp_run(d, r):
            k_end, why = r[2], r[3]
            if k_end is None:
                k_end = min(_bp_end, r[1] + d["len"])
                why = "schedule" if r[1] + d["len"] <= _bp_end else "walk_end"
            n = max(0, k_end - r[1])
            return [None if r[0] is None else int(r[0]), int(r[1]), int(k_end), why,
                    float(d["hz_of"][:n].sum() * tick / 1000.0)]

        _bp_log = {}
        for d in _br_pulse["pulses"]:
            runs = [_bp_run(d, r) for r in d["runs"]]
            e = {"trigger": d["trigger"], "ids_sha256": d["ids_sha256"], "n_ids": len(d["ids"]),
                 "schedule": d["schedule"], "runs": runs,
                 "expected_spikes_per_row": float(sum(r[4] for r in runs))}
            if d["trigger"] == "time":
                e["t_ms"] = d["t_ms"]
            else:
                e.update({"leg": d["leg"], "mode": d["mode"], "until_lift": d["until_lift"],
                          "holdoff_ms": d["holdoff_ms"], "after_ms": d["after_ms"],
                          "counts": {k: len(d[k]) for k in ("raw", "accepted", "unconfirmed",
                                                            "held_off", "lifts")},
                          "raw": [int(k) for k in d["raw"]],
                          "accepted": [[int(a), int(b)] for a, b in d["accepted"]],
                          "unconfirmed": [int(k) for k in d["unconfirmed"]],
                          "held_off": [int(k) for k in d["held_off"]],
                          "lifts": [int(k) for k in d["lifts"]]})
            _bp_log[d["name"]] = e
        LAST_EXTRAS["bristle_pulse"] = {
            "rule": ("touchdown: touchdown_pulse's tip rule on the leg (tip z, min over its "
                     "geoms, less its running 5th percentile from 0 ms; air above 0.15 mm, "
                     "ground below 0.05 mm, starting on the ground; a touchdown at k0 is "
                     "accepted at k0 + 3 unless within holdoff_ms of the last accepted k0), "
                     "the schedule starting on the accepted tick; until_lift ends it at the "
                     "leg's next air crossing. time: the schedule starts at t_ms."),
            "tick_ms": float(tick), "rows_sha256": _br_pulse["rows_sha256"],
            "n_rows": len(_br_pulse["ids"]), "pulses": _bp_log}
    if rig is not None:
        LAST_EXTRAS["rig"] = dict(rig)
    # Design 5ec96ce5 Build 1: the command stream's receipt, present only when
    # its keys were set.
    if _cmd_ou is not None:
        LAST_EXTRAS["cmd_rate_ou"] = {
            "mode": th["cmd_rate_ou_mode"], "tau_ms": float(th["cmd_rate_ou_tau_ms"]),
            "cv": float(th["cmd_rate_ou_cv"]), "rho": th.get("cmd_rate_ou_rho"),
            "stream": th.get("cmd_rate_ou_stream", 20260918),
            "rows": [int(r) for r in cmd], "ids": [str(net.ids[i]) for i in cmd],
            "side": meta.iloc[cmd]["side"].astype(str).tolist(),
            **_cmd_ou.receipt(tick, dur_ms)}
    if held is not None:
        LAST_EXTRAS["held"] = held.result()
    if prob is not None:
        LAST_EXTRAS["proboscis"] = prob.describe()
        LAST_EXTRAS["proboscis_arrays"] = prob.arrays()
    if run_kwargs:
        LAST_EXTRAS["native_run"] = {
            k: run_out[k] for k in ("native", "release_log", "release_log_sha256",
                                    "gpu_seconds", "n_spikes") if k in run_out}
    # Design 336cacb4: the release-gain window's receipt (Build 3) and the
    # release recorder (Build 4), present only when their options were set.
    for _k336 in ("graded_gain_window", "graded_release_record"):
        if _k336 in run_out:
            LAST_EXTRAS[_k336] = run_out[_k336]
    # Design 707773e7 Build 4 (study 0fc789d7): the potential recorder's
    # samples, (samples, rows) float32 in voltage_idx's row order, sample j
    # the state after tick (j + 1) * every - 1. Present only when run_kwargs
    # asked for record_voltage; gpu/retain_capture.py writes it out.
    if "voltage" in run_out and "record_voltage" in (run_kwargs or {}):
        LAST_EXTRAS["voltage"] = {
            "mv": run_out["voltage"],
            "rows": np.asarray(run_out["voltage_idx"], dtype=np.int64),
            "every_ms": run_kwargs.get("record_voltage_every_ms")}
    if graded_add_ids_sha256 is not None:
        _gset_ids = [str(meta["banc_888_id"].iloc[int(r)])
                     for r in sorted(graded_rows)]
        LAST_EXTRAS["graded_set"] = {
            "graded_add_ids_sha256": graded_add_ids_sha256,
            "added_rows": int(len(graded_add_rows)),
            "added_by_ids": int(len(graded_add_ids_rows)),
            "graded_rows": int(len(graded_rows)),
            "graded_row_order_sha256": hashlib.sha256(
                "\n".join(_gset_ids).encode()).hexdigest(),
        }
    if row_set_params is not None:
        LAST_EXTRAS["row_set_params"] = _row_set_receipt(row_set_params, meta, net.p)
    LAST_EXTRAS["final_tilt_deg"] = (float(np.mean(tilts[-win:]))
                                     if tilts else None)
    LAST_EXTRAS["mn_transferred"] = mn_transferred
    # Design 171d6f1a B1 and B2: present only when their keys are on.
    if mn_transfer is not None:
        LAST_EXTRAS["mn_transfer"] = _mn_transfer_receipt(ro.ro.mapping, mapping_eff)
    if size_pools is not None:
        LAST_EXTRAS["size_pools"] = _size_pool_receipt(meta, edges, size_pools, size_pool_map)
    LAST_EXTRAS["thc_map_rows_moved"] = thc_map_moved
    if jo_rest_receipt is not None:
        LAST_EXTRAS["jo_rest"] = jo_rest_receipt
    if labellar_receipt is not None:
        LAST_EXTRAS["labellar_rest"] = labellar_receipt
    if gust_receipt is not None:
        LAST_EXTRAS["gust_rest"] = gust_receipt
    if labellar_stim is not None:
        LAST_EXTRAS["labellar_stimulus"] = labellar_stimulus_receipt(labellar_stim)
    if tarsal_stim is not None:
        LAST_EXTRAS["tarsal_stimulus"] = [
            {**labellar_stimulus_receipt(_tp), "design": "a146ad7f"} for _tp in tarsal_stim]
    LAST_EXTRAS["muscle_gain"] = muscle_gain_receipt
    # Design fcb48834 Builds 1, 2 and 7: present only when their keys are on.
    if musc_class_gain_receipt is not None:
        LAST_EXTRAS["musc_class_gain"] = musc_class_gain_receipt
    if body_joint_drive_offset is not None:
        LAST_EXTRAS["body_joint_drive_offset"] = {
            "nonzero_dofs": int(np.count_nonzero(body_joint_drive_offset)),
            "vector_sha256": hashlib.sha256(
                np.ascontiguousarray(body_joint_drive_offset, dtype=float).tobytes()).hexdigest()}
    if musc_unit is not None:
        LAST_EXTRAS["musc_unit_path"] = musc_unit.receipt()
    if musc_kernel is not None:
        LAST_EXTRAS["musc_kernel"] = musc_kernel.receipt()
    LAST_EXTRAS["mn_speed_class_rows"] = (
        getattr(net, "_mn_speed_class_rows", mn_speed_class_rows)
        if th.get("mn_speed_class") else None)
    LAST_EXTRAS["channel_pool_map"] = _cpm
    # Design e2b1a20d B1 to B5: present only when a key is on.
    if graded_add_receipt is not None:
        LAST_EXTRAS["graded_add"] = [dict(r) for r in graded_add_receipt]
    for _k, _rows_by_type in _type_rows.items():
        if _rows_by_type:
            _vec = np.ascontiguousarray({
                "graded_gain_type": net.p.graded_gain,
                "graded_v50_type": net.p.graded_v50,
                "graded_slope_type": net.p.graded_slope,
                "e_inh_type_mv": net.p.e_inh}[_k])
            LAST_EXTRAS[_k] = {
                "types": {_t: {"rows": len(_r), "value": _v}
                          for _t, (_r, _v) in _rows_by_type.items()},
                "vector_dtype": str(_vec.dtype),
                "vector_sha256": hashlib.sha256(_vec.tobytes()).hexdigest()}
    if _nt_transfer_ids is not None:
        LAST_EXTRAS["nt_transfer_rows"] = nt_transfer_receipt(net, _nt_transfer)
    # Design fa5adfc8: present only when the key is on, so a run without it
    # carries exactly the extras it carried before.
    if th.get("v_rest_type_reset_follows"):
        LAST_EXTRAS["v_rest_type_reset"] = net.p.v_rest_type_reset_audit
    if _forced_receipt is not None:
        LAST_EXTRAS["force_spikes"] = _forced_receipt
    if _ttm_hooks is not None:
        if _ttm_hooks.m != _ttm_hooks.n_steps:
            raise RuntimeError(f"the hooks saw {_ttm_hooks.m} physics steps "
                               f"of {_ttm_hooks.n_steps}")
        LAST_EXTRAS.update(_ttm_hooks.receipts())
    if adh_state is not None or reductor_receipt is not None:
        LAST_EXTRAS["fall_control"] = {
            "release_ratio": adh_release_ratio, "dwell_ms": adh_dwell_ms,
            "tau_ms": adh_tau_ms, "reductor": reductor_receipt}
    if window_recorder is not None:
        # Design 304e8cbe: capture elements 26 to 28 are six wide and every
        # reader joins them on THIS name list, never on a position. The
        # per-leg cell counts and the hash of each leg's sorted BANC ids go
        # with them, so a reader can check it is weighting the membership the
        # run actually drove. No per-tick cell-id array is written: the
        # whole-leg branch fills every cell of a leg with one value.
        LAST_EXTRAS["window_capture_metadata"] = dict(
            window_recorder.metadata,
            camp_capture_order=list(legs_sorted),
            camp_cells_per_leg={l: int(len(camp.get(l, ()))) for l in legs_sorted},
            camp_ids_sha256={
                l: hashlib.sha256(json.dumps(
                    sorted(str(i) for i in camp.get(l, ()))).encode()).hexdigest()
                for l in legs_sorted},
            camp_adapt=_camp_adapt,
            camp_adapt_tau_ms=(_camp_adapt_tau_ms if _camp_adapt else None),
            camp_adapt_k=_camp_adapt_k,
            camp_rate_units="Hz, the commanded campaniform rate per leg; not a "
                            "delivered current and not a spike count",
            camp_x_absent="NaN where no unloading filter ran, which is every "
                          "arm with camp_adapt absent or zero")
    if window_hook is not None:
        LAST_EXTRAS["window"] = window_hook.receipt()
    if "serotonin" in run_out:
        LAST_EXTRAS["serotonin"] = dict(run_out["serotonin"])
        # design b1746728: the declared gain bound and the target set by name
        # and hash beside the state's own receipt, each only when set
        if serotonin_gain_bound is not None:
            LAST_EXTRAS["serotonin"]["gain_bound_mv_per_hz"] = list(
                serotonin_gain_bound)
        if serotonin_target_set is not None:
            LAST_EXTRAS["serotonin"]["target_set"] = {
                "name": serotonin_target_set[0],
                "ids_sha256": serotonin_target_set[1]}
    if octopamine_offset_mv is not None:
        # design 8b6e0441, Build 2: the offset, N_tgt and the set's sha256, and
        # the target rows whose spikes the walk forces (a positive drive rate
        # on any drive tick, less the rows whose drawn events reach the
        # membrane as kicks, plus the deterministic forced-spike rows), so the
        # offset reaches them and does not set their forced output (reached
        # and inert), named with their count. The target rows among the
        # feedback rows (the motor readout), which this field named until
        # 2026-10-04, are counted beside under their own name.
        _oa = np.asarray(octopamine_target_rows, dtype=np.int64)
        _fb = (np.zeros(0, dtype=np.int64) if feedback_rows is None
               else np.asarray(feedback_rows, dtype=np.int64))
        _forced = (np.zeros(net.N, dtype=bool) if _oa_forced is None
                   else _oa_forced.copy())
        if membrane_drive_rows is not None:
            _forced[np.asarray(membrane_drive_rows, dtype=np.int64)] = False
        for _rs in (_forced_ms or {}).values():
            _forced[np.asarray(_rs, dtype=np.int64)] = True
        _inert = np.intersect1d(_oa, np.flatnonzero(_forced))
        _ids = meta["banc_888_id"].astype(str).to_numpy()
        LAST_EXTRAS["octopamine_offset"] = {
            "offset_mv": octopamine_offset_mv,
            "engine": run_out.get("octopamine_offset"),
            "target_set": {"name": octopamine_target_set[0],
                           "ids_sha256": octopamine_target_set[1],
                           "n_tgt": int(len(_oa))},
            "forced_rows_reached_inert": {
                "count": int(len(_inert)),
                "rule": ("target rows given a positive drive rate on any drive "
                         "tick, less membrane-drive rows, plus deterministic "
                         "forced-spike rows (2026-10-04)"),
                "ids": sorted(_ids[_inert].tolist(), key=int)},
            "motor_readout_rows_among_targets": {
                "count": int(len(np.intersect1d(_oa, _fb)))}}
    if "octopamine_state" in run_out:   # design 8b6e0441, Build 1
        # the engine's receipt, the sets by name and hash, the excluded source
        # rows by meta id, and the declared gain bound when set
        _ids = meta["banc_888_id"].astype(str).to_numpy()
        LAST_EXTRAS["octopamine_state"] = dict(run_out["octopamine_state"])
        LAST_EXTRAS["octopamine_state"].update(
            excluded_source_ids=sorted(
                (str(_ids[r]) for r in run_out["octopamine_state"]["excluded_source_rows"]),
                key=int),
            source_set={"name": octopamine_source_set[0],
                        "ids_sha256": octopamine_source_set[1],
                        "n_src": int(len(octopamine_source_rows))},
            target_set={"name": octopamine_target_set[0],
                        "ids_sha256": octopamine_target_set[1],
                        "n_tgt": int(len(octopamine_target_rows))},
            gain_bound_mv_per_hz=(None if octopamine_gain_bound is None
                                  else list(octopamine_gain_bound)))
    if "presynaptic_inhibition" in run_out:
        LAST_EXTRAS["presynaptic_inhibition"] = dict(
            run_out["presynaptic_inhibition"])
        # design 302c1bfb, Build 8: the mode and the rows by digest
        LAST_EXTRAS["presynaptic_inhibition"].update(
            mode=_presynaptic_inhibition_mode,
            rows_sha16=rows_sha16(meta, presynaptic_inhibition_rows))
    if cmd_off_tick is not None:
        LAST_EXTRAS["cmd_off"] = {"cmd_off_ms": float(th["cmd_off_ms"]),
                                  "off_tick": cmd_off_tick,
                                  "off_ms": cmd_off_tick * tick}
    if edge_count_set is not None:    # Build B1, made where the network was built
        LAST_EXTRAS["edge_count_set"] = net.edge_count_set_receipt
    if _net_acc is not None:     # design 983f1c29 Build 4's receipt
        _nbw = float(np.sum(sim.mj_model.body_mass)
                     * np.linalg.norm(sim.mj_model.opt.gravity))
        _nt = max(int(_net_acc["ticks"]), 1)
        LAST_EXTRAS["camp_load_net_adhesion"] = {
            "legs": list(legs_sorted), "from_ms": 500.0,
            "ticks": int(_net_acc["ticks"]),
            "contact_ticks": _net_acc["n"].tolist(),
            "stance_mean": {k: [float(v / n) if n else None for v, n in
                                zip(_net_acc[k], _net_acc["n"])]
                            for k in ("gross", "adhesion", "net")},
            "window_sum_mean": dict(zip(("gross", "adhesion", "net"),
                                        (_net_acc["window"] / _nt).tolist())),
            "body_weight_model_units": _nbw,
            "window_sum_mean_body_weights": dict(zip(
                ("gross", "adhesion", "net"),
                (_net_acc["window"] / _nt / _nbw).tolist()))}
    if th.get("delay_class_scale"):   # design 983f1c29 Build 6, made at the build
        LAST_EXTRAS["delay_class_scale"] = json.loads(json.dumps(
            net.delay_class_scale_receipt))
    if th.get("delay_row_ms"):   # design 9bdd809a Build 1, made at the build
        LAST_EXTRAS["delay_row_ms"] = json.loads(json.dumps(
            net.delay_row_ms_receipt))
    if cmd_on_tick is not None or cmd2_off_tick is not None or len(cmd3):
        # Builds B2 and B3: the gates as ticks, each group's rows by digest and
        # the first and last tick on which it carried a nonzero rate
        LAST_EXTRAS["cmd_gates"] = {
            "cmd_on_tick": cmd_on_tick, "cmd_off_tick": cmd_off_tick,
            "cmd2_off_tick": cmd2_off_tick, "cmd3_off_tick": cmd3_off_tick,
            **({"cmd2_on_tick": cmd2_on_tick} if cmd2_on_tick is not None else {}),
            "tick_ms": tick,
            # critique b15a92f2 condition 4: with a scalar spike_reg k every
            # forced group is a renewal train with Erlang-k intervals
            # (k 44: CV 0.15), not Poisson
            "spike_reg": float(th.get("spike_reg", 1.0)),
            "spike_reg_mode": th.get("spike_reg_mode"),
            "groups": {name: {"rows": int(len(rows)),
                              "rows_sha16": rows_sha16(meta, rows),
                              "hz": hz, "first_nonzero_tick": cmd_gate_ticks[name][0],
                              "last_nonzero_tick": cmd_gate_ticks[name][1]}
                       for name, rows, hz in (("cmd", cmd, float(th["cmd_hz"])),
                                              ("cmd2", cmd2, cmd2_hz),
                                              ("cmd3", cmd3, cmd3_hz))}}
    if sez_rest is not None:
        LAST_EXTRAS["sez_rest"] = dict(run_out["rest_drive"], ids_sha256=sez_rest[1],
                                       matched_rows=sez_rest[2])
    if "membrane_drive" in run_out and hook_membrane_rows is not None:
        # design 302c1bfb, Build 9: the routed hook rows by digest
        LAST_EXTRAS["hook_membrane_drive"] = dict(
            run_out["membrane_drive"],
            rows_sha16=rows_sha16(meta, hook_membrane_rows))
    elif "membrane_drive" in run_out:
        LAST_EXTRAS["command_membrane_drive"] = dict(
            run_out["membrane_drive"])
    if moving_leg_scale is not None:
        LAST_EXTRAS["measured_step_leg_gains_moving"] = {
            "gains": [float(v) for v in moving_leg_gains],
            "ticks_moving_braking": moving_gain_ticks.astype(int).tolist(),
        }
    # coincidence receipt (judge, 2026-09-05): the seam's applied product per
    # run, so a vision-grade seed can show the detector fired. Read-only copy
    # of the run output; nothing about the run changes.
    if "coincidence" in run_out:
        LAST_EXTRAS["coincidence"] = dict(run_out["coincidence"])
    if "coincidence_triples" in run_out:
        LAST_EXTRAS["coincidence_triples"] = [
            dict(q) for q in run_out["coincidence_triples"]]
    LAST_EXTRAS["neural_motor_scale"] = float(neural_motor_scale)
    if _feco_polarity_mode is not None:
        LAST_EXTRAS["feco_type_polarity"] = {
            "mode": _feco_polarity_mode,
            "covered_rows": int(prop.type_polarity_rows),
            "fallback_rows": int(prop.type_polarity_fallback_rows),
            "groups": prop.counts(),
        }
    if _proprioceptor_phasic_mode is not None:
        LAST_EXTRAS["proprioceptor_phasic"] = {
            "mode": _proprioceptor_phasic_mode,
            "phasic_gain": float(prop.phasic),
            "exempt_claw_rows": int(prop.phasic_exempt_rows),
            "transformed_nonclaw_rows": int(
                len(prop.all_rows) - prop.phasic_exempt_rows),
        }
    if feco_13b_analog is not None:
        LAST_EXTRAS["feco_13b_analog"] = {
            "mode": feco_13b_analog["mode"],
            "edge_pairs": feco_13b_analog["edge_pairs"],
            "chemical_synapses_replaced": feco_13b_analog[
                "chemical_synapses_replaced"],
            "claw_source_rows": feco_13b_analog["claw_source_rows"],
            "actual_13b_rows": feco_13b_analog["actual_13b_rows"],
            "driven_target_rows": feco_13b_analog["driven_target_rows"],
            "normalization_hz": float(B.Proprioceptors.R_MAX),
            "target_rate_min_hz": feco_13b_rate_min.tolist(),
            "target_rate_max_hz": feco_13b_rate_max.tolist(),
            "saturation_samples": feco_13b_saturation_samples.tolist(),
            "chemical_path_deleted": True,
            "continuous_membrane_boundary": True,
            "banc_edge_counts_are_pair_weight_proxy": True,
        }
    if graded_13b_command is not None:
        LAST_EXTRAS["graded_13b_command"] = {
            **({"ids_sha256": graded_command_ids_sha256}
               if graded_command_ids_sha256 is not None else {}),
            "mode": graded_13b_command["mode"],
            "target_rows": int(len(graded_13b_command["rows"])),
            "target_banc_ids": [str(meta.iloc[int(row)]["banc_888_id"])
                                for row in graded_13b_command["rows"]],
            "level_rest_to_threshold": graded_13b_command["level"],
            "start_ms": graded_13b_command["start_ms"],
            "stop_ms": graded_13b_command["stop_ms"],
            "active_body_ticks": int(graded_13b_command_active_ticks[0]),
            "continuous_membrane_command": True,
            "fixed_firing_threshold_or_delay_added": False,
        }
    if measured_step is not None:
        LAST_EXTRAS["measured_step_tape"] = {
            key: measured_step[key] for key in (
                "swing_frac", "adhesion_phase",
                "commanded_stance_fraction", "flex_sha256", "ext_sha256",
                "adhesion_sha256", "out_of_range_fraction",
                "front_rom_floor", "front_rom_floor_motion_gate")
        }
    if measured_step_neural_upstream_mix is not None:
        LAST_EXTRAS["measured_step_neural_upstream_mix"] = {
            key: (int(value) if key.endswith("ticks") else float(value))
            for key, value in measured_step_neural_upstream_mix.items()
        }
    if measured_step_neural_sign_gate is not None:
        _active = int(measured_step_neural_sign_gate["active_dof_ticks"])
        LAST_EXTRAS["measured_step_neural_sign_gate"] = {
            "mode": ("prestim_homeostatic_delta_per_dof_budget_preserving"
                     if measured_step_neural_sign_gate[
                         "homeostatic_baseline"] else
                     "per_dof_running_envelope_budget_preserving"),
            "route": measured_step_neural_sign_gate["route"],
            "homeostatic_baseline": bool(
                measured_step_neural_sign_gate["homeostatic_baseline"]),
            "baseline_ticks": int(
                measured_step_neural_sign_gate["baseline_ticks"]),
            "baseline_frozen": bool(
                measured_step_neural_sign_gate["baseline_frozen"]),
            "baseline_active_dofs": int(np.count_nonzero(
                measured_step_neural_sign_gate["baseline_flex_sum"]
                + measured_step_neural_sign_gate["baseline_ext_sum"])),
            "ticks": int(measured_step_neural_sign_gate["ticks"]),
            "active_dof_ticks": _active,
            "silent_dof_ticks": int(
                measured_step_neural_sign_gate["silent_dof_ticks"]),
            "direction_changed_dof_ticks": int(
                measured_step_neural_sign_gate[
                    "direction_changed_dof_ticks"]),
            "strength_min": (None if _active == 0 else float(
                measured_step_neural_sign_gate["strength_min"])),
            "strength_mean": (None if _active == 0 else float(
                measured_step_neural_sign_gate["strength_sum"] / _active)),
            "strength_max": (None if _active == 0 else float(
                measured_step_neural_sign_gate["strength_max"])),
            "neural_envelope_by_dof": (
                measured_step_neural_sign_gate["neural_envelope_by_dof"]
                .astype(float).tolist()),
            "max_dof_budget_abs_error": float(
                measured_step_neural_sign_gate[
                    "max_dof_budget_abs_error"]),
            "sum_dof_budget_abs_error": float(
                measured_step_neural_sign_gate[
                    "sum_dof_budget_abs_error"]),
            "applied_ticks_by_class0_class1_none": (
                measured_step_neural_sign_gate[
                    "applied_ticks_by_class0_class1_none"]
                .astype(int).tolist()),
            "bypassed_ticks_by_class0_class1_none": (
                measured_step_neural_sign_gate[
                    "bypassed_ticks_by_class0_class1_none"]
                .astype(int).tolist()),
        }
    if (th.get("graded") or th.get("cbased") or th.get("shunt")
            or th.get("gap") or th.get("delay_mode")
            or th.get("threshold_mode") or _serotonin_present
            or th.get("typed_12b_v_rest_mode")
            or th.get("typed_21a_inhibitory_gain_mode")
            or th.get("typed_19b_excitatory_gain_mode")
            or th.get("typed_21a_excitatory_gain_mode")
            or feco_13b_analog is not None
            or _adapt_a is not None):
        LAST_EXTRAS["neural_physics"] = _neural_physics_receipt(net)
        if _adapt_a is not None:
            LAST_EXTRAS["neural_physics"]["run_total_spikes"] = int(
                run_out["n_spikes"])
        if gap_pair_rule_receipt:
            LAST_EXTRAS["gap_pair_rules"] = [
                dict(item) for item in gap_pair_rule_receipt]
    if neural_motor_scale_by_class is not None:
        LAST_EXTRAS["neural_motor_scale_by_class"] = {
            "scales": list(neural_motor_scale_by_class),
            "ungated_scale": float(neural_motor_scale),
            "ticks_class0_class1_none": (
                neural_motor_gate_ticks.astype(int).tolist()),
        }
    if body_drive_leg_scale is not None:
        LAST_EXTRAS["body_drive_leg_gains"] = {
            "gains_lf_lm_lh_rf_rm_rh": [
                float(v) for v in th["body_drive_leg_gains"]],
            "final_flexor_and_extensor_boundary": True,
        }
    if adh_leg_threshold_scales is not None:
        LAST_EXTRAS["adh_leg_threshold_scales"] = {
            "scales_lf_lm_lh_rf_rm_rh": [
                float(v) for v in adh_leg_threshold_scales],
            "base_threshold": float(adh_lvl),
            "effective_thresholds": [
                float(v) for v in adh_lvls],
        }
    if prop_tonic_hz is not None:
        LAST_EXTRAS["prop_tonic_hz"] = prop_tonic_hz
        LAST_EXTRAS["prop_tonic_scope"] = prop_tonic_scope
        LAST_EXTRAS["prop_tonic_scope_n"] = (
            None if _prop_scope_mask[0] is None else int(_prop_scope_mask[0].sum()))
    if _prop_tonic_pops is not None:
        LAST_EXTRAS["prop_tonic_scope"] = dict(_prop_tonic_pops)
        LAST_EXTRAS["prop_tonic_scope_n"] = (
            None if _prop_scope_mask[0] is None
            else {_k: int(_m.sum()) for _k, _m in _prop_scope_mask[0].items()})
    # CD1-SI/SJ/SK readouts on every arm (the closed loop's delivered rate is
    # the quantity of interest on the BASE arm), written here because
    # LAST_EXTRAS is cleared after the simulation and before this block.
    LAST_EXTRAS["feco_v_scale"] = {"hook": prop.v_scale_hook, "club": prop.v_scale_club}
    if th.get("feco_r_rest_hook") is not None:
        LAST_EXTRAS["feco_r_rest_hook"] = prop.r_rest_hook
    if th.get("feco_r_rest_hair") is not None:
        LAST_EXTRAS["feco_r_rest_hair"] = prop.r_rest_hair
    LAST_EXTRAS["feco_claw_gain"] = prop.claw_gain
    LAST_EXTRAS["prop_rate_mean"] = (None if not _prop_rate_acc[1] else {
        _k: round(_mv[1] / _prop_rate_acc[1], 2)
        for _k, _mv in _prop_rate_acc[0].items() if _mv[0].any()})
    if cohp8_rows is not None:
        import motormap as _MM_COHP8
        _cohp8_denominator = max(
            1, cohp8_stats["ticks"] * len(cohp8_rows))
        LAST_EXTRAS["cohp8"] = {
            "rows": int(len(cohp8_rows)),
            "banc_888_ids": [
                str(meta.iloc[int(row)]["banc_888_id"])
                for row in cohp8_rows],
            "dof_names": [
                _MM_COHP8.DOF_NAMES[int(index)] for index in cohp8_dofs],
            "ticks": int(cohp8_stats["ticks"]),
            "mean_rate_hz": float(
                cohp8_stats["rate_sum_hz"] / _cohp8_denominator),
            "above_rest_cell_tick_fraction": float(
                cohp8_stats["above_rest_cell_ticks"] / _cohp8_denominator),
            "max_rate_hz": float(cohp8_stats["max_rate_hz"]),
            "rate_law": "5 + 120 * max(0, normalised_ThC_pitch - 0.7) / 0.3",
        }
    if hp_code is not None:
        import motormap as _MM_HPC
        _hpc_n = len(hp_code["rows"])
        _hpc_rate = np.asarray(hp_code["rate_tick"], dtype=np.float64).reshape(-1, _hpc_n)
        LAST_EXTRAS["hp_row_code"] = {
            "rows": len(hp_code["rows"]),
            "banc_888_ids": [str(meta.iloc[int(row)]["banc_888_id"]) for row in hp_code["rows"]],
            "dof_names": [_MM_HPC.DOF_NAMES[int(index)] for index in hp_code["dofs"]],
            "sides": [float(v) for v in hp_code["sides"]],
            "hinges": [float(v) for v in hp_code["hinges"]],
            "lag_ticks": hp_code["lag"],
            "ticks": int(_hpc_rate.shape[0]),
            "mean_rate_hz": float(_hpc_rate.mean()) if _hpc_rate.size else 0.0,
            "above_rest_cell_tick_fraction": float((_hpc_rate > 5.0).mean()) if _hpc_rate.size else 0.0,
            "max_rate_hz": float(_hpc_rate.max()) if _hpc_rate.size else 0.0,
            "rate_law": "side -1: 5 + min(120, 800 max(0, h - p)); side +1: 5 + min(120, 800 max(0, p - 0.85)); "
                        "side 0: 5 + 120 max(0, |2p - 1| - 0.7) / 0.3; p the normalised joint position "
                        "lag_ticks body ticks earlier (p_tick), p_now_tick the position at the tick",
            "p_now_tick": np.asarray(hp_code["p_now_tick"], dtype=np.float64).reshape(-1, _hpc_n),
            "p_tick": np.asarray(hp_code["p_tick"], dtype=np.float64).reshape(-1, _hpc_n),
            "rate_tick": _hpc_rate,
        }
    if adh_knee_legs is not None:
        LAST_EXTRAS["adh_knee_legs"] = {
            "legs": [leg for leg, on in zip(
                ["lf", "lm", "lh", "rf", "rm", "rh"], adh_knee_legs) if on],
            "knee_dof_index_lf_lm_lh_rf_rm_rh": [int(v) for v in adh_knee_dof],
            "thresholds": [float(v) for v in adh_lvls],
        }
    if body_joint_bias is not None:
        LAST_EXTRAS["body_joint_rest_offsets_deg"] = {
            str(name): float(value)
            for name, value in th["body_joint_rest_offsets_deg"].items()
        }
    if body_joint_tau_act is not None:
        LAST_EXTRAS["body_joint_tau_act_ms"] = {
            str(name): float(value)
            for name, value in th["body_joint_tau_act_ms"].items()
        }
    if body_tau_deact is not None:
        LAST_EXTRAS["tau_deact_ms"] = float(body_tau_deact)
    if body_joint_drive_scale is not None:
        LAST_EXTRAS["body_joint_drive_scale"] = {
            str(name): float(value)
            for name, value in th["body_joint_drive_scale"].items()
        }
    if body_joint_f0 is not None:
        LAST_EXTRAS["body_joint_f0"] = {
            str(name): float(value)
            for name, value in th["body_joint_f0"].items()
        }
    if _nt_patch_rows:
        # The receipt a reader grades the arm on: how many rows cns patched,
        # and each row's resolved fast sign before and after. Before is read
        # from the untouched meta row (the key refuses a row with a verified
        # transmitter, so the predicted column IS what the row resolved to);
        # after is read from the built network's own transmitter series.
        _p_sign = cns.SimParams().nt_sign
        _ids_r = meta["banc_888_id"].astype(str).to_numpy()
        _pred_r = meta["neurotransmitter_predicted"].astype(object).to_numpy()
        _after_r = np.asarray(net.nt.astype(object).to_numpy())
        _rows_r = {}
        for _rid in cell_type_patch:
            _w = int(np.flatnonzero(_ids_r == str(_rid))[0])
            _rows_r[str(_rid)] = {
                "cell_type": cell_type_patch[_rid]["cell_type"],
                "neurotransmitter_before": (None if _pred_r[_w] is None
                                            else str(_pred_r[_w])),
                "neurotransmitter_after": (None if _after_r[_w] is None
                                           else str(_after_r[_w])),
                "sign_before": _p_sign.get(str(_pred_r[_w])),
                "sign_after": _p_sign.get(str(_after_r[_w])),
                "citation": str(_nt_patch_rows[_rid].get("citation")),
            }
        LAST_EXTRAS["nt_patch_rows"] = {
            "nt_patch_rows_applied": int(getattr(net, "n_cell_type_patched", 0)),
            "rows": _rows_r}
    if balance_scale_override is not None:
        LAST_EXTRAS["balance_scale_override"] = {
            "balance_scale_frozen": bool(getattr(net, "balance_scale_frozen", False)),
            "file": str(th["balance_scale_override"]),
            "sha256": str(balance_scale_sha256),
            "rows": int(np.asarray(balance_scale_override).shape[0]),
            "n_balance_capped": int(getattr(net, "n_balance_capped", 0)),
        }
    if getattr(net.p, "release_resource_rows", None) is not None:
        _rrd, _rrc, _rrk = cns.release_resource_constants(
            net.p.release_resource_U, net.p.release_resource_tau_rec_ms,
            net.p.dt)
        LAST_EXTRAS["release_resource"] = {
            "rows": [int(_r) for _r in net.p.release_resource_rows],
            "U": [float(_u) for _u in net.p.release_resource_U],
            "tau_rec_ms": [float(_t) for _t in net.p.release_resource_tau_rec_ms],
            "d_bits": [f"0x{int(_b):08x}" for _b in _rrd.view(np.uint32)],
            "c_bits": [f"0x{int(_b):08x}" for _b in _rrc.view(np.uint32)],
            "k_bits": [f"0x{int(_b):08x}" for _b in _rrk.view(np.uint32)],
        }
        if getattr(net.p, "release_resource_target_rows", None) is not None:
            LAST_EXTRAS["release_resource"]["target_rows"] = [
                int(_r) for _r in net.p.release_resource_target_rows]
    if transmitter_map:
        LAST_EXTRAS["cell_type_patch"] = ({
            "rows_patched": int(getattr(net, "n_cell_type_patched", 0)),
            "rows": sorted(cell_type_patch)} if cell_type_patch else None)
        LAST_EXTRAS["transmitter_map"] = {
            "types": dict(sorted(transmitter_map.items())),
            "rows_overridden": int(net.n_transmitter_overridden),
            "sign_flips": int(net.n_transmitter_sign_flips),
        }
    if th.get("silence_rows") is not None:
        _thr_vec = np.asarray(net.p.v_threshold, dtype=np.float64)
        LAST_EXTRAS["silence_rows"] = {
            "requested": len(th["silence_rows"]),
            "silenced_rows": int(np.count_nonzero(_thr_vec >= 1e9)),
            "threshold_mv": 1e9,
        }
        if silence_window is not None:
            LAST_EXTRAS["silence_rows"]["window"] = dict(
                run_out["silence_window"],
                rows_sha16=rows_sha16(meta, silence_window["rows"]))
    if th.get("pulse_exc_ids") is not None and "g_op" not in run_out["opsin"]:
        # Design 4ee8b7f3, Build B1: an engine without the tap (the Mac's
        # Metal library) records the light schedule and the rows, and no
        # conductance; record_rows reads the pulsed rows' spikes.
        _pe_op = run_out["opsin"]
        _pe_v = _pulse_exc_values(th)
        _pe_sched, _pe_t = [], _pe_v["pulse_start_ms"]
        while _pe_t < _pe_v["pulse_stop_ms"]:
            _pe_sched.append([_pe_t, min(_pe_t + _pe_v["pulse_on_ms"],
                                         _pe_v["pulse_stop_ms"])])
            _pe_t += _pe_v["pulse_on_ms"] + _pe_v["pulse_off_ms"]
        _pe_rows = [int(r) for r in _pe_op["slot_rows"]]
        LAST_EXTRAS["pulse_exc"] = {
            **_pe_v,
            "pulse_exc_ids_sha256": th.get("pulse_exc_ids_sha256"),
            "rows": _pe_rows,
            "ids": [str(meta["banc_888_id"].iloc[r]) for r in _pe_rows],
            "graded_rows_listed": sorted(
                set(_pe_rows) & {int(r) for r in (
                    graded_rows if graded_rows is not None else ())}),
            "e_rev_mv": float(net.p.e_exc),
            "light_schedule_ms": _pe_sched,
            "readback": _pe_op["readback"],
            "tap": None,
        }
    elif th.get("pulse_exc_ids") is not None:
        # Design 2163ba7d, Build 1's output: the light schedule in ms and,
        # per listed row, the delivered conductance per trace bin (mean over
        # the bin's steps), in leak units, float32 (bins, rows).
        _pe_op = run_out["opsin"]
        _pe_dt = float(net.p.dt)
        _pe_bin = int(round(float(trace_bin_ms) / _pe_dt))
        _pe_g = np.asarray(_pe_op["g_op"], dtype=np.float64)
        _pe_nb = _pe_g.shape[0] // _pe_bin
        _pe_gbin = _pe_g[:_pe_nb * _pe_bin].reshape(
            _pe_nb, _pe_bin, _pe_g.shape[1]).mean(axis=1).astype(np.float32)
        _pe_v = _pulse_exc_values(th)
        _pe_sched, _pe_t = [], _pe_v["pulse_start_ms"]
        while _pe_t < _pe_v["pulse_stop_ms"]:
            _pe_sched.append([_pe_t, min(_pe_t + _pe_v["pulse_on_ms"],
                                         _pe_v["pulse_stop_ms"])])
            _pe_t += _pe_v["pulse_on_ms"] + _pe_v["pulse_off_ms"]
        _pe_rows = [int(r) for r in _pe_op["slot_rows"]]
        LAST_EXTRAS["pulse_exc"] = {
            **_pe_v,
            "pulse_exc_ids_sha256": th.get("pulse_exc_ids_sha256"),
            "rows": _pe_rows,
            "ids": [str(meta["banc_888_id"].iloc[r]) for r in _pe_rows],
            "graded_rows_listed": sorted(
                set(_pe_rows) & {int(r) for r in (
                    graded_rows if graded_rows is not None else ())}),
            "e_rev_mv": float(net.p.e_exc),
            "light_schedule_ms": _pe_sched,
            "readback": _pe_op["readback"],
            "tap_sha256": _pe_op.get("tap_sha256"),
            "bin_ms": float(trace_bin_ms),
            "g_bin": _pe_gbin,
            "g_bin_sha256": hashlib.sha256(_pe_gbin.tobytes()).hexdigest(),
            "s_final": [float(x) for x in np.asarray(_pe_op["s"])[-1]],
            "g_step_max": float(_pe_g.max()) if _pe_g.size else 0.0,
        }
    elif th.get("pulse_exc_sets") is not None:
        # Design 568d05a8, Build B5's output: the shared pulse keys and light
        # schedule, every set's g, hash, rows, ids and graded rows, the slot
        # map, and with the tap the delivered conductance per trace bin in
        # slot order (columns as "rows"); without it (the Mac's Metal
        # library) no conductance, as Build B1 of design 4ee8b7f3 records.
        _pe_op = run_out["opsin"]
        _pes, _pes_v = _pulse_exc_sets(th)
        _pe_sched, _pe_t = [], _pes_v["pulse_start_ms"]
        while _pe_t < _pes_v["pulse_stop_ms"]:
            _pe_sched.append([_pe_t, min(_pe_t + _pes_v["pulse_on_ms"],
                                         _pes_v["pulse_stop_ms"])])
            _pe_t += _pes_v["pulse_on_ms"] + _pes_v["pulse_off_ms"]
        _pe_rows = [int(r) for r in _pe_op["slot_rows"]]
        _pe_set = [int(j) for j in _pe_op["slot_set"]]
        _pe_graded = {int(r) for r in (graded_rows if graded_rows is not None else ())}
        _pe_sets = []
        for _j, _s in enumerate(_pes):
            _rj = [r for r, j in zip(_pe_rows, _pe_set) if j == _j]
            _pe_sets.append({
                "g": _s["g"], "ids_sha256": _s["ids_sha256"], "rows": _rj,
                "ids": [str(meta["banc_888_id"].iloc[r]) for r in _rj],
                "graded_rows_listed": sorted(set(_rj) & _pe_graded)})
        LAST_EXTRAS["pulse_exc"] = {
            **_pes_v,
            "sets": _pe_sets,
            "rows": _pe_rows,
            "slot_set": _pe_set,
            "e_rev_mv": float(net.p.e_exc),
            "light_schedule_ms": _pe_sched,
            "readback": _pe_op["readback"],
        }
        if "g_op" not in _pe_op:
            LAST_EXTRAS["pulse_exc"]["tap"] = None
        else:
            _pe_dt = float(net.p.dt)
            _pe_bin = int(round(float(trace_bin_ms) / _pe_dt))
            _pe_g = np.asarray(_pe_op["g_op"], dtype=np.float64)
            _pe_nb = _pe_g.shape[0] // _pe_bin
            _pe_gbin = _pe_g[:_pe_nb * _pe_bin].reshape(
                _pe_nb, _pe_bin, _pe_g.shape[1]).mean(axis=1).astype(np.float32)
            LAST_EXTRAS["pulse_exc"].update({
                "tap_sha256": _pe_op.get("tap_sha256"),
                "bin_ms": float(trace_bin_ms),
                "g_bin": _pe_gbin,
                "g_bin_sha256": hashlib.sha256(_pe_gbin.tobytes()).hexdigest(),
                "s_final": [float(x) for x in np.asarray(_pe_op["s"])[-1]],
                "g_step_max": float(_pe_g.max()) if _pe_g.size else 0.0,
            })
    if th.get("slow_inh_out") is not None:
        # Design 2163ba7d, Build 2's output: the engine's out["slow_inh"]
        # (the receipt, the delivery check's conductance-times and the record
        # g_bin, float32 (5 ms bins, front-leg motor neurons)) with the ids
        _si_out = dict(run_out["slow_inh"])
        _si_v = _slow_inh_spec(th)
        LAST_EXTRAS["slow_inh_out"] = {
            **_si_out, "ids": _si_v["ids"], "ids_sha256": _si_v["ids_sha256"],
            "record_ids": [str(meta["banc_888_id"].iloc[r])
                           for r in _si_out["record_rows"]]}
    if th.get("adapt_zero_ids") is not None:
        _az_a = np.asarray(net.p.adapt_a, dtype=np.float64)
        _az_b = np.asarray(net.p.adapt_b, dtype=np.float64)
        LAST_EXTRAS["adapt_zero_ids"] = {
            "requested": len(th["adapt_zero_ids"]),
            "adapt_a_zero_rows": int(np.count_nonzero(_az_a == 0.0)),
            "adapt_b_zero_rows": int(np.count_nonzero(_az_b == 0.0)),
            "adapt_a_is_vector": _az_a.ndim == 1 and _az_a.size > 1,
            "adapt_b_is_vector": _az_b.ndim == 1 and _az_b.size > 1,
        }
    if th.get("mn_membrane_by_id") is not None:
        LAST_EXTRAS["mn_membrane_by_id"] = dict(net.p.mn_membrane_by_id_audit)
    if th.get("adapt_law_rows") is not None:
        LAST_EXTRAS["adapt_law_rows"] = dict(net.p.adapt_law_rows_audit)
    if th.get("r_in_total_by_id") is not None:
        LAST_EXTRAS["r_in_total_by_id"] = r_in_total_by_id_readback(
            net, th["r_in_total_by_id"], r_in_law=th.get("r_in_law"),
            w_syn_scale=th.get("w_syn_scale"))
    if th.get("row_membrane_by_id") is not None:
        LAST_EXTRAS["row_membrane_by_id"] = {
            **dict(net.p.row_membrane_by_id_audit),
            "readback": row_membrane_by_id_readback(net, th["row_membrane_by_id"])}
    if th.get("mn_exact_by_id") is not None:          # design 83177ced Build 1, after the walk
        LAST_EXTRAS["mn_exact_by_id"] = {
            **dict(net.p.mn_exact_by_id_audit),
            "readback": mn_exact_by_id_readback(net, th["mn_exact_by_id"])}
    if th.get("mn_reset_offset_by_id") is not None:   # study 9e0afa22 Build 1, after the walk
        _ro_eng = getattr(net, "_metal_engine", None)
        LAST_EXTRAS["mn_reset_offset_by_id"] = mn_reset_offset_by_id_check(
            net.p, net.ids, None if _ro_eng is None else _ro_eng._keep)
    if th.get("mn_vrest_by_id") is not None:          # study 7f1b4a1e B1, after the walk
        _vr_eng = getattr(net, "_metal_engine", None)
        LAST_EXTRAS["mn_vrest_by_id"] = mn_vrest_by_id_readback(
            net.p, net.ids, None if _vr_eng is None else _vr_eng._keep)
    if th.get("holddown_by_id") is not None:
        LAST_EXTRAS["holddown_by_id"] = {
            **dict(net.p.holddown_by_id_audit),
            "readback": holddown_by_id_readback(net)}
    if th.get("crossing_tau_w_mult") is not None:
        _ct_ix = np.asarray(net.p.crossing_tau_w_row_index)
        _ct_n = len(meta)
        _ct_tw = np.broadcast_to(np.asarray(net.p.tau_w, dtype=np.float32), (_ct_n,))
        _ct_b = np.broadcast_to(np.asarray(net.p.adapt_b, dtype=np.float32), (_ct_n,))
        _ct_off = np.ones(_ct_n, dtype=bool)
        _ct_off[_ct_ix] = False
        _ct_area = _ct_b[_ct_ix].astype(np.float64) * _ct_tw[_ct_ix].astype(np.float64)
        _ct_area_off = (_ct_b[_ct_off].astype(np.float64)
                        * _ct_tw[_ct_off].astype(np.float64))
        LAST_EXTRAS["crossing_tau_w_mult"] = {
            "mult": float(net.p.crossing_tau_w_mult),
            "rows_dosed": int(net.p.crossing_tau_w_rows),
            "rows_sha256_16": hashlib.sha256(",".join(sorted(
                str(r) for r in th["crossing_tau_w_mult"]["rows"])).encode()).hexdigest()[:16],
            "tau_w_scalar": float(net.p.crossing_tau_w_scalar),
            "tau_w_dosed": sorted({float(x) for x in _ct_tw[_ct_ix]}),
            "tau_w_elsewhere": sorted({float(x) for x in _ct_tw[_ct_off]}),
            "hold_b_tau": bool(getattr(net.p, "crossing_tau_w_hold_b_tau", False)),
            "adapt_b_dosed": sorted({float(x) for x in _ct_b[_ct_ix]}),
            "adapt_b_elsewhere": sorted({float(x) for x in _ct_b[_ct_off]}),
            "b_tau_dosed_min": float(_ct_area.min()),
            "b_tau_dosed_max": float(_ct_area.max()),
            "b_tau_elsewhere_min": float(_ct_area_off.min()),
            "b_tau_elsewhere_max": float(_ct_area_off.max()),
        }
    if th.get("cmd_presyn_gain") is not None:
        LAST_EXTRAS["cmd_presyn_gain"] = {
            "cmd": str(th.get("cmd")),
            "gain": float(th["cmd_presyn_gain"]),
            "rows": int(len(getattr(net.p, "cmd_presyn_gain_ids", []) or [])),
            "ids": list(getattr(net.p, "cmd_presyn_gain_ids", []) or []),
        }
    if type_out_gain_vec is not None:
        LAST_EXTRAS["type_out_gain"] = {
            "type_map": dict(_type_out_gain),
            "id_map": dict(_type_out_gain_id),
            "type_rows": dict((type_out_gain_audit or {}).get("type_rows") or {}),
            "id_rows": list((type_out_gain_audit or {}).get("id_rows") or []),
            "edges": int(getattr(net, "n_type_out_gain_edges", 0)),
        }
    if th.get("thc_yaw_moment_arm_orientation") is not None:
        import motormap as _MM_TY
        _yaw_names = [name for name in _MM_TY.DOF_NAMES
                      if name.endswith("_ThC_yaw")]
        LAST_EXTRAS["thc_yaw_moment_arm_orientation"] = {
            "mode": th["thc_yaw_moment_arm_orientation"],
            "flipped_dofs": _yaw_names,
            "flipped_dof_count": len(_yaw_names),
            "applied_to_final_motor_map": True,
            "evidence_status": "HYPOTHESIS_UNRESOLVED_ABSOLUTE_SIGN",
        }
    if th.get("thc_roll_moment_arm_orientation") is not None:
        import motormap as _MM_TR
        _roll_names = [name for name in _MM_TR.DOF_NAMES
                       if name.endswith("_ThC_roll")]
        LAST_EXTRAS["thc_roll_moment_arm_orientation"] = {
            "mode": th["thc_roll_moment_arm_orientation"],
            "flipped_dofs": _roll_names,
            "flipped_dof_count": len(_roll_names),
            "applied_to_final_motor_map": True,
            "evidence_status": "HYPOTHESIS_UNRESOLVED_ABSOLUTE_SIGN",
        }
    if neural_motor_right_class_dof_substitution is not None:
        config = neural_motor_right_class_dof_substitution
        teacher = float(config["teacher_abs_drive"])
        LAST_EXTRAS["neural_motor_right_class_dof_substitution"] = {
            "mode": (
                "right_plus_detector_zero_per_dof_budget_preserving"
                if config["detector_zero"] else
                "right_class_per_dof_budget_preserving"),
            "ticks_class0_class1_none": config[
                "ticks_class0_class1_none"].astype(int).tolist(),
            "eligible_right_ticks": int(config["eligible_right_ticks"]),
            "eligible_detector_zero_ticks": int(
                config["eligible_detector_zero_ticks"]),
            "substituted_dof_ticks": int(config["substituted_dof_ticks"]),
            "fallback_dof_ticks": int(config["fallback_dof_ticks"]),
            "zero_budget_dof_ticks": int(config["zero_budget_dof_ticks"]),
            "offered_neural_abs_drive": float(
                config["offered_neural_abs_drive"]),
            "teacher_abs_drive": teacher,
            "substituted_teacher_abs_drive": float(
                config["substituted_teacher_abs_drive"]),
            "authority_fraction": (
                0.0 if teacher == 0.0 else float(
                    config["substituted_teacher_abs_drive"]) / teacher),
            "final_abs_drive": float(config["final_abs_drive"]),
            "max_dof_budget_abs_error": float(
                config["max_dof_budget_abs_error"]),
            "sum_dof_budget_abs_error": float(
                config["sum_dof_budget_abs_error"]),
        }
    if visual_teacher_hs_thc_yaw_phase is not None:
        LAST_EXTRAS["visual_teacher_hs_thc_yaw_phase"] = (
            _visual_teacher_hs_thc_yaw_phase_receipt(
                visual_teacher_hs_thc_yaw_phase))
    if visual_relay_mode:
        LAST_EXTRAS["visual_relay"] = {
            "mode": visual_bridge["mode"],
            "input_source": visual_bridge["input_source"],
            "relay_code": visual_bridge["relay_code"],
            "input_rows": visual_bridge["input_rows"].astype(int).tolist(),
            "input_ids": [str(net.ids[row])
                          for row in visual_bridge["input_rows"]],
            "input_sides": visual_bridge["input_sides"].astype(int).tolist(),
            "dna02_rows": visual_bridge["rows"].astype(int).tolist(),
            "dna02_ids": [str(net.ids[row]) for row in visual_bridge["rows"]],
            "dna02_counts": visual_relay_counts.astype(int).tolist(),
            "dna02_stimulus_counts": (
                visual_relay_stimulus_counts.astype(int).tolist()),
            "hs_feedback_counts": (
                visual_relay_hs_counts.astype(
                    float if visual_hs_graded_mode else int).tolist()),
            "hs_feedback_stimulus_counts": (
                visual_relay_hs_stimulus_counts.astype(
                    float if visual_hs_graded_mode else int).tolist()),
            "hs_feedback_kind": (
                "mean_release_per_tick"
                if visual_hs_graded_mode else "spike_count_per_window"),
            "hs_input_boundary": (
                ("continuous_cell_relative_signed_membrane"
                 if visual_bridge["hs_signed_drive"]
                 else "continuous_cell_relative_membrane")
                if visual_direct_hs_clamp_mode else
                "poisson_spiking_actual_banc_t4t5_to_graded_hs"
                if visual_flyvis_t4t5_tape is not None else
                "poisson_spike"),
            "hs_input_signed_drive": bool(
                visual_bridge["hs_signed_drive"]),
            "hs_input_clamp": (
                _graded_drive_clamp_receipt(
                    net, visual_bridge["input_rows"],
                    visual_relay_command_min, visual_relay_command_max,
                    visual_relay_saturation_samples)
                if visual_direct_hs_clamp_mode else None),
            "direct_rendered_detector_to_hs_boundary": bool(
                visual_hs_graded_mode
                and visual_bridge["input_source"] == _RENDERED_VISUAL_INPUT),
            "direct_boundary_limitation": (
                "rendered_reichardt_detector_commands_HS_membrane_directly"
                if (visual_hs_graded_mode and visual_bridge["input_source"]
                    == _RENDERED_VISUAL_INPUT) else None),
            "hs_input_rows_spike_suppressed": bool(
                visual_direct_hs_clamp_mode),
            "hs_clamped_rows_excluded_from_spike_rng": bool(
                visual_direct_hs_clamp_mode),
            "hs_decoder_magnitude_threshold": (
                None if visual_hs_graded_mode else "mode_specific"),
            "hs_decoder_added_delay_ms": (
                0.0 if visual_hs_graded_mode else None),
            "decoded_class_ticks_by_side": (
                visual_relay_decoded_class_ticks.astype(int).tolist()),
            "raw_hs_decoded_class_ticks_by_side": (
                visual_relay_raw_hs_class_ticks.astype(int).tolist()),
            "raw_hs_decoded_class_ticks_by_epoch_side": (
                visual_relay_raw_hs_class_ticks_by_epoch.astype(int).tolist()),
            "raw_hs_detector_agreement_ticks": int(
                visual_relay_raw_hs_detector_agreement_ticks),
            "raw_hs_detector_mismatch_ticks": int(
                visual_relay_raw_hs_detector_mismatch_ticks),
            "final_ema": visual_relay_ema.astype(float).tolist(),
            "gate_ticks_by_side": visual_relay_gate_ticks.astype(int).tolist(),
            "teacher_gate_ticks_by_side": (
                visual_teacher_gate_ticks.astype(int).tolist()),
            "teacher_steering_legs": visual_bridge["teacher_steering_legs"],
            "teacher_phase_asymmetry": (
                visual_bridge["teacher_phase_asymmetry"]),
            "hs_forward_speed_phase_gain": bool(
                visual_hs_forward_speed_mode),
            "hs_forward_speed_max": float(
                visual_hs_forward_speed_max),
            "hs_forward_speed_mean": (
                None if visual_hs_forward_speed_samples == 0 else float(
                    visual_hs_forward_speed_sum
                    / visual_hs_forward_speed_samples)),
            "hs_forward_speed_strength_min_motion": (
                None if visual_hs_forward_speed_motion_ticks == 0 else float(
                    visual_hs_forward_speed_strength_min)),
            "hs_forward_speed_strength_max_motion": (
                None if visual_hs_forward_speed_motion_ticks == 0 else float(
                    visual_hs_forward_speed_strength_max)),
            "hs_forward_speed_strength_mean_motion": (
                None if visual_hs_forward_speed_motion_ticks == 0 else float(
                    visual_hs_forward_speed_strength_sum
                    / visual_hs_forward_speed_motion_ticks)),
            "hs_forward_speed_motion_ticks": int(
                visual_hs_forward_speed_motion_ticks),
            "hs_forward_speed_no_motion_ticks": int(
                visual_hs_forward_speed_no_motion_ticks),
            "teacher_leg_phase_asymmetry": (
                visual_bridge["teacher_leg_phase_asymmetry"]),
            "teacher_phase_scope": (
                visual_teacher_phase["scope"]
                if visual_teacher_phase is not None else None),
            "teacher_phase_offsets_rad": (
                visual_teacher_phase["phase_offsets"].astype(float).tolist()
                if visual_teacher_phase is not None else [0.0, 0.0]),
            "teacher_phase_asymmetric_ticks": (
                int(visual_teacher_phase["asymmetric_ticks"])
                if visual_teacher_phase is not None else 0),
            "teacher_phase_decoded_ticks_by_side": (
                visual_teacher_phase["decoded_ticks"].astype(int).tolist()
                if visual_teacher_phase is not None else [0, 0]),
            "teacher_phase_memoryless_ticks": (
                int(visual_teacher_phase["memoryless_ticks"])
                if visual_teacher_phase is not None else 0),
            "teacher_phase_memoryless_target_abs_min_rad": (
                None if visual_teacher_phase is None
                or visual_teacher_phase["memoryless_ticks"] == 0
                else float(visual_teacher_phase[
                    "memoryless_target_abs_min_rad"])),
            "teacher_phase_memoryless_target_abs_max_rad": (
                None if visual_teacher_phase is None
                or visual_teacher_phase["memoryless_ticks"] == 0
                else float(visual_teacher_phase[
                    "memoryless_target_abs_max_rad"])),
            "phase_reset_on_no_motion": (
                visual_bridge["phase_reset_on_no_motion"]),
            "phase_from_detector": visual_bridge["phase_from_detector"],
            "phase_neural_motion_control": (
                visual_bridge["phase_neural_motion_control"]),
            "phase_neutral_reset_ticks_by_segment": (
                visual_phase_reset_ticks_by_segment.astype(int).tolist()),
            "phase_brake_on_no_motion": (
                visual_bridge["phase_brake_on_no_motion"]),
            "phase_brake_deadband_deg_s": (
                visual_bridge["phase_brake_deadband_deg_s"]),
            "phase_brake_proportional_cap": (
                visual_bridge["phase_brake_proportional_cap"]),
            "phase_brake_strength_sum": float(
                visual_phase_brake_strength_sum),
            "phase_brake_strength_max": float(
                visual_phase_brake_strength_max),
            "phase_brake_filter_body_rate": (
                visual_bridge["phase_brake_filter_body_rate"]),
            "phase_brake_proportional_body_rate": (
                visual_bridge["phase_brake_proportional_body_rate"]),
            "phase_brake_memoryless_body_rate": (
                visual_bridge["phase_brake_memoryless_body_rate"]),
            "phase_brake_fixed_deadband_active": bool(
                not visual_bridge["phase_brake_proportional_body_rate"]),
            "phase_brake_filter_tau_ms": (
                visual_bridge["relay_tau_ms"]
                if visual_bridge["phase_brake_filter_body_rate"] else None),
            "phase_brake_filter_ticks": int(
                visual_phase_brake_filter_ticks),
            "phase_brake_filter_changed_ticks": int(
                visual_phase_brake_filter_changed_ticks),
            "phase_brake_filtered_abs_yaw_rate_max_deg_s": float(
                visual_phase_brake_filtered_abs_yaw_rate_max),
            "phase_brake_proportional_envelope_max_deg_s": float(
                visual_phase_brake_proportional_envelope_max),
            "phase_brake_proportional_strength_ticks": int(
                visual_phase_brake_proportional_strength_ticks),
            "phase_brake_proportional_strength_min": (
                None if visual_phase_brake_proportional_strength_ticks == 0
                else float(visual_phase_brake_proportional_strength_min)),
            "phase_brake_proportional_strength_max": (
                None if visual_phase_brake_proportional_strength_ticks == 0
                else float(visual_phase_brake_proportional_strength_max)),
            "phase_brake_proportional_strength_mean": (
                None if visual_phase_brake_proportional_strength_ticks == 0
                else float(visual_phase_brake_proportional_strength_sum)
                / float(visual_phase_brake_proportional_strength_ticks)),
            "phase_brake_ticks_by_class": (
                visual_phase_brake_ticks_by_class.astype(int).tolist()),
            "phase_brake_ticks_by_segment_class": (
                visual_phase_brake_ticks_by_segment_class.astype(int).tolist()),
            "phase_brake_neutral_reset_ticks": int(
                visual_phase_brake_neutral_reset_ticks),
            "phase_brake_opposition_violation_ticks": int(
                visual_phase_brake_opposition_violation_ticks),
            "phase_brake_static_abs_yaw_rate_max_deg_s": float(
                visual_phase_brake_static_abs_yaw_rate_max),
            "input_ticks_by_side": visual_relay_input_ticks.astype(int).tolist(),
            "negative_input_ticks_by_side": (
                visual_relay_negative_input_ticks.astype(int).tolist()),
            "input_rate_hz_ticks_by_side": (
                visual_relay_rate_sums.astype(float).tolist()),
            "input_abs_rate_hz_ticks_by_side": (
                visual_relay_abs_rate_sums.astype(float).tolist()),
            "input_signed_command_ticks_by_side": (
                None if float(visual_bridge["rate_hz"]) == 0.0 else
                (visual_relay_rate_sums
                 / float(visual_bridge["rate_hz"])).astype(float).tolist()),
            "input_abs_command_ticks_by_side": (
                None if float(visual_bridge["rate_hz"]) == 0.0 else
                (visual_relay_abs_rate_sums
                 / float(visual_bridge["rate_hz"])).astype(float).tolist()),
            "prestim_baseline_ticks": int(visual_relay_baseline_ticks),
            "hs_prestim_baseline_ticks": int(visual_relay_hs_baseline_ticks),
            "hs_prestim_baseline_mean_counts_per_tick": (
                visual_relay_hs_baseline_mean.astype(float).tolist()),
            "hs_opponent_prestim_baseline": visual_bridge[
                "hs_opponent_prestim_baseline"],
            "prestim_baseline_count_sum": (
                visual_relay_baseline_sum.astype(int).tolist()),
            "prestim_baseline_mean_counts_per_tick": (
                visual_relay_baseline_mean.astype(float).tolist()),
            "first_gate_ms": visual_relay_first_gate_ms,
            "teacher_first_gate_ms": visual_teacher_first_gate_ms,
            "gate_ticks_at_or_before_500ms": int(
                visual_relay_early_gate_ticks),
            "direct_dna02_injection": False,
            **({
                "flyvis_hss_only": visual_bridge["flyvis_hss_only"],
                "flyvis_hs_input_cell_types": (
                    meta.iloc[visual_bridge["input_rows"]]["cell_type"]
                    .astype(str).tolist()),
            } if visual_bridge["input_source"]
                 == _FLYVIS_HS_TAPE_VISUAL_INPUT else {}),
            **({"carrier_drive": visual_carrier_receipt}
               if visual_carrier_receipt is not None else {}),
            **({
                "input_groups": visual_bridge["flyvis_t4t5_groups"],
                "input_label_source": visual_bridge["flyvis_t4t5_label_source"],
                "input_stimulus_from_drive": visual_bridge[
                    "flyvis_t4t5_stimulus_from_drive"],
                "input_group_counts": np.bincount(
                    visual_bridge["input_group_indices"],
                    minlength=len(visual_bridge["flyvis_t4t5_groups"])
                ).astype(int).tolist(),
                "input_rows_graded_count": int(
                    np.asarray(net.graded, dtype=bool)[
                        visual_bridge["input_rows"]].sum()),
                "hs_feedback_rows": (
                    visual_bridge["hs_rows"].astype(int).tolist()),
                "hs_feedback_ids": [
                    str(net.ids[row]) for row in visual_bridge["hs_rows"]],
                "hs_feedback_sides": (
                    visual_bridge["hs_sides"].astype(int).tolist()),
                "direct_hs_injection": False,
                "offline_flyvis_tape": True,
                "tape_side_body_gate": False,
                "live_graded_hs_owns_body_phase": True,
                "tape_injection_scope": "all_callback_ticks",
                "body_callback_tick_ms": float(tick),
                "body_phase_time_gate": None,
                "body_phase_magnitude_threshold": None,
                "raw_hs_decoded_class_ticks_by_tape_segment": (
                    visual_relay_raw_hs_class_ticks_by_tape_segment
                    .astype(int).tolist()),
                "raw_hs_decoded_class_ticks_by_body_window": (
                    visual_relay_raw_hs_class_ticks_by_body_window
                    .astype(int).tolist()),
                "flyvis_t4t5_tape": {
                    "source_model": visual_flyvis_t4t5_tape["source_model"],
                    "movie_sha256": visual_flyvis_t4t5_tape["movie_sha256"],
                    "rate_tape_sha256": visual_flyvis_t4t5_tape[
                        "rate_tape_sha256"],
                    "tick_ms": visual_bridge[
                        "flyvis_t4t5_rate_tape_tick_ms"],
                    "ticks": int(visual_flyvis_t4t5_tape["ticks"]),
                    "tape_rows_consumed": int(
                        visual_flyvis_t4t5_tape["tape_rows_consumed"]),
                    "last_tape_index": visual_flyvis_t4t5_tape[
                        "last_tape_index"],
                    "unconsumed_tail_rows": int(
                        len(visual_bridge["flyvis_t4t5_rate_tape"])
                        - visual_flyvis_t4t5_tape["tape_rows_consumed"]),
                    "nominal_motion_window_ms": [500.0, 5500.0],
                    "stimulus_ticks": int(
                        visual_flyvis_t4t5_tape["stimulus_ticks"]),
                    "nonzero_ticks_inside_window": int(
                        visual_flyvis_t4t5_tape[
                            "nonzero_ticks_inside_window"]),
                    "nonzero_ticks_outside_window": int(
                        visual_flyvis_t4t5_tape[
                            "nonzero_ticks_outside_window"]),
                    "positive_ticks_by_group": (
                        visual_flyvis_t4t5_tape["positive_ticks_by_group"]
                        .astype(int).tolist()),
                    "rate_sum_hz_by_group": (
                        visual_flyvis_t4t5_tape["rate_sum_hz_by_group"]
                        .astype(float).tolist()),
                    "rate_sum_hz_by_group_inside_window": (
                        visual_flyvis_t4t5_tape[
                            "rate_sum_hz_by_group_inside_window"]
                        .astype(float).tolist()),
                    "rate_sum_hz_by_group_outside_window": (
                        visual_flyvis_t4t5_tape[
                            "rate_sum_hz_by_group_outside_window"]
                        .astype(float).tolist()),
                    "max_rate_hz_by_group": (
                        visual_flyvis_t4t5_tape["max_rate_hz_by_group"]
                        .astype(float).tolist()),
                    "selected_ticks_by_side": (
                        visual_flyvis_t4t5_tape["selected_ticks_by_side"]
                        .astype(int).tolist()),
                    "selected_ticks_by_epoch_side": (
                        visual_flyvis_t4t5_tape[
                            "selected_ticks_by_epoch_side"]
                        .astype(int).tolist()),
                    "zero_or_tied_ticks": int(
                        visual_flyvis_t4t5_tape["zero_or_tied_ticks"]),
                },
            } if visual_flyvis_t4t5_tape is not None else {}),
            "phase_decoder": (
                {
                    "source": "rendered_detector_sign",
                    "feedback_includes_hs_counts": False,
                    "feedback_includes_dna02_counts": False,
                } if visual_bridge["phase_from_detector"] else None),
            "dna02_phase_amplitude": {
                "enabled": bool(visual_bridge["phase_dna02_amplitude"]),
                "mode": visual_bridge["phase_dna02_amplitude_mode"],
                "feedback_includes_dna02_counts": bool(
                    visual_bridge["phase_dna02_amplitude"]),
                "normalization": (
                    "matched_side_refractory_capacity"
                    if visual_bridge["phase_dna02_amplitude_mode"] == "routed"
                    else "summed_per_cell_refractory_capacity"),
                "max_rate_hz": visual_bridge["dna02_max_rate_hz"],
                "max_rate_hz_by_side": visual_bridge[
                    "dna02_max_rate_hz_by_side"],
                "final_ema_counts_per_tick": (
                    visual_relay_dna02_ema.astype(float).tolist()),
            },
            "dna02_measured_step_return_gate": {
                "enabled": bool(visual_bridge[
                    "dna02_measured_step_return_gate"]),
                "source": "hs_matched_dna02_refractory_normalized_rate",
                "target": "ipsilateral_unloaded_measured_step_drive",
                "ticks_by_side": (
                    visual_dna02_step_return_ticks.astype(int).tolist()),
                "minimum_scale_by_side": (
                    visual_dna02_step_return_scale_min.astype(float).tolist()),
            },
            "rendered_image": (
                None if visual_image_motion is None else {
                    "width_px": 72,
                    "height_px": 24,
                    "spatial_cycles": 6.0,
                    "temporal_hz": visual_bridge["image_motion_hz"],
                    "motion_profile": visual_bridge["image_motion_profile"],
                    "body_reafference": visual_bridge[
                        "image_body_reafference"],
                    "reversal_ms": (
                        3000.0 if visual_bridge["image_motion_profile"]
                        == _SINGLE_REVERSAL_IMAGE_MOTION_PROFILE else None),
                    "contrast": visual_bridge["contrast"],
                    "rendered_frames": int(
                        visual_image_motion["rendered_frames"]),
                    "stimulus_frames": int(
                        visual_image_motion["stimulus_frames"]),
                    "stimulus_frames_by_epoch": (
                        visual_image_motion["stimulus_frames_by_epoch"]
                        .astype(int).tolist()),
                    **({} if "frames_by_segment" not in visual_image_motion
                        else {
                            "frames_by_segment": (
                                visual_image_motion["frames_by_segment"]
                                .astype(int).tolist()),
                            "selected_ticks_by_segment_side": (
                                visual_image_motion[
                                    "selected_ticks_by_segment_side"]
                                .astype(int).tolist()),
                            "zero_ticks_by_segment": (
                                visual_image_motion["zero_ticks_by_segment"]
                                .astype(int).tolist()),
                        }),
                    "detector_positive_ticks": int(
                        visual_image_motion["detector_positive_ticks"]),
                    "detector_negative_ticks": int(
                        visual_image_motion["detector_negative_ticks"]),
                    "detector_zero_ticks": int(
                        visual_image_motion["detector_zero_ticks"]),
                    "selected_ticks_by_side": (
                        visual_image_motion["selected_ticks"]
                        .astype(int).tolist()),
                    "selected_ticks_by_epoch_side": (
                        visual_image_motion["selected_ticks_by_epoch_side"]
                        .astype(int).tolist()),
                    "detector_score_sum": float(
                        visual_image_motion["score_sum"]),
                    "detector_score_abs_max": float(
                        visual_image_motion["score_abs_max"]),
                    "detector_last_score": float(
                        visual_image_motion["last_score"]),
                    "body_yaw_samples": int(
                        visual_image_motion["body_yaw_samples"]),
                    "body_yaw_min_rad": visual_image_motion[
                        "body_yaw_min_rad"],
                    "body_yaw_max_rad": visual_image_motion[
                        "body_yaw_max_rad"],
                    "body_phase_contribution_abs_max_cycles": float(
                        visual_image_motion[
                            "body_phase_contribution_abs_max_cycles"]),
                    "retinal_slip_nonzero_ticks": int(
                        visual_image_motion["retinal_slip_nonzero_ticks"]),
                    "detector_retinal_slip_match_ticks": int(
                        visual_image_motion[
                            "detector_retinal_slip_match_ticks"]),
                    "detector_retinal_slip_mismatch_ticks": int(
                        visual_image_motion[
                            "detector_retinal_slip_mismatch_ticks"]),
                    "retinal_slip_opposes_external_by_epoch": (
                        visual_image_motion[
                            "retinal_slip_opposes_external_by_epoch"]
                        .astype(int).tolist()),
                    **({} if not visual_bridge[
                        "image_motion_cancellation"] else {
                        "body_motion_cancellation": True,
                        "retinal_delta_abs_max_cycles": float(
                            visual_image_motion[
                                "retinal_delta_abs_max_cycles"]),
                        "body_delta_abs_max_cycles": float(
                            visual_image_motion[
                                "body_delta_abs_max_cycles"]),
                        "corrected_delta_abs_max_cycles": float(
                            visual_image_motion[
                                "corrected_delta_abs_max_cycles"]),
                        "raw_retinal_selected_ticks_by_side": (
                            visual_image_motion[
                                "raw_retinal_selected_ticks"]
                            .astype(int).tolist()),
                        "compensation_changed_ticks": int(
                            visual_image_motion[
                                "compensation_changed_ticks"]),
                    }),
                    "frame_sha256": visual_image_motion.get(
                        "frame_sha256",
                        visual_image_motion["frame_digest"].hexdigest()),
                    **({} if "source_model" not in visual_image_motion else {
                        "source_model": visual_image_motion["source_model"],
                        "side_tape_sha256": (
                            visual_image_motion["side_tape_sha256"]),
                        "side_tape_tick_ms": (
                            visual_bridge["flyvis_side_tape_tick_ms"]),
                    }),
                }),
        }
    if (visual_teacher_phase is not None
            and (visual_teacher_phase.get(
                "hs_outer_load_phase_slowdown", 0.0) > 0.0
                 or visual_teacher_phase.get(
                    "hs_outer_load_phase_self_normalized", False))):
        _nominal_phase_step = (
            2.0 * np.pi * float(measured_step["tick_ms"])
            / (len(measured_step["flex"]) * float(measured_step["tick_ms"])))
        _self_normalized = bool(visual_teacher_phase.get(
            "hs_outer_load_phase_self_normalized", False))
        _slow_phase_step = (None if _self_normalized else
            _nominal_phase_step * (1.0 - float(
                visual_teacher_phase["hs_outer_load_phase_slowdown"])))
        _strength_ticks = int(
            visual_teacher_phase["hs_outer_load_strength_ticks"])
        LAST_EXTRAS["measured_step_hs_outer_load_phase_slowdown"] = {
            "slowdown": float(
                visual_teacher_phase["hs_outer_load_phase_slowdown"]),
            "loaded_phase_rate_fraction": (
                None if _self_normalized else float(
                    1.0 - visual_teacher_phase[
                        "hs_outer_load_phase_slowdown"])),
            "contact_threshold_N": (
                None if _self_normalized else float(
                    visual_teacher_phase["hs_outer_load_threshold_N"])),
            **({} if not _self_normalized else {
                "mode": "per_leg_running_envelope",
                "load_envelope_N_by_leg": (
                    visual_teacher_phase["hs_outer_load_envelope_N_by_leg"]
                    .astype(float).tolist()),
                "strength_min": (None if _strength_ticks == 0 else float(
                    visual_teacher_phase["hs_outer_load_strength_min"])),
                "strength_mean": (None if _strength_ticks == 0 else float(
                    visual_teacher_phase["hs_outer_load_strength_sum"]
                    / _strength_ticks)),
                "strength_max": (None if _strength_ticks == 0 else float(
                    visual_teacher_phase["hs_outer_load_strength_max"])),
                "strength_ticks": _strength_ticks,
            }),
            "nominal_phase_step_rad": float(_nominal_phase_step),
            "slow_phase_step_rad": (
                None if _slow_phase_step is None else float(_slow_phase_step)),
            "contact_chronology": "previous tick",
            "class_to_outer_side": {"0": "right", "1": "left"},
            "leg_order": list(visual_teacher_phase["leg_order"]),
            "decoded_ticks_by_class": (
                visual_teacher_phase["hs_outer_decoded_ticks_by_class"]
                .astype(int).tolist()),
            "undecoded_ticks": int(
                visual_teacher_phase["hs_outer_undecoded_ticks"]),
            "loaded_slow_ticks_by_class_leg": (
                visual_teacher_phase[
                    "hs_outer_loaded_slow_ticks_by_class_leg"]
                .astype(int).tolist()),
            "delayed_phase_rad_by_class_leg": (
                visual_teacher_phase[
                    "hs_outer_delayed_phase_rad_by_class_leg"]
                .astype(float).tolist()),
            "final_local_phase_rad_by_leg": (
                visual_teacher_phase["hs_outer_phase_rad"]
                .astype(float).tolist()),
        }
    if measured_step_contact_gain is not None:
        contact_receipt = {
            "contact_threshold_N": float(
                measured_step_contact_gain["contact_threshold_N"]),
            "contact_chronology": "previous tick",
            "active_ticks_by_leg": (
                measured_step_contact_gain["active_ticks_by_leg"]
                .astype(int).tolist()),
            "offered_abs_drive_tick_sum": float(
                measured_step_contact_gain["offered_abs_drive_tick_sum"]),
            "added_abs_drive_tick_sum": float(
                measured_step_contact_gain["added_abs_drive_tick_sum"]),
        }
        if measured_step_contact_gain["gain_by_visual_class"] is None:
            contact_receipt["gain"] = float(
                measured_step_contact_gain["gain"])
        else:
            contact_receipt.update({
                "mode": "visual_hs_class_gated",
                "gain_by_visual_class": (
                    measured_step_contact_gain["gain_by_visual_class"]
                    .astype(float).tolist()),
                "active_ticks_by_visual_class_leg": (
                    measured_step_contact_gain[
                        "active_ticks_by_visual_class_leg"]
                    .astype(int).tolist()),
            })
        LAST_EXTRAS["measured_step_contact_gain"] = contact_receipt
    if measured_step_hs_outer_contact_gain is not None:
        LAST_EXTRAS["measured_step_hs_outer_contact_gain"] = {
            "gain": float(measured_step_hs_outer_contact_gain["gain"]),
            "contact_threshold_N": float(
                measured_step_hs_outer_contact_gain[
                    "contact_threshold_N"]),
            "contact_chronology": "previous tick",
            "class_to_outer_side": {"0": "right", "1": "left"},
            "decoded_ticks_by_class": (
                measured_step_hs_outer_contact_gain[
                    "decoded_ticks_by_class"].astype(int).tolist()),
            "outer_side_ticks_left_right": (
                measured_step_hs_outer_contact_gain[
                    "outer_side_ticks_left_right"].astype(int).tolist()),
            "undecoded_ticks": int(
                measured_step_hs_outer_contact_gain["undecoded_ticks"]),
            "active_ticks_by_leg": (
                measured_step_hs_outer_contact_gain[
                    "active_ticks_by_leg"].astype(int).tolist()),
            "active_ticks_by_class_leg": (
                measured_step_hs_outer_contact_gain[
                    "active_ticks_by_class_leg"].astype(int).tolist()),
            "offered_abs_drive_tick_sum": float(
                measured_step_hs_outer_contact_gain[
                    "offered_abs_drive_tick_sum"]),
            "added_abs_drive_tick_sum": float(
                measured_step_hs_outer_contact_gain[
                    "added_abs_drive_tick_sum"]),
        }
    if measured_step_hs_conservative_push_pull is not None:
        LAST_EXTRAS["measured_step_hs_conservative_push_pull"] = {
            "gain": float(measured_step_hs_conservative_push_pull["gain"]),
            "inner_factor": float(
                1.0 - measured_step_hs_conservative_push_pull["gain"]),
            "contact_threshold_N": float(
                measured_step_hs_conservative_push_pull[
                    "contact_threshold_N"]),
            "contact_chronology": "previous tick",
            "class_to_outer_side": {"0": "right", "1": "left"},
            "decoded_ticks_by_class": (
                measured_step_hs_conservative_push_pull[
                    "decoded_ticks_by_class"].astype(int).tolist()),
            "outer_side_ticks_left_right": (
                measured_step_hs_conservative_push_pull[
                    "outer_side_ticks_left_right"].astype(int).tolist()),
            "undecoded_ticks": int(
                measured_step_hs_conservative_push_pull["undecoded_ticks"]),
            "transfer_ticks": int(
                measured_step_hs_conservative_push_pull["transfer_ticks"]),
            "outer_active_ticks_by_class_leg": (
                measured_step_hs_conservative_push_pull[
                    "outer_active_ticks_by_class_leg"].astype(int).tolist()),
            "inner_active_ticks_by_class_leg": (
                measured_step_hs_conservative_push_pull[
                    "inner_active_ticks_by_class_leg"].astype(int).tolist()),
            "source_outer_added_abs_drive_tick_sum": float(
                measured_step_hs_conservative_push_pull[
                    "source_outer_added_abs_drive_tick_sum"]),
            "inner_removed_abs_drive_tick_sum": float(
                measured_step_hs_conservative_push_pull[
                    "inner_removed_abs_drive_tick_sum"]),
            "transferred_abs_drive_tick_sum": float(
                measured_step_hs_conservative_push_pull[
                    "transferred_abs_drive_tick_sum"]),
            "candidate_outer_added_abs_drive_tick_sum": float(
                measured_step_hs_conservative_push_pull[
                    "candidate_outer_added_abs_drive_tick_sum"]),
            "max_outer_factor": float(
                measured_step_hs_conservative_push_pull["max_outer_factor"]),
            "max_tick_budget_abs_error": float(
                measured_step_hs_conservative_push_pull[
                    "max_tick_budget_abs_error"]),
            "sum_tick_budget_abs_error": float(
                measured_step_hs_conservative_push_pull[
                    "sum_tick_budget_abs_error"]),
        }
    if measured_step_hs_saturation_aware_push_pull is not None:
        config = measured_step_hs_saturation_aware_push_pull
        LAST_EXTRAS["measured_step_hs_saturation_aware_push_pull"] = {
            "gain": float(config["gain"]),
            "outer_factor_cap": float(config["outer_factor_cap"]),
            "contact_threshold_N": float(config["contact_threshold_N"]),
            "contact_chronology": "previous tick",
            "class_to_outer_side": {"0": "right", "1": "left"},
            "decoded_ticks_by_class": (
                config["decoded_ticks_by_class"].astype(int).tolist()),
            "outer_side_ticks_left_right": (
                config["outer_side_ticks_left_right"].astype(int).tolist()),
            "undecoded_ticks": int(config["undecoded_ticks"]),
            "transfer_ticks": int(config["transfer_ticks"]),
            "saturation_ticks": int(config["saturation_ticks"]),
            "outer_active_ticks_by_class_leg": (
                config["outer_active_ticks_by_class_leg"].astype(int).tolist()),
            "inner_active_ticks_by_class_leg": (
                config["inner_active_ticks_by_class_leg"].astype(int).tolist()),
            "source_outer_added_abs_drive_tick_sum": float(
                config["source_outer_added_abs_drive_tick_sum"]),
            "requested_inner_removal_abs_drive_tick_sum": float(
                config["requested_inner_removal_abs_drive_tick_sum"]),
            "transferred_abs_drive_tick_sum": float(
                config["transferred_abs_drive_tick_sum"]),
            "untransferred_abs_drive_tick_sum": float(
                config["untransferred_abs_drive_tick_sum"]),
            "candidate_outer_added_abs_drive_tick_sum": float(
                config["candidate_outer_added_abs_drive_tick_sum"]),
            "min_inner_factor": float(config["min_inner_factor"]),
            "max_outer_factor": float(config["max_outer_factor"]),
            "max_tick_budget_abs_error": float(
                config["max_tick_budget_abs_error"]),
            "sum_tick_budget_abs_error": float(
                config["sum_tick_budget_abs_error"]),
        }
    z_end = float(np.mean(zs[-win:])) if zs else 0.0
    # ⚠️ THE GATE, added after the first search's 4.02 mm winner turned out to
    # have FALLEN OVER (verified: seed 0 upright 0.56, height 0.51, against
    # 0.97-0.99 / 1.17-1.32 for seeds that stood). Displacement alone cannot
    # tell walking from toppling-and-sliding. A fallen fly scores ZERO.
    score = disp if (upright > 0.90 and z_end > 1.0) else 0.0
    return score, disp, upright, z_end


def main():
    rng = np.random.default_rng(20260811)
    meta = pd.read_feather("data/banc_888_meta.feather")
    edges = pd.read_feather("data/banc_888_edgelist_simple_v3.feather")
    log = (json.load(open("results-walk-search2.json"))
           if os.path.exists("results-walk-search2.json") else [])
    net_cache = {}
    t0 = time.time()
    best = max(log, key=lambda r: r["disp"]) if log else None
    for it in range(120):
        if it < 70 or best is None:
            th = sample(rng)
            phase = "random"
        else:
            th = neighbour(best["theta"], rng)
            phase = "hill"
        try:
            d, raw, up, z = evaluate(net_cache, meta, edges, th)
        except Exception as e:
            print(f"  eval failed: {type(e).__name__} {e}", flush=True)
            continue
        rec = {"iter": len(log), "phase": phase, "theta": th, "disp": d,
               "raw_disp": raw, "upright": round(up, 2), "z": round(z, 2)}
        log.append(rec)
        if best is None or d > best["disp"]:
            best = rec
            print(f"[{len(log):>3}] {phase:<6} score {d:6.2f} (raw {raw:.2f}, "
                  f"up {up:.2f}, z {z:.2f})  ** BEST **  "
                  f"{ {k: (round(v,3) if isinstance(v,float) else v) for k,v in th.items()} }",
                  flush=True)
        else:
            print(f"[{len(log):>3}] {phase:<6} score {d:6.2f} (raw {raw:.2f}, "
                  f"up {up:.2f}, z {z:.2f})", flush=True)
        json.dump(log, open("results-walk-search2.json", "w"), indent=1)
    print(f"\nbest: {best['disp']:.2f} mm  {best['theta']}")
    print(f"wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
