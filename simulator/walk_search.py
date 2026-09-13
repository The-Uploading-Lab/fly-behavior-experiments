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


def _motor_sign_vector(mode, thc_yaw_moment_arm_orientation=None):
    """Resolve measured signs and an opt-in tied ThC-yaw orientation probe.

    The six absolute ThC-yaw motor-map signs are not among the 24 signs with
    geometry-derived evidence. ``"alternate"`` is therefore a binary BUILD
    hypothesis about that unresolved coordinate convention, not a measured
    biological correction. Absent remains the historical path exactly.
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
}


def _apply_mn_transfer(mapping, mn_ids, spec):
    """th["mn_transfer"]: absent = bit-identical. {"set": <name>} rewrites the
    named cells' rows in a copy of the mapping to the target muscle's entries,
    exactly as motormap.build_map writes them (live and hypothesised), and
    rebuilds the raw matrix from that copy. The cell set is unchanged, so the
    row order is the readout's; asserted."""
    import motormap as MM2
    if not isinstance(spec, dict) or spec.get("set") not in _MN_TRANSFER_SETS:
        raise ValueError(f"th['mn_transfer'] must be absent or {{'set': one of {sorted(_MN_TRANSFER_SETS)}}}")
    ts = _MN_TRANSFER_SETS[spec["set"]]
    ids = {str(i) for i in ts["ids"]}; targets = MM2.MUSCLE_TO_DOF[ts["to_muscle"]]   # BANC ids are strings in the mapping
    mapping = mapping.copy()
    hit = mapping["banc_888_id"].astype(str).isin(ids)
    if mapping.loc[hit, "banc_888_id"].nunique() != len(ids) or not mapping.loc[hit, "dof"].notna().all():
        raise ValueError("mn_transfer: every cell in the set must be a mapped leg motor neuron")
    new = []
    for cid, grp in mapping[hit].groupby("banc_888_id", sort=False):
        base = grp.iloc[0].to_dict()
        for joint_axis, sign, hypo in targets:
            new.append({**base, "muscle": ts["to_muscle"], "dof": f"{base['leg']}_{joint_axis}", "joint_axis": joint_axis,
                        "sign": float(sign), "gain": 0.0 if hypo else 1.0, "hypothesised": hypo, "status": "mapped", "transferred": True})
    import pandas as pd
    mapping = pd.concat([mapping[~hit], pd.DataFrame(new)], ignore_index=True)
    M_raw, ids2 = MM2.to_matrix(mapping)
    if list(ids2) != list(mn_ids):
        raise ValueError("mn_transfer changed the motor-neuron row set")
    return mapping, M_raw, len(ids)


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


def length_delay_vector(meta, mean_ms, cap_ms, groups):
    """MEASURED-STRUCTURE DIRECTIVE (Robin, 2026-09-01): per-neuron
    presynaptic delay proportional to BANC whole-neuron cable length,
    scaled so its MEAN equals `mean_ms` (default 1.8, the verified scalar
    operating point, so the dial isolates delay HETEROGENEITY from a mean
    shift), clipped to [0.1, cap_ms], then quantised via _quantise_delays.
    Semantics from lane B's 2026-09-01 implementation, canonicalised here;
    whole-neuron cable is a labelled PROXY for spike-initiation-to-synapse
    path, which BANC does not carry.
    """
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
    for name, value in (type_map or {}).items():
        g = float(value)
        if not np.isfinite(g) or g < 0.0:
            raise ValueError(
                f"type_out_gain: {name} must be finite and >= 0, got {value!r}")
        m = labels == str(name)
        n = int(m.sum())
        if n == 0:
            raise ValueError(f"type_out_gain: {name} matches no row")
        vec[m] = g
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
    """Type-tied chloride reversals with two exact adult-fly anchors.

    l-LNv (-48 mV) and MN5 (-74 mV) use published values. Other biological
    groups receive reproducible hypotheses within that measured envelope;
    a common shift preserves the historical population mean of -70 mV.
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


def size_factors(meta, edges):
    """Memoised on the frames' content: the groupby over all 13.6M edges cost
    0.75 s and evaluate() calls this up to three times per seed with the
    same frames (compute seat, 2026-09-02). Returns fresh dict copies."""
    key = _frame_fingerprint(meta, edges)
    hit = _SIZE_FACTORS_CACHE.get(key)
    if hit is None:
        hit = (_size_factors_uncached(meta, edges), (meta, edges))
        if len(_SIZE_FACTORS_CACHE) >= 4:
            _SIZE_FACTORS_CACHE.pop(next(iter(_SIZE_FACTORS_CACHE)))
        _SIZE_FACTORS_CACHE[key] = hit
    return tuple(dict(d) for d in hit[0])


def _size_factors_uncached(meta, edges):
    """The measured MN size principle (Azevedo 2020; recipe MUSCLE-SPLIT.md):
    within each muscle pool, rank MNs by total input synapse count (the
    r=0.94 anatomical-size proxy, Lesser 2024). Per-spike force gains
    log-spaced over the measured ~770x span (10 / 1 / 0.013 uN); input-
    resistance factors log-spaced over the measured 150-700 MOhm (largest
    MN 0.5x, smallest 2.33x -- geometric-mean-neutral). Singleton pools
    stay neutral: no gradient information exists for them.
    Returns (gain, r_factor) dicts keyed by banc id."""
    import motormap as MM
    mapping = MM.build_map(meta)
    base = mapping[~mapping["hypothesised"]].drop_duplicates("banc_888_id")
    insyn = edges.groupby("post")["count"].sum()
    gains, rfac, vrest = {}, {}, {}
    for _, pool in base.groupby(["muscle", "leg"]):
        ids = sorted(pool["banc_888_id"], key=lambda i: -insyn.get(i, 0))
        n = len(ids)
        for k, i in enumerate(ids):
            if n == 1:
                gains[i], rfac[i] = 1.0, 1.0
            else:
                f = k / (n - 1)               # 0 = largest/fast, 1 = smallest/slow
                gains[i] = 770.0 ** (-f)
                rfac[i] = 0.5 * (700.0 / 150.0) ** f
                # measured resting-Vm gradient: fast -68, slow -48 mV
                vrest[i] = -68.0 + 20.0 * f
    return gains, rfac, vrest


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


def graded_rows_for(sel, meta):
    """Rows a graded selector names, as a tuple of core-row indices.

    Lifted out of evaluate() unchanged (lane A, 2026-09-02) so a second
    selector can extend a theta's graded set: th["graded_add"] names rows
    added beside th["graded"], and they carry th["graded_gain_add"] while
    the base rows keep th["graded_gain"]. A champion's graded rows are
    then untouched by the extension, which is what a paired test needs.
    """
    graded_rows = ()
    if sel == "hyp":
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


def _refuse_unknown_theta_keys(th):
    """A theta key this tree does not read is a claim the run does not carry.

    Every key in evaluate is read with th.get(...), so a key the running tree has
    not merged does NOTHING and raises nothing: the lane runs a different body
    from the theta it was handed and gets a plausible number. Measured
    2026-09-06 11:30 (lane E, 213 commits behind main): the judge's grip-15
    theta's `delay_region_scale` was ignored, arm A stood 0 of 12 against the
    judge's 9 of 12, and a contradiction was nearly filed.

    The read-key set is THIS tree's own sources (walk_search, body, cns,
    regression_walk), the same quoted-string read lanes/judge/theta_keys_gate.py
    makes, so a lane's unported mechanism is known on the lane's tree and a
    judge theta with a newer key is unknown on a stale one. Underscore-prefixed
    keys and the theta-file wrapper keys are metadata and tolerated.

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
        for fn in ("walk_search.py", "body.py", "cns.py", "regression_walk.py"):
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


def evaluate(net_cache, meta, edges, th, seed=0, dur_ms=4000.0, capture=None,
             playback=None, record_rows=None, trace_out=None):
    """capture: optional list; if given, (xyz, quat) appended per control
    tick. THE one sanctioned way to get trajectories -- side scripts that
    hand-copy this drive path go stale every time a mechanism lands
    (battery_test and direction_test2 already did; see 2026-08-13).
    record_rows/trace_out (2026-08-17, capture-only, default off =
    bit-identical): extra neuron rows to record; the run's binned spike
    trace (rows_sorted, trace) is appended to trace_out. Readout only --
    the record set never feeds back into the simulation."""
    _refuse_unknown_theta_keys(th)
    sign_vector = _motor_sign_vector(
        th.get("signs"), th.get("thc_yaw_moment_arm_orientation"))
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
    graded_add_rows = ()
    if th.get("graded_add"):
        if th.get("graded_gain_add") is None:
            raise ValueError("graded_add requires graded_gain_add")
        _base = set(graded_rows)
        graded_add_rows = tuple(
            r for r in graded_rows_for(th["graded_add"], meta)
            if r not in _base)
        graded_rows = tuple(sorted(_base | set(graded_add_rows)))
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
    _serotonin_present = any(key_ in th for key_ in _serotonin_keys)
    serotonin_source_rows = serotonin_target_rows = ()
    serotonin_tau_ms = serotonin_gain_mv_per_hz = None
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
        if (not np.isfinite(serotonin_gain_mv_per_hz)
                or not (serotonin_gain_mv_per_hz == 0.0
                        or 0.001 <= serotonin_gain_mv_per_hz <= 0.5)):
            raise ValueError(
                "serotonin_gain_mv_per_hz must be zero or in the mapped 0.001-0.5 prior")
        serotonin_source_rows, serotonin_target_rows = (
            serotonin_vnc_5ht7_rows(meta))
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
                    "pre_nt, sign, gain)")
        receptor_classes = tuple(dict(_entry) for _entry in receptor_classes)
    _presynaptic_inhibition_mode = th.get("presynaptic_inhibition_mode")
    presynaptic_inhibition_rows = ()
    if _presynaptic_inhibition_mode == "hook_shunt":
        # Use the production encoder's membership rather than a parallel
        # selector: this includes the one hook-subclass / hair-class conflict
        # that body.Proprioceptors has always routed as hook.
        _prop_membership = B.Proprioceptors(
            meta, None, claw_polarity=th["claw"], hook_polarity=th["hook"])
        presynaptic_inhibition_rows = tuple(sorted(
            np.concatenate((_prop_membership.groups["hook_flex"],
                            _prop_membership.groups["hook_ext"])).tolist()))
        if not (th.get("cbased") and th.get("shunt")):
            raise ValueError(
                "hook presynaptic shunt requires true conductance shunting")
    elif _presynaptic_inhibition_mode not in (None, ""):
        raise ValueError(
            "unknown presynaptic_inhibition_mode: "
            f"{_presynaptic_inhibition_mode!r}")
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
    _pg_by_id_key = th.get("presyn_gain_by_id")
    if _pg_by_id_key:
        import hashlib as _hl
        key = key + ("presyn_gain_by_id", _hl.sha256(json.dumps(
            {str(k): float(v) for k, v in _pg_by_id_key.items()},
            sort_keys=True).encode()).hexdigest()[:16])
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
    if receptor_classes:
        key = key + ("receptor_classes", tuple(
            (str(_e.get("pre_nt", "*")), str(_e["post_cell_class"]),
             float(_e.get("sign", 1.0)), round(float(_e.get("gain", 1.0)), 6))
            for _e in receptor_classes))
    if _serotonin_present:
        # Extend only for an explicitly enabled map. Historical thetas retain
        # the exact old cache key and path.
        key = key + (
            "serotonin_5ht7_08b",
            round(serotonin_tau_ms, 6),
            round(serotonin_gain_mv_per_hz, 9))
    if _presynaptic_inhibition_mode:
        key = key + ("presynaptic_inhibition",
                     _presynaptic_inhibition_mode,
                     len(presynaptic_inhibition_rows))
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
    if _graded_13b_gain is not None:
        key = key + (
            "graded_13b_gain", round(_graded_13b_gain, 9),
            _graded_13b_gain_mode,
            int(th.get("graded_13b_gain_permutation_seed", 0)))
    if gap_pair_weights:
        key = key + ("gap_pair_weights",
                     tuple(round(float(value), 7)
                           for value in gap_pair_weights))
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
        key = key + ("graded_add", th["graded_add"],
                     round(float(th["graded_gain_add"]), 6))
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
    # graded_v50 / graded_slope were theta-set and engine-ignored
    # (80254). Extend the key only when set so absent thetas stay
    # byte-identical.
    if th.get("graded_v50") is not None:
        key = key + ("gv50", round(float(th["graded_v50"]), 3))
    if th.get("graded_slope") is not None:
        key = key + ("gslope", round(float(th["graded_slope"]), 3))
    if th.get("coincidence_gain"):
        key = key + (
            "coincidence", round(float(th["coincidence_gain"]), 6),
            round(float(th.get("coincidence_lag_ms", 60.0)), 3),
            tuple(sorted(map(str, th.get("coincidence_target_types") or ()))),
            tuple(sorted(map(str, th.get("coincidence_fast_types") or ()))),
            tuple(sorted(map(str, th.get("coincidence_slow_types") or ()))),
            round(float(th.get("coincidence_tau_ms", 5.0)), 3))
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
    if th.get("membrane_mode"):
        key = key + ("membrane", th["membrane_mode"],
                     int(th.get("membrane_seed", -1)))
    if th.get("adaptation_parameter_mode"):
        key = key + (
            "adaptation_parameters", th["adaptation_parameter_mode"])
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
    if th.get("silence_rows") is not None:
        key = key + ("silence_rows",
                     tuple(sorted(str(x) for x in th["silence_rows"])))
    if th.get("cmd_presyn_gain") is not None:
        key = key + (
            "cmd_presyn_gain",
            str(th.get("cmd", "")),
            round(float(th["cmd_presyn_gain"]), 6))
    if key not in net_cache:
        p = cns.SimParams()
        p.balance_target, p.w_syn = 1.10, 0.65
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
        if type_out_gain_vec is not None:
            p.type_out_gain = type_out_gain_vec
        if _serotonin_present:
            p.serotonin_source_rows = serotonin_source_rows
            p.serotonin_target_rows = serotonin_target_rows
            p.serotonin_tau_ms = serotonin_tau_ms
            p.serotonin_gain_mv_per_hz = serotonin_gain_mv_per_hz
        if _presynaptic_inhibition_mode:
            p.presynaptic_inhibition_rows = presynaptic_inhibition_rows
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
        if graded_add_rows:
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
        if th.get("graded_slope") is not None:
            _gs = float(th["graded_slope"])
            if _gs <= 0.0:
                raise ValueError("graded_slope must be positive")
            p.graded_slope = _gs
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
        elif _dmode == "mcns_active_proxy":
            _mcns_structure = _load_mcns_active_structure(meta)
            p.delay_per_neuron = _mcns_structure[0]
            p.structure_active_mask = _mcns_structure[2]
        elif _dmode not in (None, ""):
            raise ValueError(f"unknown delay_mode: {_dmode!r}")
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
        if _cgain:
            _ct = meta["cell_type"].astype("string")
            if "fafb_alignment_cell_type" in meta.columns:
                _ct = _ct.where(_ct.notna(),
                                meta["fafb_alignment_cell_type"].astype("string"))
            _lab = np.array([x if isinstance(x, str) else ""
                             for x in _ct.to_numpy()], dtype=object)

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
        if use_size:
            _, rfac, vrest = size_factors(meta, edges)
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
        _vrmode = th.get("v_rest_mode")
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
            _merged_vrest = dict(p.v_rest_override or {})
            _merged_vrest.update(_vr_overrides)
            p.v_rest_override = _merged_vrest
        elif _vrmode not in (None, ""):
            raise ValueError(f"unknown v_rest_mode: {_vrmode!r}")
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
            _thr = np.broadcast_to(
                np.asarray(p.v_threshold, dtype=np.float64),
                (len(meta),)).copy()
            _thr[_silence_rows] = 1e9
            p.v_threshold = _thr
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
        net_cache[key] = cns.CNS(meta, _network_edges, p)
    net = net_cache[key]
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
    if _camp_src not in (None, "load", "load_matched"):
        raise ValueError(f"th['camp_src'] must be absent, 'load' or "
                         f"'load_matched', got {_camp_src!r}")
    if _camp_src == "load_matched" and "camp_match_factor" not in th:
        raise ValueError("camp_src='load_matched' requires "
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
                            # CD1-SN: absent = 1.0 = bit-identical
                            claw_gain=th.get("feco_claw_gain"),
                            # CD1-TN: absent = linear = bit-identical
                            claw_code=th.get("feco_claw_code"),
                            claw_flex_frac=th.get("feco_claw_flex_frac"),
                            reflex_map=th.get("feco_reflex_map"),
                            claw_width=th.get("feco_claw_width"))
    # measured joint ranges (Haustein 2024) opt-in via th["ranges"]
    body_joint_bias = _resolve_body_joint_bias(th)
    body_joint_tau_act = _resolve_body_joint_tau_act(th)
    body_joint_drive_scale = _resolve_body_joint_drive_scale(th)
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
                 f0_hz=th.get("f0_hz"), zeta=float(th.get("zeta", 0.8)),
                 exact_filters=(th.get("filt") == "exact"),
                 joint_bias_rad=body_joint_bias,
                 drive_scale_vec=body_joint_drive_scale,
                 joint_integrator=th.get("joint_int", "euler"))
    # gap.abdomen_actuation step 2 (judge, 2026-09-03): th["abd_gain"] > 0
    # builds the abdomen adapter -- ten actuated abdomen DOFs on the fly,
    # BANC's abdominal motor neurons appended to the feedback rows, a
    # second first-order Body driving them (see abdomen.py). Absent or
    # 0 = not built = bit-identical.
    abd = None
    if float(th.get("abd_gain", 0.0) or 0.0) > 0.0:
        import abdomen as AB
        abd = AB.AbdomenAdapter(th, net)
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
            size_g, _, _ = size_factors(meta, edges)
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
            # mn_gain_scale (longevity lane, 2026-09-11): one scalar on the
            # per-MN force gains, applied AFTER every overlay so it is a
            # uniform dose on top of the holder's fitted gains, never a
            # refit of them (MUSCLE.md: the muscle layer is instrument, not
            # knob). It is the neuromuscular-transmission damage arm of the
            # paired baseline/dose ladder; bounded (0, 1] so the key can only
            # weaken. Absent or 1.0 = bit-identical. The muscle_gain receipt
            # below does not see gvec, so the value travels in the theta and
            # its sha256 is the arm's identity.
            if th.get("mn_gain_scale") is not None:
                _mgs = float(th["mn_gain_scale"])
                if not (0.0 < _mgs <= 1.0):
                    raise ValueError("mn_gain_scale must be in (0, 1]")
                gvec = gvec * _mgs
            M_raw = M_raw * gvec[:, None]
        M, muscle_gain_receipt = finalize_motor_matrix(
            M_raw, mapping_eff, ro.ro.mn_ids, muscle_gain)
    else:
        M = ro.M / np.maximum(np.abs(ro.M).sum(axis=0, keepdims=True), 1.0)
    # CD1-MK: apply the measured signs to the final map, after secondary
    # moment arms and per-MN gains. Applying them only inside JointReadout
    # would leave r23's hypothesised FTi rows at the old +1 pin.
    M = _apply_motor_sign_vector(M, sign_vector)
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
        occupied = np.concatenate([cmd, cmd2])
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
    if _camp_src == "load_matched":
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
    gust_rest = bool(th.get("gust_rest"))
    gust_by_leg = {}
    if gust_rest:
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
    # adh_gain/mu ported from lane-a's rig 2026-08-24 (judge): both are
    # ledgered mechanisms (CD1-N adhesion dial, CD1-Q pair_friction) that
    # render_split already honors; a battery instrument that silently
    # drops them mis-measures any theta carrying them (found when r23's
    # adh_gain 31.4 ran as 40 in a judge battery). Absent keys keep the
    # factory values = bit-identical.
    _make_fly = make_locomotion_fly if abd is None else AB.make_fly
    if float(th.get("esc_ttm", 0.0) or 0.0) > 0.0:
        # Enabled TTM owner resolves frozen inv/ CTr hinge names.
        fly = _make_fly(name="inv",
                        add_adhesion=True,
                        adhesion_gain=float(th.get("adh_gain", 40.0)))
    else:
        fly = _make_fly(name=f"s{int(time.time()*10)%100000}",
                        add_adhesion=True,
                        adhesion_gain=float(th.get("adh_gain", 40.0)))
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
    sim = Simulation(world)
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
        joint_angles=(bod.theta if abd is None else abd.joint_angles(bod.theta)),
        adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()
    body_geoms = (body_contact_geoms(sim.mj_model)
                  if thorax_br_rows is not None else None)
    # cocon_scale: the pilot measured k saturated (mean 97 of [45,105]) --
    # tanh(cocon/drive_scale) pins near 1 during activity, so modulation needs
    # its own scale. Searched, per the parameter rule.
    stiff = (MS.StiffnessController(sim, fly,
                                    (both_mask if abd is None
                                     else abd.pad_mask(both_mask)),
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
    adh_motor_balance = _resolve_motor_balance_adhesion(th, use_split)
    adh_leg_threshold_scales = _resolve_adhesion_leg_threshold_scales(th)
    if adh_leg_threshold_scales is not None and not adh_flex:
        raise ValueError("adh_leg_threshold_scales requires th['adh']='flex'")
    adh_knee_legs = _resolve_adhesion_knee_legs(th)
    if adh_knee_legs is not None and not adh_flex:
        raise ValueError("adh_knee_legs requires th['adh']='flex'")
    if adh_flex or adh_motor_balance:
        import motormap as MM5
        adh_legs = ["lf", "lm", "lh", "rf", "rm", "rh"]  # get_legs_order()
        adh_dofs = [np.array([i for i, nm in enumerate(MM5.DOF_NAMES)
                              if nm.startswith(leg_ + "_")])
                    for leg_ in adh_legs]
    if adh_knee_legs is not None:
        adh_knee_dof = [int(d[[MM5.DOF_NAMES[i].endswith("_FTi_pitch")
                               for i in d]][0]) for d in adh_dofs]
    if adh_flex:
        adh_lvl = float(th.get("adh_thresh", 1.0)) * th["drive_scale"]
        adh_lvls = (np.full(6, adh_lvl, dtype=float)
                    if adh_leg_threshold_scales is None else
                    adh_lvl * adh_leg_threshold_scales)
    prev_loads = {}
    g2l = leg_of_geom(sim.mj_model)
    # CD1-KH: absent/None -> _g2s unused and nothing changes.
    _camp_load = th.get("camp_load")
    if _camp_load not in (None, "per_segment"):
        raise ValueError(f"th['camp_load'] must be absent or "
                         f"'per_segment', got {_camp_load!r}")
    _g2s = seg_of_geom(sim.mj_model) if _camp_load == "per_segment" else None
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
    cscale = 5.0 / tick
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
            if "adhesion" in _nm.lower() or "abdomen" in _nm.lower():
                continue
            _j = int(sim.mj_model.actuator_trnid[_u, 0])
            if _j < 0:
                continue
            pos_act.append(_u)
            pos_qadr.append(int(sim.mj_model.jnt_qposadr[_j]))

    per = int(round(tick / 1000.0 / sim.timestep))
    xy0, xy = None, None
    prev_roll = None
    quats, zs = [], []
    tilts = []  # v3.2 posture angle per tick (re-freeze event 2026-08-31)
    legs_sorted = sorted(LEG_KEYS.values())
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

    def _cmd_rates():
        """Command rates for this tick; ungated unless the seam is on."""
        if cmd_side is None:
            return np.full(len(cmd), th["cmd_hz"])
        return _vision_gated_cmd_rates(
            th["cmd_hz"], cmd_side, vision_cmd_side_gain, visual_relay_side,
            vision_cmd_side_swap, visual_relay_magnitude)

    def _cmd2_rates():
        """Second-command rates; ungated unless its own seam is on."""
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
    if prop_tonic_scope not in _PROP_SCOPES:
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
        _, _, _vrest_map = size_factors(meta, edges)
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
                        body_yaw_rad=_body_yaw(sim.mj_data.qpos),
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
                    sim.mj_data.qpos, sim.mj_data.qvel)
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
            bod.step2(df, de, tick)
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
        if abd is not None:
            abd.step(abd_counts, cscale, tick)
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
            onoff = np.array([bod.a_flex[d].mean() <= adh_lvls[i]
                              for i, d in enumerate(adh_dofs)])
            if adh_knee_legs is not None:
                for i, d in enumerate(adh_dofs):
                    if adh_knee_legs[i]:
                        onoff[i] = bod.a_ext[adh_knee_dof[i]] <= adh_lvls[i]
        elif adh_motor_balance and t >= float(th.get("settle_ms", 0.0)):
            onoff = _motor_balance_adhesion_onoff(
                bod.a_flex, bod.a_ext, adh_dofs)
        else:
            onoff = np.ones(6, dtype=bool)
        act = LocomotionAction(joint_angles=(bod.theta if abd is None
                                             else abd.joint_angles(bod.theta)),
                               adhesion_onoff=onoff)
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
        for _ in range(per):
            apply_locomotion_action(sim, fly.name, act)
            if _ttm_bridge is not None:
                _ttm_bridge.apply_before_step(sim)
            sim.step()
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
        loads = contact_loads(sim, g2l)
        prev_loads = loads
        p_ = np.array(sim.mj_data.qpos[:2], dtype=float)
        if t >= 500 and xy0 is None:
            xy0 = p_.copy()
        xy = p_
        quats.append(abs(float(sim.mj_data.qpos[3])))
        # v3.2 tilt measure (re-freeze event 2026-08-31): posture-only
        # uprightness. qpos[3:7] is (w,x,y,z); the body-z-vs-world-up
        # angle is acos(1 - 2(x^2 + y^2)), yaw-independent. Additive:
        # nothing downstream changes unless a consumer reads LAST_EXTRAS.
        _qx, _qy = float(sim.mj_data.qpos[4]), float(sim.mj_data.qpos[5])
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
            capture.append((np.array(sim.mj_data.qpos[:3], dtype=float),
                            np.array(sim.mj_data.qpos[3:7], dtype=float),
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
                              sim.mj_data.qfrc_constraint[:3]],
                             [float(x) for x in
                              sim.mj_data.qfrc_actuator[:3]],
                             [float(x) for x in
                              sim.mj_data.qfrc_passive[:3]],
                             [float(x) for x in
                              sim.mj_data.qfrc_bias[:3]]],
                            # CD1-AX (2026-08-22): element 16, APPENDED.
                            # The simulator's OWN translational
                            # acceleration and velocity. CD1-AW found
                            # every force channel agreeing with each
                            # other and disagreeing with my second
                            # difference of position; this replaces the
                            # suspect quantity. Readout only.
                            [[float(x) for x in sim.mj_data.qacc[:3]],
                             [float(x) for x in sim.mj_data.qvel[:3]]],
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
                            *([[float(x) for x in abd.bod.theta]]
                              if abd is not None else [])))
        zs.append(float(sim.mj_data.qpos[2]))
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
        else:
            cr = [np.full(len(cells), _camp_gain_leg(leg)
                          * float(np.clip(loads[leg] / th["F_ref"], 0, 1)))
                  for leg, cells in camp.items() if len(cells)]
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
        if _prop_rate_acc[0] is None:
            _prop_rate_acc[0] = {k: [np.isin(idx, np.concatenate(
                [prop.groups[g] for g in gs])), 0.0] for k, gs in (
                ("claw", ("claw_flex", "claw_ext")), ("hook", ("hook_flex", "hook_ext")),
                ("club", ("club",)), ("hair", ("hair",)), ("actcamp", ("campaniform",)))}
        for _k, _mv in _prop_rate_acc[0].items():
            if _mv[0].any():
                _mv[1] += float(r[_mv[0]].mean())
        _prop_rate_acc[1] += 1

        def _pack(parts_i, parts_r):
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
                qw, qx, qy, qz = (float(sim.mj_data.qpos[k]) for k in range(3, 7))
                roll = np.degrees(np.arctan2(
                    2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
                hz = 5.0 if abs(roll) > 16.0 else 0.0
                parts_i = list(parts_i) + [halt_rows]
                parts_r = list(parts_r) + [np.full(len(halt_rows), hz)]
            if drop_hp_rows is not None and len(drop_hp_rows):
                parts_i = list(parts_i) + [drop_hp_rows]
                parts_r = list(parts_r) + [dropped_hp_rates(bod, drop_hp_dofs)]
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
                    hz = 5.0 if float(loads.get(leg, 0.0)) > 1.0 else 0.0
                    g_idx.append(cells)
                    g_r.append(np.full(len(cells), hz))
                parts_i = list(parts_i) + [np.concatenate(g_idx)]
                parts_r = list(parts_r) + [np.concatenate(g_r)]
            if thorax_cs_rows is not None and len(thorax_cs_rows):
                planted = sum(float(loads.get(leg, 0.0)) for leg in loads) > 1.0
                parts_i = list(parts_i) + [thorax_cs_rows]
                parts_r = list(parts_r) + [
                    np.full(len(thorax_cs_rows), 5.0 if planted else 0.0)
                ]
            if wheeler_rows is not None and len(wheeler_rows):
                qw, qx, qy, qz = (float(sim.mj_data.qpos[k]) for k in range(3, 7))
                roll = np.degrees(np.arctan2(
                    2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
                hz = 5.0 + 120.0 * float(np.clip(abs(roll) / 180.0, 0.0, 1.0))
                parts_i = list(parts_i) + [wheeler_rows]
                parts_r = list(parts_r) + [np.full(len(wheeler_rows), hz)]
            if neck_rows is not None and len(neck_rows):
                qw, qx, qy, qz = (float(sim.mj_data.qpos[k]) for k in range(3, 7))
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
                qw, qx, qy, qz = (float(sim.mj_data.qpos[k]) for k in range(3, 7))
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
                roll = body_roll_deg(sim.mj_data.qpos)
                droll = 0.0 if prev_roll is None else (roll - prev_roll) / (tick / 1000.0)
                prev_roll = roll
                hz = wing_cs_rollvel_rate(droll)
                parts_i = list(parts_i) + [wing_cs_rows]
                parts_r = list(parts_r) + [np.full(len(wing_cs_rows), hz)]
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
    # force_spikes_ms (longevity lane, 2026-09-12): {cell_type: [t_ms, ...]},
    # every row of that exact cell_type is forced to spike at each listed
    # time, on top of the closed-loop drive. The escape protocol's "one
    # spike in the giant fibre" inside the walking rig. Absent = bit-identical
    # (net.run's force_spikes stays None).
    _force_spikes = None
    if th.get("force_spikes_ms"):
        _fs = {}
        _ct = meta["cell_type"].fillna("").astype(str).to_numpy()
        _pos = pd.Series(np.arange(net.N), index=np.asarray(net.ids).astype(str))
        for _type, _times in dict(th["force_spikes_ms"]).items():
            _rows_meta = np.flatnonzero(_ct == str(_type))
            _rows = [int(_pos[str(_i)]) for _i in
                     meta["banc_888_id"].astype(str).to_numpy()[_rows_meta]
                     if str(_i) in _pos.index]
            if not _rows:
                raise ValueError(f"force_spikes_ms: no network rows for cell_type {_type!r}")
            for _t in list(_times):
                _fs.setdefault(float(_t), []).extend(_rows)
        _force_spikes = {k: np.asarray(sorted(set(v)), dtype=np.int64)
                         for k, v in _fs.items()}
    run_out = net.run(dur_ms, seed=seed, record=rec,
                      record_trace_every=5.0, drive_fn=drive_fn,
                      force_spikes=_force_spikes,
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
                          if visual_direct_hs_clamp_mode else False))
    if trace_out is not None:
        trace_out.append((rec, run_out.get("trace")))
    disp = float(np.linalg.norm(xy - xy0)) if xy0 is not None else 0.0
    win = int(round(1000.0 / tick))   # gate windows = last 1 s, tick-invariant
    upright = float(np.mean(quats[-win:])) if quats else 0.0
    LAST_EXTRAS.clear()
    LAST_EXTRAS["final_tilt_deg"] = (float(np.mean(tilts[-win:]))
                                     if tilts else None)
    LAST_EXTRAS["mn_transferred"] = mn_transferred
    LAST_EXTRAS["thc_map_rows_moved"] = thc_map_moved
    LAST_EXTRAS["muscle_gain"] = muscle_gain_receipt
    LAST_EXTRAS["mn_speed_class_rows"] = (
        getattr(net, "_mn_speed_class_rows", mn_speed_class_rows)
        if th.get("mn_speed_class") else None)
    LAST_EXTRAS["channel_pool_map"] = _cpm
    if "serotonin" in run_out:
        LAST_EXTRAS["serotonin"] = dict(run_out["serotonin"])
    if "presynaptic_inhibition" in run_out:
        LAST_EXTRAS["presynaptic_inhibition"] = dict(
            run_out["presynaptic_inhibition"])
    if "membrane_drive" in run_out:
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
    # CD1-SI/SJ/SK readouts on every arm (the closed loop's delivered rate is
    # the quantity of interest on the BASE arm), written here because
    # LAST_EXTRAS is cleared after the simulation and before this block.
    LAST_EXTRAS["feco_v_scale"] = {"hook": prop.v_scale_hook, "club": prop.v_scale_club}
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
    if body_joint_drive_scale is not None:
        LAST_EXTRAS["body_joint_drive_scale"] = {
            str(name): float(value)
            for name, value in th["body_joint_drive_scale"].items()
        }
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
