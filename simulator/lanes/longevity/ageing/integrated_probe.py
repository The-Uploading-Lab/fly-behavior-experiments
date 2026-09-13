# Portability adaptation: local source manifest replaces research fleet references.
"""One fixed candidate on the first lane's full-CNS walking/escape/flight rig.

Imports the first lane's implementation; changes neither its source nor its
base theta. The same candidate applies in the walking and escape protocols.
Development seed 0 is a replay, not fresh validation. See integrated-build.md.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ESC = ROOT / "lanes/longevity/escape"
for path in (ROOT, ESC):
    sys.path.insert(0, str(path))

import numpy as np
import jumpfly_arms as JA
import jumpfly_cns as JC
import walk_search as W
from lanes.longevity.ageing.motor_rate import MotorRateWindow, MotorReference
from lanes.longevity.ageing.wing_power import limit_stroke_amplitudes


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def finite_metrics(value, missing, path="walking_regression"):
    if isinstance(value, dict):
        return {k: finite_metrics(v, missing, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_metrics(v, missing, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        missing.append(path)
        return None
    return value


def candidate_theta(theta, candidate, meta=None):
    result = deepcopy(theta)
    excursion_gain = candidate.get("joint_excursion_gain")
    if excursion_gain is not None:
        import motormap
        gains = excursion_gain if isinstance(excursion_gain, dict) else {"all": excursion_gain}
        allowed_joints = ({"CTr_pitch", "FTi_pitch"} | set(motormap.DOF_NAMES)
                          if isinstance(excursion_gain, dict) else {"all"})
        if (not gains or set(gains) - allowed_joints
                or any(type(g) not in (int, float) or not np.isfinite(g) or not .5 <= g <= 2 for g in gains.values())):
            raise ValueError("Joint excursion gain must be finite in [0.5, 2]")
        previous = result.get("body_joint_drive_scale") or {}
        result["body_joint_drive_scale"] = {name: float(previous.get(name, result["drive_scale"]))
                                            / gains.get(name, gains.get(name.split("_", 1)[1], gains.get("all", 1.)))
                                            for name in motormap.DOF_NAMES}
    stiffness = candidate.get("walking_stiffness_floor")
    if stiffness is not None:
        if type(stiffness) not in (int, float) or not np.isfinite(stiffness) or not 20 <= stiffness <= 140:
            raise ValueError("Walking stiffness floor must be finite in [20, 140]")
        result["k_min"] = float(stiffness)
    adhesion_scale = candidate.get("walking_adhesion_scale")
    if adhesion_scale is not None:
        if (type(adhesion_scale) not in (int, float) or not np.isfinite(adhesion_scale)
                or not .5 <= adhesion_scale <= 1.):
            raise ValueError("Walking adhesion scale must be finite in [0.5, 1]")
        result["adh_gain"] = float(result["adh_gain"]) * adhesion_scale
    load_filter = candidate.get("load_stiffness_activation_filter", False)
    if type(load_filter) is not bool or (load_filter and result.get("k_mode") != "load"):
        raise ValueError("The boolean stiffness filter requires the existing load mode")
    mv = candidate.get("power_threshold_mv")
    overrides = dict(candidate.get("threshold_type_mv") or {})
    allowed = set(JA.FL3.POWER_TYPES) | set(JA.FL3.STEERING_ROLES)
    if set(overrides) - allowed:
        raise ValueError("This candidate may change only registered wing motor types")
    if mv is not None:
        overrides = {**{kind: mv for kind in JA.FL3.POWER_TYPES}, **overrides}
    if overrides:
        if not all(type(v) in (int, float) and np.isfinite(v) and -70 <= v <= -20
                   for v in overrides.values()):
            raise ValueError("Wing thresholds must be finite mV values in [-70, -20]")
        if result.get("threshold_mode") not in (None, "", "by_type", "by_exact_type"):
            raise ValueError("Wing threshold does not compose with this threshold mode")
        result["threshold_mode"] = "by_exact_type"
        levels = dict(result.get("threshold_type_mv") or {})
        levels.update({kind: float(value) for kind, value in overrides.items()})
        result["threshold_type_mv"] = levels
    jo_gain = candidate.get("antennal_gf_gain")
    if jo_gain is not None:
        gains = jo_gain if isinstance(jo_gain, dict) else {"both": jo_gain}
        expected_sides = {"left", "right"} if isinstance(jo_gain, dict) else {"both"}
        if (set(gains) != expected_sides
                or any(type(g) not in (int, float) or not np.isfinite(g) or not 0 < g <= 4 for g in gains.values())):
            raise ValueError("Antennal-to-GF gain must be finite in (0, 4]")
        if meta is None:
            raise ValueError("Antennal gain requires canonical neuron metadata")
        from lanes.longevity.ageing.threat_input import AntennalPulse
        sensory = AntennalPulse(meta)
        gf = meta.loc[meta["cell_type"].astype(str).eq("DNp01")]
        if len(gf) != 2 or gf["banc_888_id"].nunique() != 2:
            raise ValueError("Expected the two canonical giant fibres")
        if isinstance(jo_gain, dict) and ("side" not in gf or set(gf["side"]) != {"left", "right"}
                                        or not gf["super_class"].eq("descending").all()):
            raise ValueError("Side-specific gain requires one canonical descending GF per side")
        result["row_pair_seams"] = deepcopy(result.get("row_pair_seams") or []) + [{
            "pre_ids": sensory.ids,
            "post_ids": (gf if side == "both" else gf.loc[gf["side"].eq(side)])["banc_888_id"].astype(str).tolist(),
            "gain": float(gain)} for side, gain in gains.items()]
    recurrent_gain = candidate.get("lplc2_recurrent_gain")
    if recurrent_gain is not None:
        if type(recurrent_gain) not in (int, float) or not np.isfinite(recurrent_gain) or not 0 <= recurrent_gain <= 1:
            raise ValueError("LPLC2 recurrent gain must be finite in [0, 1]")
        if meta is None:
            raise ValueError("LPLC2 recurrence requires canonical neuron metadata")
        population = meta.loc[meta["cell_type"].astype(str).eq("LPLC2")]
        if (len(population) < 2 or population["banc_888_id"].nunique() != len(population)
                or not population["super_class"].astype(str).eq("visual_projection").all()):
            raise ValueError("Expected unique LPLC2 visual projection neurons")
        ids = population["banc_888_id"].astype(str).tolist()
        result["row_pair_seams"] = deepcopy(result.get("row_pair_seams") or []) + [{
            "pre_ids": ids, "post_ids": ids.copy(), "gain": float(recurrent_gain)}]
    lplc2_gf_gain = candidate.get("lplc2_gf_gain")
    if lplc2_gf_gain is not None:
        if type(lplc2_gf_gain) not in (int, float) or not np.isfinite(lplc2_gf_gain) or not 0 <= lplc2_gf_gain <= 1:
            raise ValueError("LPLC2-to-GF gain must be finite in [0, 1]")
        if meta is None:
            raise ValueError("LPLC2-to-GF gain requires canonical neuron metadata")
        population = meta.loc[meta["cell_type"].astype(str).eq("LPLC2")]
        gf = meta.loc[meta["cell_type"].astype(str).eq("DNp01")]
        if (len(population) < 2 or population["banc_888_id"].nunique() != len(population)
                or not population["super_class"].astype(str).eq("visual_projection").all()
                or len(gf) != 2 or gf["banc_888_id"].nunique() != 2):
            raise ValueError("Expected canonical LPLC2 and both giant fibres")
        result["row_pair_seams"] = deepcopy(result.get("row_pair_seams") or []) + [{
            "pre_ids": population["banc_888_id"].astype(str).tolist(),
            "post_ids": gf["banc_888_id"].astype(str).tolist(), "gain": float(lplc2_gf_gain)}]
    right_visual_gain = candidate.get("right_visual_gf_gain")
    if right_visual_gain is not None:
        if (type(right_visual_gain) not in (int, float) or not np.isfinite(right_visual_gain)
                or not 1. <= right_visual_gain <= 1.5):
            raise ValueError("Right visual-to-GF gain must be finite in [1, 1.5]")
        if meta is None:
            raise ValueError("Right visual gain requires canonical neuron metadata")
        visual = meta.loc[meta["cell_type"].astype(str).isin(("LC4", "LPLC2"))]
        gf = meta.loc[meta["cell_type"].astype(str).eq("DNp01")]
        if (set(visual["cell_type"]) != {"LC4", "LPLC2"}
                or not visual["banc_888_id"].is_unique
                or not visual["super_class"].eq("visual_projection").all()
                or len(gf) != 2 or not gf["banc_888_id"].is_unique
                or not gf["super_class"].eq("descending").all()
                or "side" not in gf or set(gf["side"]) != {"left", "right"}):
            raise ValueError("Expected canonical LC4/LPLC2 and one descending GF per side")
        result["row_pair_seams"] = deepcopy(result.get("row_pair_seams") or []) + [{
            "pre_ids": visual["banc_888_id"].astype(str).tolist(),
            "post_ids": gf.loc[gf["side"].eq("right"), "banc_888_id"].astype(str).tolist(),
            "gain": float(right_visual_gain)}]
    return result


@contextmanager
def patched(obj, name, value):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield original
    finally:
        setattr(obj, name, original)


def source_hashes():
    paths = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py" and ".venv" not in path.parts:
                paths.add(path)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def run(candidate, protocol, seed=0, duration_ms=3000.0):
    native_ttm = candidate.get("native_ttm", False)
    if type(native_ttm) is not bool:
        raise ValueError("native_ttm must be a boolean")
    complete_ttm = candidate.get("complete_ttm_twitch", False)
    if type(complete_ttm) is not bool or (complete_ttm and not native_ttm):
        raise ValueError("Completing the twitch requires native TTM feedback")
    started = time.perf_counter()
    # Portable reproduction: immutable local manifest replaces fleet Git refs.
    from portable_integrity import verify_sources
    base = verify_sources()
    initial_sources = source_hashes()
    capture, rates_log, trajectory, muscle_trace = [], [], [], []
    state = {"body_s": 0.0, "cns_s": 0.0, "evaluate_s": 0.0, "rates": None,
             "count_sum": None, "net_neurons": None, "ttm_spikes_by_side": [[], []],
             "load_stiffness_filter": None}
    window = MotorRateWindow(JC.MN_WINDOW_TICKS * JC.TICK_MS, JC.TICK_MS)
    power_window = MotorRateWindow(candidate.get("power_window_ms", JC.MN_WINDOW_TICKS * JC.TICK_MS), JC.TICK_MS)
    original_run = JA.cns.CNS.run
    original_evaluate = W.evaluate
    original_body = JA.ColourBody
    body_ref = []
    ttm_continuation = None
    physics_context = nullcontext()
    if complete_ttm:
        from lanes.longevity.ageing.complete_ttm import apply_after_takeoff
        ttm_continuation = {"post_takeoff_physics_steps": 0, "active_twitch_physics_steps": 0,
                            "peak_torque_by_side": {"L": 0., "R": 0.},
                            "first_active_time_ms": None, "last_active_time_ms": None,
                            "scope": "Existing101uN twitch continues after takeoff at physics steps; flight pose controller retained"}
        original_step = JA.mj.mj_step

        def physics_step(model, data, *args, **kwargs):
            if body_ref and model is body_ref[0].h["model"] and data is body_ref[0].h["data"]:
                if args or kwargs:
                    raise ValueError("The observed winged body must use single physics steps")
                apply_after_takeoff(body_ref[0], model, data, ttm_continuation)
            return original_step(model, data, *args, **kwargs)

        physics_context = patched(JA.mj, "mj_step", physics_step)

    class Body(original_body):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            body_ref.append(self)
            self.power_budget = None
            if candidate.get("wing_power_budget", False):
                if candidate["wing_power_budget"] is not True:
                    raise ValueError("wing_power_budget must be a boolean")
                self.power_budget = {"updates": 0, "limited_updates": 0, "minimum_scale": 1.,
                    "scope": "shared cubic stroke-amplitude proxy at fixed frequency; not aerodynamic work"}
                original_update = self.layer.update

                def power_limited_update(*args, **kwargs):
                    command = original_update(*args, **kwargs)
                    last = self.layer.last
                    available = tuple(max(0., value + last["da_alt"]) if value > 0. else 0.
                                      for value in (last["a_power"]["left"], last["a_power"]["right"]))
                    amplitudes, info = limit_stroke_amplitudes(command.amp, available)
                    self.layer.cmd = replace(command, amp=amplitudes)
                    last["power_budget"] = info
                    self.power_budget["updates"] += 1
                    self.power_budget["limited_updates"] += int(info["scale"] < 1.)
                    self.power_budget["minimum_scale"] = min(self.power_budget["minimum_scale"], info["scale"])
                    return self.layer.cmd

                self.layer.update = power_limited_update
            for side in ("left", "right"):
                for muscle in self.layer.steer[side].values():
                    muscle.gain_deg *= float(candidate.get("steering_gain_scale", 1.0))
            original_kinematics = self.layer.steering_kinematics
            reference_hz = float(candidate.get("b1_reference_hz", 0.0))
            self.neural_reference_active = False
            self.feed_absolute = self.layer.feed
            reference_interval = candidate.get("motor_reference_ms")
            if reference_interval is not None and reference_hz:
                raise ValueError("Use one steering reference")
            self.motor_reference = (None if reference_interval is None else MotorReference(
                reference_interval, JC.TICK_MS,
                np.isin(self.index.types, list(JA.FL3.POWER_TYPES)), self.index.sides))

            def programmed_feed(rates):
                # The inherited landing tape already uses the calibrated
                # zero-steering coordinate system. A neural tonic reference
                # belongs only to actual CNS rates, not to that tape.
                self.neural_reference_active = False
                self.feed_absolute(rates)

            self.layer.feed = programmed_feed

            def kinematics(side):
                values = original_kinematics(side)
                if self.neural_reference_active:
                    values["front"] -= (self.layer.steer[side]["b1"].gain_deg
                                         * reference_hz / JA.FL3.STEER_UNIT_HZ)
                return values

            self.layer.steering_kinematics = kinematics

        def feed(self, rates):
            value = state.get("drive_rates")
            self.neural_reference_active = candidate.get("wing_source", "cns") != "steady" and value is not None
            if not self.neural_reference_active:
                value = self.steady
            elif self.motor_reference is not None:
                value = (self.steady if self.motor_reference.baseline is None
                         else self.motor_reference.apply(value))
            self.feed_absolute(value)

        def advance_to(self, t_s):
            begin = time.perf_counter()
            super().advance_to(t_s)
            state["body_s"] += time.perf_counter() - begin
            if not trajectory or t_s - trajectory[-1][0] >= 0.009999:
                d, h = self.h["data"], self.h
                trajectory.append([float(t_s), *map(float, d.xpos[h["thorax"]]),
                                   float(JC.JF.upright(h))])
                if state["rates"] is not None:
                    rates_log.append([float(t_s), *state["rates"].tolist()])
                if self.t_wing0 is not None:
                    muscle_trace.append({"time_s": float(t_s),
                        "power_equivalent_hz": {s: float(p.c) for s, p in self.layer.power.items()},
                        "power_input_hz": {s: float(p["all"]) for s, p in self.layer.pooled.items()},
                        "wing_amplitude": list(map(float, self.layer.cmd.amp)),
                        "wing_shift_rad": list(map(float, self.layer.cmd.shift)),
                        "external_altitude_amplitude": float(self.layer.last.get("da_alt", 0.0)),
                        "front_reflex_deg": float(self.layer.last.get("front_deg", 0.0)),
                        "power_budget": deepcopy(self.layer.last.get("power_budget"))})

    def instrument_run(net, *args, **kwargs):
        rows = np.asarray(kwargs["feedback_rows"])
        body = body_ref[0]
        power_mask = np.isin(body.index.types, list(JA.FL3.POWER_TYPES))
        positions = np.array([int(np.flatnonzero(rows == row)[0]) for row in body.index.rows])
        ttm_positions = [int(np.flatnonzero(rows == row)[0]) for row in state["ttm_rows"]]
        original_drive = kwargs["drive_fn"]
        state["net_neurons"] = int(net.N)
        state["count_sum"] = np.zeros(len(positions))

        def drive(t_ms, feedback):
            if feedback is not None:
                if native_ttm:
                    body.ttm(t_ms * 1e-3, float(feedback[ttm_positions[0]]),
                             float(feedback[ttm_positions[1]]))
                for side, position in enumerate(ttm_positions):
                    if feedback[position]:
                        state["ttm_spikes_by_side"][side].append(float(t_ms))
                counts = np.asarray(feedback)[positions].astype(float)
                if body.motor_reference is not None:
                    body.motor_reference.observe(t_ms, counts)
                state["count_sum"] += counts
                state["rates"] = window.push(counts)
                state["drive_rates"] = state["rates"].copy()
                state["drive_rates"][power_mask] = power_window.push(counts[power_mask])
            return original_drive(t_ms, feedback)

        kwargs["drive_fn"] = drive
        before = time.perf_counter()
        try:
            return original_run(net, *args, **kwargs)
        finally:
            state["cns_s"] += time.perf_counter() - before

    def evaluate(cache, meta, edges, theta, **kwargs):
        theta = candidate_theta(theta, candidate, meta)
        state["theta"] = deepcopy(theta)
        ids = meta["banc_888_id"].astype(str).to_numpy()
        state["ttm_rows"] = [int(np.flatnonzero(ids == cell)[0]) for cell in JC.TTMN_IDS]
        kwargs["capture"] = capture
        before = time.perf_counter()
        filter_context = nullcontext()
        if candidate.get("load_stiffness_activation_filter", False):
            import muscle_split as MS
            from lanes.longevity.ageing.load_stiffness import activation_controller
            state["load_stiffness_filter"] = {}
            controller = activation_controller(MS.StiffnessController, float(theta["tick_ms"]),
                                               state["load_stiffness_filter"])
            filter_context = patched(MS, "StiffnessController", controller)
        with filter_context:
            result = original_evaluate(cache, meta, edges, theta, **kwargs)
        state["evaluate_s"] += time.perf_counter() - before
        state["evaluate_result"] = result
        return result

    # Each restored reference is the pre-existing object, including on errors.
    with patched(JA, "ColourBody", Body), patched(W, "evaluate", evaluate), \
            patched(JA.cns.CNS, "run", instrument_run), physics_context:
        result = JA.run("control", seed, dur_ms=duration_ms,
                        flash_ms=600.0 if protocol == "escape" else 1e9,
                        land_at_s=2.5 if protocol == "escape" else None,
                        gate_ms=-1.0 if native_ttm else 100.0,
                        cmd_hz=104.0)

    body = body_ref[0]
    r = np.asarray(body.rows)
    times = r[:, 0]
    takeoff = body.t_off
    flight = (times > (takeoff + 0.02 if takeoff is not None else duration_ms)) & (times < 2.5)
    trajectory = np.asarray(trajectory)
    rate_array = np.asarray(rates_log)
    neuron_types = np.asarray(body.index.types)
    by_type = {}
    for kind in sorted(set(neuron_types)):
        mask = neuron_types == kind
        rest_window = (rate_array[:, 0] >= 0.3) & (rate_array[:, 0] < 0.55)
        flight_window = (rate_array[:, 0] >= (takeoff + 0.2 if takeoff is not None else duration_ms)) & (rate_array[:, 0] < 2.4)
        by_type[kind] = {"rest_hz": float(rate_array[rest_window, 1:][:, mask].mean()) if rest_window.any() else None,
                         "flight_hz": float(rate_array[flight_window, 1:][:, mask].mean()) if flight_window.any() else None,
                         "spikes": int(state["count_sum"][mask].sum()), "cells": int(mask.sum())}

    out = {"schema": "longevity-integrated-probe-v1", "recorded_utc": datetime.now(timezone.utc).isoformat(),
           "source_manifest_sha256": base,
           "baseline_source_commit": "4cd49ae74223b6616c73466e3e71f277527fd52d",
           "argv": sys.argv,
           "candidate": candidate, "candidate_sha256": identity(candidate), "protocol": protocol,
           "seed": seed, "seed_role": "development replay; not fresh validation", "duration_ms": duration_ms,
           "theta_sha256": identity(state["theta"]), "engine": {"backend": state["theta"].get("backend"),
               "numpy_version": np.__version__, "neurons": state["net_neurons"],
               "metal_refusals": ["gap junctions", "per-neuron spike regularity"]},
           "result": {key: value for key, value in result.items() if key != "raster"},
           "flight_min_upright": float(r[flight, 2].min()) if flight.any() else None,
           "flight_clear_fraction": float((r[flight, 3] > 0.15).mean()) if flight.any() else None,
           "first_low_foot_after_takeoff_ms": (float(r[flight & (r[:, 3] <= 0.15), 0][0] * 1000)
                                                if np.any(flight & (r[:, 3] <= 0.15)) else None),
           "touchdown_ms": None if getattr(body, "t_touch", None) is None else float(body.t_touch * 1000),
           "winged_final_min_foot_z_mm": float(r[-1, 3]),
           "wing_motor_rates": by_type,
           "ttm_spikes_by_side_ms": {side: state["ttm_spikes_by_side"][i] for i, side in enumerate(("left", "right"))},
           "ttm_mapping": "every native feedback tick" if native_ttm else "stimulus-window gate",
           "winged_track": trajectory.tolist(),
           "muscle_trace": muscle_trace,
           "wing_power_budget": body.power_budget,
           "load_stiffness_filter": state["load_stiffness_filter"],
           "ttm_force_continuation": ttm_continuation,
           "motor_reference": None if body.motor_reference is None else body.motor_reference.describe(),
           "timing_s": {"total": time.perf_counter() - started, "evaluate": state["evaluate_s"],
                        "cns_inclusive": state["cns_s"], "winged_body": state["body_s"]},
           "limitations": ["two bodies share one CNS", "prescribed wingbeat", "external attitude/altitude feedback",
                           "timed leg fold/landing", "development seed only"]
                          + ([] if native_ttm else ["stimulus-window TTM gate"])}
    if protocol == "walk":
        import regression_walk as RW
        def replay(cache, meta, edges, theta, **kwargs):
            kwargs["capture"].extend(capture)
            return state["evaluate_result"]
        with patched(W, "evaluate", replay), patched(RW, "DUR", duration_ms), \
                patched(RW, "_G", {"th": state["theta"], "cache": None, "meta": None, "edges": None}):
            out["unavailable_walking_metrics"] = []
            out["walking_regression"] = finite_metrics(RW._run(seed), out["unavailable_walking_metrics"])
    out["source_sha256"] = source_hashes()
    out["changed_local_sources"] = [path for path, digest in initial_sources.items()
                                      if out["source_sha256"].get(path) != digest]
    model_inputs = [ESC / "theta-wj1-jump-film.json", ESC / "jumpfly-stance-v2-best.json",
                    JC.L.FL1_RECEIPT, JC.L.FL2B_RECEIPT]
    out["model_input_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in model_inputs}
    out["data_sha256"] = {name: sha(ROOT / "data" / name) for name in
                           ("banc_888_meta.feather", "banc_888_edgelist_simple_v3.feather")}
    out["source_manifest_sha256_at_end"] = verify_sources()
    out["review_required"] = bool(out["changed_local_sources"]
                                  or base != out["source_manifest_sha256_at_end"])
    out["timing_s"]["total_with_hashing"] = time.perf_counter() - started
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--protocol", choices=("walk", "escape"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration-ms", type=float, default=3000.0)
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text())
    result = run(candidate, args.protocol, duration_ms=args.duration_ms)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"file": str(output), "candidate": candidate, "protocol": args.protocol,
                      "result": result["result"], "flight_min_upright": result["flight_min_upright"],
                      "timing_s": result["timing_s"]}, allow_nan=False), flush=True)
    if result["review_required"]:
        raise SystemExit("Runtime or authority advanced: result saved; review before another run")


if __name__ == "__main__":
    main()
