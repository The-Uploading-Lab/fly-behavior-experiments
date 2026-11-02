"""The load pathway: real ground-contact forces driving the campaniform cells.

WHY THIS IS THE UNIQUELY-DETERMINED NEXT BUILD. The alternation work showed the
half-centres already alternate where both pools engage, and that 14 of the 21
silent pools are actively SUPPRESSED -- losing sides of half-centre fights that
nothing ever un-freezes. The insect-walking literature (Ekeberg/Buschges/Cruse)
says the un-freezer is LOAD: stance yields to swing when the leg unloads. Until
now campaniform sensilla were fed muscle activation as a stand-in -- a signal
that cannot say "this leg just lost the ground".

WHAT THIS DOES. Runs the CNS and the MuJoCo body in one loop:

    CNS spikes --5 ms--> muscle filter -> joint angles -> flygym steps 50x
        -> per-leg CONTACT FORCE read from MuJoCo
        -> campaniform rate = camp_gain * clip(F_leg / F_ref, 0, 1)
        -> injected into that leg's campaniform cells, next CNS window
    (claw/hook/club position+velocity encoding unchanged, from body.py)

⚠️ THE DATA LIMITATION, recorded up front: BANC's leg campaniform annotation is
wildly asymmetric -- 54 front-leg cells, 4 middle, 4 hind. The load channel
this builds is anatomically front-leg dominated. If load release only works on
front legs, that is the annotation's shape, not a modelling choice.

⚠️ PRE-REGISTERED A/B, before any run. LOAD-CLOSED (forces drive campaniform)
vs LOAD-CONSTANT (campaniform held at the closed run's own mean rate --
identical average drive, zero timing information; the loop_necessity
discipline). 3 seeds each. Metrics:
  primary   RELEASED POOLS: antagonist pools silent (<100 wtd spikes) under
            LOAD-CONSTANT that fire under LOAD-CLOSED. Success >= 5.
  secondary engaged-joint count; per-leg load oscillation (does stance/swing
            switching happen at all: load crossings of F_ref/2 per second).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import body as B
import cns
import closed_loop as CL
import motormap as MM
from nobody import Readout
from stage0 import PROPRIOCEPTOR_CLASSES, LEG_PARTS

LEG_KEYS = {("front_leg", "L"): "lf", ("front_leg", "R"): "rf",
            ("middle_leg", "L"): "lm", ("middle_leg", "R"): "rm",
            ("hind_leg", "L"): "lh", ("hind_leg", "R"): "rh"}
CAMP_GAIN = 120.0     # Hz at full load -- searched parameter, unmeasured
F_REF = 35.0          # measured in the pilot: per-leg mean load 30-39


def camp_by_leg(meta, net):
    cc = meta["cell_class"].astype(str)
    bp = meta["body_part_sensory"].astype(str)
    side = meta["side"].astype(str).str.upper().str[:1]
    out = {}
    for (part, s), leg in LEG_KEYS.items():
        m = (cc == "campaniform_sensillum_neuron") & (bp == part) & (side == s)
        out[leg] = np.flatnonzero(m.to_numpy())
    return out


def leg_of_geom(model):
    """geom id -> leg key, from MuJoCo geom names."""
    import mujoco
    tags = {"LF": "lf", "LM": "lm", "LH": "lh",
            "RF": "rf", "RM": "rm", "RH": "rh"}
    m = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        up = name.upper()
        for tag, leg in tags.items():
            if tag in up and ("TARSUS" in up or "TIBIA" in up
                              or "CLAW" in up):
                m[g] = leg
                break
    return m


def contact_loads(sim, geom2leg):
    """Per-leg total contact force magnitude (N)."""
    import mujoco
    d = sim.mj_data
    model = getattr(sim, "mj_model", None) or d.model if hasattr(d, "model") \
        else sim.mj_model
    loads = {k: 0.0 for k in LEG_KEYS.values()}
    buf = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        leg = geom2leg.get(con.geom1) or geom2leg.get(con.geom2)
        if leg:
            mujoco.mj_contactForce(model, d, c, buf)
            loads[leg] += float(np.linalg.norm(buf[:3]))
    return loads


def run_loop(net, meta, seed, mode, dur_ms=4000.0, verbose=False):
    """One closed run. mode: 'closed' or 'constant' (rates from a prior run)."""
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import (
        LocomotionAction, apply_locomotion_action, make_locomotion_fly)
    import mujoco

    ro = Readout(net)
    prop = B.Proprioceptors(meta, net, claw_polarity=1, hook_polarity=-1)
    bod = B.Body(tau_joint_ms=15.0, drive_scale=0.003)
    M = ro.M / np.maximum(np.abs(ro.M).sum(axis=0, keepdims=True), 1.0)
    cmd = net.select(cell_type="DNg100")
    camp = camp_by_leg(meta, net)
    camp_all = np.concatenate([v for v in camp.values() if len(v)])

    fly = make_locomotion_fly(name=f"ld{seed}{mode[:2]}", add_adhesion=True)
    cam = None
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
    sim = Simulation(world)
    sim.reset()
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=bod.theta, adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()
    model = sim.mj_model if hasattr(sim, "mj_model") else None
    if model is None:
        import mujoco as mj
        model = sim.mj_data.model if hasattr(sim.mj_data, "model") else None
    g2l = leg_of_geom(model)
    per = int(round(0.005 / sim.timestep))

    load_hist = []
    const_rates = None
    if mode == "constant":
        const_rates = run_loop.mean_rates      # set by a prior closed run

    def drive_fn(t, counts):
        bod.step(M.T @ counts.astype(float), 5.0)
        act = LocomotionAction(joint_angles=bod.theta,
                               adhesion_onoff=np.ones(6, dtype=bool))
        for _ in range(per):
            apply_locomotion_action(sim, fly.name, act)
            sim.step()
        loads = contact_loads(sim, g2l)
        load_hist.append([loads[k] for k in sorted(LEG_KEYS.values())])
        idx, r = prop.rates(bod)                 # position/velocity channels
        if mode == "closed":
            cr = []
            for leg, cells in camp.items():
                if not len(cells):
                    continue
                rate = CAMP_GAIN * float(np.clip(loads[leg] / F_REF, 0, 1))
                cr.append(np.full(len(cells), rate))
            camp_r = np.concatenate(cr) if cr else np.empty(0)
        else:
            camp_r = const_rates
        return (np.concatenate([idx, camp_all, cmd]),
                np.concatenate([r, camp_r, np.full(len(cmd), 50.0)]))

    r = net.run(dur_ms, seed=seed, record=np.sort(ro.rows),
                record_trace_every=1.0, drive_fn=drive_fn,
                feedback_rows=ro.rows, drive_every_ms=5.0)
    tr = r["trace"][500:]
    rates = tr[:, ro.unsort].astype(float)
    loads_arr = np.array(load_hist)
    if mode == "closed":
        # store the mean campaniform rate per cell for the constant control
        mr = []
        for leg, cells in camp.items():
            if not len(cells):
                continue
            lm = np.mean([np.clip(l / F_REF, 0, 1)
                          for l in loads_arr[:, sorted(LEG_KEYS.values()).index(leg)]])
            mr.append(np.full(len(cells), CAMP_GAIN * lm))
        run_loop.mean_rates = np.concatenate(mr) if mr else np.empty(0)

    pools = []
    rows_arr = np.array(ro.rows)
    for k in range(42):
        m = ro.M[:, k]
        for sgn in (1, -1):
            sel = (m * sgn) > 0
            if sel.any():
                pools.append(float(rates[:, sel].sum()))
    crossings = 0
    if len(loads_arr):
        for j in range(loads_arr.shape[1]):
            s = (loads_arr[:, j] > F_REF / 2).astype(int)
            crossings += int(np.abs(np.diff(s)).sum())
    return {"pools": pools, "mn_spikes": int(tr.sum()),
            "load_mean": loads_arr.mean(0).round(6).tolist() if len(loads_arr) else [],
            "load_crossings_per_s": crossings / (dur_ms / 1000.0),
            "n_spikes": int(r["n_spikes"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    meta = pd.read_feather("data/banc_888_meta.feather")
    edges = pd.read_feather("data/banc_888_edgelist_simple_v3.feather")
    p = cns.SimParams()
    p.balance_target, p.w_syn, p.delay = 1.10, 0.65, 1.8
    p.r_in_alpha = 0.10
    net = cns.CNS(meta, edges, p)
    camp = camp_by_leg(meta, net)
    print("campaniform cells per leg:",
          {k: len(v) for k, v in camp.items()}, flush=True)

    if args.pilot:
        r = run_loop(net, meta, 0, "closed", dur_ms=1500.0, verbose=True)
        print(f"pilot: MN {r['mn_spikes']:,}, per-leg mean load {r['load_mean']}")
        print(f"load crossings/s {r['load_crossings_per_s']:.1f}")
        print("If the loads are all ~0 or all huge, recalibrate F_REF.")
        return

    out = {"closed": [], "constant": []}
    for seed in range(args.seeds):
        rc = run_loop(net, meta, seed, "closed")
        out["closed"].append(rc)
        rk = run_loop(net, meta, seed, "constant")
        out["constant"].append(rk)
        pc = np.array(rc["pools"]); pk = np.array(rk["pools"])
        released = int(((pk < 100) & (pc >= 100)).sum())
        print(f"seed {seed}: released pools {released}, closed MN "
              f"{rc['mn_spikes']:,} vs const {rk['mn_spikes']:,}, "
              f"crossings/s {rc['load_crossings_per_s']:.1f}", flush=True)
        out.setdefault("released", []).append(released)
        json.dump(out, open("results-load-loop.json", "w"), indent=1)
    rel = out.get("released", [])
    print(f"\nreleased pools per seed: {rel}  "
          f"(pre-registered success: >= 5 median)")
    print(f"wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()


def camp_extra_by_leg(meta, n):
    """HYPOTHESISED dense campaniform set (parameter rule, 2026-08-18).

    Measured: the rig sees 5 leg campaniform cells in the male and 61
    in the female, and the female's are 53/61 on the FRONT legs -- both
    animals walk on four nearly-unsensed legs. This adds n hypothesised
    load sensors per leg by designating unclassified leg sensory
    neurons (deterministic: lowest banc ids first, so a run is
    reproducible) as campaniform. n = 0 returns empty and the rig is
    bit-identical. Nothing here claims these cells ARE campaniform;
    it asks whether load-sensor DENSITY is what limits balance."""
    import numpy as np
    if not n:
        return {leg: np.array([], dtype=int) for leg in LEG_KEYS.values()}
    cc = meta["cell_class"].astype(str)
    sc = meta["super_class"].astype(str)
    bp = meta["body_part_sensory"].astype(str)
    side = meta["side"].astype(str).str.upper().str[:1]
    known = {"campaniform_sensillum_neuron", "chordotonal_organ_neuron",
             "hair_plate_neuron"}
    ids = meta["banc_888_id"].to_numpy()
    out = {}
    for (part, s), leg in LEG_KEYS.items():
        m = (sc.str.contains("sensory", case=False, na=False)
             & (bp == part) & (side == s) & ~cc.isin(known))
        idx = np.flatnonzero(m.to_numpy())
        out[leg] = idx[np.argsort(ids[idx])][:n] if len(idx) else idx
    return out


# --- CD1-KH (2026-08-24): per-segment contact load. PURELY ADDITIVE ---
# leg_of_geom() reads the geom name, TESTS it for TARSUS/TIBIA/CLAW, and
# then keeps only the leg -- the segment identity is parsed and dropped
# on the next line. contact_loads() then sums every contact on a leg into
# one scalar, and walk_search broadcasts that one number to all of the
# leg's campaniform sensilla. Campaniforms are cuticle STRAIN detectors
# distributed across leg segments (Robin, 2026-08-24), so the collapse
# throws away exactly the distinction they exist to report.
#
# These two functions are NEW. Nothing above is modified, so every
# existing call path is bit-identical by construction.
SEG_TAGS = ("TARSUS", "TIBIA", "CLAW")


def seg_of_geom(model):
    """geom id -> (leg, segment). The segment leg_of_geom throws away."""
    import mujoco
    tags = {"LF": "lf", "LM": "lm", "LH": "lh",
            "RF": "rf", "RM": "rm", "RH": "rh"}
    m = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        up = name.upper()
        for tag, leg in tags.items():
            if tag not in up:
                continue
            for s in SEG_TAGS:
                if s in up:
                    m[g] = (leg, s)
                    break
            if g in m:
                break
    return m


def contact_loads_by_segment(sim, geom2seg):
    """Per-(leg, segment) contact force magnitude (N).

    Same accumulation as contact_loads(), not collapsed across segments.
    sum over SEG_TAGS of this == contact_loads()[leg] for every leg,
    which is the invariant CD1-KH's probe checks.
    """
    import mujoco
    d = sim.mj_data
    model = getattr(sim, "mj_model", None) or d.model if hasattr(d, "model") \
        else sim.mj_model
    loads = {(k, s): 0.0 for k in LEG_KEYS.values() for s in SEG_TAGS}
    buf = np.zeros(6)
    for c in range(d.ncon):
        con = d.contact[c]
        key = geom2seg.get(con.geom1) or geom2seg.get(con.geom2)
        if key:
            mujoco.mj_contactForce(model, d, c, buf)
            loads[key] += float(np.linalg.norm(buf[:3]))
    return loads
