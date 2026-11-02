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
import json, os, time
import numpy as np, pandas as pd
import body as B
import cns
from nobody import Readout
from load_loop import (LEG_KEYS, camp_by_leg, leg_of_geom, contact_loads)

SPACE = {
    "camp_gain": (20.0, 300.0), "F_ref": (10.0, 80.0),
    "drive_scale": (0.001, 0.008), "tau_joint": (8.0, 30.0),
    "adapt_b": (0.0, 1.2), "cmd_hz": (20.0, 120.0),
}
CATS = {"claw": (1, -1), "hook": (1, -1),
        "cmd": ("DNg100", "DNp09", "MDN")}


def sample(rng):
    th = {k: float(rng.uniform(*v)) for k, v in SPACE.items()}
    for k, opts in CATS.items():
        th[k] = opts[rng.integers(len(opts))]
    return th


def neighbour(th, rng):
    out = dict(th)
    for k, (lo, hi) in SPACE.items():
        out[k] = float(np.clip(th[k] + rng.normal(0, 0.15) * (hi - lo), lo, hi))
    for k, opts in CATS.items():
        if rng.random() < 0.25:
            out[k] = opts[rng.integers(len(opts))]
    return out


def evaluate(net_cache, meta, edges, th, seed=0, dur_ms=4000.0):
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import (
        LocomotionAction, apply_locomotion_action, make_locomotion_fly)
    key = round(th["adapt_b"], 3)
    if key not in net_cache:
        p = cns.SimParams()
        p.balance_target, p.w_syn, p.delay = 1.10, 0.65, 1.8
        p.r_in_alpha = 0.10
        if th["adapt_b"] > 0:
            p.adapt_b, p.tau_w = th["adapt_b"], 100.0
        if len(net_cache) >= 3:
            net_cache.pop(next(iter(net_cache)))
        net_cache[key] = cns.CNS(meta, edges, p)
    net = net_cache[key]
    ro = Readout(net)
    prop = B.Proprioceptors(meta, net, claw_polarity=th["claw"],
                            hook_polarity=th["hook"])
    bod = B.Body(tau_joint_ms=th["tau_joint"], drive_scale=th["drive_scale"])
    M = ro.M / np.maximum(np.abs(ro.M).sum(axis=0, keepdims=True), 1.0)
    # two-channel muscle split (MUSCLE-SPLIT.md): opt-in via th["k_gain"] > 0.
    # Default off; the else-branches below are the original code, untouched.
    use_split = float(th.get("k_gain", 0.0)) > 0.0
    if use_split:
        import muscle_split as MS
        M_flex, M_ext, both_mask = MS.split_map(M)
    cmd = net.select(cell_type=th["cmd"])
    camp = camp_by_leg(meta, net)
    camp_all = np.concatenate([v for v in camp.values() if len(v)])
    fly = make_locomotion_fly(name=f"s{int(time.time()*10)%100000}",
                              add_adhesion=True)
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.5], Rotation3D("quat", [1, 0, 0, 0]))
    sim = Simulation(world)
    sim.reset()
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=bod.theta, adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()
    # cocon_scale: the pilot measured k saturated (mean 97 of [45,105]) --
    # tanh(cocon/drive_scale) pins near 1 during activity, so modulation needs
    # its own scale. Searched, per the parameter rule.
    stiff = (MS.StiffnessController(sim, fly, both_mask,
                                    th.get("k_min", 45.0), th["k_gain"],
                                    th["drive_scale"]
                                    * th.get("cocon_scale", 1.0))
             if use_split else None)
    # k_mode "load" = load-closed per-leg stiffness (round 5); default "cocon"
    k_load = use_split and th.get("k_mode", "cocon") == "load"
    if k_load:
        import motormap as MM
        stiff.set_leg_map(MM.DOF_NAMES, th.get("F_ref_k", 35.0))
    prev_loads = {}
    g2l = leg_of_geom(sim.mj_model)
    per = int(round(0.005 / sim.timestep))
    xy0, xy = None, None
    quats, zs = [], []
    legs_sorted = sorted(LEG_KEYS.values())

    def drive_fn(t, counts):
        nonlocal xy0, xy, prev_loads
        if use_split:
            c = counts.astype(float)
            bod.step2(M_flex.T @ c, M_ext.T @ c, 5.0)
            if k_load:
                stiff.apply_load(prev_loads)
            else:
                stiff.apply(bod.cocon)
        else:
            bod.step(M.T @ counts.astype(float), 5.0)
        act = LocomotionAction(joint_angles=bod.theta,
                               adhesion_onoff=np.ones(6, dtype=bool))
        for _ in range(per):
            apply_locomotion_action(sim, fly.name, act)
            sim.step()
        loads = contact_loads(sim, g2l)
        prev_loads = loads
        p_ = np.array(sim.mj_data.qpos[:2], dtype=float)
        if t >= 500 and xy0 is None:
            xy0 = p_.copy()
        xy = p_
        quats.append(abs(float(sim.mj_data.qpos[3])))
        zs.append(float(sim.mj_data.qpos[2]))
        idx, r = prop.rates(bod)
        cr = [np.full(len(cells), th["camp_gain"]
                      * float(np.clip(loads[leg] / th["F_ref"], 0, 1)))
              for leg, cells in camp.items() if len(cells)]
        return (np.concatenate([idx, camp_all, cmd]),
                np.concatenate([r, np.concatenate(cr),
                                np.full(len(cmd), th["cmd_hz"])]))

    net.run(dur_ms, seed=seed, record=np.sort(ro.rows),
            record_trace_every=5.0, drive_fn=drive_fn,
            feedback_rows=ro.rows, drive_every_ms=5.0)
    disp = float(np.linalg.norm(xy - xy0)) if xy0 is not None else 0.0
    upright = float(np.mean(quats[-200:])) if quats else 0.0
    z_end = float(np.mean(zs[-200:])) if zs else 0.0
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
