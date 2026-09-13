"""Visual-drive escape runner (longevity lane, 2026-09-12; the consultant's
shared dependency for the starvation / ibuprofen / control arms).

A light-off or loom drives the giant fibre through its LC4 (114 cells, 354
synapses onto the two GFs) and LPLC2 (181 cells, 576 synapses) inputs; the
escape circuit downstream is the search-v3 fit (junction/leak 30, AP waveform,
measured DLMn constants and refractory periods); an ESCAPE is a giant-fibre
spike inside the stimulus window; escape PROBABILITY is the fraction of
Poisson-seeded trials that escape. The stimulus strength (one Poisson rate on
LC4 and LPLC2 for `stim_ms`) is the one fitted scalar, set so the control
arm's probability matches the measured young value (above 80%, Gaitanidis
2025 light-off) and then frozen for every arm.

Arm B, starvation (Gaitanidis 2025 Fig 5 and Fig 7: the loss sits at the
LC4-to-GF synapse; LPLC2 drive keeps its escape): the LC4 -> GF synapse counts
are scaled by p (1.0 control, 0.5, 0.0), LPLC2 untouched. Predictions
declared before the run (consultant note 12:2x, Karolina's arm B): escape
probability falls monotonically with p; latency, when the GF fires, does not
change; an LPLC2-weighted stimulus keeps its escape at p 0 (the dissociation).
Arm B falls if LC4 at 0.5 leaves probability unchanged while LPLC2 at 0.5
moves it.

Seeds: Poisson trial seeds 0..N-1 on the no-body circuit. These are not
walking-battery seeds and grade no walking result; recorded per run.

    PYTHONPATH=. .venv/bin/python lanes/longevity/escape/escape_visual.py --calibrate
    PYTHONPATH=. .venv/bin/python lanes/longevity/escape/escape_visual.py --arms --rate <Hz>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import cns  # noqa: E402
from escape_test import find_cells  # noqa: E402

HERE = Path(__file__).resolve().parent
# search v3 fit (2026-09-12-escape-search-grid-v3.json, w 30 member)
W_JUNCTION, AP_MV, AP_MS, GAP_DELAY = 30.0, 60.0, 0.6, 0.3
K_DLM, D_CHEM, TAU_SYN_DLM, TAU_MEM_DLM, GAP_DLM_MV = 8.0, 0.2, 1.0, 4.4, 4.7
PSP5 = 0.1575
# GF input resistance (2026-09-12). The GF's cable is 8,384 and 11,568 um
# against a CNS median of 353 (24-33x), and it carried the default R_in. The
# per-cell value DERIVED from structure (median/cable, 0.042 and 0.030; rank 2)
# is the default. With BANC's own GF visual counts it fired on nothing
# (2026-09-12-gf-rin-check.json) and a searched 0.2 was needed; with the
# transferred counts below it fires on LC4 alone at 80 Hz in 16 ms, on LPLC2
# alone at 30 Hz half the time, on both always (2026-09-12-gf-transfer-check.json),
# so the searched value is retired. GF_RIN is kept only for the record.
GF_RIN = 0.2   # retired searched value; the default is "derived"
# GF visual input counts, TRANSFERRED (2026-09-12). BANC v888 carries 177 LC4
# and 288 LPLC2 synapses per GF; FAFB counts 2,442 LC4 and 1,366 LPLC2 onto
# one GF (von Reyn 2017, Ache 2019) and the hemibrain 2,290 and 1,443
# (Scheffer 2020): a 13x and 5x under-count on the GF lateral dendrite with
# the LC4:LPLC2 ratio inverted, two reconstructions agreeing. Per the
# transfer rule (AGENTS.md, 2026-09-04) a count difference two other
# reconstructions mark as an annotation gap is a correction to BANC, kept and
# labelled. Each GF's existing LC4 and LPLC2 edges are scaled so their sums
# match the mean of the two reconstructions.
GF_LC4_SYN, GF_LPLC2_SYN = 2366.0, 1405.0
LIGHTOFF_HZ, LOOM_HZ, WINDOW_MS = 40.0, 30.0, 100.0   # frozen on the young control (2026-09-12-escape-visual-calibration-transfer.json)


def correct_gf_visual_input(edges, meta):
    ct = meta["cell_type"].astype(str); ids = meta["banc_888_id"].astype(str)
    lc4 = set(ids[ct.str.fullmatch("LC4") | ct.str.startswith("LC4_")]); lplc2 = set(ids[ct.str.fullmatch("LPLC2") | ct.str.startswith("LPLC2_")])
    edges = edges.copy(); pre = edges["pre"].astype(str); post = edges["post"].astype(str); log = {}
    for g in ids[ct.eq("DNp01")]:
        for name, pool, target in (("LC4", lc4, GF_LC4_SYN), ("LPLC2", lplc2, GF_LPLC2_SYN)):
            m = pre.isin(pool) & post.eq(g); have = float(edges.loc[m, "count"].sum())
            edges.loc[m, "count"] = np.round(edges.loc[m, "count"] * target / have).astype(edges["count"].dtype)
            log[f"{name}->{g}"] = (have, float(edges.loc[m, "count"].sum()))
    return edges, log


class VisualEscape:
    def __init__(self, p_lc4: float = 1.0, p_lplc2: float = 1.0, g_junction: float = 1.0, gf_rin="derived", edge_fn=None, params_fn=None):
        # g_junction scales the GF -> TTMn and GF -> PSI coupling (the ageing site of
        # the search held-out: x0.25 reproduced the old jump latency), p_lc4 the
        # LC4 -> GF synapse counts (Gaitanidis 2025 Fig 5/7 loss; starvation arm B).
        self.meta = pd.read_feather("data/banc_888_meta.feather")
        edges = pd.read_feather("data/banc_888_edgelist_simple_v3.feather")
        ct = self.meta["cell_type"].astype(str); ids = self.meta["banc_888_id"]
        self.gf, self.ttm, self.dlm, self.psi = find_cells(self.meta)
        gf_ids = set(ids[ct.eq("DNp01")])
        self.lc4 = np.flatnonzero(ct.str.fullmatch("LC4") | ct.str.startswith("LC4_"))
        self.lplc2 = np.flatnonzero(ct.str.fullmatch("LPLC2") | ct.str.startswith("LPLC2_"))
        lc4_ids, lplc2_ids = set(ids.iloc[self.lc4]), set(ids.iloc[self.lplc2])
        edges, self.transfer_log = correct_gf_visual_input(edges, self.meta)
        if edge_fn is not None:
            edges = edge_fn(edges)   # e.g. per-synapse deletion (escape_factorial)
        for pre_ids, p in ((lc4_ids, p_lc4), (lplc2_ids, p_lplc2)):
            if p != 1.0:
                mask = edges["pre"].isin(pre_ids) & edges["post"].isin(gf_ids)
                edges.loc[mask, "count"] = np.maximum(0, np.round(edges.loc[mask, "count"] * p)).astype(edges["count"].dtype)
                self.scaled = int(mask.sum())
        self.p_lc4, self.p_lplc2, self.g_junction = p_lc4, p_lplc2, g_junction
        sids = ids.astype(str).to_numpy(); dlm_ids = [sids[i] for i in self.dlm]
        pairs = [(int(a), int(b)) for a in self.gf for b in self.ttm] + [(int(a), int(b)) for a in self.gf for b in self.psi]
        p0 = cns.SimParams(); p0.balance_target, p0.w_syn, p0.delay = 0.35, 0.5, D_CHEM
        n0 = cns.CNS(self.meta, edges, p0); wsum = np.asarray(n0.W[self.dlm][:, self.psi].sum(axis=1)).ravel(); del n0
        pp = cns.SimParams(); pp.balance_target, pp.w_syn, pp.delay = 0.35, 0.5, D_CHEM
        pp.gap_pairs, pp.gap_pair_weights = tuple(pairs), tuple([W_JUNCTION * g_junction] * len(pairs))
        pp.gap_ap_mv, pp.gap_ap_ms, pp.gap_delay_ms = AP_MV, AP_MS, GAP_DELAY
        pp.gap_rectify = True   # measured: GF -> TTMn/PSI junctions rectify (Phelan 2008)
        pp.v_rest_override = {b: -45.0 - GAP_DLM_MV for b in dlm_ids}
        pp.r_in_override = {b: float(K_DLM * GAP_DLM_MV / (ws * PSP5)) for b, ws in zip(dlm_ids, wsum)}
        cable = self.meta["l2_cable_length_um"]; med = float(cable.median())
        self.gf_rin = {sids[i]: (float(med / cable.iloc[i]) if gf_rin == "derived" else float(GF_RIN if gf_rin is None else gf_rin)) for i in self.gf}
        pp.r_in_override.update(self.gf_rin)
        ts = np.full(len(self.meta), float(pp.tau_syn), dtype=np.float32); ts[self.dlm] = TAU_SYN_DLM; pp.tau_syn = ts
        tm = np.full(len(self.meta), float(pp.tau_mem), dtype=np.float32); tm[self.dlm] = TAU_MEM_DLM; pp.tau_mem = tm
        tr = np.full(len(self.meta), float(pp.t_refrac), dtype=np.float32); tr[self.dlm] = 7.0; tr[self.ttm] = 1.0; pp.t_refrac = tr
        # the giant fibre keeps its raw E/I (balance_exempt, lane E's T4a
        # precedent): the network-wide rule scaled one glutamatergic input
        # (PVLP010, 132 synapses) 7.7x and the loom drove the GF to -90 mV
        # (measured 2026-09-12); the animal's GF fires to a loom. Special-cell
        # rule, rank 3; PVLP010's sign is a conflict candidate.
        ex = np.zeros(len(self.meta), dtype=bool); ex[self.gf] = True; pp.balance_exempt = ex
        if params_fn is not None:
            params_fn(pp)   # e.g. adaptation, w_syn scale (escape_factorial)
        self.net = cns.CNS(self.meta, edges, pp)

    def trial(self, rate_hz: float, seed: int, stim_ms: float = 30.0, window_ms: float = 40.0, lc4_w: float = 1.0, lplc2_w: float = 1.0) -> dict:
        drive = {}
        if lc4_w > 0:
            drive[tuple(int(i) for i in self.lc4)] = rate_hz * lc4_w
        if lplc2_w > 0:
            drive[tuple(int(i) for i in self.lplc2)] = rate_hz * lplc2_w
        r = self.net.run(window_ms, drive=drive, seed=seed, drive_window=(0.0, stim_ms), first_spike=True, record=np.arange(self.net.N))
        tf = np.asarray(r["t_first_ms"], dtype=float)
        def first(rows):
            v = tf[rows]; return float(np.nanmin(v)) if np.isfinite(v).any() else None
        gf_t = first(self.gf)
        return {"seed": seed, "escape": gf_t is not None and gf_t <= window_ms, "gf_ms": gf_t, "ttm_ms": first(self.ttm), "dlm_ms": first(self.dlm)}


def probability(ve: VisualEscape, rate_hz: float, n: int, **kw) -> dict:
    kw.setdefault("window_ms", WINDOW_MS)   # light-off jump latency 60-80 ms measured (Trimarchi and Schneiderman 1995)
    rows = [ve.trial(rate_hz, s, **kw) for s in range(n)]
    esc = [r for r in rows if r["escape"]]
    return {"rate_hz": rate_hz, "n": n, "p_escape": len(esc) / n, "gf_ms_mean": float(np.mean([r["gf_ms"] for r in esc])) if esc else None,
            "ttm_ms_mean": float(np.mean([r["ttm_ms"] for r in esc if r["ttm_ms"] is not None])) if esc else None, "trials": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--calibrate", action="store_true"); ap.add_argument("--arms", action="store_true"); ap.add_argument("--aged", action="store_true")
    ap.add_argument("--rate", type=float, default=None); ap.add_argument("--n", type=int, default=24); a = ap.parse_args()
    t0 = time.time()
    if a.calibrate:
        ve = VisualEscape(1.0, 1.0); out = []
        for rate in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
            r = probability(ve, rate, a.n); r.pop("trials"); out.append(r)
            print(f"rate {rate:5.1f} Hz: p_escape {r['p_escape']:.2f}  GF {r['gf_ms_mean']}  TTM {r['ttm_ms_mean']}", flush=True)
        (HERE / "2026-09-12-escape-visual-calibration.json").write_text(json.dumps(out, indent=1))
    # Two stimulus types, each with its own frozen rate (calibrated on the
    # young control to the measured >80%): LIGHT-OFF, the Gaitanidis assay (a
    # 20 ms dark flash), drives LC4 (dimming-responsive) with LPLC2 at 0.2
    # (assumed: LPLC2 is loom-selective and weak to full-field dimming,
    # Klapoetke 2017); LOOM drives both pools equally. The measured
    # constraints: LC4 silenced (TeTxLC) removes the retina-evoked GF response;
    # LPLC2 optogenetics alone escapes ~85% at any age (Gaitanidis 2025).
    # rates frozen on the derived GF and transferred counts (2026-09-12-escape-visual-calibration-transfer.json)
    STIMS = {"lightoff": (LIGHTOFF_HZ, {"lplc2_w": 0.2}), "loom": (LOOM_HZ, {}),
             "lc4_opto": (50.0, {"lplc2_w": 0.0}), "lplc2_opto": (50.0, {"lc4_w": 0.0})}
    if a.arms:
        out = {}
        for p in (1.0, 0.5, 0.0):
            ve = VisualEscape(p_lc4=p, p_lplc2=1.0)
            for stim, (rate, kw) in STIMS.items():
                r = probability(ve, rate, a.n, **kw); r.pop("trials"); r["stim"] = stim; out[f"p_lc4={p}/{stim}"] = r
                print(f"LC4 x{p:.1f} {stim:10s} {rate:5.1f} Hz: p_escape {r['p_escape']:.2f}  GF {r['gf_ms_mean']}  TTM {r['ttm_ms_mean']}", flush=True)
        (HERE / "2026-09-12-escape-visual-arms.json").write_text(json.dumps(out, indent=1))
    if a.aged:
        # ageing state per the consultant's design: g(t) at the junction (latency),
        # p(t) at LC4 -> GF (probability); ibuprofen = p(t) collapse delayed
        # (aged females, 0.5 uM from day 30, J Transl Med 2026), so the aged
        # ibuprofen arm keeps more of p at the same g. Declared readings: g alone
        # moves latency and not probability; p alone moves probability; the
        # ibuprofen arm sits between aged and young on probability at aged latency.
        out = {}
        for label, g, p in (("young", 1.0, 1.0), ("aged_junction", 0.25, 1.0), ("aged", 0.25, 0.5),
                            ("aged_ibuprofen", 0.25, 0.75), ("aged_starved", 0.25, 0.0)):
            ve = VisualEscape(p_lc4=p, p_lplc2=1.0, g_junction=g)
            for stim in ("lightoff", "loom"):
                rate, kw = STIMS[stim]
                r = probability(ve, rate, a.n, **kw); r.pop("trials"); r.update(g_junction=g, p_lc4=p, stim=stim); out[f"{label}/{stim}"] = r
                print(f"{label:15s} {stim:8s} g {g:.2f} p {p:.2f}: p_escape {r['p_escape']:.2f}  GF {r['gf_ms_mean']}  TTM {r['ttm_ms_mean']}", flush=True)
        (HERE / "2026-09-12-escape-visual-aged.json").write_text(json.dumps(out, indent=1))
    print(f"wall {time.time() - t0:.0f} s")
