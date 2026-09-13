"""Capture the delivered model without adding the legacy walking schedule."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lanes.longevity.ageing import capture_candidate as CC


def walking_receipt(integrated):
    receipt = deepcopy(integrated["result"])
    receipt.update(track=[[t, x, y] for t, x, y, z, up in integrated["winged_track"]],
                   walk_fraction_input=1.0, schedule_s=[[0., integrated["duration_ms"] * .001]],
                   schedule_is_imposed=False, rest="no imposed rest schedule; continuous neural joint transfer")
    return receipt


def extra_parity(actual, expected, original):
    # The original steady-control receipt predates the voltage-only reader.
    # Reconstruct its runtime identity from its recorded theta and the
    # unchanged adapt_a=2 value; never invent an earlier voltage trace.
    legacy = (expected.get("candidate") == {"id": "steady-control", "wing_source": "steady"}
              and "runtime_theta_sha256" not in expected and "gf_membrane" not in expected)
    comparison_expected = deepcopy(expected)
    if legacy:
        comparison_expected["runtime_theta_sha256"] = CC.IP.identity({
            "base_theta_sha256": expected["theta_sha256"],
            "gf_adapt_a_by_id": {"720575941509145950": 2., "720575941451068597": 2.}})
    result = original(actual, comparison_expected)
    fields = ("winged_track", "ttm_spikes_by_side_ms", "wing_motor_rates", "sensory_spikes",
              "theta_sha256", "model_input_sha256", "data_sha256")
    if not legacy:
        fields += ("gf_membrane",)
    for field in fields:
        if actual.get(field) != expected.get(field):
            result["different_fields"].append(field)
    result["exact"] = not result["different_fields"]
    result["scope"] += "; complete recorded trajectory, side-specific motor spikes and model input identities also compared"
    result["unavailable_in_original"] = ["gf_membrane"] if legacy else []
    if legacy:
        result["expected_runtime_identity_origin"] = "derived from original theta hash and unchanged GF adapt_a=2; original receipt preserved"
        result["derived_expected_runtime_theta_sha256"] = comparison_expected["runtime_theta_sha256"]
    else:
        result["scope"] += "; GF voltage also compared"
    return result


def verify_delivery(delivery_path):
    delivery = json.loads(delivery_path.read_text())
    if not delivery.get("files"):
        raise ValueError("delivery must identify its source and receipt files")
    for entry in delivery["files"]:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(ROOT) or CC.IP.sha(path) != entry["sha256"]:
            raise ValueError(f"delivered source or receipt changed: {entry['path']}")
    return delivery, CC.IP.sha(delivery_path)


def run(candidate, protocol, expected, out, delivery_path=None):
    delivery_path = (Path(delivery_path) if delivery_path is not None else
                     Path(__file__).with_name("delivery-antennal-gain105.json"))
    delivery, delivery_sha = verify_delivery(delivery_path)
    original_em, original_parity = CC.EM.run, CC.parity
    completed = []

    def record_integrated(*args, **kwargs):
        value = original_em(*args, **kwargs)
        completed.append(value)
        return value

    def direct_walking(arm, seed, render_path=None, dur_ms=6000.):
        # CC has installed the candidate entry point here. Calling it
        # directly avoids WA.run's PUPPET_GATE and its unused outer track.
        CC.IP.JA.run(arm, seed, dur_ms=dur_ms)
        return walking_receipt(completed[-1])

    with CC.IP.patched(CC.EM, "run", record_integrated), \
            CC.IP.patched(CC, "parity", lambda a, e: extra_parity(a, e, original_parity)):
        if protocol == "walk":
            with CC.IP.patched(CC.CP.WA, "run", direct_walking):
                manifest = CC.run(candidate, protocol, expected, out)
        else:
            manifest = CC.run(candidate, protocol, expected, out)
    if len(completed) != 1:
        raise RuntimeError("expected one integrated capture")
    if verify_delivery(delivery_path) != (delivery, delivery_sha):
        raise RuntimeError("delivery changed during capture; saved result requires review")
    manifest.update(delivery_commit=delivery["delivery_commit"],
                    delivery_manifest=str(delivery_path), delivery_manifest_sha256=delivery_sha,
                    limits=completed[0]["limitations"],
                    production_wrapper_sha256=CC.IP.sha(__file__))
    if protocol == "walk":
        manifest["walking_schedule_mode"] = "continuous_neural_joint_transfer; no_imposed_rest_schedule"
    (Path(out) / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--protocol", choices=("puff", "visual", "sham", "walk"), required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--delivery", type=Path,
                   help="frozen delivery manifest; default is the original gain105 delivery")
    a = p.parse_args()
    m = run(a.candidate, a.protocol, a.expected, a.out, a.delivery)
    print(json.dumps({"out": str(a.out), "scientific_parity": m["scientific_parity"]}), flush=True)
