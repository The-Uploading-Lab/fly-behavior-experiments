"""Apply the stored Karolina intervention to the delivered full fly and record it."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lanes.longevity.ageing import integrated_probe as IP
from lanes.longevity.ageing import escape_membrane as EM
import matched_body as M
from capture_delivered import verify_delivery
from successor_body import dense_readout

HERE = Path(__file__).resolve().parent
BASE = ROOT/'lanes/longevity/ageing/candidate-fixed-native-walk-visual125-lplc2-quarter-complete-ttm.json'
DELIVERY = HERE/'delivery-fixed-native-walk-visual125-lplc2-quarter-complete-ttm.json'
DECLARATION = HERE/'KAROLINA-COMPARISON.md'
ADAPT = {'adapt_a': 1.0, 'adapt_b': 2.0, 'tau_w': 100.0}
ARMS = ('control', 'ibuprofen', 'deprivation_b_half')


def apply_intervention(theta, arm, meta):
    result = deepcopy(theta)
    if arm == 'ibuprofen':
        result.update(ADAPT)
    elif arm == 'deprivation_b_half':
        visual = meta.loc[meta.cell_type.astype(str).isin(('LC4', 'LPLC2'))]
        gf = meta.loc[meta.cell_type.astype(str).eq('DNp01')]
        if (set(visual.cell_type) != {'LC4', 'LPLC2'} or len(gf) != 2
                or not visual.banc_888_id.is_unique or not gf.banc_888_id.is_unique):
            raise ValueError('canonical visual/GF identities do not match')
        result['row_pair_seams'] = deepcopy(result.get('row_pair_seams') or []) + [{
            'pre_ids': visual.banc_888_id.astype(str).tolist(),
            'post_ids': gf.banc_888_id.astype(str).tolist(), 'gain': 0.5}]
    elif arm != 'control':
        raise ValueError('undeclared intervention')
    return result


def run(arm, protocol, seed, out):
    if out.exists() or Path.cwd().resolve() != ROOT:
        raise ValueError('use a new output directory from the lane checkout')
    verify_delivery(DELIVERY)
    out.mkdir(parents=True)
    started = time.perf_counter()
    candidate = json.loads(BASE.read_text())
    if arm != 'control':
        candidate.update(id=candidate['id']+'-'+arm, karolina_intervention=arm)
    before_sources = IP.source_hashes()
    bodies, live, theta_change = [], [], []
    original_body, original_neural = IP.JA.ColourBody, IP.JA.cns.CNS.run
    original_theta = IP.candidate_theta

    class ObservedBody(original_body):
        def __init__(self, *a, **kw):
            kw['render_path'] = None
            super().__init__(*a, **kw)
            self.pose_time, self.pose_qpos, self.pose_xyz = [], [], []
            self.next_pose = 0.0
            bodies.append(self)

        def _capture(self):
            d = self.h['data']
            if d.time + 1e-9 >= self.next_pose:
                self.pose_time.append(float(d.time))
                self.pose_qpos.append(d.qpos.copy())
                self.pose_xyz.append(d.xpos[self.h['thorax']].copy())
                self.next_pose = len(self.pose_time)/240.0

    def changed_theta(theta, config, meta=None):
        base = original_theta(theta, config, meta)
        changed = apply_intervention(base, arm, meta)
        difference = {k: {'before': base.get(k), 'after': changed.get(k)}
                      for k in set(base) | set(changed) if base.get(k) != changed.get(k)}
        expected = set(ADAPT) if arm == 'ibuprofen' else {'row_pair_seams'} if arm == 'deprivation_b_half' else set()
        if set(difference) != expected:
            raise ValueError(f'unexpected parameter delta: {set(difference)}')
        theta_change.append(difference)
        return changed

    def observed_neural(net, *a, **kw):
        actual = {key: np.broadcast_to(np.asarray(getattr(net.p, key)), (net.N,)) for key in ADAPT}
        if arm == 'ibuprofen' and any(not np.all(actual[k] == v) for k, v in ADAPT.items()):
            raise ValueError('ibuprofen parameters did not reach every neuron')
        live.append({'neurons': net.N, 'parameters': M.parameter_fingerprint(net.p),
                     'adaptation': {k: {'min': float(v.min()), 'max': float(v.max())} for k, v in actual.items()}})
        return original_neural(net, *a, **kw)

    duration = 6000. if protocol == 'walk' else 3000.
    try:
        with IP.patched(IP, 'candidate_theta', changed_theta), \
                IP.patched(IP.JA, 'ColourBody', ObservedBody), \
                IP.patched(IP, 'run', M.seeded_call(IP.run, seed)), \
                IP.patched(IP.JA.cns.CNS, 'run', observed_neural):
            integrated = EM.run(candidate, protocol=protocol, duration_ms=duration)
        if len(bodies) != 1 or len(live) != 1 or len(theta_change) != 1 or integrated['seed'] != seed:
            raise ValueError('unexpected simulation count or seed')
        b = bodies[0]
        state = np.asarray(b.rows)
        row = dense_readout(integrated, state, protocol)
        times = np.asarray(b.pose_time)
        if len(times) != int(duration*240/1000) or np.max(np.diff(times)) > 1/240+.000201:
            raise ValueError('incomplete 240 Hz pose observation')
        if integrated['review_required'] or any(IP.sha(ROOT/p) != h for p, h in before_sources.items()):
            raise ValueError('source or authority changed during execution')
        verify_delivery(DELIVERY)
        np.savez_compressed(out/'poses.npz', time_s=times, qpos=b.pose_qpos,
                            thorax_xyz=b.pose_xyz, body_state=state)
        IP.JA.mj.mj_saveModel(b.h['model'], str(out/'body.mjb'), None)
        (out/'integrated-receipt.json').write_text(json.dumps(integrated, indent=2, allow_nan=False)+'\n')
        (out/'readout.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
        manifest = {'schema': 'karolina-r2-body-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'arm': arm, 'protocol': protocol, 'seed': seed, 'duration_ms': duration,
                    'declaration_sha256': IP.sha(DECLARATION), 'baseline_delivery_sha256': IP.sha(DELIVERY),
                    'theta_changes': theta_change[0], 'live_neural_parameters': live[0],
                    'source_sha256': IP.source_hashes(), 'source_commit': IP.git('rev-parse', 'HEAD'),
                    'files': {p.name: IP.sha(p) for p in out.iterdir()},
                    'timing_s': {**integrated['timing_s'], 'worker_total': time.perf_counter()-started},
                    'status': 'complete', 'exposure': 'R2 measurements read before run; no fitting',
                    'parameter_time_model': 'fixed intervention; no exposure-time kinetics'}
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'out': str(out), 'arm': arm, 'protocol': protocol, 'seed': seed,
                          'retention_pass': row['successor_retention_pass'],
                          'adaptation': live[0]['adaptation'], 'timing_s': manifest['timing_s']}), flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'error': f'{type(exc).__name__}: {exc}',
             'elapsed_s': time.perf_counter()-started, 'arm': arm, 'protocol': protocol, 'seed': seed}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--protocol', choices=('walk', 'puff', 'visual', 'sham'), required=True)
    parser.add_argument('--seed', type=int, choices=(0, 1), default=0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.arm, args.protocol, args.seed, args.out)
