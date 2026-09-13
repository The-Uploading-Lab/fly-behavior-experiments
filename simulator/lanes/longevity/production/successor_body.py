"""Read dense successor captures and run the four declared numerical replays."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

import matched_body as M
from capture_delivered import verify_delivery
from read_body import load_capture, validate_state

ROOT, HERE, IP, EM = M.ROOT, M.HERE, M.IP, M.EM
DELIVERY = HERE / 'delivery-fixed-native-walk-bilateral25.json'
CANDIDATE = ROOT / 'lanes/longevity/ageing/candidate-fixed-native-walk-bilateral25.json'
DECLARATION = HERE / 'SUCCESSOR-BODY-COHORT.md'
PLAN = [('puff', 0), ('walk', 0), ('visual', 0), ('sham', 0),
        ('walk', 1), ('walk', 2), ('walk', 3), ('puff', 1)]


def trial_path(protocol, seed):
    if (protocol, seed) not in PLAN:
        raise ValueError('case outside the declared successor comparison')
    return (HERE / 'captures' / f'shared-{protocol}-s0' if seed == 0 else
            HERE / 'successor-body-v1' / f'shared-{protocol}-s{seed}')


def dense_readout(integrated, state, protocol):
    row = M.readout(integrated, state, protocol)
    result = integrated['result']
    quiet = not result['gf_spikes_ms'] and not any(integrated['ttm_spikes_by_side_ms'].values())
    if protocol == 'walk':
        gates = {'both_bodies_standing': row['cohort_retention_pass'],
                 'no_uncued_gf_or_ttm': quiet, 'no_takeoff': result['t_off_ms'] is None}
    elif protocol == 'sham':
        state = validate_state(state)
        gates = {'dense_posture_and_supported_finish': row['cohort_retention_pass'],
                 'upright_after_initialization': bool((state[state[:, 0] >= .3, 2] >= .5).all()),
                 'no_gf_or_ttm': quiet, 'no_takeoff': result['t_off_ms'] is None}
    else:
        gates = {'dense_flight_and_return_retention': row['cohort_retention_pass'],
                 'timely_gf': row['neural_trigger'], 'timely_takeoff': row['body_takeoff']}
    row.update(successor_retention_gates=gates, successor_retention_pass=all(gates.values()),
               live_neural_parameters_sha256=integrated['gf_membrane']['runtime_parameters']['sha256'])
    return row


def runtime_reference_review(path, manifest):
    """Default cohorts have no exception for a flagged runtime."""
    return False


def read_case(protocol, seed):
    path = trial_path(protocol, seed)
    manifest = json.loads((path / 'manifest.json').read_text())
    delivery, delivery_sha = verify_delivery(DELIVERY)
    if manifest['delivery_manifest_sha256'] != delivery_sha:
        raise ValueError('capture belongs to a different model delivery')
    if seed == 0:
        load_capture(path)
        archive = path / 'capture.npz'
    else:
        if (manifest['schema'] != 'longevity-successor-body-numerical-v1'
                or (manifest['runtime_review_required'] and not runtime_reference_review(path, manifest))
                or (manifest['protocol'], manifest['seed']) != (protocol, seed)
                or manifest['declaration_sha256'] != IP.sha(DECLARATION)):
            raise ValueError('unreviewed or mismatched numerical case')
        for name, expected in manifest['files'].items():
            if IP.sha(path / name) != expected:
                raise ValueError(f'numerical artifact changed: {name}')
        archive = path / 'body-state.npz'
    integrated = json.loads((path / 'integrated-receipt.json').read_text())
    expected_protocol = {'puff': 'antennal-puff', 'sham': 'antennal-sham', 'visual': 'escape', 'walk': 'walk'}[protocol]
    if (integrated['candidate'] != delivery['candidate_parameters']
            or integrated['candidate_sha256'] != delivery['candidate_sha256']
            or integrated['seed'] != seed or integrated['protocol'] != expected_protocol
            or integrated['duration_ms'] != (6000. if protocol == 'walk' else 3000.)
            or (integrated['review_required'] and not runtime_reference_review(path, manifest))
            or integrated['gf_membrane']['runtime_parameters']['sha256'] != delivery['runtime_parameters_sha256']):
        raise ValueError('integrated model, runtime parameters, seed or protocol differs')
    for field in ('engine', 'model_input_sha256', 'data_sha256'):
        if integrated[field] != delivery[field]:
            raise ValueError(f'delivered model identity differs: {field}')
    with np.load(archive, allow_pickle=False) as arrays:
        state = validate_state(arrays['body_state'])
        if len(state) != int(integrated['duration_ms']):
            raise ValueError('dense body observation window is incomplete')
        row = dense_readout(integrated, state, protocol)
    return {'model': delivery['candidate'], 'protocol': protocol, 'seed': seed,
            'source': str(path.relative_to(ROOT)), 'manifest_sha256': IP.sha(path / 'manifest.json'),
            'engine': integrated['engine'], 'model_input_sha256': integrated['model_input_sha256'],
            'data_sha256': integrated['data_sha256'], 'row': row}


def check_prerequisites(protocol, seed):
    """Require captured seed-zero retention before the original cohort advances."""
    if not all(read_case(mode, 0)['row']['successor_retention_pass'] for mode in ('puff', 'walk', 'visual', 'sham')):
        raise ValueError('complete the seed-zero capture and dense retention gate first')
    for earlier in range(1, seed):
        if not read_case(protocol, earlier)['row']['successor_retention_pass']:
            raise ValueError('an earlier seed closed this protocol')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=('walk', 'puff'), required=True)
    parser.add_argument('--seed', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if (args.protocol, args.seed) not in PLAN or Path.cwd().resolve() != ROOT:
        raise ValueError('run a declared nonzero case from the checkout')
    if args.out.resolve() != trial_path(args.protocol, args.seed) or args.out.exists():
        raise ValueError('the declared output path must be new')
    delivery, delivery_sha = verify_delivery(DELIVERY)
    check_prerequisites(args.protocol, args.seed)
    candidate = json.loads(CANDIDATE.read_text())
    before = IP.source_hashes()
    bodies, parameters = [], []
    original_body, original_neural = IP.JA.ColourBody, IP.JA.cns.CNS.run

    class ObservedBody(original_body):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            bodies.append(self)

    def observed_neural(net, *a, **kw):
        parameters.append(M.parameter_fingerprint(net.p))
        return original_neural(net, *a, **kw)

    duration = 6000. if args.protocol == 'walk' else 3000.
    with IP.patched(IP.JA, 'ColourBody', ObservedBody), \
            IP.patched(IP, 'run', M.seeded_call(IP.run, args.seed)), \
            IP.patched(IP.JA.cns.CNS, 'run', observed_neural):
        integrated = EM.run(candidate, protocol=args.protocol, duration_ms=duration)
    if len(bodies) != 1 or len(parameters) != 1 or integrated['seed'] != args.seed:
        raise RuntimeError('unexpected body, neural-run count or seed')
    state = validate_state(np.asarray(bodies[0].rows))
    row = dense_readout(integrated, state, args.protocol)
    changed = [path for path, expected in before.items() if IP.sha(ROOT / path) != expected]
    drift = bool(changed or integrated['review_required'] or verify_delivery(DELIVERY) != (delivery, delivery_sha))
    args.out.mkdir(parents=True)
    (args.out / 'integrated-receipt.json').write_text(json.dumps(integrated, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(args.out / 'body-state.npz', body_state=state)
    (args.out / 'readout.json').write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
    manifest = {'schema': 'longevity-successor-body-numerical-v1',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'protocol': args.protocol, 'seed': args.seed, 'candidate': candidate,
                'delivery_manifest_sha256': delivery_sha, 'declaration_sha256': IP.sha(DECLARATION),
                'source_sha256': IP.source_hashes(), 'changed_sources': changed,
                'full_runtime_parameters': parameters[0],
                'files': {p.name: IP.sha(p) for p in sorted(args.out.iterdir())},
                'runtime_review_required': drift, 'command': [sys.executable, *sys.argv],
                'scope': 'fixed selected model on reused development seeds; no independent animal or model validation'}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    if drift:
        raise RuntimeError('result preserved; source or authority review is required')
    verified = read_case(args.protocol, args.seed)
    print(json.dumps({'protocol': args.protocol, 'seed': args.seed,
                      'retention_pass': verified['row']['successor_retention_pass'],
                      'gates': verified['row']['successor_retention_gates'],
                      'timing_s': integrated['timing_s']}), flush=True)


if __name__ == '__main__':
    main()
