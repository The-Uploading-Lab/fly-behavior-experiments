"""Verify and summarize the source-review snapshot, using only the standard library."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, entry in manifest['files'].items():
        path = ROOT / name
        data = path.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError(f'Snapshot mismatch: {name}')
        if path.suffix == '.py':
            compile(data, str(path), 'exec')
    return manifest


def main():
    manifest = verify()
    theta = json.loads((ROOT / 'config/retained-build.json').read_text())
    print('Snapshot hashes and Python syntax: PASS')
    print('Formal champion: wm1; retained research build: 7167953a (2026-10-15)')
    print('Research run source:', manifest['source_commit'])
    print(f'{len(theta)} runtime keys; source review only, no full-network replay performed.')
    for name, keys in {
        'Neural': ('cbased', 'shunt', 'delay', 'delay_mode', 'adapt_a', 'adapt_b', 'tau_w', 'graded'),
        'Muscle/body': ('drive_scale', 'f0_hz', 'zeta', 'joint_int', 'k_mode', 'k_min', 'k_gain', 'cocon_scale'),
        'Sensory/command': ('cmd', 'cmd_hz', 'spike_reg', 'tick_ms', 'camp_src', 'camp_gain', 'camp_match_factor', 'F_ref', 'phasic', 'proprioceptor_phasic_mode'),
    }.items():
        print(name + ':')
        for key in keys:
            print(f'  {key}: {theta[key]}')


if __name__ == '__main__':
    main()
