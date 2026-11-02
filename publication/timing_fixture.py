"""Small CPU correctness fixture, not a whole-network simulation.

An excitatory source connects to one silent inhibitory target. A single
forced source spike tests presynaptic delivery and neural-step rounding.
The inhibitory target has no outgoing edge; its label preserves the engine's
inhibition-construction assertion without disabling that guard.
"""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'delay-model'
OUTPUT=ROOT.parent/'outputs/publication/timing-fixture.json'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
from analyse import verify
verify()
os.chdir(SOURCE);sys.path.insert(0,str(SOURCE))
import cns

meta=pd.DataFrame(dict(banc_888_id=['source','target'],
                       neurotransmitter_verified=['acetylcholine','gaba'],
                       neurotransmitter_predicted=['acetylcholine','gaba'],
                       super_class=['other','other'],cell_type=['fixture','fixture']))
edges=pd.DataFrame(dict(pre=['source'],post=['target'],count=[1]))
rows=[]
for step in (.1,.05):
    for delay in (0.,.04,.05,.06,.15,1.8,6.24,23.623):
        traces=[];onsets=[]
        for delivery in ('bucket','queue'):
            for target_delay in (.1,9.):
                params=cns.SimParams(dt=step,backend='numpy',delay_per_neuron=np.array([delay,target_delay]),
                                     delay_delivery=delivery,v_threshold=100.)
                with contextlib.redirect_stdout(io.StringIO()):
                    network=cns.CNS(meta,edges,params)
                    result=network.run(26.,force_spikes={0.:[0]},record_voltage=[1])
                trace=np.asarray(result['voltage']).ravel()
                changed=np.flatnonzero(trace!=params.v_rest)
                assert len(changed),'No postsynaptic response'
                first=int(changed[0]);expected=max(1,int(np.rint(delay/step)))
                assert first==expected,(step,delay,delivery,first,expected)
                traces.append(trace);onsets.append(first)
                rows.append(dict(step_ms=step,presynaptic_delay_ms=delay,target_delay_ms=target_delay,
                                 delivery=delivery,first_changed_voltage_index=first,
                                 observed_latency_ms=first*step,rounding_error_ms=first*step-delay))
        assert len(set(onsets))==1,'The target delay changed incoming latency'
        assert all(np.array_equal(traces[0],t) for t in traces[1:]),'Queue/bucket or target-delay mismatch'
out=dict(status='passed',kind='Two-neuron CPU correctness fixture',cases=rows,
         source_sha256=hashlib.sha256((SOURCE/'cns.py').read_bytes()).hexdigest(),
         checks=['No postsynaptic voltage change before the source delay slot',
                 'Changing target delay does not change the incoming response',
                 'Zero delay still takes at least one neural step',
                 'Queue and bucket give identical voltage traces for this one-input circuit'],
         limits=['Does not establish full-network queue/bucket equivalence or Metal parity.',
                 'Does not test graded transmission, converging inputs or body-step convergence.',
                 'A voltage sample is read after its step update; latency here uses the engine step index.'])
OUTPUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],cases=len(rows),checks=out['checks'])))
