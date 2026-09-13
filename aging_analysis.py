# Portability adaptation of the aging lane analysis; fixed formulas and eligibility rules retained.
"""Read fixed ageing captures; preserve unavailable values and failed cases."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter

from portable_integrity import ROOT as PACKAGE_ROOT
ROOT=PACKAGE_ROOT/"simulator"

def metrics(t,xy,start,end):
    clock=np.arange(start,end+1e-10,1/120.)
    if clock[-1]<end-1e-10:clock=np.r_[clock,end]
    a=np.column_stack([np.interp(clock,t,xy[:,i]) for i in range(2)])
    a=median_filter(a,size=(5,1),mode='reflect')
    ix=np.unique(np.r_[np.arange(0,len(a),3),len(a)-1])
    length=float(np.linalg.norm(np.diff(a[ix],axis=0),axis=1).sum())
    net=float(np.linalg.norm(a[-1]-a[0]))
    return {'speed_mm_s':length/(end-start),'path_mm':length,'net_mm':net,
            'straightness':net/length if length else None}

def case(day,protocol,directory):
    receipt=directory/'integrated-receipt.json'
    if not receipt.exists():return None
    r=json.loads(receipt.read_text());npz=directory/'poses.npz'
    z=np.load(npz);state=z['body_state'];times=state[:,0]
    out={'age_days':day,'efficacy':2**(-day/45),'protocol':protocol,'seed':0,
         'trial_n':1,'receipt':str(receipt),'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),
         'capture':str(npz),'review_required':r['review_required'],
         'full_neurons':r['engine']['neurons'],'model':r['candidate']['id'],
         'model_source_commit':r.get('baseline_source_commit'),'aging_model':'global-synaptic-efficacy-half45-v1',
         'aging_applied_neurons':r.get('global_ageing',{}).get('assigned_neurons',188508 if day==0 else None),
         'new_run':True}
    if out['review_required']:raise ValueError(f'Unresolved source review: {receipt}')
    if day and out['aging_applied_neurons']!=188508:raise ValueError('Missing global coverage')
    if protocol=='walk':
        from lanes.longevity.ageing.walking_battery import assess
        gate=assess(r,json.loads((ROOT/'CONSTRAINT-REGISTRY.json').read_text()))
        out.update(standing=gate['standing'],internal_standing=gate['internal_body_standing'],
                   displayed_standing=gate['displayed_body_standing_without_takeoff'])
        out['status']='standing' if gate['standing'] else 'failed_standing; walking statistics censored'
        out.update({k:None for k in ('speed_mm_s','path_mm','net_mm','straightness','internal_speed_mm_s',
                                     'window_speed_mean_mm_s','window_speed_sd_mm_s')})
        out['window_n']=0;out['trial_sd_mm_s']=None
        if gate['standing']:
            t=z['time_s'];xy=z['thorax_xyz'][:,:2]
            if t[0]>.3 or t[-1]<5.99:raise ValueError('Walking source shorter than fixed interval')
            out.update(metrics(t,xy,.3,5.99));out['internal_speed_mm_s']=r['walking_regression'].get('speed')
            windows=[metrics(t,xy,.3+i,1.3+i) for i in range(5)]
            out['windows']=windows;out['window_n']=5
            speeds=[x['speed_mm_s'] for x in windows]
            out['window_speed_mean_mm_s']=float(np.mean(speeds))
            out['window_speed_sd_mm_s']=float(np.std(speeds,ddof=1))
        out['observation_s']=5.69
    else:
        onset=600.;horizon=1000*100/240;pre=(times>=.5)&(times<.6)
        if times[-1]*1000 < onset+horizon:raise ValueError('Insufficient response followup')
        eligible=bool(pre.any() and np.all(state[pre,2]>=.5) and np.all(state[pre,3]<=.15))
        gf=r['result']['gf_spikes_ms'];ttm=sorted(v for a in r['ttm_spikes_by_side_ms'].values() for v in a)
        native=r['result']['t_off_ms']
        first=lambda values:next((float(v)-onset for v in values if onset<=v<=onset+horizon),None)
        dep=None if native is None else float(native)-onset
        departed=bool(dep is not None and 0<=dep<=horizon)
        if native is not None and native<onset:eligible=False
        out.update(eligible=eligible,preinput_gf=sum(v<onset for v in gf),
            preinput_departure=native is not None and native<onset,
            gf_latency_ms=first(gf),ttm_latency_ms=first(ttm),
            departure_latency_ms=dep if departed else None,
            departure_by_416_7ms=departed,no_departure_through_ms=horizon if not departed and eligible else None,
            native_takeoff_time_ms=native,flight_clear_fraction=r['flight_clear_fraction'],
            final_upright=r['result']['upright_final'],
            status='pre-input support/posture failure' if not eligible else 'departure' if departed else 'no departure; right-censored',
            first_visible_movement_ms=None,movement_scope='Native support-loss proxy; arbitrary visible movement not separately detected')
    return out
