"""Numpy-only comparison using thresholds preregistered in plan.json."""
from pathlib import Path
import hashlib
import json
import numpy as np
here = Path(__file__).resolve().parent
plan = json.loads((here/'plan.json').read_text())
checks = []
def check(name, reference, observed):
    assert reference.shape == observed.shape
    error = float(np.max(np.abs(reference.astype(np.float64)-observed.astype(np.float64))))
    normalized = error / max(1., float(np.max(np.abs(reference))))
    passed = bool(np.isfinite(reference).all() and np.isfinite(observed).all()
                  and error <= plan['max_absolute_logit_error'] and normalized <= plan['max_normalized_error'])
    checks.append({'name': name, 'max_absolute_error': error, 'normalized_error': normalized, 'passed': passed})
with np.load(here/'torch_outputs.npz') as t, np.load(here/'mlx_outputs.npz') as m:
    assert set(t.files) == set(m.files)
    for k in t.files:
        check('cross-runtime/'+k, t[k], m[k])
    for label, arrays in [('torch',t),('mlx',m)]:
        for i in range(len(plan['prompts'])):
            prefix = np.concatenate([arrays[f'prefix_{i}_{j}'] for j in range(len(plan['common_continuation']))],axis=1)
            check(f'{label}/matched-prefix-{i}', arrays[f'matched_{i}'], prefix)
            check(f'{label}/cache-{i}', arrays[f'matched_{i}'], arrays[f'cached_{i}'])
result = {'gate_passed': all(c['passed'] for c in checks), 'checks': checks,
          'maximum_absolute_error': max(c['max_absolute_error'] for c in checks),
          'tolerance_declared_before_execution': True, 'max_absolute_tolerance':plan['max_absolute_logit_error'],
          'real_weights_or_competition_data_read': False,
          'plan_sha256':hashlib.sha256((here/'plan.json').read_bytes()).hexdigest(),
          'torch':json.loads((here/'torch_result.json').read_text()),
          'mlx':json.loads((here/'mlx_result.json').read_text()),'limits':plan['limits']}
(here/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
raise SystemExit(0 if result['gate_passed'] else 1)
