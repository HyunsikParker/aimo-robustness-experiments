"""MLX-only process reads exported synthetic float32 weights."""
import os
os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import sys
from pathlib import Path
import json
import importlib.metadata
import types
import numpy as np
import mlx.core as mx
# Bypass only the convenience package initializer, which eagerly imports tokenizer/
# Transformers utilities and thereby Torch despite USE_TORCH=0 in this environment.
# Model, attention, rope, activation, and cache source modules are imported unchanged.
package_root = Path(importlib.metadata.distribution('mlx-lm').locate_file('mlx_lm'))
for package, folder in [('mlx_lm', package_root), ('mlx_lm.models', package_root/'models')]:
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(folder)]
    sys.modules[package] = namespace
from mlx_lm.models.qwen3 import Model, ModelArgs
from mlx_lm.models.cache import make_prompt_cache

here = Path(__file__).resolve().parent
plan = json.loads((here / 'plan.json').read_text())
assert 'torch' not in sys.modules
assert importlib.metadata.version('mlx-lm') == '0.31.3'
model = Model(ModelArgs(**plan['config']))
with np.load(here / 'synthetic_weights.npz', allow_pickle=False) as stored:
    model.load_weights([(k, mx.array(stored[k])) for k in stored.files], strict=True)
model.eval()
mx.eval(model.parameters())
results = {}
for i, prompt in enumerate(plan['prompts']):
    continuation = plan['common_continuation']
    seq = prompt + continuation[:-1]
    full = np.array(model(mx.array([seq])))
    results[f'full_{i}'] = full
    results[f'matched_{i}'] = full[:, len(prompt)-1:len(prompt)-1+len(continuation)]
    for j in range(len(continuation)):
        results[f'prefix_{i}_{j}'] = np.array(model(mx.array([prompt+continuation[:j]])))[:, -1:]
    cache = make_prompt_cache(model)
    cached = [np.array(model(mx.array([prompt]), cache=cache))[:, -1:]]
    for tok in continuation[:-1]:
        cached.append(np.array(model(mx.array([[tok]]), cache=cache)))
    results[f'cached_{i}'] = np.concatenate(cached, axis=1)
np.savez(here / 'mlx_outputs.npz', **results)
assert 'torch' not in sys.modules
report = {'mlx_lm': importlib.metadata.version('mlx-lm'), 'mlx': importlib.metadata.version('mlx'),
          'import_mode': 'Unmodified model modules; bypass eager package initializer to keep Torch absent',
          'torch_imported': False, 'default_device': str(mx.default_device()),
          'dtype': str(model.model.embed_tokens.weight.dtype),
          'rope_mscale_effective': model.layers[0].self_attn.rope.mscale}
(here / 'mlx_result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
