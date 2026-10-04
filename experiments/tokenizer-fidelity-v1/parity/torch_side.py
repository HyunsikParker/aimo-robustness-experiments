"""Host-version definition; synthetic weights; Torch-only process."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', USE_FLAX='0')
import sys
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
import transformers
import sympy
import mpmath
from transformers import Qwen3Config, Qwen3ForCausalLM

here = Path(__file__).resolve().parent
plan = json.loads((here / 'plan.json').read_text())
assert transformers.__version__ == '5.13.0'
assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)
torch.set_num_threads(2)
torch.manual_seed(plan['seed'])
config = Qwen3Config(**plan['config'])
config._attn_implementation = 'eager'
model = Qwen3ForCausalLM(config).float().cpu().eval()
weights = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
np.savez(here / 'synthetic_weights.npz', **weights)
results = {}
with torch.no_grad():
    for i, prompt in enumerate(plan['prompts']):
        continuation = plan['common_continuation']
        seq = prompt + continuation[:-1]
        full = model(torch.tensor([seq]), use_cache=False).logits.numpy()
        results[f'full_{i}'] = full
        results[f'matched_{i}'] = full[:, len(prompt)-1:len(prompt)-1+len(continuation)]
        for j in range(len(continuation)):
            prefix = prompt + continuation[:j]
            results[f'prefix_{i}_{j}'] = model(torch.tensor([prefix]), use_cache=False).logits[:, -1:].numpy()
        pref = model(torch.tensor([prompt]), use_cache=True)
        cached = [pref.logits[:, -1:].numpy()]
        for tok in continuation[:-1]:
            pref = model(torch.tensor([[tok]]), past_key_values=pref.past_key_values, use_cache=True)
            cached.append(pref.logits.numpy())
        results[f'cached_{i}'] = np.concatenate(cached, axis=1)
np.savez(here / 'torch_outputs.npz', **results)
report = {'transformers': transformers.__version__, 'torch': torch.__version__,
          'sympy': sympy.__version__, 'mpmath': mpmath.__version__,
          'device': 'cpu', 'dtype': str(next(model.parameters()).dtype),
          'mlx_imported': False, 'parameter_count': sum(x.size for x in weights.values()),
          'rope_parameters_effective': config.rope_parameters,
          'rope_attention_scaling_effective': model.model.rotary_emb.attention_scaling,
          'weights_sha256': hashlib.sha256((here / 'synthetic_weights.npz').read_bytes()).hexdigest()}
(here / 'torch_result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
