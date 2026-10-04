# Reproduction map

## 1. Verify the release

```sh
python3 tools/verify_release.py
```

The verifier uses Python's standard library. It checks the release manifest, Python syntax, frozen gate decisions, confusion-matrix arithmetic, generation totals, and stored tokenizer/parity evidence. It does not rerun the experiments. A matching hash proves identity, not correctness of the underlying measurement.

## 2. Prepare public development inputs

The legacy source is `aimo-interp/baselines` commit `7e8839966750059b6b1d247a12ab56552c79b342`. The current source is `aimo-interp/train-main-v2` revision `23791d9b120132087877b0459f953d5959fc8455`. Model and tokenizer revision: `6e8885a6ff5c1dc5201574c8fd700323f23c25fa`. `public-inputs.json` records complete input and output digests, exposed group identities, and the three pilot groups.

Use a disposable working copy with pandas, pyarrow, transformers, and tokenizers installed, then run:

```sh
python3 tools/prepare_inputs.py --fetch
```

For offline preparation, provide `--baseline-csv`, `--current-parquet`, and `--tokenizer-dir` instead. Only three tokenizer files are read. The helper recreates 109 legacy development/tuning cases, 53 current development rows, three label-free pilot cases, and 109 native token-ID pairs. It never opens the local sealed holdout, loads weights, fits a model, or submits a bundle. Raw public downloads remain subject to upstream terms.

## 3. Local mechanisms

| Stage | Code | Input and recorded environment | Reproduction boundary |
|---|---|---|---|
| Representation extraction | `experiments/representation-extraction-kaggle-v1/aimo-representation-extraction.py` | Pinned public CSV, solve prompt, and model; torch 2.10.0+cu128 | Forward passes only; extraction includes all 137 public prompts, including final prompts. The published script now accepts a relative output directory or `AIMO_EXTRACTION_OUTPUT`. No representation array is distributed. |
| Extraction audit | `experiments/representation-extraction-audit/validate_output.py` | Exactly the three extracted output files | Validates bounded numeric artifacts; do not use a new extraction to overwrite preserved evidence. |
| Development search | `experiments/representation-probe-falsifier-v1/development_screen.py` | Reviewed representation array and row map; scikit-learn version in `development_results.json` | 82 development examples, 740 configurations, 25 folds each; final labels used only mechanically for the original stratified allocation. |
| Ensembles | `experiments/representation-probe-falsifier-v1/development_ensembles.py` | Identity-pinned development record | Fourteen development-only ensembles. |
| Frozen tuning | `experiments/representation-probe-falsifier-v1/tuning_gate.py` | Both pinned development records, same representation array | Five candidates, no retuning. All fail; `open_final=false`, `selected_candidate=null`. The script does not compute a final result. |
| Layer geometry | `experiments/layer-geometry-v1/experiment.py` | Same representation array and frozen split | One geometry candidate. Previously exposed tuning data; no independent confirmation. |
| Tokenizer control | `experiments/tokenizer-fidelity-v1/check.py` | Prepared legacy inputs and pinned tokenizer | `python3 experiments/tokenizer-fidelity-v1/check.py local`; no weights. Stored local environment: transformers 5.6.2, tokenizers 0.22.2, huggingface-hub 1.23.0. Stored host environment: transformers 5.13.0, tokenizers 0.22.2, huggingface-hub 1.22.0. |
| Runtime parity | `experiments/tokenizer-fidelity-v1/parity/` | Seeded synthetic float32 weights, plan.json, transformers 5.13.0, mlx-lm 0.31.3 | Run Torch and MLX scripts in separate processes, then compare.py. No real model weights or competition data are used. Stored comparison is provided; release checks do not rerun it. |
| Newline intervention | `experiments/intervention-jsd-native-v1/extract.py`, then `fit.py` | Prepared native token IDs and exposed labels; original checkpoint precision on MLX | Extraction must exit before the one CPU fit. The earlier `intervention-jsd-v1/extract.py` is retained as the failed-tokenizer path, not an alternative passing method. |
| Sampled-answer screen | `experiments/answer-agreement-pilot-v1/pilot.py`, `audit.py` | Three current development groups, no robustness labels | MLX, two samples per problem, 2,048 tokens per sample, 1,800 inference seconds, 24 GiB cap. Saved raw generations are not distributed. `audit.py --self-test` exercises only the answer parser. |
| Batch allocation | `experiments/batch-prevalence-rank-v1/runner.py` | Prepared 53 current development rows; Python 3.13.12, pandas 3.0.1, scikit-learn 1.8.0 | Seven grouped folds, seed 20260921. `--check` verifies input identity without fitting. The public registration is a protocol extract; its digest differs from the private registration. |

Run model-dependent stages only as deliberate reproductions with suitable local hardware. The failed gates do not authorize additional tuning or opening a final set. Timings, stochastic generations, GPU numerics, and exact byte identities can vary across environments. No fresh model experiment was run to prepare this release.

Frozen JSON records are provided beside the code. Before any deliberate rerun, copy them out of the disposable working directory and preserve their hashes. Several scripts refuse to overwrite completed outputs. The release verifier checks the preserved records, not new rerun outputs.

## 4. Numeric forest component

The release contains the original restricted converter and data-only predictor, not the full upstream generation pipeline or a new submission bundle. In a disposable working copy, obtain the organizer's pinned source separately:

```sh
git clone https://github.com/aimo-interp/getting-started.git
git -C getting-started checkout de794053debe75a711400696e66b60937cebbb1b
python3 audit/convert_official_forest.py
python3 audit/test_numeric_forest.py
```

The converter accepts only the reviewed 1,007,536-byte artifact with SHA-256 `bc90bd8eed4d9f6383cbb9ae1626896afea90b0e71c46d9a56bc87d376f85592` and a restricted serialization schema. Expected numeric output SHA-256: `c5deba2bffbcac89e909c1b77cb6ae8bc26bc891ca5f55fb12f971b6987c4164`. The tests compare 1,024 synthetic vectors, including missing values, to tolerance 1e-15, check threshold decisions, and reject a changed artifact hash before parsing. They load no transformer or language-model weights.

The original converter was built against scikit-learn 1.8.0. The release check environment uses numpy 2.4.3, scikit-learn 1.8.0, joblib 1.5.3, and safetensors 0.7.0. Do not deserialize a different artifact or treat a filename as provenance.

## 5. Hosted control reproduction

The scored rules have no training data or learned parameters. They return native Python `True` for each input problem in order. Python's standard library and the pinned organizer repository suffice to rebuild the archives:

```sh
cd getting-started
python3 scripts/build.py solutions
python3 scripts/build.py solutions --small
cd ..
```

Main archive SHA-256: `07dcb7ef79933501b3b6f47f49f29a70bc6e101c3128b990a61e82b7e02011e9`.
Small archive SHA-256: `4a3718e38f915a49ab1a74c5d99f45ee2814fb2845ebb1da3cb0daaa0b249e0a`.
At the recorded October 4 checkpoint, phase 30038 used Main task 36237 and Small task 36241. Submission IDs were 959561 and 959564. The Small archive adds an empty root-level `small.txt` marker. Accuracy on a later task revision can change. The subsequent HTTP 500 event has no submission ID and no score to reproduce.
