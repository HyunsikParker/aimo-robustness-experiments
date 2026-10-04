# AIMO robustness experiments

Code and frozen records for a study of robustness prediction in the AIMO Interpretability Challenge. The local predictive gates failed. This repository documents those negative results; it does not contain a new ranked predictor.

| Mechanism | Recorded result | Boundary |
|---|---|---|
| Prompt-final linear probes | Five frozen candidates each reached 16/27 tuning accuracy | All failed the accuracy gate; final predictions and scores were not computed |
| Relative layer geometry | 16/27 tuning accuracy | Failed its gate on reused tuning data |
| Matched-prefix newline sensitivity | 19/27 tuning accuracy | Failed its gate on reused tuning data |
| Tokenizer control | Automatic loader erased the contrast in 109/109 cases; generic loader preserved it in 109/109 | Tokenizer fidelity, not predictor accuracy |
| Sampled-answer feasibility | No completed answers in six attempts | Five token caps and one time cap; no robustness accuracy measurement |
| Batch-prevalence allocation | 41/53 accuracy, 0.694231 balanced accuracy | Missed both plain-accuracy gates by one classification |
| Numeric forest conversion | Synthetic predictions and threshold decisions matched the reviewed forest | Engineering equivalence, not a new competition score |

The initial representation extraction included final prompts. “Unopened” in the historical records means unscored, not unseen. Later predictive follow-ups reused exposed tuning data and are exploratory. The release contains no final-row input file, representation array, generated answer, or row-level label file.

## Reproduction

Run `python3 tools/verify_release.py` to verify file identities and recorded arithmetic without downloading data, fitting a model, or opening a holdout. [REPRODUCTION.md](REPRODUCTION.md) maps each mechanism to its inputs, code, environment, and limits. `source-provenance.json` lists original hashes and the small portability changes; it does not claim that changed files retain their old hashes.

`tools/prepare_inputs.py --fetch` rebuilds the exposed public inputs and native token IDs from pinned organizer resources. It checks every output against the recorded hash and downloads no weights. Run it in a disposable working copy, not in an immutable evidence archive.

The original numeric forest code is included. The organizer's starter-kit runtime, serialized forest, model weights, and tokenizer files are not redistributed. Fetch upstream resources separately under their own terms; this repository grants no license to those resources.

## Hosted deployment checkpoint

On October 4, 2026, unchanged organizer always-true controls finished the current Main and Small validation tasks with accuracies 0.4761904762 and 0.5, full coverage, and no invalid predictions. A later upload of the unchanged official uncertainty-profiling Small bundle returned HTTP 500 before creating a submission. The worker did not run. None of the experimental mechanisms above has a score from that upload.

The implementation for the scored controls remains at [organizer commit de794053](https://github.com/aimo-interp/getting-started/tree/de794053debe75a711400696e66b60937cebbb1b). Exact score reproduction requires the corresponding organizer task and evaluation data; downloadable development data are not a substitute. This code release is not a technical-report submission, award, or payout receipt.
