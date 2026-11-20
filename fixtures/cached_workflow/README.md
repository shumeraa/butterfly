# Offline cached-workflow example

Run commands from the repository root. This fixture uses one local photo for
every original and test image. It does not transform the photo or infer its contents.
Two question IDs from the supplied example have six custom tests each. Each
question has four cached candidates: two mapped answers, an unmapped answer,
and an intentional RuntimeError.

## Setup

```bash
python3 -m venv .venv-dummy
source .venv-dummy/bin/activate
python -m pip install -r requirements-dummy.txt
```

The photo and program pickle have already been prepared locally. They are ignored
by Git. To recreate them, supply your local source photo and generation output:

```bash
python scripts/prepare_cached_workflow.py \
  --image '/path/to/photo.jpg' \
  --tests '/path/to/test_out.json'
```

## 1. Execute cached programs and save all results

```bash
python generate.py --dummy_execution \
  --config_path fixtures/cached_workflow/config.yaml \
  --code_path fixtures/cached_workflow/cached_programs.p
```

No LLM, tokenizer, segmentation model, GPU, or API calls are used. The dummy
backend only interprets a string-literal return or an intentional RuntimeError;
it rejects arbitrary programs. It uses a lightweight GQA-shaped dataset adapter
and exact-string accuracy. The actual generate.py baseline, suite, instrumentation,
scoring, final-selection, and cache branches run unchanged.

The first run makes 8 baseline executions, 36 transformed-test executions
(the broken candidates' custom executions are skipped), and 2 final evaluations.
State is saved in `output/cached_workflow_dummy/data.p`:

- `generated_code`: the cached fixture programs.
- `original_program_outputs`: original-image outputs/errors and cache fingerprints.
- `selected_unit_tests`: shared raw answer rules, including mappings.
- `candidate_unit_tests`: concrete expectations for each candidate.
- `unit_test_results`: outputs, expected answers, errors, and scores.
- `code_outputs`: final selected-program outputs used during replay.

Open `output/cached_workflow_dummy/dummy_summary.json` for a readable view of
baselines, shared rules, candidate expectations, errors, and scores.

## 2. Replay the complete cache

```bash
python generate.py --dummy_execution \
  --config_path fixtures/cached_workflow/config.yaml \
  --reload 1 --execute_unit_tests 0 --recompute_unit_tests 0
```

Expect `Unit tests already executed..` and no new dummy execution calls.

## 3. Recompute scores from saved outputs

```bash
python generate.py --dummy_execution \
  --config_path fixtures/cached_workflow/config.yaml \
  --reload 1 --execute_unit_tests 0 --recompute_unit_tests 1
```

Expect `Recomputing unit tests...`, identical scores, and no new executions.
Do not use `--custom_unit_tests_path` for these replay commands; the YAML
`metamorphic.tests_path` already supplies the custom tests without forcing execution.

`dummy_execution.jsonl` in the output directory logs each execution batch.
After all three commands on a fresh output directory its batch counts remain
`[8, 36, 2]`. For a fresh independent experiment, override `output_dir` using
`--config_options output_dir=./output/another_dummy_run` on all three commands.

## Expected behavior

| Question | Baseline answers | Custom-test scores |
| --- | --- | --- |
| `006149` | left, right, above, error | 5/6, 5/6, 1, 0 |
| `006108` | yes, no, maybe, error | 3/6, 5/6, 1, 0 |

All fixture programs are constant-return programs. Mapped expectations can
therefore intentionally fail, while unmapped answers stay invariant. This also
shows why metamorphic consistency does not establish correctness: `above` and
`maybe` win the suite, but final question accuracy is 0 for these dummy labels.
`any` keys in the sample mappings are literal keys, not wildcards.

The normal model-backed path remains available by omitting `--dummy_execution`.
The dummy mode is restricted to cached, metamorphic-only, single-iteration runs.

```bash
python -m unittest discover -s tests -v
```
