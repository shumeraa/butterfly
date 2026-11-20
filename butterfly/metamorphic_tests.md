# Using custom metamorphic tests with the full pipeline

`generate.py` can generate ViperGPT-style candidate programs and evaluate them
with native ViUniT tests, custom metamorphic tests, or both. `train.py` supports
the same suite configuration and baseline-answer rules.

## 1. Prepare the real inputs

Use the project's normal model-backed environment and experiment configuration:
original question images, dataset annotations, visual-program generator, and
ViperGPT execution models. The lightweight `requirements-dummy.txt` environment
is only for the offline example. Install `requirements-transforms.txt` as well
if your production environment does not already include Pillow and NumPy.

Run your custom-test generation step and save its output as JSON. Each key must
match a dataset question ID (`sample_id`; GQA uses `questionId`). Keep IDs as
strings, including leading zeros.

```json
{
  "006149": [
    {
      "description": "horizontal flip",
      "image_paths": ["/path/to/transformed/image_flipped.png"],
      "answer": {"left": "right", "right": "left"}
    },
    {
      "description": "grayscale",
      "image_paths": ["/path/to/transformed/image_gray.png"],
      "answer": ""
    }
  ]
}
```

Supply a nonempty test list for every question in the run. Each test currently
needs one existing transformed image. Paths may be absolute or relative to the
working directory; run the commands below from the repository root.

The JSON describes the tests; loading it does not create missing images. Your
upstream generation step must actually save the transformed images. In
particular, the recently merged MR1/MR2 manifest exporter still contains
commented-out image-saving operations. The dummy photo copies are only suitable
for checking the workflow, not a real transformation experiment.

## 2. Configure the experiment

Add these settings to your existing experiment YAML, retaining its dataset,
model, prompt, and execution settings:

```yaml
output_dir: ./output/metamorphic_experiment

do_unit_test: true
run_fixed_code: false

unit_test_generation:
  use_program: false

unit_test_suite:
  mode: hybrid
  viunit_count: 3
  metamorphic_count: 3
  metamorphic:
    tests_path: /path/to/custom_tests.json

execution:
  metamorphic_baselines: true
```

| Setting | Behavior |
| --- | --- |
| `mode: viunit` | Native ViUniT tests only. |
| `mode: metamorphic` | Custom tests only; skips native caption/image generation. |
| `mode: hybrid` | Combines native ViUniT and custom tests. |
| `viunit_count` | Native test budget; `null` inherits `unit_test_sampling.num_unit_tests`. |
| `metamorphic_count` | Custom test limit per question; pre-generated lists use the first N tests. |
| `metamorphic_baselines: true` | Executes each candidate on the original image to resolve custom answer rules. |

Set `metamorphic_count` high enough to include all supplied tests if that is your
intention. Native test sampling remains controlled by `unit_test_sampling`.
Custom/hybrid suites and baseline instrumentation require `use_program: false`.

## 3. Generate programs and evaluate them

```bash
python generate.py --config_path /path/to/experiment.yaml \
  --execute_unit_tests 1 --recompute_unit_tests 1
```

For a fresh run, use a new output directory and omit `--dummy_execution`,
`--code_path`, `--data_path`, and `--reload 1`. Candidate programs are generated
by the configured model. Custom-test creation is a separate upstream step;
`tests_path` is the integration point for its generated dictionary.

The evaluation flow is:

1. Generate candidate visual programs for each question.
2. Run each candidate on the original image and store its baseline output/error.
3. Load custom test rules and images; in hybrid mode, also generate/sample native tests.
4. Resolve custom expectations separately for each candidate using its baseline.
5. Execute candidates on the test images, score them, and select the highest-scoring programs.
6. Evaluate the selected programs on the original questions against dataset answers.

For training, use the same suite settings in your training experiment YAML,
retain the project's training/model settings, and enable `do_train: true`:

```bash
python train.py --config_path /path/to/training_experiment.yaml \
  --execute_unit_tests 1 --recompute_unit_tests 1
```

## How custom answers are resolved

| Test's `answer` | Expected answer for a candidate |
| --- | --- |
| `""` | That candidate's original output, unchanged. |
| Mapping containing the original output | The mapped value. |
| Mapping without the original output | The original output, unchanged. |
| Nonempty string | That explicit answer. |

Mappings use exact string keys; `"any"` is not a wildcard. A failed baseline
marks all custom tests for that candidate as failed, with score zero and its
error message. Those custom executions are skipped; native ViUniT tests still run.
Native ViUniT expected answers remain independent of candidate baselines.

Shared custom rules are not overwritten. For example, two candidates originally
answering `left` and `right` receive flip expectations of `right` and `left`,
respectively, from the same test dictionary.

## Results and cached runs

The run's `output_dir/data.p` contains candidate programs, baseline outputs,
candidate-specific expectations, test scores/errors, and selected programs
(`max_accuracy_code`). `Iteration accuracy` measures selected-program correctness
against the dataset answers; it is separate from the unit-test score used for selection.

Metamorphic consistency does not guarantee correctness. In particular, an
unsupported answer can score well because missing mapping entries assume
invariance. Interpret custom-test scores alongside original-question accuracy.

To reuse saved execution results with the same experiment configuration:

```bash
python generate.py --config_path /path/to/experiment.yaml \
  --reload 1 --execute_unit_tests 0 --recompute_unit_tests 0
```

To recompute scores from cached outputs, change `--recompute_unit_tests` to `1`.
Changed inputs/settings can invalidate caches and trigger execution. Keep the
original and transformed images with the run. Use a fresh output directory after
changing generator code or when retrying cached baseline failures.

Prefer YAML `tests_path` for these workflows. The evaluation-only shortcut
`--custom_unit_tests_path` requires execution and recomputation enabled, and turns
`viunit` mode into `hybrid` when custom tests are supplied.

## Offline example and validation

The [dummy cached-workflow guide](../fixtures/cached_workflow/README.md) includes
ready-to-run commands using one photo and hard-coded programs, without model
calls. It exercises baseline execution, answer mappings, errors, and cache reuse.

```bash
python -m unittest discover -s tests -v
```

The CPU checks and dummy workflow have been exercised. Full model-backed
ViperGPT evaluation/training has not yet been verified end to end. The standalone
`butterfly/demo.py` and `butterfly/demo_simple.py` do not use this suite configuration.
