# Running the metamorphic test pipeline

The normal pipeline generates visual programs, runs each on its original image,
creates transformed-image tests, and scores each program against those tests.
**MR1 and MR2 image operations run when the custom tests are created.** You do
not need to create the transformed images separately.

## Choose what to run

| Run | What it verifies | Environment |
| --- | --- | --- |
| Local transformation checks | Real pixel operations and saved paths; MR2 detections are mocked | CPU test dependencies |
| Real pipeline, MR1 only | Generated visual programs evaluated on flips/grayscale | Production ViperGPT/model environment |
| Real pipeline, MR1 + MR2 | Also applies object color fill and bounding-box occlusion | Production environment plus Grounding DINO/SAM |

Skipping MR2 does **not** remove the visual-program models or their dependencies.
The lightweight test environment cannot run the normal model-backed pipeline.

## 1. Check image creation locally

From the repository root, in a local Python environment:

```bash
python -m pip install -r requirements-metamorphic-tests.txt
python -m unittest discover -s tests -p 'test_materialized_tests.py' -v
```

These checks apply and save real transformations to small synthetic images,
verify the unit-test paths and answer rules, and check cache reuse. MR2 uses
mocked masks/boxes, so no model weights or GPU are needed. This does not evaluate
real visual programs or the quality of object detections.

## 2. Prepare real questions and images

The source labels live in `examples/metamorphic/labels.json`. Replace that file
or change `labels_path` in your experiment config to use your own LLM output.
The format is a list of question-ID/effect pairs:

```json
[
  ["q1", {
    "hflip": {"equivariant": {"left": "right", "right": "left"}},
    "vflip": "invariant",
    "grayscale": "invariant",
    "mr2": {
      "cup": {
        "fill_color": {"red": "invariant"},
        "occlude_bbox": "unsafe"
      }
    }
  }]
]
```

IDs must match dataset question IDs exactly, including leading zeros. Labels
specify transformation rules; the question text, original image, and real answer
come from the dataset annotations.

For the supplied 200 GQA labels, download only the 198 corresponding images:

```bash
python scripts/download_gqa_subset.py --dry-run
python scripts/download_gqa_subset.py --output-dir output/gqa_subset
```

The downloader uses local validation annotations and selective ZIP requests.
It saves `annotations.json`, `tuples.json`, and `images/` under `output/gqa_subset`
and reuses existing downloads. Use `--labels` and `--annotations` for other inputs.
The example pipeline configs already point to these downloaded originals.

## 3. Configure the production environment

Follow the project's [installation instructions](../README.md#installation) for
the visual-program generator and ViperGPT execution models. Adjust the model
names/paths inherited from `viunit_configs/base.yaml` for your environment.
The existing production setup targets Linux/CUDA; installing the CPU test
requirements alone is not sufficient.

For MR2, also install its detection/segmentation dependencies in that environment:

```bash
python -m pip install -r requirements-masking.txt
```

Grounding DINO/SAM weights load on first use and are reused. They support CPU,
but a GPU is recommended for a full dataset. To force CPU, set
`unit_test_suite.metamorphic.segmenter.device: cpu` in the experiment YAML.

## 4. Run the pipeline

Start with MR1 only:

```bash
python generate.py --config_path viunit_configs/GQA/eval/metamorphic_mr1_only.yaml
```

Then include MR2:

```bash
python generate.py --config_path viunit_configs/GQA/eval/metamorphic_labels.yaml
```

Run from the repository root. These commands generate and execute real visual
programs. Their output directories are `output/gqa_metamorphic_mr1_only` and
`output/gqa_metamorphic_labels`, respectively.

Both configs enable baseline execution and consume raw labels. The key settings
for adapting your own experiment are:

```yaml
unit_test_generation:
  use_program: false
unit_test_suite:
  mode: metamorphic # viunit | metamorphic | hybrid
  metamorphic_count: 3
  metamorphic:
    labels_path: ./examples/metamorphic/labels.json
    # mr2_enabled: false  # Exclude MR2 before selecting tests.
execution:
  metamorphic_baselines: true
```

`hybrid` adds native ViUniT tests, whose budget is controlled by `viunit_count`.
Native test sampling remains controlled by `unit_test_sampling`. `train.py`
supports the same suite settings in your training experiment configuration.

### What happens during a run

1. Generate candidate visual programs and execute each on the original image.
2. Select up to three valid transformations per question: equivariant first,
   then seeded random invariant choices. Unsafe/null effects are excluded.
3. Apply each selected operation independently to the original image and save a PNG.
4. Resolve the answer rule against each candidate's own baseline output.
5. Run that candidate on the saved transformed images and score its outputs.
6. Select programs by test score, then measure correctness against dataset answers.

Color fill edits matched segmentation-mask pixels. Box occlusion fills matching
rectangles with white. A missing object detection fails test creation instead of
silently recording an unchanged image.

An empty answer rule preserves the baseline. A mapping changes it; exact keys
win over `any`, and list values are acceptable alternatives. Without a matching
key or `any`, the baseline stays unchanged. A failed baseline fails that
candidate's custom tests with an error message. Native ViUniT answers stay fixed.

## Results and reruns

Transformed PNGs and their `tests.json` manifest are stored under
`<output_dir>/metamorphic/<cache-key>/`. The manifest records actual image paths
and transformation/answer rules. Original images are left intact.

`<output_dir>/data.p` stores programs, baselines, test scores, and selected
programs (`max_accuracy_code`). `Iteration accuracy` compares selected outputs
with dataset answers; it is not the metamorphic test score. A consistently wrong
program can score well on metamorphic tests.

To reuse cached execution results with the same config:

```bash
python generate.py --config_path viunit_configs/GQA/eval/metamorphic_labels.yaml \
  --reload 1 --execute_unit_tests 0 --recompute_unit_tests 0
```

Set `--recompute_unit_tests 1` to rescore saved outputs. Changed inputs/settings
or missing transformed images can invalidate caches and trigger execution.
Use a new output directory after changing generation code or retrying unchanged
baseline failures. To reuse real cached programs on a fresh run, pass
`--code_path /path/to/cached_data.p`; question order must match the current dataset.

## Optional: exercise cache branches without models

This separate smoke mode uses constant-return programs and one shared photo.
It checks orchestration and caching, **not image transformations or visual reasoning**.
All generated smoke inputs go under `output/`; none are source fixtures.

```bash
python scripts/prepare_cached_workflow.py --image /path/to/any.jpg
python generate.py --dummy_execution \
  --config_path output/metamorphic_smoke/inputs/config.yaml \
  --code_path output/metamorphic_smoke/inputs/cached_programs.p
```

Replay with the same config, `--dummy_execution --reload 1`, and
`--execute_unit_tests 0 --recompute_unit_tests 0`. This mode explicitly sets
`materialize_images: false`. Do not carry that setting or `--dummy_execution`
into a real transformation run.

To use an already selected test dictionary instead of raw labels, set `tests_path`
in place of `labels_path`. Records with `transformation` and `parameters` create
images automatically; path-only records must supply existing `image_paths`.

Pixel-level transformation tests, mocked MR2 integration, and cache-smoke runs
have been verified. Full model-backed ViperGPT and real MR2 inference have not
yet been verified end to end.
