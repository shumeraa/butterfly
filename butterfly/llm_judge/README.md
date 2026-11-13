# GQA question-only pilot

Independent scripts prepare a permanent sample, classify question text with
vLLM or the Claude API, and evaluate deterministic transformation rules. All implementation,
dependencies, and tests live in this directory. No images are loaded or edited.
Python 3.9+ is required. Run the examples from the repository root, or use the
absolute path to each script from another working directory.

## CPU setup and permanent sample

Dataset preparation uses only Python's standard library. Only CPU rule
evaluation and its Excel exports require `openpyxl`; inference does not:

```bash
python -m pip install -r butterfly/llm_judge/requirements-cpu.txt

python butterfly/llm_judge/prepare_dataset.py \
  --dataset /work/11808/shumera/ls6/butterfly_shared_resources/datasets/gqa/questions/val_balanced_questions.json \
  --experiment-dir experiment --seed 42
```

Alternatively, source `export_paths.sh` **on the execution machine**, then omit
`--dataset` and optionally select `--split train` (default: `val`). The existing
export script establishes **`GQA_IMAGE_PATH`**, an images directory, and does not
export a questions path. Preparation checks its parent directory and the
`questions/` and `questions1.2/` children for the selected balanced-question JSON.
Missing or ambiguous matches require `--dataset`; the script does not assume any
cluster directory layout or copy the paths from `export_paths.sh`.

The dataset must be a GQA question-ID-to-record JSON mapping. All discovered
`types.detailed` counts are printed. Types with fewer than two questions are
ignored with a terminal message; preparation fails if no eligible types remain.
Sorted eligible types and IDs plus a seeded RNG select two distinct questions
per included type. `questions.jsonl` contains only ID, detailed type, and question text.
`sample_summary.json` records source checksum, selection seed, discovered counts,
selected IDs, skipped types with their counts, and the sample checksum. Existing
sample files are never overwritten.
Reuse these files for every model and prompt revision; choose a new experiment
directory to deliberately make a different sample.

## GPU inference

Install vLLM in the target GPU environment using that environment's supported
installation/container process. `eval_tests/qwen_smoke.py` supplied the reference
`LLM`/`SamplingParams` flow; its model, GPU count, and cluster paths are not assumed.

```bash
python butterfly/llm_judge/run_inference.py \
  --questions experiment/questions.jsonl \
  --run-dir experiment/runs/run_001 \
  --model Qwen/Qwen3-4B \
  --tensor-parallel-size 1 --disable-thinking
```

Choose a model available in your execution environment; a local model directory
is also accepted. `--help` lists dtype, GPU memory fraction, context length,
token budgets, temperature, and generation seed settings. The default context
length is 4096 and token budgets are 256/1024 for Calls 1/2. Increase them if your
model needs more context or emits long reasoning. Chat mode uses the model's
own tokenizer chat template with one user message. `--disable-thinking` passes
`enable_thinking=False` to that template where supported. `--prompt-mode plain`
uses the exact rendered prompt directly for models expecting raw prompts.

The Slurm/Apptainer launcher `run_inference.sh` follows
`eval_tests/qwen_apptainer.sh`, including its partition, account, time limit,
container version, cache setup, and GPU smoke check. Override Slurm resources
with `sbatch` options for your cluster. Submit from the repository root:

```bash
export WORK=/path/to/shared/work
sbatch butterfly/llm_judge/run_inference.sh \
  --questions experiment/questions.jsonl \
  --run-dir experiment/runs/run_001
```

`WORK` is required at runtime. `IMAGE` defaults to
`$WORK/containers/vllm-0.18.0.sif`; the image must include vLLM. Inference writes
JSONL/CSV and Markdown artifacts using Python's standard library. It does not
need openpyxl, install Excel packages, or modify the container's Python path.
Excel export happens in the subsequent CPU step.

The launcher defaults to the reference model `Qwen/Qwen3.8-27B`, bfloat16, and
tensor parallelism of two. Override with `--model`, `--dtype`, and
`--tensor-parallel-size`, or the `MODEL`, `DTYPE`, and `TENSOR_PARALLEL_SIZE`
environment variables. All other Python arguments are forwarded, including
`--resume`. The GPU/dependency check must pass before inference starts.

Set `REPO_ROOT` if submitting outside the repository root; Slurm executes a spool
copy of the shell script. Direct execution also works in an allocated GPU session
and locates the repository from the shell script. Relative input/output paths use
the submission directory (or current directory for direct execution). Work,
repository, launch, and cache directories are bound into the container. Use
`APPTAINER_BINDPATH` for additional external dataset/model/output directories.
`CACHE_ROOT` overrides the cache parent (default: `WORK`); `PYTHON_BIN` selects
the **container's** Python executable (default: `python3`). The launcher loads
`tacc-apptainer/1.4.1` when the `module` command is available; customize
`APPTAINER_MODULE`, or set it to an empty string to use an already loaded runtime.

The v2 templates in `prompts.py` exactly reproduce the three specified prompts,
including their two examples. Only question text is substituted. IDs, gold types,
and all other dataset information remain outside model input. Call 1 selects
BINARY/OPEN; both branches run Call 2. There are no LLM retries or guessed labels.
Lenient parsing removes closed multiline thinking blocks and Markdown fences;
Call 2 also extracts a single unambiguous valid JSON object from surrounding prose.
Unmodified raw text, validation errors, and per-call repair flags are retained.
Whitespace alone does not count as a repair. Failed parses have repair flag false;
unattempted calls have null flags and raw responses. Generation exceptions are
recorded explicitly, with no fabricated response.

Each run writes:

- `results.jsonl`: canonical responses, parsed predictions, diagnostics, and progress.
- `review.csv`: required review columns; objects and flags are JSON cells;
  `correct` and `notes` start blank.
- `summary.md`: per-type, answer-type, topic, and flag counts; parse/generation
  failures and repairs (repair rates use received responses as denominator).
- `run_metadata.json`: source selection including sampling seed, sample checksum
  and IDs, model, exact prompt templates/version, timestamp, and run settings.

Each received response is atomically checkpointed before the next call. If an
interrupt occurs after Call 1, resumption runs only the remaining Call 2. Completed
IDs, including failures, are not retried. An in-flight call without a received and
saved response can run again. Atomic replacement keeps one row per ID. Partial
review artifacts are regenerated on normal exceptions or Ctrl-C; after a hard
kill, resume regenerates them from `results.jsonl`. A run directory lock prevents
concurrent writers (Linux/Unix `fcntl`, consistent with the vLLM runtime).

Resume with the **same command and settings**, adding `--resume`. Sample checksum,
model, prompts, source selection, and settings must match. A completed run can
regenerate review artifacts with `--resume` without importing vLLM. Use another
run directory to retry failures or change prompts/settings. Review artifacts are
regenerated on resume: save manual corrections in a separate workbook first.

## Claude API inference (no GPU)

`run_claude_inference.py` uses the existing `CLAUDE_API_KEY` environment variable
explicitly, with `claude-haiku-5-5` as the default model. It needs Python 3.10+,
the Anthropic SDK and network access to the Claude API. It runs directly
in your Python environment and does not use Slurm, Apptainer, or vLLM:

```bash
python -m pip install -r butterfly/llm_judge/requirements-claude.txt
python butterfly/llm_judge/run_claude_inference.py \
  --questions experiment/questions.jsonl \
  --run-dir experiment/runs/claude_001
```

The model identifier and request settings follow Anthropic's
[Haiku 5.5 migration guide](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide).
The runner omits temperature/top_p/top_k and defaults to disabled thinking with
low effort for the short classification responses. Use `--thinking adaptive`
and `--effort low|medium|high` to change those settings; thinking consumes the
same output budget, so increase `--call1-max-tokens`/`--call2-max-tokens` if needed
(defaults: 256/1024). No API generation seed is assumed. The sample selection
seed remains recorded in the dataset metadata.

The same exact question-only prompts, validation, permanent sample, and review
exports are used. Only the rendered question prompt is sent in each independent
API request. The API key is never put in prompts or run metadata. `results.jsonl`
also stores `call1_api_response` and `call2_api_response`, including the full
returned content blocks, message/model IDs, stop reasons, and token usage. Raw
text fields concatenate only text blocks without trimming; the full envelopes
retain thinking blocks and other provider fields. Truncation, refusal, and other
non-`end_turn` stop reasons are explicit invalid outputs even if the text happens
to parse. Pass the resulting `results.jsonl` to `evaluate_rules.py` for CPU
rule evaluation and Excel exports. Claude inference also requires no openpyxl.

The SDK retries transient errors twice by default (`--max-retries`), with a
120-second per-attempt timeout (`--timeout`). If an API request still fails, the
runner saves its diagnostic history in `api_errors`, exits with an error, and
leaves that question incomplete. Rerun the same command with `--resume` to retry
the unanswered call while retaining previously received responses. Completed
responses, including parse failures/refusals, are never automatically retried.
Ctrl-C similarly preserves completed calls. An interrupted request may already
have been processed remotely even if no response was saved locally. Use a new
run directory when changing model/request settings. A fully completed run can
regenerate its exports with `--resume` without the API key or SDK.

## CPU transformation labels and Excel export

Run this step in a CPU environment with openpyxl installed, outside the GPU
container. Use `results.jsonl` to retain all raw responses and inference/provider
diagnostics in the canonical workbook:

```bash
python -m pip install -r butterfly/llm_judge/requirements-cpu.txt
python butterfly/llm_judge/evaluate_rules.py \
  --input experiment/runs/run_001/results.jsonl \
  --output-dir experiment/runs/run_001/labels
```

The input can also be `review.csv` or a manually corrected `.xlsx` workbook.
For Excel input, select a sheet with `--sheet Predictions` or omit `--sheet` to
use its first sheet. JSONL predictions are flattened from `call1`/`call2` while
preserving every original field. Explicit flat fields take precedence, allowing
manual corrections. CSV retains only the fields present in that CSV; JSONL is
preferred for complete provenance. Older `predictions.xlsx` exports still work.

Inputs contain one question per row. Objects and flags must be JSON arrays;
names and appearance-identification
features are validated without keyword lists. `answer_type` or `call1` is accepted.
When correcting an invalid inference, also correct its `call1_valid`/`call2_valid`
field if present: an explicitly false validity field makes the row unevaluable.
The original error/raw fields remain available as provenance and are not used to
infer semantic labels. No GQA gold type is used to correct predictions.

Exactly two workbooks are created:

- `transformation_labels_raw.xlsx`: canonical `Questions`, `GlobalLabels`,
  `LocalEdits`, and `Warnings` sheets. Input fields, order, and diagnostics are
  retained, alongside normalized fields. If an input field collides with a
  computed column, its original value is kept under a unique `source_` prefix.
  `source_row` disambiguates duplicate or missing IDs. Labels are full words;
  traceable reason codes and evaluation statuses accompany them.
- `transformation_labels_review.xlsx`: formatted `Review`, `LocalEditReview`, and
  `Legend` sheets with filters, frozen headings, wrapping, colored labels,
  conditional edit checks, and blank `correct?`/`review_notes` columns.

Override filenames with `--raw-output` and `--review-output`. The input file
and output paths must be distinct. Source formula-like strings are written as
literal text; Excel's cell length limit raises an error rather than silently
truncating data. Extremely long model responses remain available in JSONL even
if Excel export fails.

Rules use only validated answer type, topic, ordered objects/id_features, and the
16 declared flags. All six global edits, both occlusion scenarios, and three
local edits per mentioned object are recorded deterministically. No objects means
mentioned-object occlusion is `not_applicable` with a blank label and no local
rows. Invalid predictions have `evaluation_error`, blank labels, and a warning;
recognizable object entries also receive blank local error rows. Binary edits of
non-target objects are always unsafe. Equivariant color/material edits require a
later value/answer-option check; an executor must downgrade to unsafe when the
new value does not map to the other offered choice. Unmentioned occlusion assumes
an object could be selected later; it does not locate or invent one.

Labels are **predicted by rules, not visually verified**. They are not edited
images or recomputed answers. Corrected predictions can be reevaluated entirely
on CPU without repeating inference.

## Verification

```bash
python -m unittest discover -s butterfly/llm_judge/tests -v
```

The suite needs `openpyxl`, no dataset/model/GPU. It covers the four requested
examples, exhaustive flag/branch/topic rule precedence, per-object identities,
conditional checks, malformed/missing predictions, exact prompts, repair parsing,
portable dataset resolution, reproducible sampling, checkpoint/resume behavior,
metadata identity, and Excel provenance/formatting. Actual model execution must
be verified on the target GPU machine.

Spatial roles use `LR_ID`/`UD_ID` for object identification and
`LR_ANSWER`/`UD_ANSWER` for the relation being asked about. Relevant identity
flags make flips/rotation unsafe, taking precedence over answer flags. Only
answer flags can trigger binary equivariance; OPEN answers remain unsafe.
Existing TEXT/BODY/SUPPORT blockers retain precedence. Generic `LR`/`UD` flags
remain descriptive and do not trigger spatial labels by themselves. Rerun Call 2
with prompt version `gqa-question-only-v2` (in a new run directory), or manually
annotate the new flags, before reevaluating older predictions. Existing workbooks
are not automatically relabeled.
