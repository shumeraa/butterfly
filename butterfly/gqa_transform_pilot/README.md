# GQA transformation pilot

Can a model tell how an image transformation will affect a visual question's answer, just from the question? This pilot asks Llama 3 8B Instruct to make predictions, then uses human review to check them.

We sample 100 distinct images from GQA's balanced validation set and choose one question per image, using seed 42. Each question gets six predictions: horizontal flip, vertical flip, 180° rotation, grayscale, blur, and noise. The model sees only the question and instructions. It can predict an answer relation or choose `unsafe` when it cannot tell reliably.

## Run on the cluster

From the repository root:

```bash
sbatch butterfly/gqa_transform_pilot/gqa_transform_pilot.sh
```

The launcher uses `butterfly/start_env.sh` for the environment and image paths. Run prediction on a GPU compute node. Use `--model` for a local checkpoint, or `--annotations`, `--images`, and `--output` to change paths. Choose a fresh output directory for each run.

## Review the predictions

Copy the whole output folder (by default, `plots/gqa_transform_pilot`) to your laptop. Open `review.html` in a browser to compare the original and transformed images alongside the question, GQA answer, prediction, and expected answer. No server is needed.

Review all 600 cases in `review.csv`, editing only `status` and `review_note`:

- `correct`: the transformed image supports the expected answer.
- `incorrect`: the image contradicts it, or the prediction is malformed or cannot be applied.
- `uncertain`: there isn't enough visual evidence to decide.
- `invalid_source`: the original image, question, or GQA answer prevents a fair review. Mark all six rows for that question.

`unsafe` is an abstention, prefilled as `uncertain`; leave it that way unless the source is invalid. Malformed or inapplicable predictions start as `incorrect`. Keep the other CSV fields unchanged, including leading zeros in IDs.

## Generate the report

After labeling, run this from the repository root, adjusting the review folder path as needed:

```bash
python3 butterfly/gqa_transform_pilot/report.py \
  --csv gqa_transform_pilot/review.csv \
  --output gqa_transform_pilot/report
```

Reporting needs only Python 3. Keep `review_manifest.json` beside the CSV. You'll get a Markdown report and CSV summaries overall, by transformation, and by GQA type and group. They include accuracy, coverage, abstentions, and review status counts, with denominators shown. Invalid sources are excluded from rates; accuracy is `correct / (correct + incorrect)`.

The output folder also keeps the sampled records, exact prompts, raw responses, and run settings. Results describe these sampled images; they don't establish guarantees from question text alone.
