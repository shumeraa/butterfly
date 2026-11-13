# <img src="assets/viunit_logo.png" alt="Logo" height="100"> <br> Visual Unsupervised Unit Testing


## Framework Overview
<img src="assets/ViUnit Demo.png"/>

## Installation

Go to butterfly/ls6_env_install.sh for lonestar 6 specific installation directions.

```
conda create -n viunit_env python=3.10
conda activate viunit_env

conda install -c conda-forge gcc_linux-64=9.5.0
conda install -c conda-forge compilers
conda install -c conda-forge glib

export CC=gcc
export CXX=g++

conda install nvidia/label/cuda-12.1.0::cuda-toolkit

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
alias ld="ld -L $LD_LIBRARY_PATH"

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install pip
unset PYTHONPATH
python -m pip install -r requirements.txt

bash download_models.sh
```

:exclamation::exclamation: Ensure that the `cudatoolkit` version matches the one used to compile `pytorch` otherwise `GroundingDino`'s kernel for multiscale deformable attention will produce an error. 


## Datasets
### GQA
Download the GQA images from [here](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip). 
Set the environment variable `GQA_IMAGE_PATH` to the unzipped directory of the images. 
### WinoGround
Download the WinoGround images from [here](https://huggingface.co/datasets/facebook/winoground). 
Set the environment variable `WINOGROUND_IMAGE_PATH` to the image path. The annotation file should be placed under `my_datasets/annotation_files/WinoGround`.


### SugarCREPE
Download the [COCO-2017](https://cocodataset.org/#download) validation split used in [SugarCREPE](https://github.com/RAIVNLab/sugar-crepe).
Set the environment variable `COCO_VAL2017` to the image path.  

## Configs
Refer to [ViperGPT](https://github.com/cvlab-columbia/viper) for information about the API configs in `viper_configs/`.

`ViUniT` configs are found `/viunit_configs`.

### Config Parameters

#### **General Settings**

- **`seed`**  
  Sets the random seed for reproducibility.

- **`output_dir`**  
  Specifies the directory where the output files, including models and logs, will be saved.

- **`llm_engine`**  
  Defines the language model engine to be used. Options are `"hf"` (Hugging Face) or `"vllm"`. VLLM is about 25x faster, and is used in reported experiments. 

- **`do_unit_test`**  
  Specifies whether to perform unit testing.

---

#### **Data Settings**

- **`task`**  
  Used to define the task in question. Options: `VQA` (Visual Question Answering), `ITM` (Image-Text Matching)

- **`dataset_name`**  
  Specifies the dataset to be used. Options: `GQA`, `WinoGround`, `SugarCREPE`

- **`annotation_file`**  
  Path to the annotation file for the dataset. For training use the train split annotation file. 

- **`image_root`**  
  Defines the root directory for images associated with the dataset.

- **`sample_ids_file: null`**  
  If specified, this file would contain sample IDs for selective data loading. Set to `null` by default. File should consist of sample_ids split by `\n`.

- **`num_samples`**  
  Limits the number of samples to be loaded for the task. Randomly selected

- **`sample_per_type`**  
  Specifies the number of samples per type. `-1` indicates no limit. For GQA type is the question `group`, for WinoGround the instance `tag`, and for SugarCREPE the `question_type`.

---

#### **Training Settings**

- **`train_prompt_file`**  
  Path to the training prompt file. Does not include the full API. 

- **`num_improve_steps`**  
  Specifies the number of improvement steps for training. Namely, at each policy iteration how many times to iterate over the dataset. 

- **`batch_size`**  
  Defines the *per device* batch size for training.

- **`train_max_iters`**  
  Sets the number of iterations for training.

- **`train_kwargs`**  
  Additional training arguments including:
  - **`only_correct: True`**: Focuses training on correct responses on the training datasets. 
  - **`unit_test_reward: 'all'`**: Specifies reward calculation during unit testing. Can be one of `all`, `ratio`, `threshold`. All only accepts examples that pass all unit tests (`reward`=1), `ratio` sets reward equal to unit test score and considers all examples, unless `only_correct` is set, and `threshold` sets `reward`=1 to all examples that pass the threshold score on the unit tests. 
  - **`unit_test_reward_weight`**: Specifies a weight to apply to the reward such that `reward*=unit_test_reward_weight`.  
  - **`threshold`**: Threshold value for `unit_test_reward`=`'threshold'`
  - **`warmup_ratio`**: Warmup period ratio for learning rate scheduling.
  - **`max_grad_norm`**: Limits gradient norm to prevent exploding gradients.
  - **`lr_scheduler_type`**: Specifies the learning rate scheduler type.
  - **`learning_rate`**: Sets the learning rate.
  - **`lora_config`**: Configuration for Low-Rank Adaptation (LoRA) with parameters like `r`, `lora_alpha`, `lora_dropout`, and `target_modules`.
---

#### **Visual Program Generation**

- **`model_name`**  
  Specifies the model used for generating visual programs.

- **`half`**  
  Uses half-precision to save memory.

- **`model_kwargs`**  
  Additional model arguments, such as loading in 4-bit precision.

- **`generation`**  
  Configuration for generating visual programs, including prompt files, in-context examples, and generation parameters like `top_p`, `temperature`, `max_new_tokens`, and sampling strategies.
  - `num_return_sequences` specifies number of programs to generate!

---

#### **Unit Test Generation**

- **`model_name`**  
  Defines the model used for generating unit tests.

- **`half`**  
  Uses half-precision for efficiency.

- **`use_program`**  
  Indicates whether to condition unit test generation on program implementation. 

- **`generation`**  
  Contains generation settings similar to those in visual program generation but tailored for unit testing.

---

#### **Image Generation**

- **`image_scorer`**  
  Specifies whether to use an image scorer.

- **`image_source`**  
  Defines the source for image generation, with options like `'web'`, `'cc12m'`, and `'diffusion'`.

- **`top_k`**  
  Limits the number of images to be considered for rerakning in `'web'` and `'cc12m'`

- **`return_k`**  
  Number of images to be returned for each example.

- **`reranking_model_name`**  
  Specifies the model used for reranking generated images.

- **`generation`**  
  Details the parameters for generating images, including diffusion model settings, inference steps, and prompt configurations.

---

#### **Unit Test Sampling**

- **`strategy`**  
  Defines the strategy for sampling unit tests. Options: `coverage_by_answer` (maximize different answers, then fill rest of options by maximizing difference in inputs), `coverage` (maximize input difference), `random`, `answer_only` (maximize different asnwers, then randomly fill rest of spots).

- **`model_name`**  
  Model used for encoding descriptions for coverage sampling.

- **`filter_long_answers`**  
  Filters out unit tests with long answers (more than 5 words).

- **`num_unit_tests`**  
  Specifies the number of unit tests to generate.

---

#### **Execution Settings**

- **`pass_threshold`**  
  Sets the threshold for passing execution criteria for unit testing.

- **`feedback_max_iters`**  
  Maximum iterations for feedback loops (reprompting for correction). If set at 1 no feedback is conducted.

- **`image_agg`**  
  Aggregation method for images, with options like `'max'` (if at least one passes the unit test it is correct), `'mean'` (average across unit tests), or `'majority'` (if most images pass, then unit test passes).

- **`code_agg`**  
  Aggregation method for code execution, with options like `'single'` (randomly break ties of top scoring programs), `'vote'` (majority output of top voting programs).

- **`use_fixed_code_when_low_confidence`**  
  Option to use fixed code when confidence is low for answer refusal experiments.

- **`synonym_checker`**  
  Option to enable a synonym checker for unit testing instead of accuracy function.

- **`error_penalty`**  
  Penalties for errors in syntax and runtime. Default to 0.


## Run Inference
`
python generate.py --config_path /path/to/config
`
There are additional arguments that can be passed in the `generate.py` file that help with loading cached results. 
```
--data_path          Specify the path to a pickle file containing all data necessary for the generation. This option facilitates replicating previous runs.

--unit_test_path     Indicate the path to a pickle file containing unit tests from a different run that you wish to copy.

--load_images        Set to 0 or 1 to determine whether to load images from the unit tests. The default value is 1.

--load_selected      Set to 0 or 1 to decide whether to load sampled unit tests as they are, or to perform resampling. The default value is 1.

--execute_unit_tests Set to 0 or 1 to control whether to re-execute the unit tests. The default setting is 0.

--recompute_unit_tests Enable this option to recompute the accuracy of code on unit tests without re-execution. This is useful for applying thresholds. Default is set to 0.

--reload             Set to 0 or 1 to decide if stored data in the output directory should be reloaded. The default setting is 0.

--lora_path          Specify this option to use a Lora path for the visual program generator.
```


## Run Train
`
python train.py --config_path /path/to/config
`
There are additional arguments that can be passed in the `train.py` file that help with loading cached results. 
```
--data_path          Specify the path to a pickle file containing all data necessary for the generation. This option facilitates replicating previous runs.

--unit_test_path     Indicate the path to a pickle file containing unit tests from a different run that you wish to copy.

--load_images        Set to 0 or 1 to determine whether to load images from the unit tests. The default value is 1.

--load_selected      Set to 0 or 1 to decide whether to load sampled unit tests as they are, or to perform resampling. The default value is 1.

--execute_unit_tests Set to 0 or 1 to control whether to re-execute the unit tests. The default setting is 0.

--recompute_unit_tests Enable this option to recompute the accuracy of code on unit tests without re-execution. This is useful for applying thresholds. Default is set to 0.

--reload             Set to 0 or 1 to decide if stored data in the output directory should be reloaded. The default setting is 0.

--lora_path          Specify this option to use a Lora path for the visual program generator.
```

## Demo
An `ipython` demo file is provided in `demo.ipynb`

## Ethical Considerations
This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people's lives, rights, or safety. For further guidance on use cases, refer to our [AUP](https://www.salesforce.com/content/dam/web/en_us/www/documents/legal/Agreements/policies/ExternalFacing_Services_Policy.pdf) and [AI AUP](https://www.salesforce.com/content/dam/web/en_us/www/documents/legal/Agreements/policies/ai-acceptable-use-policy.pdf).


## Disclaimer
* Users need to make their own assessment regarding any obligations or responsibilities under the corresponding licenses or terms and conditions pertaining to the original datasets and data. This repository is being released for research purposes only. 
* The Meta WinoGround dataset is linked out to and not included in the repo. 
* The GQA dataset is released by Stanford and the origin of the data is not described - the CC-BY 4.0 logo is simply listed at the bottom of the dataset download page. The dataset is based on the image-text pairs of COCO (https://cocodataset.org/#termsofuse), which follows flickr TOU (https://www.flickr.com/creativecommons/) having a variety of licenses.



## Acknowledgements
API code adapted from [ViperGPT](https://github.com/cvlab-columbia/viper)

## Image transformations

`image_transforms/` applies transformations to Pillow images using NumPy. Each function
returns a new image.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-transforms.txt
python -m unittest discover -s tests -v
```

```python
from PIL import Image
from image_transforms import hsv

with Image.open('input.png') as source:
    result = hsv(source, delta_h=30, delta_s=-0.2, delta_v=0.1)
result.save('adjusted.png')
```

CLI usage:

```bash
python -m image_transforms input.png flipped.png flip left
python -m image_transforms input.png rotated.png rotate -30
python -m image_transforms input.png sepia.png color sepia
python -m image_transforms input.png blurred.png blur 2
python -m image_transforms input.png noisy.png noise 0.05 --seed 42
python -m image_transforms input.png adjusted.png hsv --delta-h 30 --delta-s -0.2 --delta-v 0.1
```

| Transform | Input and conventions |
| --- | --- |
| `flip` | `up` reflects top/bottom; `left` reflects left/right. |
| `rotate` | Signed degrees, positive counterclockwise. Expands canvas by default; CLI `--no-expand` keeps original dimensions. New areas are black for RGB, transparent for RGBA. |
| `color_filter` / CLI `color` | `grayscale`, `sepia`, `invert`, `warm`, `cool`. Warm adds `(20, 0, -20)` to RGB; cool adds `(-20, 0, 20)`; values clip to byte range. |
| `blur` | Nonnegative Gaussian radius in pixels. |
| `noise` | Gaussian standard deviation in `[0, 1]` relative to the full RGB range; optional nonnegative integer seed. Noise is independent per pixel/channel, then clipped. |
| `hsv` | Additive hue delta in degrees, saturation/value deltas in `[-1, 1]`. Hue wraps; saturation/value clamp to `[0, 1]`. Brightness uses HSV value. |
| Python-only `scale` | Positive independent X/Y factors; omit Y for uniform scaling. Dimensions round to the nearest integer, at least one pixel. Bicubic interpolation resizes pixels and alpha together. A shared primitive, **not** exposed in the six-filter MR1 registry or CLI. |

Color operations convert to RGB or RGBA and preserve alpha unchanged. Flips and
rotations transform alpha with the image. Grayscale and other non-RGB modes are
converted to RGB; palette transparency is converted to RGBA. RGB results round
to 8-bit values. Zero blur/noise/HSV deltas preserve RGB/RGBA pixels exactly.
The CLI applies EXIF orientation before filtering, rejects the same input/output
path, and uses the output extension to select the encoder. Use PNG for lossless
output and transparency; existing output files are overwritten.

Preview all transforms on the same source:

```bash
python -m scripts.preview_transforms 'assets/ViUnit Demo.png' outputs/transform_preview.png
```

The preview downsizes the source before filtering, so its pixel-based blur radius
is measured at preview resolution.

### High-level filters for the reliability prompt

`image_transforms.high_level` wraps the low-level functions with the exact six
keys and fixed strengths in `PROMPT` in
`origin/llm_reliability_experiment:butterfly/gqa_transform_pilot/gqa_transform_pilot.py`
(commit `fdcab4c`). Python modules are the primary interface:

```python
from image_transforms import HIGH_LEVEL_TRANSFORMS, apply_transform, transform_images

# One operation, using an exact prompt key.
blurred = apply_transform(image, 'blur')
noisy = apply_transform(image, 'noise', seed=42)

# All six variants independently start from image, not from the previous variant.
variants = transform_images(image, seed=42)
flipped = variants['horizontal_flip']
```

| Key | Low-level operation |
| --- | --- |
| `horizontal_flip` | `flip(image, 'left')` |
| `vertical_flip` | `flip(image, 'up')` |
| `rotate_180` | `rotate(image, 180)` |
| `grayscale` | `color_filter(image, 'grayscale')` (luminance grayscale) |
| `blur` | `blur(image, 2)` (radius in pixels) |
| `noise` | `noise(image, 0.05, seed=seed)` (RGB standard deviation 12.75) |

The six named functions are also available directly in `image_transforms.high_level`,
e.g. `high_level.blur(image)`. The existing top-level low-level functions still
require their original parameters, so `image_transforms.blur(image, radius)` and
`image_transforms.noise(image, amount, seed=...)` are unchanged.

`transform_images` returns an ordered dict keyed by `HIGH_LEVEL_TRANSFORMS`.
Pass the pilot's per-image `noise_seed` to reproduce its images; omitting the
seed gives fresh noise. Seeds affect only noise. Unknown names raise `ValueError`;
these functions do not parse LLM text/JSON, infer operations, or map answers.
That processing belongs in the calling pipeline.

For the reliability pilot, replace its local `transform_images` helper with
`from image_transforms import transform_images` after bringing this package into
that branch. Its existing RGB input/export path produces the same pixels.
The new API also preserves RGBA alpha according to the low-level conventions;
pass `image.convert('RGB')` explicitly to reproduce the old helper's RGB-only
behavior for transparent inputs. These six filters use only Pillow and NumPy;
SAM masking remains a separate operation.

A thin CLI wrapper is available for manual smoke tests:

```bash
python -m image_transforms input.png output.png apply horizontal_flip
python -m image_transforms input.png output.png apply noise --seed 42
```

### Mask objects by word

Install `requirements-masking.txt` to enable Grounding DINO + SAM:

```python
from image_transforms import mask_objects

result = mask_objects(image, 'person')  # Fill matching objects with white.
result.save('whitened.png')
```

Use `fill='black'` for a black fill. All matching objects are masked; no matches
return an unchanged copy. Alpha and other pixels are preserved. Models load on
first use and are reused. To set thresholds or device, pass a reusable backend:

```python
from image_transforms import GroundedSAM, mask_objects

segmenter = GroundedSAM(device='cuda', box_threshold=0.35, text_threshold=0.25)
result = mask_objects(image, 'red car', segmenter=segmenter)
```

The CLI also supports masking:

```bash
python -m image_transforms input.png whitened.png mask person --device cuda
```

To bundle the package and dependencies for Colab, run
`python -m scripts.export_colab_bundle`. The ZIP is written to
`outputs/butterfly-transforms-colab.zip`.

Local unit tests use synthetic/mocked masks and require only Pillow and NumPy.
Actual model inference is a separate GPU smoke test; detection and segmentation
quality must still be checked for the intended dataset.

### MR2: segment, reuse a transform, composite back

`transform_masked` accepts an explicit boolean `[H, W]` mask and an existing
transform **callable**. It does not reimplement the MR1 filters. Segment once and
reuse that mask for independent variants:

```python
from PIL import Image, ImageOps
from image_transforms import GroundedSAM, hsv, scale, transform_masked
from image_transforms import high_level

with Image.open('input.png') as source:
    image = ImageOps.exif_transpose(source).convert('RGB')
segmenter = GroundedSAM(device='cuda')
masks = segmenter.segment_instances(image, 'person')  # boolean [N, H, W]
if not len(masks):
    raise ValueError('No person detected; skip this source')
mask = masks[0]  # Inspect masks first; detector order is not a semantic identity.

variants = {
    'local_grayscale': transform_masked(image, mask, high_level.grayscale),
    'local_blur': transform_masked(image, mask, high_level.blur),
    'local_noise': transform_masked(image, mask, high_level.noise, seed=42),
    'local_color': transform_masked(image, mask, hsv, delta_h=30),
    'local_brightness': transform_masked(image, mask, hsv, delta_v=.1),
    'local_flip': transform_masked(
        image, mask, high_level.horizontal_flip, geometry=True, fill='white'),
    'taller_person': transform_masked(
        image, mask, scale, geometry=True, scale_x=1, scale_y=1.25,
        anchor='bottom_center', fill='white'),
}
for name, result in variants.items():
    result.save(f'{name}.png')
```

All outputs preserve the scene canvas dimensions. `scale_x` changes object width;
`scale_y` changes height. Omitting `scale_y` uses the X factor for both axes.
Scaling remains a low-level primitive, not another MR1 prompt key.

**Appearance mode** (the default) transforms a copy of the full image, then copies
only selected pixels back. Dimensions and alpha must be preserved; outside-mask
pixels remain byte-identical. Blur can sample background across mask boundaries;
this is not mask-normalized blur. Blur/noise perturb detail, not replace texture.

**Geometry mode** (`geometry=True`) extracts a tight RGBA object layer and passes
it to the existing flip/rotate/scale callable. Transparent areas of the bounding
box do not carry scene background into the edit. Pixels and alpha move together.
Placement uses a named anchor (`center` by default); `bottom_center` keeps a
person's feet at the original bottom edge, subject to integer rounding. The nine
anchors are top/center/bottom crossed with left/center/right, using `center` for
the middle and e.g. `top_left`, `center_right`, or `bottom_center` elsewhere.
Known geometric callables (including `functools.partial` wrappers) require
`geometry=True`, avoiding an accidental full-scene flip copied into a local mask.
For custom callable wrappers, the caller must choose the correct mode explicitly.
Visible clipping raises an error unless `allow_clipping=True`.

**Exposed old-object pixels are not silently retained.** If the new object does
not fully cover the original mask (including partial-alpha coverage), geometry
requires one of:

- `background=clean_background`: a full-size Pillow image of the scene without
  the object. Only pixels inside the original mask are copied from it.
- `fill='white'` (or another Pillow color): explicit non-photorealistic filling
  of the original mask before compositing. The examples above use this policy.

Without either, such edits raise `ValueError`. There is no automatic inpainting,
clean-background recovery, or depth-aware occlusion. New object pixels can cover
other scene objects; review geometric outputs accordingly. Imperfect segmentation
can leave original-object outlines outside the mask; inspect/refine masks before
using geometry. A passing composition test does not establish mask quality.

For one-off edits, `transform_objects` performs segmentation and then delegates
to the same compositor:

```python
from image_transforms import transform_objects

result = transform_objects(
    image, 'person', scale, segmenter=segmenter, instance=0,
    geometry=True, scale_x=1, scale_y=1.25,
    anchor='bottom_center', fill='white',
)
```

With `instance=None`, appearance edits apply to the union of all matches, as
`mask_objects` does. Geometry requires an explicit instance index when multiple
objects match, rather than stretching their combined arrangement. An explicit
index outside the detections raises `ValueError`; no detections with
`instance=None` return an unchanged copy. Empty explicit masks also return a copy
without invoking the transform. Callers should record these no-target cases as
skipped rather than successful perturbations.

`GroundedSAM.segment()` retains its original union-mask contract;
`segment_instances()` exposes individual masks before unioning. Phrase-based
spatial targeting is not guaranteed: inspect the instance masks, then select an
index or supply an explicit mask. A partial mask can also select just part of an
object. These composition APIs and `scale` do not import Torch/Transformers until
a segmentation backend is constructed. Python is the interface for object edits;
the existing CLI and six-filter reliability-prompt contract are unchanged.

Finally, an edit is not automatically answer-preserving. Select or relabel VQA
cases according to whether the changed color, texture, geometry, or visibility
is relevant to the question.
