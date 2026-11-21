# ViUniT environment (Lonestar6, A100)

Working env: conda env `viunit_env` at `/work/11808/shumera/ls6/conda/envs/viunit_env`
(python 3.10, torch 2.5.1+cu121, transformers 4.46.3, accelerate 0.33.0, vllm 0.6.4.post1,
xformers 0.0.28.post3, bitsandbytes 0.43.2, diffusers 0.30.3).

## Use the existing env
```bash
source butterfly/environment/activate.sh      # activates env + sets model/dataset paths
bash butterfly/environment/test_env.sh        # end-to-end check (needs a GPU node)
```
`test_env.py` (python version of `demo.ipynb`) generates visual programs (CodeLlama / ViperGPT API),
generates visual unit tests (Llama-3 + sampler + image generator), executes every program on every
unit-test image, and writes `plots/env_test/results.json` (+ programs and test images).
Existing `butterfly/demo.py` and `demo_simple.py` are the older smoke tests.

## Recreate from scratch
Files here were exported from `viunit_env`:
- `environment.yml` – conda env (conda + pip sections)
- `conda-explicit.txt` – exact conda packages (`conda create -n NAME --file conda-explicit.txt`)
- `pip-freeze.txt` – pip packages

```bash
conda env create -n viunit_env_new -f butterfly/environment/environment.yml
conda activate viunit_env_new
```
Then edit the env name in `activate.sh`. Models are expected under
`/work/11808/shumera/ls6/butterfly_shared_resources/models` (see `download_models.sh`).

## Notes
- Do NOT run `conda create --clone` of this env interactively: it takes >10 min.
- Past failure "bitsandbytes 8-bit requires accelerate" is fixed by `accelerate==0.33.0` (present).
- `VLLM_ATTENTION_BACKEND=XFORMERS` is required on A100 here (set in `activate.sh`).
- Set `HF_TOKEN` in your shell for gated models (Llama-3, CodeLlama); never commit it.
- flash-attn is NOT installed: the prebuilt wheel in `butterfly/wheels` needs GLIBC>=2.32 (LS6 has 2.28). The ViUniT/ViperGPT path doesn't need it (only the optional HF engine references it).
- XVLM (`base_models` repo) is not installed, so `test_env.py` uses `viper_configs/env_test_config.yaml` (= `demo_config.yaml` with `xvlm: False`). `verify_property` uses CLIP.
- If torch reports 0 GPUs while nvidia-smi works, `CUDA_VISIBLE_DEVICES` is set to empty; `activate.sh` unsets it.
