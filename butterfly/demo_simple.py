
    
  
#!/usr/bin/env python3
"""ViUniT smoke test: generate (caption, answer) pairs and synthetic images.

Uses the original ViUniT prompt builder, parser and sampler; bypasses
main_simple_lib/vision_processes and replaces LMD+ with plain SD 1.5.
"""
import argparse
import json
import os
from pathlib import Path

PROJECT = Path('/work/11808/shumera/ls6/projects/butterfly')
DEFAULT_IMAGE = Path('/work/11808/shumera/ls6/butterfly_shared_resources/datasets/coco/val2017/000000482319.jpg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=DEFAULT_IMAGE)
    parser.add_argument('--question', default='How many people are in the image?')
    parser.add_argument('--config', type=Path, default=PROJECT / 'viunit_configs/base.yaml')
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'plots/simple_unit_tests')
    parser.add_argument('--num-tests', type=int, default=2)
    parser.add_argument('--sd-model', default='runwayml/stable-diffusion-v1-5',
                        help='Plain SD checkpoint for the smoke test; replaces LMD+.')
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Paths in ViUniT YAML are relative to the repository root.
    os.chdir(PROJECT)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from unit_test_generation.processing import get_unit_test_prompt, extract_unit_tests
    from unit_test_generation.unit_test_sampling import TextSampler

    with Image.open(args.image) as im:
        im.verify()
    if not torch.cuda.is_available():
        raise RuntimeError('This script requires a GPU compute node (not a login node).')

    cfg = OmegaConf.load(args.config)
    model_id = cfg.unit_test_generation.model_name
    gen_cfg = cfg.unit_test_generation.generation
    system_prompt = Path(gen_cfg.prompt_file).read_text(encoding='utf-8')
    examples = Path(gen_cfg.in_context_examples_file).read_text(encoding='utf-8')

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    prompts = get_unit_test_prompt([args.question], system_prompt, examples, model_id, tokenizer)

    print('Loading text model:', model_id, flush=True)
    llm = LLM(model=model_id, dtype='float16', tensor_parallel_size=1,
              gpu_memory_utilization=0.80, max_model_len=4096,
              trust_remote_code=True)
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512, n=3)
    tests = []
    for attempt in range(2):
        result = llm.generate(prompts, params, use_tqdm=False)
        for candidate in result[0].outputs:
            tests.extend(extract_unit_tests(candidate.text))
        if tests:
            break
        print(f'No parseable tests on attempt {attempt + 1}; retrying', flush=True)
    if not tests:
        raise RuntimeError('No parseable ViUniT unit tests generated')

    # Avoid loading both the 8B LLM and the diffusion model simultaneously.
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    sample_cfg = cfg.unit_test_sampling
    sampler = TextSampler(model_name=sample_cfg.model_name,
                          sampling_strategy=sample_cfg.strategy,
                          filter_long_answers=sample_cfg.filter_long_answers)
    tests = sampler.sample(tests, args.num_tests)
    if not tests:
        raise RuntimeError('Sampling returned no tests')
    del sampler
    gc.collect()
    torch.cuda.empty_cache()

    # Diffusion is intentionally plain SD, NOT the default LMD+ pipeline.
    from diffusers import StableDiffusionPipeline
    print('Loading diffusion model:', args.sd_model, flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model, torch_dtype=torch.float16,
                                                  safety_checker=None)
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (caption, answer) in enumerate(tests, 1):
        caption, answer = str(caption), str(answer)
        print(f'Test {i}: {caption} -> {answer}', flush=True)
        image = pipe(prompt=caption, num_inference_steps=args.steps,
                     guidance_scale=7.5,
                     generator=torch.Generator(device='cuda').manual_seed(args.seed + i)).images[0]
        filename = f'test_{i:02d}.png'
        image.save(args.output_dir / filename)
        records.append({'description': caption, 'expected_answer': answer,
                        'generated_image': filename})

    metadata = {'source_image': str(args.image), 'original_question': args.question,
                'text_model': model_id, 'image_model': args.sd_model,
                'note': 'Source image pixels do not condition generation; plain SD replaces LMD+.',
                'unit_tests': records}
    (args.output_dir / 'unit_tests.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print('Saved results to:', args.output_dir, flush=True)


if __name__ == '__main__':
    main()


