#!/usr/bin/env python3
"""Headless ViUniT smoke test for a Slurm GPU batch job."""

import ast
import copy
import os
from pathlib import Path

# These must be set before importing modules that use them.
PROJECT = Path('/work/11808/shumera/ls6/projects/butterfly')
PLOT_PATH = PROJECT / 'plots'
BASE_CONFIG = PROJECT / 'viunit_configs/base.yaml'
os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', os.environ['CONDA_PREFIX'])
os.environ['CONFIG_NAMES'] = 'demo_config'
PLOT_PATH.mkdir(parents=True, exist_ok=True)

import omegaconf
from PIL import Image
from rich.syntax import Syntax
from tqdm import tqdm
from transformers import AutoTokenizer

from unit_test_generation.processing import extract_unit_tests, get_unit_test_prompt, get_grounded_diffusion_prompt
from unit_test_generation.unit_test_sampling import TextSampler
from utils import (
    extract_python_code,
    get_base_prompt,
    get_fixed_code,
    initialize_image_generator,
)
from viper_configs import viper_config
from main_simple_lib import forward, execute_code


def log(message):
    print(message, flush=True)


def get_image(entry):
    """Unwrap the (metadata, image) structure returned by ViUniT."""
    image = entry[1]
    if isinstance(image, list):
        image = image[0]
    if not isinstance(image, Image.Image):
        raise TypeError(f'Expected PIL image, got {type(image).__name__}')
    return image.convert('RGB')


def get_unit_tests(query, tokenizer, sampler, generator, config, system_prompt, examples, grounded_system_prompt, grounded_examples, num_unit_tests=2):
    log('Generating unit tests...')
    prompts = get_unit_test_prompt([query], system_prompt, examples, viper_config.llm.model_id, tokenizer)
    unit_tests = []
    for attempt in range(2):
        output = forward(
            model_name='llm_general', prompt=copy.deepcopy(prompts), queues=None,
            min_new_tokens=10, max_new_tokens=180, return_full_text=False,
            top_p=0.9, do_sample=True, num_return_sequences=3,
        )
        # Preserve original ViUniT response parsing behavior.
        if not isinstance(output, str):
            raise TypeError(f'Unexpected LLM output type: {type(output).__name__}')
        unit_tests = extract_unit_tests(output.split('assistant')[-1])
        if unit_tests:
            break
    if not unit_tests:
        raise RuntimeError('The LLM returned no parseable unit tests after two attempts')

    log(f'Sampling {num_unit_tests} unit tests...')
    unit_tests = sampler.sample(unit_tests, num_unit_tests)
    if not unit_tests:
        raise RuntimeError('The sampler returned zero unit tests')

    if config['image_generation']['image_source'] == 'diffusion' and 'lmd' in config['image_generation']['diffusion_model_name']:
        prompts = get_grounded_diffusion_prompt(
            [ut[0].replace('"', '') for ut in unit_tests],
            grounded_system_prompt, grounded_examples,
            viper_config.llm.model_id, tokenizer,
        )
        responses = []
        log('Generating grounded diffusion prompts...')
        for prompt in tqdm(prompts):
            response = forward(
                model_name='llm_general', prompt=[prompt], queues=None,
                min_new_tokens=10, max_new_tokens=320, return_full_text=False,
            )
            responses.append(response.split('assistant')[-1])
        images = generator.batch_fetch_image([ut[0] for ut in unit_tests], responses)
    else:
        log('Generating images...')
        images = generator.batch_fetch_image([ut[0] for ut in unit_tests])

    per_test = int(config['image_generation']['return_k'])
    expected = len(unit_tests) * per_test
    if len(images) != expected:
        raise RuntimeError(f'Expected {expected} generated images, got {len(images)}')
    grouped_images = [images[i * per_test:(i + 1) * per_test] for i in range(len(unit_tests))]
    return unit_tests, grouped_images


def save_unit_tests(query, unit_tests, grouped_images):
    summary = [f'Question: {query}', '']
    for i, (unit_test, image_group) in enumerate(zip(unit_tests, grouped_images), start=1):
        line = f'Unit test {i}: {unit_test[0]} | Expected answer: {unit_test[1]}'
        log(line)
        summary.append(line)
        for j, entry in enumerate(image_group, start=1):
            path = PLOT_PATH / f'unit_test_{i}_image_{j}.png'
            get_image(entry).save(path)
            log(f'Saved image: {path}')
            summary.append(str(path))
    (PLOT_PATH / 'unit_tests.txt').write_text('\n'.join(summary) + '\n')


def get_program(query):
    log('Generating candidate visual programs...')
    outputs = forward(
        model_name='codellama', prompt=[query], queues=None,
        num_return_sequences=3, do_sample=True, top_p=0.9, max_new_tokens=320,
    )
    if isinstance(outputs, str):
        outputs = [outputs]
    return [extract_python_code(output) for output in outputs]


def exec_code(image, code):
    code = extract_python_code(code)
    if 'def execute_command(image)' not in code:
        code = 'def execute_command(image):\n' + code
    code = ast.unparse(ast.parse(code))
    syntax = Syntax(code, 'python', theme='light', line_numbers=True, start_line=0)
    executable = code.replace(
        'def execute_command(image)',
        'def execute_command(image, my_fig, time_wait_between_lines, syntax)',
    )
    img = image.convert('RGB')
    execute_code((executable, syntax), img, show_intermediate_steps=False)


def execute_programs(programs, unit_tests, grouped_images):
    for program_index, program in enumerate(programs, start=1):
        path = PLOT_PATH / f'candidate_program_{program_index}.py'
        path.write_text(program + '\n')
        log(f'Program {program_index} saved: {path}')
        for test_index, (unit_test, image_group) in enumerate(zip(unit_tests, grouped_images), start=1):
            for image_index, entry in enumerate(image_group, start=1):
                log(f'Running program {program_index}, test {test_index}, image {image_index}; expected={unit_test[1]!r}')
                try:
                    exec_code(get_image(entry), program)
                    log('Execution completed (output correctness not scored).')
                except Exception as exc:
                    log(f'Program execution failed: {type(exc).__name__}: {exc}')


def main():
    log(f'Writing outputs to {PLOT_PATH}')
    config = omegaconf.OmegaConf.load(BASE_CONFIG)
    llm_model_name = viper_config.llm.model_id
    codex_model_name = viper_config.codex.codellama_model_name
    token = os.getenv('HF_ACCESS_TOKEN')

    unit_cfg = config['unit_test_generation']['generation']
    system_prompt = Path(unit_cfg['prompt_file']).read_text()
    examples = Path(unit_cfg['in_context_examples_file']).read_text()
    correction_cfg = config['visual_program_generator']['generation']
    # These are loaded to match the original demo's setup.
    Path(correction_cfg['correction_prompt_file']).read_text()
    grounded_cfg = config['image_generation']['generation']
    grounded_system_prompt = Path(grounded_cfg['prompt_file']).read_text().strip()
    grounded_examples = Path(grounded_cfg['in_context_examples_file']).read_text().strip()
    get_base_prompt(
        correction_cfg['prompt_file'], correction_cfg['in_context_examples_file'],
        correction_cfg['num_in_context_examples'],
    )

    log('Loading tokenizers...')
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True, token=token)
    AutoTokenizer.from_pretrained(codex_model_name, trust_remote_code=True, token=token)

    log('Initializing text sampler and image generator...')
    sampler = TextSampler(
        model_name=config['unit_test_sampling']['model_name'],
        sampling_strategy=config['unit_test_sampling']['strategy'],
        filter_long_answers=config['unit_test_sampling']['filter_long_answers'],
    )
    config['image_generation']['return_image'] = True
    generator = initialize_image_generator(config)

    query = 'What color are the pillows on the dark couch?'
    log(f'Question: {query}')
    unit_tests, images = get_unit_tests(
        query, llm_tokenizer, sampler, generator, config,
        system_prompt, examples, grounded_system_prompt, grounded_examples,
        num_unit_tests=2,
    )
    save_unit_tests(query, unit_tests, images)
    programs = get_program(query)
    execute_programs(programs, unit_tests, images)
    log('ViUniT smoke test finished.')


if __name__ == '__main__':
    main()
