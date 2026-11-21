#!/usr/bin/env python3
"""End-to-end environment check for ViUniT (python version of demo.ipynb).

For each question: generate visual-program candidates (CodeLlama/ViperGPT API),
generate visual unit tests (Llama-3 + sampler + image generator), execute every
program on every unit-test image, and score output vs. expected answer.
Results go to <output>/results.json.

Run via: sbatch/bash butterfly/environment/test_env.sh   (needs GPU)
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT = Path('/work/11808/shumera/ls6/projects/butterfly')
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'butterfly'))

QUESTIONS = [
    'What color are the pillows on the dark couch?',
    'How many people are in the image?',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, default=PROJECT / 'plots/env_test')
    ap.add_argument('--num-tests', type=int, default=3)
    ap.add_argument('--num-programs', type=int, default=3)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # xvlm repo isn't installed; use a config with xvlm disabled (demo.py would override CONFIG_NAMES)
    import os
    os.environ['CONFIG_NAMES'] = 'env_test_config'
    os.chdir(PROJECT)
    import viper_configs  # caches merged config before demo.py imports it
    import demo  # butterfly/demo.py: imports ViperGPT API + ViUniT helpers (sets CONFIG_NAMES)
    import omegaconf
    from transformers import AutoTokenizer
    from unit_test_generation.unit_test_sampling import TextSampler
    from utils import get_base_prompt, initialize_image_generator
    from viper_configs import viper_config
    import os

    demo.PLOT_PATH = args.output_dir
    cfg = omegaconf.OmegaConf.load(demo.BASE_CONFIG)
    gen = cfg['unit_test_generation']['generation']
    sys_prompt = Path(gen['prompt_file']).read_text()
    examples = Path(gen['in_context_examples_file']).read_text()
    g = cfg['image_generation']['generation']
    g_sys = Path(g['prompt_file']).read_text().strip()
    g_ex = Path(g['in_context_examples_file']).read_text().strip()
    tok = AutoTokenizer.from_pretrained(viper_config.llm.model_id, trust_remote_code=True,
                                        token=os.getenv('HF_ACCESS_TOKEN'))
    sampler = TextSampler(model_name=cfg['unit_test_sampling']['model_name'],
                          sampling_strategy=cfg['unit_test_sampling']['strategy'],
                          filter_long_answers=cfg['unit_test_sampling']['filter_long_answers'])
    cfg['image_generation']['return_image'] = True
    generator = initialize_image_generator(cfg)

    results = []
    for q in QUESTIONS:
        demo.log(f'=== {q}')
        tests, imgs = demo.get_unit_tests(q, tok, sampler, generator, cfg, sys_prompt, examples,
                                          g_sys, g_ex, num_unit_tests=args.num_tests)
        programs = demo.get_program(q)[:args.num_programs]
        qres = {'question': q, 'unit_tests': [list(map(str, t)) for t in tests], 'programs': []}
        for pi, prog in enumerate(programs):
            (args.output_dir / f'q{len(results)}_prog{pi}.py').write_text(prog)
            outs = []
            for ti, (t, group) in enumerate(zip(tests, imgs)):
                for entry in group:
                    try:
                        code = demo.extract_python_code(prog)
                        if 'def execute_command(image)' not in code:
                            code = 'def execute_command(image):\n' + code
                        code = demo.ast.unparse(demo.ast.parse(code))
                        syn = demo.Syntax(code, 'python')
                        code = code.replace('def execute_command(image)',
                                            'def execute_command(image, my_fig, time_wait_between_lines, syntax)')
                        out = demo.execute_code((code, syn), demo.get_image(entry), show_intermediate_steps=False)
                        status = 'ok'
                    except Exception as e:
                        out, status = f'{type(e).__name__}: {e}', 'error'
                    passed = status == 'ok' and str(out).strip().lower() == str(t[1]).strip().lower()
                    outs.append({'test': ti, 'expected': str(t[1]), 'output': str(out), 'status': status, 'passed': passed})
            qres['programs'].append({'index': pi, 'results': outs,
                                     'executed_ok': sum(o['status'] == 'ok' for o in outs),
                                     'passed': sum(o['passed'] for o in outs), 'total': len(outs)})
        for ti, group in enumerate(imgs):
            for ii, entry in enumerate(group):
                demo.get_image(entry).save(args.output_dir / f'q{len(results)}_test{ti}_img{ii}.png')
        results.append(qres)
        (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2))

    ran = sum(p['executed_ok'] for r in results for p in r['programs'])
    tot = sum(p['total'] for r in results for p in r['programs'])
    demo.log(f'ENV TEST DONE: {ran}/{tot} program executions ran without error; see {args.output_dir}/results.json')
    sys.exit(0 if ran > 0 else 1)


if __name__ == '__main__':
    main()
