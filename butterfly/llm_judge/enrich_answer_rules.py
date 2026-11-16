#!/usr/bin/env python3
"""Enrich saved question-only predictions with per-transformation answer rules."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from answer_rule_prompts import PROMPT, PROMPT_VERSION, TRANSFORMATIONS, render
from common import atomic_text, clean_response, read_jsonl, save_json, save_jsonl
from evaluate_rules import evaluate, normalize, read_predictions

FIELDS = {'mode', 'answer_domain', 'mapping', 'constant_answer', 'conditions', 'reason'}


def validate_rule(rule, answer_type, analysis):
    """Validate structure and useful constraints; semantic claims remain predictions."""
    if not isinstance(rule, dict) or set(rule) != FIELDS:
        raise ValueError('Expected exactly the six answer-rule fields')
    mode = rule['mode']
    if mode not in ('MAP', 'CONSTANT', 'UNRESOLVED'):
        raise ValueError('Invalid mode')
    domain = rule['answer_domain']
    if not isinstance(domain, list) or any(not isinstance(x, str) or not x.strip() for x in domain):
        raise ValueError('answer_domain must contain nonempty strings')
    if len(set(domain)) != len(domain):
        raise ValueError('Duplicate answer-domain entries')
    options = analysis['answer_options']
    if options and domain != options:
        raise ValueError('answer_domain must preserve explicit choices and their order')
    if mode != 'UNRESOLVED' and (not domain or (answer_type == 'BINARY' and len(domain) != 2)):
        raise ValueError('A resolved binary rule requires exactly two answers')
    if mode != 'UNRESOLVED' and answer_type == 'BINARY' and not options and domain != ['yes', 'no']:
        raise ValueError('Implicit binary answers must be yes/no; otherwise use UNRESOLVED')
    mapping = rule['mapping']
    if not isinstance(mapping, dict):
        raise ValueError('mapping must be an object')
    if mode == 'MAP':
        if set(mapping) != set(domain) or rule['constant_answer'] is not None:
            raise ValueError('MAP must cover the domain and have null constant_answer')
        for outputs in mapping.values():
            if (not isinstance(outputs, list) or not outputs
                    or any(not isinstance(x, str) or x not in domain for x in outputs)
                    or len(set(outputs)) != len(outputs)):
                raise ValueError('Mapped outputs must be nonempty unique subsets of the domain')
        if all(set(outputs) == set(domain) for outputs in mapping.values()):
            raise ValueError('Uninformative mapping: use UNRESOLVED')
        if all(len(outputs) == 1 for outputs in mapping.values()) and len({v[0] for v in mapping.values()}) == 1:
            raise ValueError('Constant mappings must use CONSTANT mode')
    elif mapping:
        raise ValueError('Only MAP may contain a mapping')
    if mode == 'CONSTANT':
        if not isinstance(rule['constant_answer'], str) or rule['constant_answer'] not in domain:
            raise ValueError('constant_answer must belong to answer_domain')
    elif rule['constant_answer'] is not None:
        raise ValueError('constant_answer must be null outside CONSTANT')
    if not isinstance(rule['conditions'], list):
        raise ValueError('conditions must be a list')
    for condition in rule['conditions']:
        if (not isinstance(condition, dict) or set(condition) != {'condition', 'verified_from_input'}
                or not isinstance(condition['condition'], str) or not condition['condition'].strip()
                or type(condition['verified_from_input']) is not bool):
            raise ValueError('Invalid condition')
    if not isinstance(rule['reason'], str) or not rule['reason'].strip():
        raise ValueError('reason must be nonempty')
    return rule


def parse_rule(raw, answer_type, analysis):
    try:
        rule = json.loads(clean_response(raw))
        validate_rule(rule, answer_type, analysis)
        return rule, True, clean_response(raw) != raw.strip(), None
    except (ValueError, TypeError, AttributeError) as exc:
        return None, False, False, str(exc)


def classify(rule):
    if rule['mode'] == 'UNRESOLVED':
        return 'uninformative', 'unresolved'
    informative = 'partial' if rule['mode'] == 'MAP' and any(len(v) > 1 for v in rule['mapping'].values()) else 'exact'
    # All conditions are still model assertions, including claims of verification.
    return informative, 'conditional_prediction' if rule['conditions'] else 'prediction'


def make_tasks(rows):
    tasks = []
    for source_row, source in enumerate(rows, 1):
        try:
            answer_type, analysis = normalize(source)
            globals_, locals_ = evaluate(answer_type, analysis['topic'], analysis['objects'],
                                   analysis['question_flags'], analysis['answer_options'])
            candidates = [(name, 'equivariant') for name, (label, _) in globals_.items() if label == 'C']
            for edits in locals_:
                candidates.extend((name, 'local_equivariant') for name, (label, _) in edits.items() if label == 'C')
            error = None
        except (ValueError, TypeError) as exc:
            candidates, answer_type, analysis, error = [], None, None, str(exc)
        # Keep a task-free source record for review, including invalid source rows.
        for transformation, selection in candidates or [(None, 'invalid_input' if error else 'not_applicable')]:
            tasks.append({'source_row': source_row, 'id': source.get('id'), 'question': source.get('question'),
                          'answer_type': answer_type, 'analysis': analysis, 'transformation': transformation,
                          'selection': selection, 'source_error': error, 'complete': transformation is None,
                          'call3_raw': None, 'call3': None, 'call3_valid': None,
                          'call3_parse_repaired': None, 'error': None})
    return tasks


def run_pipeline(tasks, generate, checkpoint):
    for task in tasks:
        if task['complete']:
            continue
        prompt = render(task['question'], task['answer_type'], task['analysis'], task['transformation'])
        print(f'{task["id"]}: {task["transformation"]}', flush=True)
        try:
            response = generate(prompt)
        except Exception as exc:
            # Never save provider exception bodies, which may contain credentials.
            task['error'] = 'Generation failed: ' + type(exc).__name__
            checkpoint()
            raise RuntimeError(task['error'] + '; resume to retry the unanswered request') from None
        raw = response['text']
        task.update({'call3_raw': raw, 'generation': {k: v for k, v in response.items() if k != 'text'}})
        rule, valid, repaired, error = parse_rule(raw, task['answer_type'], task['analysis'])
        if not response.get('completed', False):
            rule, valid, repaired, error = None, False, False, 'Generation did not complete normally'
        task.update({'call3': rule, 'call3_valid': valid, 'call3_parse_repaired': repaired,
                     'error': error, 'complete': True})
        if valid:
            task['informativeness'], task['rule_status'] = classify(rule)
        checkpoint()


def export(directory, tasks):
    columns = ['source_row', 'id', 'question', 'transformation', 'selection', 'source_error',
               'call3_valid', 'call3', 'informativeness', 'rule_status', 'error', 'correct', 'notes']
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for task in tasks:
        row = {key: task.get(key) for key in columns}
        row['call3'] = json.dumps(row['call3'], ensure_ascii=False) if row['call3'] is not None else ''
        writer.writerow(row)
    atomic_text(directory / 'review.csv', output.getvalue())
    counts = Counter()
    for task in tasks:
        status = task['selection'] if task['transformation'] is None else (
            task.get('rule_status') or ('invalid_response' if task['complete'] else 'pending'))
        counts[status] += 1
    atomic_text(directory / 'summary.md', '# Answer-rule enrichment\n\n'
                + '\n'.join(f'- {key}: {value}' for key, value in sorted(counts.items()))
                + '\n\nRules are model predictions. Conditions are not independently verified.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Saved results.jsonl, CSV, or corrected XLSX')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--provider', choices=['anthropic', 'vllm'], default='anthropic')
    parser.add_argument('--model', help='Defaults to the Claude runner model; required for vLLM')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.max_tokens, args.tensor_parallel_size, args.max_model_len, args.timeout) <= 0 or args.max_retries < 0:
        parser.error('Budgets, timeout and parallelism must be positive; retries must be nonnegative')
    if args.provider == 'vllm' and not args.model:
        parser.error('--model is required for vllm')
    from run_claude_inference import DEFAULT_MODEL, generate_claude
    args.model = args.model or DEFAULT_MODEL
    client = None
    try:
        import fcntl
        if args.input.resolve() == (args.run_dir / 'results.jsonl').resolve():
            raise ValueError('Input and enrichment results must be distinct')
        # Never write enrichment artifacts into the source directory.
        if args.run_dir.resolve() == args.input.parent.resolve():
            raise ValueError('Use a separate enrichment directory')
        _, rows = read_predictions(args.input)
        expected = make_tasks(rows)
        identity = {'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    'prompt_version': PROMPT_VERSION, 'prompt_template': PROMPT,
                    'transformations': TRANSFORMATIONS,
                    'settings': {k: v for k, v in vars(args).items() if k not in ('input', 'run_dir', 'resume')}}
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with (args.run_dir / '.run.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another enrichment process is using this directory') from None
            metadata_path = args.run_dir / 'run_metadata.json'
            result_path = args.run_dir / 'results.jsonl'
            if args.resume:
                metadata = json.loads(metadata_path.read_text())
                if any(metadata.get(k) != v for k, v in identity.items()):
                    raise ValueError('Resume requires identical input, prompts, transformations and settings')
                tasks = read_jsonl(result_path)
                keys = ('source_row', 'id', 'question', 'answer_type', 'analysis', 'transformation', 'selection', 'source_error')
                if ([{k: t[k] for k in keys} for t in tasks] != [{k: t[k] for k in keys} for t in expected]):
                    raise ValueError('Saved tasks differ from source tasks')
            else:
                if any(p.name != '.run.lock' for p in args.run_dir.iterdir()):
                    raise ValueError('Use an empty run directory or --resume')
                tasks = expected
                save_json(metadata_path, {**identity, 'input_file': str(args.input.resolve()),
                                         'timestamp': datetime.now(timezone.utc).isoformat()})
            def checkpoint():
                save_jsonl(result_path, tasks)
            checkpoint()
            try:
                if any(not t['complete'] for t in tasks):
                    if args.provider == 'anthropic':
                        key = os.environ.get('CLAUDE_API_KEY')
                        if not key:
                            raise ValueError('CLAUDE_API_KEY is not set')
                        from anthropic import Anthropic
                        client = Anthropic(api_key=key, timeout=args.timeout, max_retries=args.max_retries)
                        # Reuse existing provider defaults and independent-request behavior.
                        args.call2_max_tokens, args.thinking, args.effort = args.max_tokens, 'disabled', 'low'
                        def generate(prompt):
                            response = generate_claude(client, args, prompt, 3)
                            return {'text': ''.join(b.text for b in response.content if b.type == 'text'),
                                    'completed': response.stop_reason == 'end_turn',
                                    'api_response': response.model_dump(mode='json')}
                    else:
                        from vllm import LLM, SamplingParams
                        llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
                                  max_model_len=args.max_model_len, seed=args.seed)
                        tokenizer = llm.get_tokenizer()
                        def generate(prompt):
                            chat = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
                            response = llm.generate([chat], SamplingParams(temperature=0, seed=args.seed,
                                                       max_tokens=args.max_tokens))[0].outputs[0]
                            return {'text': response.text, 'completed': response.finish_reason == 'stop',
                                    'finish_reason': response.finish_reason,
                                    'generated_tokens': len(response.token_ids)}
                    run_pipeline(tasks, generate, checkpoint)
            finally:
                checkpoint()
                export(args.run_dir, tasks)
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; use --resume to continue.\n')
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    finally:
        if client is not None:
            client.close()


if __name__ == '__main__':
    main()
