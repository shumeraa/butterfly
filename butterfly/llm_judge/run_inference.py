#!/usr/bin/env python3
"""Run one question-only vLLM call per question and save durable results and review artifacts."""
import argparse
import copy
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

from common import atomic_text, parse_response, read_jsonl, save_json, save_jsonl
from evaluate_rules import build_labels, count_verdicts
from prompts import PROMPT, PROMPT_VERSION, render

REVIEW_COLUMNS = ['id', 'detailed_type', 'question', 'target', 'flags', 'valid', 'parse_repaired',
                  'correct', 'notes']


def new_result(question):
    return {**question, 'raw': None, 'prediction': None, 'valid': None, 'parse_repaired': None,
            'attempted': False, 'error': None, 'complete': False}


def recover_failures(results):
    """Reparse failed responses, archiving unresolved attempts before retrying."""
    for row in results.values():
        if row.get('valid'):
            continue
        row['error'] = None
        row['complete'] = False
        parsed, valid, repaired, _ = parse_response(row.get('raw'))
        if row.get('finish_reason') == 'length':
            valid = False
        if valid:
            row.update({'prediction': parsed, 'valid': True, 'parse_repaired': repaired,
                        'attempted': True, 'complete': True})
            continue
        if row.get('attempted') or row.get('raw') is not None:
            keys = [k for k in row if k not in {'id', 'detailed_type', 'question', 'attempt_history'}]
            row.setdefault('attempt_history', []).append({k: copy.deepcopy(row[k]) for k in keys})
        for key in ('raw', 'prediction', 'valid', 'parse_repaired', 'finish_reason', 'generated_tokens'):
            row.pop(key, None)
        row.update({'raw': None, 'prediction': None, 'valid': None, 'parse_repaired': None, 'attempted': False})


def record_response(row, response):
    row['attempted'] = True
    # String callbacks remain supported for CPU callers and tests.
    raw = response['text'] if isinstance(response, dict) else response
    if isinstance(response, dict):
        row['finish_reason'] = response.get('finish_reason')
        row['generated_tokens'] = response.get('generated_tokens')
    row['raw'] = raw
    parsed, valid, repaired, error = parse_response(raw)
    if row.get('finish_reason') == 'length':
        parsed, valid, repaired = None, False, False
        error = 'generation reached its token/context limit before completing the answer'
    row.update({'prediction': parsed, 'valid': valid, 'parse_repaired': repaired,
                'error': error, 'complete': True})
    if not valid:
        print(f'{row["id"]}: {error}', file=sys.stderr)


def run_pipeline(questions, results, generate_batch, checkpoint, batch_size=1):
    """Generate pending questions in batches; checkpoint after each batch.

    generate_batch(prompts) returns one response per prompt, in order. If a batch
    raises, its questions are retried one at a time so a single bad request is
    recorded as that question's generation error instead of failing the batch.
    Completed IDs are never repeated.
    """
    pending = [results.setdefault(q['id'], new_result(q)) for q in questions]
    pending = [row for row in pending if not row['complete']]
    for start in range(0, len(pending), batch_size):
        rows = pending[start:start + batch_size]
        prompts = [render(row['question']) for row in rows]
        try:
            responses = generate_batch(prompts)
            if len(responses) != len(rows):
                raise RuntimeError('generator returned the wrong number of responses')
            outcomes = list(zip(rows, responses))
        except Exception:
            outcomes = []
            for row, prompt in zip(rows, prompts):
                try:
                    outcomes.append((row, generate_batch([prompt])[0]))
                except Exception as exc:
                    outcomes.append((row, exc))
        for row, response in outcomes:
            if isinstance(response, Exception):
                row.update({'attempted': True, 'valid': False, 'complete': True,
                            'error': f'generation error: {type(response).__name__}: {response}'})
                print(f'{row["id"]}: {row["error"]}', file=sys.stderr)
            else:
                record_response(row, response)
        checkpoint()


def export_artifacts(directory, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        prediction = row['prediction'] or {}
        writer.writerow({**row, 'target': prediction.get('target'),
                         'flags': json.dumps(prediction.get('flags'), ensure_ascii=False)
                         if prediction else None, 'correct': None, 'notes': None})
    atomic_text(directory / 'review.csv', stream.getvalue())

    labels = build_labels(rows)
    save_json(directory / 'labels.json', labels)

    valid = [r for r in rows if r['valid']]
    lines = ['# Question-only classification run', '', f'Questions saved: {len(rows)}',
             f'Completed: {sum(bool(r["complete"]) for r in rows)}', '']
    counters = {
        'Detailed types': Counter(r['detailed_type'] for r in rows),
        'Targets': Counter('(none)' if r['prediction']['target'] is None else 'object' for r in valid),
        'Flags': Counter(flag for r in valid for flag in r['prediction']['flags']),
    }
    for name, counts in counters.items():
        lines.extend([f'## {name}', ''] + [f'- {key}: {count}' for key, count in sorted(counts.items())] + [''])
    lines.extend(['## Verdicts (code-derived)', ''])
    for name, counts in count_verdicts(labels).items():
        lines.append(f'- {name}: ' + (', '.join(f'{k} {v}' for k, v in sorted(counts.items())) or 'none'))
    lines.extend(['', '## Parsing and generation', ''])
    attempted = sum(bool(r['attempted']) for r in rows)
    responses = sum(r['raw'] is not None for r in rows)
    parse_failures = sum(r['raw'] is not None and not r['valid'] for r in rows)
    generation_failures = sum(bool(r['attempted']) and r['raw'] is None for r in rows)
    repairs = sum(r['parse_repaired'] is True for r in rows)
    rate = repairs / responses if responses else 0
    lines.append(f'- {attempted} attempted, {responses} responses, {parse_failures} parse failures, '
                 f'{generation_failures} generation failures, {repairs} repaired ({rate:.1%} of responses).')
    for row in rows:
        if row['error']:
            lines.append(f'- Error {row["id"]}: {row["error"]}')
    atomic_text(directory / 'summary.md', '\n'.join(lines) + '\n')


def load_sample(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError('Sample is empty')
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'id', 'detailed_type', 'question'}:
            raise ValueError('Sample rows must contain only id, detailed_type, question')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
            raise ValueError('Sample fields must be nonempty strings')
        if row['id'] in ids:
            raise ValueError(f'Duplicate sample ID: {row["id"]}')
        ids.add(row['id'])
    return rows


def create_run_directory(directory, resume=False):
    """Reserve a unique directory for a fresh run before acquiring its lock."""
    if resume:
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    candidate = directory
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True)
        except FileExistsError:
            suffix += 1
            candidate = directory.with_name(f'{directory.name}_{suffix}')
        else:
            return candidate


def create_run_directory(directory, resume=False):
    """Reserve a unique directory for a fresh run before acquiring its lock."""
    if resume:
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    candidate = directory
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True)
        except FileExistsError:
            suffix += 1
            candidate = directory.with_name(f'{directory.name}_{suffix}')
        else:
            return candidate


def prepare_run(args, questions):
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    settings = {key: value for key, value in vars(args).items()
                if key not in {'questions', 'run_dir', 'resume', 'recover_failures'}}
    checksum = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    selection = None
    summary = args.questions.parent / 'sample_summary.json'
    if summary.is_file():
        selection = json.loads(summary.read_text(encoding='utf-8'))
        if selection.get('sample_sha256') != checksum:
            raise ValueError('Sample checksum differs from sample_summary.json')
    identity = {'sample_sha256': checksum, 'sample_ids': [r['id'] for r in questions],
                'model_identifier': args.model, 'prompt_version': PROMPT_VERSION,
                'prompt_template': PROMPT, 'settings': settings, 'dataset_selection': selection}
    metadata_path = directory / 'run_metadata.json'
    result_path = directory / 'results.jsonl'
    if metadata_path.exists():
        if not args.resume:
            raise ValueError('Run already exists; use --resume or a new --run-dir')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(k) != v for k, v in identity.items()):
            raise ValueError('Resume requires the same sample, model, prompt, and settings')
    else:
        if args.resume or result_path.exists():
            raise ValueError('Cannot resume without matching run_metadata.json; choose a fresh run directory')
        save_json(metadata_path, {**identity, 'questions_file': str(args.questions),
                                 'timestamp': datetime.now(timezone.utc).isoformat()})
    existing = read_jsonl(result_path) if result_path.exists() else []
    results = {}
    lookup = {r['id']: r for r in questions}
    for row in existing:
        qid = row['id']
        if qid in results or qid not in lookup or any(row[k] != lookup[qid][k] for k in lookup[qid]):
            raise ValueError('Results have duplicate, unknown, or changed sample records')
        results[qid] = row
    for row in questions:
        results.setdefault(row['id'], new_result(row))
    if getattr(args, 'recover_failures', False):
        recover_failures(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Output directory; fresh runs append _1, _2, etc. if it already exists')
    parser.add_argument('--model', required=True, help='Hugging Face identifier or execution-machine model directory')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--recover-failures', action='store_true',
                        help='With --resume, reparse failed responses and retry unresolved questions once')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', default='auto')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Questions per vLLM batch (also max concurrent sequences); progress is checkpointed per batch')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--prompt-mode', choices=['chat', 'plain'], default='chat')
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument('--disable-thinking', dest='disable_thinking', action='store_true', default=True,
                          help='Pass enable_thinking=False to tokenizer chat template (default)')
    thinking.add_argument('--enable-thinking', dest='disable_thinking', action='store_false',
                          help='Use the tokenizer chat template default thinking behavior')
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.batch_size <= 0:
        parser.error('--max-tokens and --batch-size must be positive')
    if args.recover_failures and not args.resume:
        parser.error('--recover-failures requires --resume')
    try:
        import fcntl
        questions = load_sample(args.questions)
        args.run_dir = create_run_directory(args.run_dir, args.resume)
        print(f'Run directory: {args.run_dir}', flush=True)
        with (args.run_dir / '.run.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Another inference process is using this run directory') from exc
            results = prepare_run(args, questions)
            def checkpoint():
                save_jsonl(args.run_dir / 'results.jsonl', [results[q['id']] for q in questions])
            checkpoint()
            try:
                if any(not row['complete'] for row in results.values()):
                    from vllm import LLM, SamplingParams
                    llm = LLM(model=args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
                              max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
                              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
                    tokenizer = llm.get_tokenizer() if args.prompt_mode == 'chat' else None
                    def generate_batch(prompts):
                        if tokenizer is not None:
                            kwargs = {'enable_thinking': False} if args.disable_thinking else {}
                            prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
                                       tokenize=False, add_generation_prompt=True, **kwargs) for p in prompts]
                        sampling = SamplingParams(temperature=args.temperature,
                                   max_tokens=args.max_tokens,
                                   seed=args.seed)
                        responses = []
                        for request in llm.generate(prompts, sampling):
                            output = request.outputs[0]
                            token_ids = getattr(output, 'token_ids', None)
                            responses.append({'text': output.text, 'finish_reason': getattr(output, 'finish_reason', None),
                                              'generated_tokens': len(token_ids) if token_ids is not None else None})
                        return responses
                    run_pipeline(questions, results, generate_batch, checkpoint, args.batch_size)
            finally:
                checkpoint()
                export_artifacts(args.run_dir, [results[q['id']] for q in questions])
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; saved responses can be resumed with --resume.\n')
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
