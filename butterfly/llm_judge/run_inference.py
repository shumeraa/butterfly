#!/usr/bin/env python3
"""Run two question-only vLLM calls and save durable results and review artifacts."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

from common import atomic_text, parse_response, read_jsonl, save_json, save_jsonl
from excel_io import write_workbook
from prompts import PROMPT_VERSION, TEMPLATES, render

REVIEW_COLUMNS = ['id', 'detailed_type', 'question', 'answer_type', 'topic', 'objects',
                  'question_flags', 'call1_valid', 'call2_valid', 'call1_parse_repaired',
                  'call2_parse_repaired', 'correct', 'notes']


def new_result(question):
    return {**question, 'call1': None, 'call2': None, 'call1_raw': None, 'call2_raw': None,
            'call1_valid': None, 'call2_valid': None, 'call1_parse_repaired': None,
            'call2_parse_repaired': None, 'call1_attempted': False, 'call2_attempted': False,
            'error': None, 'complete': False}


def run_pipeline(questions, results, generate, checkpoint):
    """Checkpoint each response before continuing; completed IDs are never repeated."""
    for question in questions:
        row = results.setdefault(question['id'], new_result(question))
        if row['complete']:
            continue
        for call in (1, 2):
            prefix = f'call{call}'
            if row[prefix + '_attempted']:
                continue
            if call == 2 and not row['call1_valid']:
                break
            template = TEMPLATES['call1' if call == 1 else row['call1']]
            try:
                raw = generate(render(template, question['question']), call)
            except Exception as exc:
                row[prefix + '_attempted'] = True
                row[prefix + '_valid'] = False
                row['error'] = f'{prefix} generation error: {type(exc).__name__}: {exc}'
                row['complete'] = True
                checkpoint()
                break
            row[prefix + '_attempted'] = True
            row[prefix + '_raw'] = raw
            parsed, valid, repaired, error = parse_response(raw, None if call == 1 else row['call1'])
            row[prefix] = parsed
            row[prefix + '_valid'] = valid
            row[prefix + '_parse_repaired'] = repaired
            if error:
                row['error'] = f'{prefix}: {error}'
            row['complete'] = call == 2 or not valid
            checkpoint()
            if not valid:
                print(f'{question["id"]}: {row["error"]}', file=sys.stderr)
                break


def export_artifacts(directory, rows):
    reviews = []
    for row in rows:
        parsed = row['call2'] or {}
        reviews.append({**row, 'answer_type': row['call1'], 'topic': parsed.get('topic'),
                        'objects': parsed.get('objects'), 'question_flags': parsed.get('question_flags'),
                        'correct': None, 'notes': None})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS, extrasaction='ignore')
    writer.writeheader()
    for row in reviews:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    atomic_text(directory / 'review.csv', stream.getvalue())
    extra = ['call1_raw', 'call2_raw', 'error', 'complete', 'call1_attempted', 'call2_attempted']
    write_workbook(directory / 'predictions.xlsx', {'Predictions': (REVIEW_COLUMNS + extra, reviews)})
    lines = ['# Question-only classification run', '', f'Questions saved: {len(rows)}',
             f'Completed: {sum(bool(r["complete"]) for r in rows)}', '']
    counters = {
        'Detailed types': Counter(r['detailed_type'] for r in rows),
        'Answer types': Counter(r['call1'] for r in rows if r['call1_valid']),
        'Topics': Counter(r['call2']['topic'] for r in rows if r['call2_valid']),
        'Question flags': Counter(flag for r in rows if r['call2_valid'] for flag in set(r['call2']['question_flags'])),
    }
    for name, counts in counters.items():
        lines.extend([f'## {name}', ''] + [f'- {key}: {count}' for key, count in sorted(counts.items())] + [''])
    lines.extend(['## Parsing and generation', ''])
    for call in ('call1', 'call2'):
        attempted = sum(bool(r[call + '_attempted']) for r in rows)
        responses = sum(r[call + '_raw'] is not None for r in rows)
        parse_failures = sum(r[call + '_raw'] is not None and not r[call + '_valid'] for r in rows)
        generation_failures = sum(bool(r[call + '_attempted']) and r[call + '_raw'] is None for r in rows)
        repairs = sum(r[call + '_parse_repaired'] is True for r in rows)
        rate = repairs / responses if responses else 0
        lines.append(f'- {call}: {attempted} attempted, {responses} responses, {parse_failures} parse failures, '
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
    counts = Counter()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'id', 'detailed_type', 'question'}:
            raise ValueError('Sample rows must contain only id, detailed_type, question')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in row):
            raise ValueError('Sample fields must be nonempty strings')
        if row['id'] in ids:
            raise ValueError(f'Duplicate sample ID: {row["id"]}')
        ids.add(row['id'])
        counts[row['detailed_type']] += 1
    if any(count != 2 for count in counts.values()):
        raise ValueError('Expected exactly two questions per discovered detailed type')
    return rows


def prepare_run(args, questions):
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    settings = {key: value for key, value in vars(args).items() if key not in {'questions', 'run_dir', 'resume'}}
    checksum = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    selection = None
    summary = args.questions.parent / 'sample_summary.json'
    if summary.is_file():
        selection = json.loads(summary.read_text(encoding='utf-8'))
        if selection.get('sample_sha256') != checksum:
            raise ValueError('Sample checksum differs from sample_summary.json')
    identity = {'sample_sha256': checksum, 'sample_ids': [r['id'] for r in questions],
                'model_identifier': args.model, 'prompt_version': PROMPT_VERSION,
                'prompt_templates': TEMPLATES, 'settings': settings, 'dataset_selection': selection}
    metadata_path = directory / 'run_metadata.json'
    result_path = directory / 'results.jsonl'
    if metadata_path.exists():
        if not args.resume:
            raise ValueError('Run already exists; use --resume or a new --run-dir')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(k) != v for k, v in identity.items()):
            raise ValueError('Resume requires the same sample, model, prompts, and settings')
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
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model', required=True, help='Hugging Face identifier or execution-machine model directory')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', default='auto')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--call1-max-tokens', type=int, default=256)
    parser.add_argument('--call2-max-tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--prompt-mode', choices=['chat', 'plain'], default='chat')
    parser.add_argument('--disable-thinking', action='store_true', help='Pass enable_thinking=False to tokenizer chat template')
    args = parser.parse_args()
    try:
        # Check Excel dependency before paying for GPU initialization.
        import openpyxl  # noqa: F401
        import fcntl
        questions = load_sample(args.questions)
        args.run_dir.mkdir(parents=True, exist_ok=True)
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
                              max_model_len=args.max_model_len, max_num_seqs=1,
                              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
                    tokenizer = llm.get_tokenizer() if args.prompt_mode == 'chat' else None
                    def generate(prompt, call):
                        if tokenizer is not None:
                            kwargs = {'enable_thinking': False} if args.disable_thinking else {}
                            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                       tokenize=False, add_generation_prompt=True, **kwargs)
                        sampling = SamplingParams(temperature=args.temperature,
                                   max_tokens=args.call1_max_tokens if call == 1 else args.call2_max_tokens,
                                   seed=args.seed)
                        return llm.generate([prompt], sampling)[0].outputs[0].text
                    run_pipeline(questions, results, generate, checkpoint)
            finally:
                checkpoint()
                export_artifacts(args.run_dir, [results[q['id']] for q in questions])
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; saved responses can be resumed with --resume.\n')
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
