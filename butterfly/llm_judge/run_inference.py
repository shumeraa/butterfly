#!/usr/bin/env python3
"""Run two question-only vLLM calls and save durable results and review artifacts."""
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
from prompts import PROMPT_VERSION, TEMPLATES, render

REVIEW_COLUMNS = ['id', 'detailed_type', 'question', 'answer_type', 'topic', 'objects',
                  'answer_options', 'question_flags', 'call1_valid', 'call2_valid', 'call1_parse_repaired',
                  'call2_parse_repaired', 'correct', 'notes']


def new_result(question):
    return {**question, 'call1': None, 'call2': None, 'call1_raw': None, 'call2_raw': None,
            'call1_valid': None, 'call2_valid': None, 'call1_parse_repaired': None,
            'call2_parse_repaired': None, 'call1_attempted': False, 'call2_attempted': False,
            'error': None, 'complete': False}


def recover_failures(results):
    """Reparse failed responses, archiving unresolved attempts before retrying."""
    for row in results.values():
        if row.get('call2_valid'):
            continue
        row['error'] = None
        row['complete'] = False
        for call in (1, 2):
            prefix = f'call{call}'
            if row.get(prefix + '_valid'):
                continue
            raw = row.get(prefix + '_raw')
            parsed, valid, repaired, _ = parse_response(raw, None if call == 1 else row['call1'])
            if row.get(prefix + '_finish_reason') == 'length':
                valid = False
            if valid:
                row.update({prefix: parsed, prefix + '_valid': True,
                            prefix + '_parse_repaired': repaired, prefix + '_attempted': True})
                continue
            # A new Call 1 can select a different branch; invalidate Call 2 too.
            for reset_call in range(call, 3):
                reset_prefix = f'call{reset_call}'
                keys = [key for key in row if key == reset_prefix or key.startswith(reset_prefix + '_')]
                if row.get(reset_prefix + '_attempted') or row.get(reset_prefix + '_raw') is not None:
                    row.setdefault('attempt_history', []).append({key: copy.deepcopy(row[key]) for key in keys})
                for key in keys:
                    row[key] = False if key.endswith('_attempted') else None
            break
        row['complete'] = bool(row.get('call2_valid'))


def import_recovery(directory, questions, model):
    """Reuse responses only when their model, sample, and actual prompts match."""
    metadata = json.loads((directory / 'run_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('settings', {}).get('provider') == 'anthropic':
        raise ValueError('--recover-from supports vLLM runs only')
    if metadata.get('model_identifier') != model:
        raise ValueError('Recovery requires the same model identifier')
    rows = read_jsonl(directory / 'results.jsonl')
    lookup = {q['id']: q for q in questions}
    results = {}
    for row in rows:
        qid = row['id']
        if qid in results or qid not in lookup or any(row[k] != lookup[qid][k] for k in lookup[qid]):
            raise ValueError('Recovery results have duplicate, unknown, or changed sample records')
        for call in (1, 2):
            if not row.get(f'call{call}_attempted'):
                continue
            template = 'call1' if call == 1 else row['call1']
            if metadata.get('prompt_templates', {}).get(template) != TEMPLATES.get(template):
                raise ValueError(f'Recovery prompt differs for {qid} call{call}; use a fresh run')
        results[qid] = copy.deepcopy(row)
    if metadata.get('sample_ids') != [q['id'] for q in questions]:
        raise ValueError('Recovery requires the same sample IDs and order')
    for question in questions:
        results.setdefault(question['id'], new_result(question))
    recover_failures(results)
    return results


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
                response = generate(render(template, question['question']), call)
            except Exception as exc:
                row[prefix + '_attempted'] = True
                row[prefix + '_valid'] = False
                row['error'] = f'{prefix} generation error: {type(exc).__name__}: {exc}'
                row['complete'] = True
                checkpoint()
                break
            row[prefix + '_attempted'] = True
            # String callbacks remain supported for CPU callers and tests.
            raw = response['text'] if isinstance(response, dict) else response
            if isinstance(response, dict):
                row[prefix + '_finish_reason'] = response.get('finish_reason')
                row[prefix + '_generated_tokens'] = response.get('generated_tokens')
            row[prefix + '_raw'] = raw
            parsed, valid, repaired, error = parse_response(raw, None if call == 1 else row['call1'])
            if row.get(prefix + '_finish_reason') == 'length':
                parsed, valid, repaired = None, False, False
                error = 'generation reached its token/context limit before completing the answer'
            row[prefix] = parsed
            row[prefix + '_valid'] = valid
            row[prefix + '_parse_repaired'] = repaired
            row['error'] = f'{prefix}: {error}' if error else None
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
                        'objects': parsed.get('objects'), 'answer_options': parsed.get('answer_options'),
                        'question_flags': parsed.get('question_flags'),
                        'correct': None, 'notes': None})
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS, extrasaction='ignore')
    writer.writeheader()
    for row in reviews:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    atomic_text(directory / 'review.csv', stream.getvalue())
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
                if key not in {'questions', 'run_dir', 'resume', 'recover_failures', 'recover_from'}}
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
    recovery_source = getattr(args, 'recover_from', None)
    if recovery_source is not None:
        if args.resume or metadata_path.exists() or result_path.exists():
            raise ValueError('--recover-from requires a fresh run directory without --resume')
        recovered = import_recovery(recovery_source, questions, args.model)
    if metadata_path.exists():
        if not args.resume:
            raise ValueError('Run already exists; use --resume or a new --run-dir')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if any(metadata.get(k) != v for k, v in identity.items()):
            raise ValueError('Resume requires the same sample, model, prompts, and settings')
    else:
        if args.resume or result_path.exists():
            raise ValueError('Cannot resume without matching run_metadata.json; choose a fresh run directory')
        recovery = {'recovery_source': str(recovery_source.resolve())} if recovery_source is not None else {}
        save_json(metadata_path, {**identity, **recovery, 'questions_file': str(args.questions),
                                 'timestamp': datetime.now(timezone.utc).isoformat()})
    if recovery_source is not None:
        return recovered
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
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument('--recover-failures', action='store_true',
                          help='With --resume, reparse failed responses and retry unresolved calls once')
    recovery.add_argument('--recover-from', type=Path,
                          help='Import a previous run into a fresh directory, reusing matching prompts and reparsing failures')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', default='auto')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--call1-max-tokens', type=int, default=256)
    parser.add_argument('--call2-max-tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--prompt-mode', choices=['chat', 'plain'], default='chat')
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument('--disable-thinking', dest='disable_thinking', action='store_true', default=True,
                          help='Pass enable_thinking=False to tokenizer chat template (default)')
    thinking.add_argument('--enable-thinking', dest='disable_thinking', action='store_false',
                          help='Use the tokenizer chat template default thinking behavior')
    args = parser.parse_args()
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
                        output = llm.generate([prompt], sampling)[0].outputs[0]
                        token_ids = getattr(output, 'token_ids', None)
                        return {'text': output.text, 'finish_reason': getattr(output, 'finish_reason', None),
                                'generated_tokens': len(token_ids) if token_ids is not None else None}
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
