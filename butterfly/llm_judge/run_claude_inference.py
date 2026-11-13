#!/usr/bin/env python3
"""Run the question-only pilot with Claude Haiku 5.5 and CLAUDE_API_KEY."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

from common import parse_response, save_jsonl
from prompts import TEMPLATES, render
# This module imports vLLM only inside its own CLI, so sharing these helpers is CPU-only.
from run_inference import export_artifacts, new_result, load_sample, prepare_run

DEFAULT_MODEL = 'claude-haiku-5-5'


class APIRequestFailed(RuntimeError):
    """An unanswered API request was checkpointed and can be resumed."""


def generate_claude(client, args, prompt, call):
    # Haiku 5.5 rejects temperature/top_p/top_k; do not inherit vLLM settings.
    return client.messages.create(
        model=args.model,
        max_tokens=args.call1_max_tokens if call == 1 else args.call2_max_tokens,
        thinking={'type': args.thinking},
        output_config={'effort': args.effort},
        messages=[{'role': 'user', 'content': prompt}],
    )


def run_claude_pipeline(questions, results, generate, checkpoint):
    """Retain full responses; retry unanswered requests only when explicitly resumed."""
    for question in questions:
        row = results.setdefault(question['id'], new_result(question))
        if row['complete']:
            continue
        for call in (1, 2):
            prefix = f'call{call}'
            # A received response, including an invalid one, is never requested again.
            if row[prefix + '_raw'] is not None:
                continue
            if call == 2 and not row['call1_valid']:
                break
            template = TEMPLATES['call1' if call == 1 else row['call1']]
            print(f'{question["id"]}: {prefix}', flush=True)
            try:
                response = generate(render(template, question['question']), call)
            except Exception as exc:
                # Keep credentials and potentially sensitive exception bodies out of artifacts.
                status = getattr(exc, 'status_code', None)
                message = f'{prefix} API request failed: {type(exc).__name__}'
                if isinstance(status, int):
                    message += f' (HTTP {status})'
                row[prefix + '_attempted'] = True
                row[prefix + '_valid'] = False
                row[prefix + '_parse_repaired'] = None
                row['error'] = message
                row.setdefault('api_errors', []).append({
                    'call': call, 'error': message,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                })
                checkpoint()
                raise APIRequestFailed(message + '; resume this run to retry the unanswered call') from None
            # Store provider envelope, content blocks, stop reason, usage, and message/model IDs.
            row[prefix + '_api_response'] = response.model_dump(mode='json')
            raw = ''.join(block.text for block in response.content if block.type == 'text')
            row[prefix + '_raw'] = raw
            row[prefix + '_attempted'] = True
            parsed, valid, repaired, error = parse_response(raw, None if call == 1 else row['call1'])
            if response.stop_reason != 'end_turn':
                parsed, valid = None, False
                error = f'API stop_reason={response.stop_reason}; response is not a completed answer'
            row[prefix] = parsed
            row[prefix + '_valid'] = valid
            row[prefix + '_parse_repaired'] = repaired
            row['error'] = f'{prefix}: {error}' if error else None
            row['complete'] = call == 2 or not valid
            checkpoint()
            if not valid:
                print(f'{question["id"]}: {row["error"]}', file=sys.stderr)
                break


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(provider='anthropic')
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--call1-max-tokens', type=positive_int, default=256)
    parser.add_argument('--call2-max-tokens', type=positive_int, default=1024)
    parser.add_argument('--thinking', choices=['disabled', 'adaptive'], default='disabled')
    parser.add_argument('--effort', choices=['low', 'medium', 'high'], default='low')
    parser.add_argument('--timeout', type=float, default=120.0, help='Timeout per API attempt in seconds')
    parser.add_argument('--max-retries', type=int, default=2, help='SDK retries for transient API errors')
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_retries < 0:
        parser.error('--timeout must be positive and --max-retries must be nonnegative')
    try:
        import openpyxl  # noqa: F401
        import fcntl
        questions = load_sample(args.questions)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with (args.run_dir / '.run.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Another inference process is using this run directory') from exc
            # The key is read only here, never placed in args, metadata, or prompts.
            # Completed runs can export artifacts without credentials or the SDK.
            client = None
            try:
                results = prepare_run(args, questions)
                def checkpoint():
                    save_jsonl(args.run_dir / 'results.jsonl', [results[q['id']] for q in questions])
                checkpoint()
                try:
                    if any(not row['complete'] for row in results.values()):
                        api_key = os.environ.get('CLAUDE_API_KEY')
                        if not api_key:
                            raise ValueError('CLAUDE_API_KEY is not set')
                        from anthropic import Anthropic
                        client = Anthropic(api_key=api_key, timeout=args.timeout, max_retries=args.max_retries)
                        run_claude_pipeline(questions, results,
                                            lambda prompt, call: generate_claude(client, args, prompt, call),
                                            checkpoint)
                finally:
                    checkpoint()
                    export_artifacts(args.run_dir, [results[q['id']] for q in questions])
            finally:
                if client is not None:
                    client.close()
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; add --resume to continue from saved responses.\n')
    except (APIRequestFailed, ValueError, OSError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
