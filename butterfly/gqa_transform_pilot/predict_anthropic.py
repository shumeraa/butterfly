#!/usr/bin/env python3
"""Run the GQA transformation pilot through Anthropic, without a local model or GPU."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT))
from butterfly.gqa_transform_pilot import gqa_transform_pilot as pilot

DEFAULT_MODEL = 'claude-haiku-5-5'


def predict(args):
    if not os.environ.get('ANTHROPIC_API_KEY', '').strip():
        raise ValueError('Set ANTHROPIC_API_KEY in your environment before running')
    if args.max_tokens < 1:
        raise ValueError('--max-tokens must be positive')
    records = pilot.prepare_records(args)
    # Check every selected image before making any paid requests.
    from PIL import Image
    for record in records:
        with Image.open(record['source_image']) as image:
            image.convert('RGB').load()
    try:
        import anthropic
    except ImportError:
        raise ValueError('Install the API dependencies: pip install -r '
                         'butterfly/gqa_transform_pilot/requirements-anthropic.txt') from None

    args.output.mkdir(parents=True)
    pilot.write_json(args.output / 'sampled_records.json', records)
    settings = dict(started_at=datetime.now(timezone.utc).isoformat(), provider='anthropic',
                    model=args.model, annotations=str(args.annotations.resolve()),
                    images=str(args.images.resolve()), sample_size=args.sample_size,
                    sampling_seed=args.seed, max_tokens=args.max_tokens,
                    thinking={'type': 'disabled'}, timeout_seconds=120, max_retries=2,
                    note='Seed controls dataset sampling and noise only; API output is not deterministic.',
                    blur_radius=2, noise_amount=.05, noise_stddev=12.75,
                    versions={p: importlib.metadata.version(p) for p in ('anthropic', 'Pillow', 'numpy')})
    pilot.write_json(args.output / 'settings.json', settings)
    prompts = [dict(image_id=r['image_id'], question_id=r['question_id'],
                    prompt=pilot.PROMPT.format(question=r['question'])) for r in records]
    pilot.write_json(args.output / 'prompts.json', prompts)
    responses = []
    print(f'Using {args.model}; requesting {len(records)} predictions (six relations each).', flush=True)
    with anthropic.Anthropic(timeout=120, max_retries=2) as client:
        for index, prompt in enumerate(prompts, 1):
            try:
                message = client.messages.create(
                    model=args.model, max_tokens=args.max_tokens, thinking={'type': 'disabled'},
                    messages=[{'role': 'user', 'content': prompt['prompt']}])
            except anthropic.APIError as exc:
                # Avoid printing request headers or credentials in exception diagnostics.
                status = getattr(exc, 'status_code', None)
                raise RuntimeError(f'Anthropic request {index} failed: {type(exc).__name__}'
                                   f' (HTTP {status if status is not None else "unavailable"}). '
                                   f'{len(responses)} completed responses are saved in {args.output}. '
                                   'Check your API key, model access, billing, and network.') from None
            raw = ''.join(block.text for block in message.content if block.type == 'text')
            parsed, error = pilot.parse_prediction(raw)
            responses.append(dict(**prompt, raw_response=raw, parsed=parsed, prediction_error=error,
                                  finish_reason=message.stop_reason, response_id=message.id,
                                  model=message.model, usage=message.usage.model_dump(mode='json'),
                                  api_response=message.model_dump(mode='json')))
            pilot.write_json(args.output / 'predictions.json', responses)
            print(f'Completed {index}/{len(records)}' + (' (malformed response)' if error else ''), flush=True)
    pilot.export_review(args.output, records, responses)
    print(f'Review {len(records) * 6} cases in {args.output / "review.html"}; edit review.csv', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--annotations', type=Path,
                        default=ROOT / 'my_datasets/annotation_files/GQA/balanced_val_annotation.json')
    parser.add_argument('--images', type=Path,
                        default=Path(os.environ['GQA_IMAGE_PATH']) if os.environ.get('GQA_IMAGE_PATH') else None)
    parser.add_argument('--output', type=Path, default=ROOT / 'plots/gqa_transform_pilot_anthropic')
    parser.add_argument('--sample-size', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-tokens', type=int, default=1024)
    args = parser.parse_args()
    try:
        predict(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
