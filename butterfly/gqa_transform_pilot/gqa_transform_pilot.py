#!/usr/bin/env python3
"""Text-only GQA transformation prediction and offline human review."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
TRANSFORMS = ('horizontal_flip', 'vertical_flip', 'rotate_180', 'grayscale', 'blur', 'noise')
RELATIONS = {'same', 'swap_left_right', 'swap_up_down', 'swap_both', 'flip_yes_no', 'unsafe'}
PROMPT = '''Given a question, predict how its CORRECT ANSWER should change under
each image transformation. The question remains exactly unchanged.

You receive only the question, not the image or original answer.
Choose a relation justified by the question and transformation without
assuming additional scene content.

Predict the correct answer's semantic change. Do not predict whether
a particular model will become confused or make recognition errors.

Transformations:
- horizontal_flip: mirrors the image left-to-right
- vertical_flip: mirrors the image top-to-bottom
- rotate_180: rotates the image by 180 degrees
- grayscale: converts colors to grayscale
- blur: Gaussian blur, radius 2 pixels
- noise: Gaussian noise, standard deviation 0.05 x 255

Choose exactly one relation per transformation:
- same: preserve the original answer text
- swap_left_right: replace left with right and right with left IN THE ANSWER
- swap_up_down: exchange up/down, above/below, top/bottom,
  and upper/lower IN THE ANSWER
- swap_both: apply both directional substitutions IN THE ANSWER
- flip_yes_no: negate the answer: yes becomes no, and no becomes yes
- unsafe: no reliable answer mapping can be inferred

Decision rules:

1. Identify what the question asks the answer to express:
   object identity, color, existence, direction, attribute, etc.

2. Distinguish a direction requested AS THE ANSWER from a direction
   used TO SELECT AN OBJECT.
   - "Which side is the dog on, left or right?" requests a direction.
   - "What color is the dog on the left?" uses direction to select a dog.
     A horizontal flip can select a different dog. Its color is unknown,
     so use unsafe, not swap_left_right.

3. Choose directional swaps based on the answer components that change,
   not merely the axes affected by the image transformation.
   - Under rotate_180, left/right answers use swap_left_right.
   - Under rotate_180, top/bottom answers use swap_up_down.
   - Use swap_both only when the answer requires both substitutions.

4. Use flip_yes_no only when the transformation logically negates
   the proposition in both directions. Recognition difficulty does
   not imply negation. A reversed spatial relation is not necessarily
   its logical complement: objects may be absent, aligned, or multiple.

5. Flips and rotations preserve object presence, identity, color,
   and intrinsic size unless the question's reference or requested
   property depends on image orientation.
   An unusual or upside-down appearance alone does not justify unsafe.

6. For grayscale, distinguish color-dependent questions or references
   from questions answerable using shape, presence, or geometry.
   Use unsafe when losing color prevents a reliable answer mapping.

7. For blur and noise, consider whether the requested information
   depends on fine details. Do not abstain solely because any
   degradation might hypothetically confuse a model.

8. Multiple conditions or an open-ended answer do not automatically
   require unsafe. Use unsafe when a relevant condition or referent
   changes in a way whose answer cannot be inferred.

Respond with a JSON object only, with one entry per transformation.
Do not include Markdown code fences, introductory text, or trailing commentary.
Each entry must contain "relation" and a short "reason".
Use exactly the six transformation keys listed above; each value must be an object
with exactly the fields "relation" (an allowed relation string) and "reason" (a string).

Question: {question}'''
FIELDS = ['image_id', 'question_id', 'type', 'group', 'question', 'answer',
          'transformation', 'relation', 'reason', 'expected_answer', 'prediction_error',
          'original_image', 'transformed_image', 'noise_seed', 'status', 'review_note']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sample_records(annotations, count=100, seed=42):
    if count < 1 or count > len(annotations):
        raise ValueError(f'sample size must be between 1 and {len(annotations)}')
    rng = random.Random(seed)
    records = []
    for image_id in rng.sample(sorted(annotations), count):
        questions = sorted(annotations[image_id], key=lambda q: str(q['questionId']))
        if not questions:
            raise ValueError(f'No questions for image {image_id}')
        q = rng.choice(questions)
        records.append(dict(image_id=image_id, question_id=str(q['questionId']),
                            question=q['question'], answer=q['answer'],
                            type=q['type'], group=q['group'],
                            noise_seed=int.from_bytes(hashlib.sha256(
                                f'{seed}:{image_id}'.encode()).digest()[:4], 'big')))
    return records


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def parse_prediction(raw):
    """Strictly validate the entire response; never extract or repair JSON."""
    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(value, dict) or set(value) != set(TRANSFORMS):
            raise ValueError('expected exactly the six transformation keys')
        for name, entry in value.items():
            if not isinstance(entry, dict) or set(entry) != {'relation', 'reason'}:
                raise ValueError(f'{name}: expected relation and reason fields')
            if not isinstance(entry['relation'], str) or entry['relation'] not in RELATIONS:
                raise ValueError(f'{name}: invalid relation')
            if not isinstance(entry['reason'], str):
                raise ValueError(f'{name}: reason must be a string')
        return value, ''
    except (ValueError, TypeError) as exc:
        return None, str(exc)


def parse_for_review(raw, allow_json_wrapper=False):
    """Optionally accept a single JSON object after prose or inside a code fence."""
    parsed, strict_error = parse_prediction(raw)
    if parsed is not None or not allow_json_wrapper or not isinstance(raw, str):
        return parsed, strict_error, ''
    start = raw.find('{')
    if start < 0:
        return None, strict_error, ''
    try:
        _, end = json.JSONDecoder(object_pairs_hook=unique_object).raw_decode(raw[start:])
    except ValueError:
        return None, strict_error, ''
    # Do not choose between multiple objects or discard trailing commentary.
    if raw[start + end:].strip() not in ('', '```'):
        return None, strict_error, ''
    parsed, error = parse_prediction(raw[start:start + end])
    if parsed is None:
        return None, error, ''
    return parsed, '', f'JSON wrapper accepted explicitly; strict parser error: {strict_error}'


def expected_answer(answer, relation):
    if relation == 'unsafe':
        return '', ''
    if relation == 'same':
        return answer, ''
    if relation == 'flip_yes_no':
        if answer.strip().lower() not in ('yes', 'no'):
            return '', 'flip_yes_no requires an original answer of yes or no'
        return ('no' if answer.strip().lower() == 'yes' else 'yes'), ''
    mapping = {}
    if relation in ('swap_left_right', 'swap_both'):
        mapping.update(left='right', right='left')
    if relation in ('swap_up_down', 'swap_both'):
        mapping.update(up='down', down='up', above='below', below='above',
                       top='bottom', bottom='top', upper='lower', lower='upper')
    if not mapping:
        return '', 'invalid relation'

    def substitute(match):
        word = match.group()
        replacement = mapping[word.lower()]
        return replacement.upper() if word.isupper() else (
            replacement.capitalize() if word.istitle() else replacement)
    return re.sub(r'\b(' + '|'.join(mapping) + r')\b', substitute, answer, flags=re.I), ''


def transform_images(image, seed):
    from butterfly.image_transforms import transforms as t
    original = image.convert('RGB')
    return dict(horizontal_flip=t.flip(original, 'left'), vertical_flip=t.flip(original, 'up'),
                rotate_180=t.rotate(original, 180), grayscale=t.color_filter(original, 'grayscale'),
                blur=t.blur(original, 2), noise=t.noise(original, .05, seed=seed))


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_review(output, records, responses, *, allow_json_wrapper=False, exported_images=None):
    from PIL import Image
    rows = []
    sections = []
    (output / 'images').mkdir()
    for index, (record, response) in enumerate(zip(records, responses)):
        parsed, error, format_warning = parse_for_review(response['raw_response'], allow_json_wrapper)
        response.update(parsed=parsed, prediction_error=error)
        if allow_json_wrapper:
            response.update(strict_prediction_error=parse_prediction(response['raw_response'])[1],
                            format_warning=format_warning)
        original_path = f'images/{index:03d}_original.png'
        if exported_images is not None:
            shutil.copyfile(exported_images / Path(original_path).name, output / original_path)
        else:
            with Image.open(record['source_image']) as source:
                original = source.convert('RGB')
            original.save(output / original_path)
            transformed = transform_images(original, record['noise_seed'])
        for name in TRANSFORMS:
            transformed_path = f'images/{index:03d}_{name}.png'
            if exported_images is not None:
                shutil.copyfile(exported_images / Path(transformed_path).name, output / transformed_path)
            else:
                transformed[name].save(output / transformed_path)
            entry = parsed[name] if parsed else {'relation': '', 'reason': ''}
            expected, mapping_error = expected_answer(record['answer'], entry['relation'])
            prediction_error = error or mapping_error
            abstain = entry['relation'] == 'unsafe'
            row = {k: record[k] for k in ('image_id', 'question_id', 'type', 'group',
                                         'question', 'answer', 'noise_seed')}
            row.update(transformation=name, **entry, expected_answer=expected,
                       prediction_error=prediction_error, original_image=original_path,
                       transformed_image=transformed_path,
                       status='incorrect' if prediction_error else ('uncertain' if abstain else ''),
                       review_note=('Abstention: model returned unsafe.' if abstain else
                                    ('Invalid prediction: ' + prediction_error if prediction_error else '')))
            if format_warning:
                row['review_note'] = (row['review_note'] + ' ' + format_warning).strip()
            rows.append(row)
            e = lambda v: html.escape(str(v), quote=True)
            sections.append(f'<section><h2>Row {len(rows)}: {e(record["image_id"])} / {e(name)}</h2>'
                            f'<p><b>Question:</b> {e(record["question"])}</p>'
                            f'<p>GQA answer: {e(record["answer"])} | Relation: {e(entry["relation"])}'
                            f' | Expected transformed answer: {e(expected)}</p>'
                            f'<p>Reason: {e(entry["reason"])} | Error: {e(prediction_error)}</p>'
                            f'<p>{e(format_warning)}</p>'
                            f'<figure><img loading="lazy" src="{original_path}"><figcaption>Original</figcaption></figure>'
                            f'<figure><img loading="lazy" src="{transformed_path}"><figcaption>{e(name)}</figcaption></figure></section>')
    write_csv(output / 'review.csv', rows, FIELDS)
    write_json(output / 'review_manifest.json', rows)
    (output / 'review.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>GQA transformation review</title>'
        '<style>body{font:16px sans-serif;margin:24px}section{border-top:1px solid #aaa;padding:16px 0}'
        'figure{display:inline-block;margin:8px;vertical-align:top}img{max-width:43vw;max-height:420px}</style>'
        '<h1>GQA transformation review</h1><p>Edit status and review_note in review.csv. '
        'correct: expected answer supported; incorrect: contradicted or prediction invalid; '
        'uncertain: insufficient visual evidence; invalid_source: source image/question/answer prevents evaluation '
        '(mark all six rows). unsafe is an abstention and stays uncertain unless the source is invalid. '
        'See the pilot instructions for reporting.</p>' + '\n'.join(sections), encoding='utf-8')
    write_json(output / 'predictions.json', responses)


def rebuild_review(args):
    """Re-export saved responses into a fresh folder without making model calls."""
    if args.output.exists():
        raise ValueError(f'Output already exists; choose a fresh directory: {args.output}')
    records = json.loads((args.input / 'sampled_records.json').read_text(encoding='utf-8'))
    responses = json.loads((args.input / 'predictions.json').read_text(encoding='utf-8'))
    if not records or len(records) != len(responses):
        raise ValueError('Saved records and responses must be nonempty and have matching lengths')
    exported_images = args.input / 'images'
    reuse_images = all((exported_images / f'{i:03d}_{name}.png').is_file()
                       for i in range(len(records)) for name in ('original', *TRANSFORMS))
    for record, response in zip(records, responses):
        if any(record[k] != response[k] for k in ('image_id', 'question_id')):
            raise ValueError('Saved response IDs do not match sampled records')
        if not reuse_images and not Path(record['source_image']).is_file():
            raise ValueError(f'Missing selected image: {record["source_image"]}; '
                             'provide the complete exported images folder or rebuild where the dataset is available')
    settings = json.loads((args.input / 'settings.json').read_text(encoding='utf-8'))
    settings['review_rebuild'] = dict(source=str(args.input.resolve()),
                                    rebuilt_at=datetime.now(timezone.utc).isoformat(),
                                    allow_json_wrapper=args.allow_json_wrapper,
                                    reused_exported_images=reuse_images,
                                    note='Fresh review labels; original run and raw responses preserved.')
    args.output.mkdir(parents=True)
    write_json(args.output / 'sampled_records.json', records)
    write_json(args.output / 'settings.json', settings)
    if (args.input / 'prompts.json').is_file():
        write_json(args.output / 'prompts.json', json.loads(
            (args.input / 'prompts.json').read_text(encoding='utf-8')))
    export_review(args.output, records, responses, allow_json_wrapper=args.allow_json_wrapper,
                  exported_images=exported_images if reuse_images else None)
    recovered = sum(bool(r.get('format_warning')) for r in responses)
    print(f'Rebuilt {len(records) * 6} review rows in {args.output}; '
          f'accepted wrappers in {recovered}/{len(responses)} responses. Review labels start fresh.')


def prepare_records(args):
    if args.images is None:
        raise ValueError('Set GQA_IMAGE_PATH (source butterfly/start_env.sh) or pass --images')
    annotations = json.loads(args.annotations.read_text(encoding='utf-8'))
    records = sample_records(annotations, args.sample_size, args.seed)
    for record in records:
        image_id = record['image_id']
        candidates = [args.images / f'{image_id}{ext}' for ext in ('.jpg', '.png', '.jpeg')]
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise ValueError(f'Missing selected image {image_id} in {args.images}')
        record['source_image'] = str(source.resolve())
    if args.output.exists():
        raise ValueError(f'Output already exists; choose a fresh directory: {args.output}')
    return records


def predict(args):
    records = prepare_records(args)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('predict requires a GPU compute node; do not run models on a laptop/login node')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import importlib.metadata
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [PROMPT.format(question=r['question']) for r in records]
    chats = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                                           add_generation_prompt=True) for p in prompts]
    for chat in chats:
        if len(tokenizer.encode(chat, add_special_tokens=False)) + args.max_tokens > 4096:
            raise ValueError('Prompt plus output budget exceeds 4096 tokens')
    args.output.mkdir(parents=True)
    write_json(args.output / 'sampled_records.json', records)
    settings = dict(started_at=datetime.now(timezone.utc).isoformat(),
                    model=args.model, annotations=str(args.annotations.resolve()),
                    images=str(args.images.resolve()), sample_size=args.sample_size, seed=args.seed,
                    temperature=0, dtype='float16', tensor_parallel_size=1,
                    gpu_memory_utilization=.80, max_model_len=4096, max_tokens=args.max_tokens,
                    blur_radius=2, noise_amount=.05, noise_stddev=12.75,
                    versions={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'vllm', 'Pillow', 'numpy')})
    write_json(args.output / 'settings.json', settings)
    responses = [dict(question_id=r['question_id'], image_id=r['image_id'],
                      prompt=p, chat_prompt=c) for r, p, c in zip(records, prompts, chats)]
    write_json(args.output / 'prompts.json', responses)
    llm = LLM(model=args.model, dtype='float16', tensor_parallel_size=1,
              gpu_memory_utilization=.80, max_model_len=4096, seed=args.seed, trust_remote_code=True)
    results = llm.generate(chats, SamplingParams(temperature=0, seed=args.seed,
                                               max_tokens=args.max_tokens, n=1))
    if len(results) != len(records):
        raise RuntimeError('Model returned an unexpected number of responses')
    for response, result in zip(responses, results):
        response.update(raw_response=result.outputs[0].text,
                        finish_reason=result.outputs[0].finish_reason)
    write_json(args.output / 'predictions.json', responses)
    export_review(args.output, records, responses)
    print(f'Review {len(records) * 6} cases in {args.output / "review.html"}; edit review.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('predict')
    p.add_argument('--model', default='meta-llama/Meta-Llama-3-8B-Instruct')
    p.add_argument('--annotations', type=Path, default=ROOT / 'my_datasets/annotation_files/GQA/balanced_val_annotation.json')
    p.add_argument('--images', type=Path, default=Path(os.environ['GQA_IMAGE_PATH']) if os.environ.get('GQA_IMAGE_PATH') else None)
    p.add_argument('--output', type=Path, default=ROOT / 'plots/gqa_transform_pilot')
    p.add_argument('--sample-size', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.set_defaults(func=predict)
    p = commands.add_parser('rebuild-review', help='Re-export saved predictions without running a model')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-json-wrapper', action='store_true',
                   help='Explicitly accept a single JSON object after introductory prose or in a code fence')
    p.set_defaults(func=rebuild_review)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    main()
