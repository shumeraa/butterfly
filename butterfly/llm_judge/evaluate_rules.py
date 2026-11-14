#!/usr/bin/env python3
"""Evaluate semantic transformation rules; write canonical and human-review Excel files."""
import argparse
import csv
import json
from pathlib import Path

from common import read_jsonl, validate_predictions
from excel_io import read_workbook, write_workbook

GLOBAL = ('hflip', 'vflip', 'rot180', 'grayscale', 'blur', 'noise')
LOCAL = ('segment_mr1_grayscale', 'segment_mr1_blur', 'segment_mr1_noise')
LABELS = {'I': 'invariant', 'C': 'equivariant', 'U': 'unsafe'}


def read_predictions(path, sheet=None):
    """Accept inference artifacts or corrected workbooks; retain every source field."""
    suffix = path.suffix.lower()
    if suffix == '.xlsx':
        return read_workbook(path, sheet)
    if sheet is not None:
        raise ValueError('--sheet applies only to .xlsx input')
    if suffix == '.jsonl':
        rows = read_jsonl(path)
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError('JSONL predictions must contain one object per row')
        # Add flat prediction fields without overwriting manual corrections or
        # dropping nested call2, full provider responses, and inference diagnostics.
        for row in rows:
            row.setdefault('answer_type', row.get('call1'))
            parsed = row.get('call2')
            for field in ('topic', 'objects', 'answer_options', 'question_flags'):
                row.setdefault(field, parsed.get(field) if isinstance(parsed, dict) else None)
        headers = list(dict.fromkeys(key for row in rows for key in row))
        return headers, rows
    if suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames
            if not headers or any(not key for key in headers) or len(set(headers)) != len(headers):
                raise ValueError('CSV headers must be nonempty and unique')
            rows = list(reader)
            if any(None in row for row in rows):
                raise ValueError('CSV row has more cells than its header')
        return headers, rows
    raise ValueError('Input must be a .jsonl, .csv, or .xlsx file')


def evaluate(answer_type, topic, objects, flags, answer_options=()):
    """Return ordered global and per-object (label, reason) results."""
    validate_predictions(answer_type, {'topic': topic, 'objects': objects,
                                       'answer_options': list(answer_options), 'question_flags': flags})
    b = answer_type == 'BINARY'
    f = set(flags)
    def has(*values):
        return bool(f.intersection(values))
    g = {}
    def spatial(name, blockers, identities, answers):
        if has(*blockers):
            g[name] = ('U', 'spatial_blocker:' + ','.join(sorted(f.intersection(blockers))))
        elif has(*identities):
            g[name] = ('U', 'spatial_identity_change:' + ','.join(sorted(f.intersection(identities))))
        elif has(*answers):
            g[name] = ('C' if b else 'U', 'binary_spatial_change' if b else 'open_spatial_unsafe')
        else:
            g[name] = ('I', 'no_spatial_dependency')
    spatial('hflip', ('TEXT', 'BODY'), ('LR_ID',), ('LR_ANSWER',))
    spatial('vflip', ('TEXT', 'SUPPORT'), ('UD_ID',), ('UD_ANSWER',))
    for name in ('hflip', 'vflip'):
        if has('MULTI_CONDITION') and g[name][0] == 'C':
            g[name] = ('U', 'multi_condition_equivariance_unsafe')
    flips = (g['hflip'][0], g['vflip'][0])
    if 'U' in flips:
        g['rot180'] = ('U', 'flip_component_unsafe')
    elif flips == ('C', 'C'):
        g['rot180'] = ('U', 'flip_equivariance_may_cancel')
    elif 'C' in flips:
        g['rot180'] = ('C', 'one_equivariant_flip')
    else:
        g['rot180'] = ('I', 'both_flips_invariant')
    gray = has('COLORW', 'COMPARE') or topic == 'COLOR'
    g['grayscale'] = ('U', 'color_or_comparison_dependency') if gray else ('I', 'no_color_dependency')
    blur = has('TEXT', 'MATW', 'ACTW', 'COMPARE') or topic in {'MATERIAL', 'ACTION'} or (not b and topic == 'OBJECT')
    g['blur'] = g['noise'] = ('U', 'detail_sensitive') if blur else ('I', 'no_detail_dependency')
    local = []
    for idx, obj in enumerate(objects):
        edits = {}
        if obj['role'] != 'REFERENT':
            local.append(edits)
            continue
        if b and idx > 0:
            edits = {name: ('U', 'binary_non_target') for name in LOCAL}
        elif has('EXISTW', 'TEXT', 'BODY', 'ACTW') or topic == 'ACTION':
            edits = {name: ('U', 'segment_sensitive') for name in LOCAL}
        else:
            for name in LOCAL:
                global_name = name.removeprefix('segment_mr1_')
                feature = 'COLOR' if global_name == 'grayscale' else 'MATERIAL'
                if feature in obj['id_features']:
                    edits[name] = ('U', 'local_' + feature.lower() + '_identity')
                else:
                    label, reason = g[global_name]
                    edits[name] = ('I' if label == 'I' else 'U', 'global_mr1:' + reason)
        local.append(edits)
    return g, local


def parse_cell(value, name):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError as exc:
            raise ValueError(f'{name}: invalid JSON: {exc}') from exc
    if value is None:
        raise ValueError(f'{name}: missing field')
    return value


def normalize(row):
    for key in ('id', 'detailed_type', 'question'):
        if row.get(key) is None or not str(row[key]).strip():
            raise ValueError(f'{key}: missing field')
    for key in ('call1_valid', 'call2_valid'):
        if row.get(key) in (False, 'false', 'False', 0):
            raise ValueError(f'{key} indicates invalid inference; correct predictions and validity together')
    answer = row.get('answer_type') if row.get('answer_type') is not None else row.get('call1')
    data = {'topic': row.get('topic'), 'objects': parse_cell(row.get('objects'), 'objects'),
            'answer_options': parse_cell(row.get('answer_options'), 'answer_options'),
            'question_flags': parse_cell(row.get('question_flags'), 'question_flags')}
    validate_predictions(answer, data)
    return answer, data


def build_outputs(headers, rows):
    questions, globals_, locals_, warnings, reviews = [], [], [], [], []
    for number, source in enumerate(rows, 2):
        provenance = {'source_row': number, 'id': source.get('id'),
                      'detailed_type': source.get('detailed_type'), 'question': source.get('question')}
        q = dict(source)
        q['source_row'] = number
        r = dict(provenance)
        notices = []
        try:
            answer, data = normalize(source)
            g, local = evaluate(answer, data['topic'], data['objects'], data['question_flags'], data['answer_options'])
            q.update({'normalized_answer_type': answer, 'normalized_topic': data['topic'],
                      'normalized_objects': data['objects'], 'normalized_question_flags': data['question_flags'],
                      'normalized_answer_options': data['answer_options'],
                      'evaluation_status': 'evaluated', 'evaluation_error': None})
            r.update({'answer_type': answer, 'topic': data['topic'],
                      'objects': '; '.join(f'{i}: {o["name"]} ({o["role"]}) [{", ".join(o["id_features"])}]' for i, o in enumerate(data['objects'])),
                      'answer_options': data['answer_options'],
                      'question_flags': ', '.join(data['question_flags']), 'evaluation_status': 'evaluated'})
            summaries = []
            for idx, edits in enumerate(local):
                obj = data['objects'][idx]
                if not edits:
                    continue
                summaries.append(f'{idx}: {obj["name"]}: ' + ', '.join(f'{k}={v[0]}' for k, v in edits.items()))
                for name in LOCAL:
                    label, reason = edits[name]
                    entry = {**provenance, 'object_index': idx, 'object_name': obj['name'],
                             'object_role': obj['role'],
                             'id_features': obj['id_features'], 'is_target': idx == 0,
                             'allowed_for_edit': not (answer == 'BINARY' and idx > 0),
                             'transformation': name, 'label': LABELS[label], 'reason_code': reason,
                             'evaluation_status': 'evaluated'}
                    locals_.append(entry)
            r['local_summary'] = '; '.join(summaries) or 'No REFERENT objects: local edits not applicable'
        except (ValueError, TypeError) as exc:
            g = {name: (None, 'invalid_input') for name in GLOBAL}
            q.update({'evaluation_status': 'evaluation_error', 'evaluation_error': str(exc)})
            r.update({'evaluation_status': 'evaluation_error', 'local_summary': '',
                      'objects': source.get('objects'), 'answer_options': source.get('answer_options'),
                      'question_flags': source.get('question_flags')})
            # Preserve identifiable object targets even when the prediction cannot be evaluated.
            try:
                error_objects = parse_cell(source.get('objects'), 'objects')
            except ValueError:
                error_objects = []
            if isinstance(error_objects, list):
                for idx, obj in enumerate(error_objects):
                    if isinstance(obj, dict) and obj.get('role') == 'ANSWER_OPTION':
                        continue
                    for name in LOCAL:
                        locals_.append({**provenance, 'object_index': idx,
                            'object_name': obj.get('name') if isinstance(obj, dict) else None,
                            'object_role': obj.get('role') if isinstance(obj, dict) else None,
                            'id_features': obj.get('id_features') if isinstance(obj, dict) else None,
                            'is_target': idx == 0, 'allowed_for_edit': None,
                            'transformation': name, 'label': None, 'reason_code': 'invalid_input',
                            'evaluation_status': 'evaluation_error'})
            notices.append(str(exc))
            warnings.append({**provenance, 'warning_code': 'invalid_input', 'message': str(exc), 'source_record': source})
        for name in GLOBAL:
            code, reason = g[name]
            status = 'evaluation_error' if q['evaluation_status'] == 'evaluation_error' else 'not_applicable' if code is None else 'evaluated'
            q[name] = r[name] = LABELS.get(code)
            globals_.append({**provenance, 'transformation': name, 'label': LABELS.get(code),
                             'reason_code': reason, 'evaluation_status': status})
        r.update({'warnings': '\n'.join(notices), 'correct?': None, 'review_notes': None})
        questions.append(q)
        reviews.append(r)
    # Source columns retain their original names even if they collide with computed fields.
    computed = ['source_row', 'normalized_answer_type', 'normalized_topic', 'normalized_objects',
                'normalized_question_flags', 'normalized_answer_options', *GLOBAL, 'evaluation_status', 'evaluation_error']
    collisions = set(headers).intersection(computed)
    for q, source in zip(questions, rows):
        for key in collisions:
            backup = 'source_' + key
            while backup in headers or backup in q:
                backup = 'source_' + backup
            q[backup] = source[key]
    question_headers = list(dict.fromkeys(headers + computed + [k for q in questions for k in q]))
    gh = ['source_row', 'id', 'detailed_type', 'question', 'transformation', 'label', 'reason_code', 'evaluation_status']
    lh = ['source_row', 'id', 'detailed_type', 'question', 'object_index', 'object_name', 'object_role', 'id_features', 'is_target',
          'allowed_for_edit', 'transformation', 'label', 'reason_code', 'evaluation_status']
    wh = ['source_row', 'id', 'detailed_type', 'question', 'object_index', 'transformation', 'warning_code', 'message', 'source_record']
    rh = ['source_row', 'id', 'detailed_type', 'question', 'answer_type', 'topic', 'objects', 'answer_options', 'question_flags', *GLOBAL,
          'local_summary', 'evaluation_status', 'warnings', 'correct?', 'review_notes']
    legend = [{'meaning': 'Labels are predicted by rules, not visually verified.'},
              {'meaning': 'I = invariant; C = equivariant (predicted answer change); U = unsafe.'},
              {'meaning': 'Blank labels: evaluation error; inspect status and Warnings.'},
              {'meaning': 'Local segment_mr1 grayscale/blur/noise apply only to REFERENT objects; C maps to unsafe.'}]
    raw = {'Questions': (question_headers, questions), 'GlobalLabels': (gh, globals_),
           'LocalEdits': (lh, locals_), 'Warnings': (wh, warnings)}
    review = {'Review': (rh, reviews), 'LocalEditReview': (lh + ['correct?', 'review_notes'], locals_),
              'Legend': (['meaning'], legend)}
    return raw, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='results.jsonl, review.csv, or corrected .xlsx predictions')
    parser.add_argument('--sheet', help='Excel input sheet name; default is the first sheet')
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--raw-output', type=Path)
    parser.add_argument('--review-output', type=Path)
    args = parser.parse_args()
    raw_path = args.raw_output or args.output_dir / 'transformation_labels_raw.xlsx'
    review_path = args.review_output or args.output_dir / 'transformation_labels_review.xlsx'
    try:
        paths = [args.input.resolve(), raw_path.resolve(), review_path.resolve()]
        if len(set(paths)) != 3:
            raise ValueError('Input, raw output, and review output must be distinct paths')
        headers, rows = read_predictions(args.input, args.sheet)
        raw, review = build_outputs(headers, rows)
        write_workbook(raw_path, raw)
        write_workbook(review_path, review, review=True)
        print(f'Saved {raw_path} and {review_path}; {len(raw["Warnings"][1])} warnings')
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
