"""CPU-only schemas, parsing, and durable artifact helpers."""
import json
import os
from pathlib import Path
import re
import tempfile

TOPICS = {
    'BINARY': {'POSITION', 'COLOR', 'MATERIAL', 'EXISTS', 'IDENTITY', 'ACTION', 'SCENE', 'OTHER'},
    'OPEN': {'COLOR', 'MATERIAL', 'OBJECT', 'ACTION', 'POSITION', 'SCENE', 'OTHER'},
}
FLAGS = {'LR', 'UD', 'SUPPORT', 'TEXT', 'BODY', 'COLORW', 'MATW', 'EXISTW',
         'CATW', 'COMPARE', 'ACTW', 'AND'}


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_jsonl(path, rows):
    atomic_text(path, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def validate_predictions(answer_type, value):
    if answer_type not in TOPICS:
        raise ValueError('answer_type must be BINARY or OPEN')
    if not isinstance(value, dict):
        raise ValueError('Call 2 must be a JSON object')
    if not isinstance(value.get('topic'), str) or value['topic'] not in TOPICS[answer_type]:
        raise ValueError('topic is missing or invalid for answer_type')
    objects = value.get('objects')
    if not isinstance(objects, list):
        raise ValueError('objects must be a list')
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get('name'), str) or not obj['name'].strip():
            raise ValueError('each object needs a nonempty string name')
        features = obj.get('id_features')
        if not isinstance(features, list) or any(not isinstance(f, str) or f not in {'COLOR', 'MATERIAL'} for f in features):
            raise ValueError('id_features must be a list containing only COLOR/MATERIAL')
    flags = value.get('question_flags')
    if not isinstance(flags, list) or any(not isinstance(f, str) or f not in FLAGS for f in flags):
        raise ValueError('question_flags must be a list of declared flags')
    return value


def clean_response(raw):
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
    text = re.sub(r'^```(?:json|text)?\s*\n?', '', text, flags=re.I)
    text = re.sub(r'\n?```$', '', text).strip()
    return text


def parse_response(raw, branch=None):
    """Return (parsed, valid, repaired, error), retaining no invented fields."""
    if not isinstance(raw, str):
        return None, False, False, 'response is not text'
    cleaned = clean_response(raw)
    if branch is None:
        if cleaned in TOPICS:
            return cleaned, True, cleaned != raw.strip(), None
        return None, False, False, 'Call 1 must be exactly BINARY or OPEN'
    try:
        try:
            value = json.loads(raw)
            repaired = False
        except ValueError:
            try:
                value = json.loads(cleaned)
            except ValueError:
                decoder = json.JSONDecoder()
                candidates = []
                for match in re.finditer(r'\{', cleaned):
                    try:
                        candidate, _ = decoder.raw_decode(cleaned[match.start():])
                        validate_predictions(branch, candidate)
                        candidates.append(candidate)
                    except (ValueError, TypeError):
                        pass
                if len(candidates) != 1:
                    raise ValueError('expected one unambiguous valid JSON prediction object')
                value = candidates[0]
            repaired = True
        validate_predictions(branch, value)
        return value, True, repaired, None
    except (ValueError, TypeError) as exc:
        return None, False, False, str(exc)
