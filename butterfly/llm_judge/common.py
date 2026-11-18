"""CPU-only schemas, parsing, and durable artifact helpers."""
import json
import os
from pathlib import Path
import re
import tempfile

FLAGS = {
    'COLOR_QUERY', 'COLOR_VERIFY', 'COLOR_CHOOSE',
    'LR_ID', 'LR_ANSWER', 'LR_SIDE', 'LR_YN',
    'UD_ID', 'UD_ANSWER', 'UD_SIDE', 'UD_YN',
    'SUPPORT', 'GRAVITY', 'REF_SURFACE',
    'TEXT', 'MATW', 'COMPARE', 'MULTI_CONDITION',
}


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


def clean_response(raw):
    text = re.sub(r'<think>.*?</think>', '', raw, flags=re.S)
    # Some chat templates supply the opening tag in the prompt, so only
    # the closing tag appears in the generated completion.
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[1]
    # Never extract a label or a JSON example from unfinished thinking.
    if '<think>' in text:
        return ''
    text = text.strip()
    text = re.sub(r'^```(?:json|text)?\s*\n?', '', text, flags=re.I)
    text = re.sub(r'\n?```$', '', text).strip()
    return text


def validate_prediction(value):
    """Validate {"target": str|null, "flags": [declared flags]} and return a normalized copy."""
    if not isinstance(value, dict) or 'target' not in value or 'flags' not in value:
        raise ValueError('response must be a JSON object with keys "target" and "flags"')
    target = value['target']
    if target is not None:
        if not isinstance(target, str):
            raise ValueError('target must be a string or null')
        target = target.strip()
        if not target or target.lower() in {'null', 'none'}:
            target = None
    flags = value['flags']
    if not isinstance(flags, list) or any(not isinstance(f, str) or f not in FLAGS for f in flags):
        raise ValueError('flags must be a list of declared flags')
    return {'target': target, 'flags': list(dict.fromkeys(flags))}


def parse_response(raw):
    """Return (prediction, valid, repaired, error), retaining no invented fields."""
    if not isinstance(raw, str):
        return None, False, False, 'response is not text'
    cleaned = clean_response(raw)
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
                        candidates.append(validate_prediction(candidate))
                    except (ValueError, TypeError):
                        pass
                if len(candidates) != 1:
                    raise ValueError('expected one unambiguous valid JSON prediction object')
                value = candidates[0]
            repaired = True
        return validate_prediction(value), True, repaired, None
    except (ValueError, TypeError) as exc:
        return None, False, False, str(exc)
