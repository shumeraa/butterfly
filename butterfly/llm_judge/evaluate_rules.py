#!/usr/bin/env python3
"""Deterministic verdicts and expected answers from the LLM's {target, flags} output.

Implements Parts 2-6 of SIMPLE_PROMPTS.md. Standard library only, so it also runs
inside the inference container. CLI: results.jsonl -> labels.json.
"""
import argparse
import json
from pathlib import Path
import re

from common import read_jsonl, save_json, validate_prediction

GLOBAL = ('hflip', 'vflip', 'grayscale')
LOCAL = ('fill_color', 'occlude_bbox')

# Lexicons are over-inclusive on purpose: a missed word under a negated gate creates
# a fake verdict, while an extra word only costs coverage.
NEGATION = {'not', 'no', 'never', 'none', 'neither', 'nor', 'without', 'nobody', 'nothing'}
LR_WORDS = {'left', 'right', 'leftmost', 'rightmost'}
UD_WORDS = {'top', 'bottom', 'above', 'below', 'upper', 'lower', 'over', 'under',
            'beneath', 'underneath', 'up', 'down'}
BODY_WORDS = {'hand', 'hands', 'arm', 'arms', 'leg', 'legs', 'foot', 'feet', 'eye', 'eyes',
              'ear', 'ears', 'shoulder', 'shoulders', 'knee', 'knees', 'elbow', 'elbows',
              'wrist', 'wrists', 'ankle', 'ankles', 'hip', 'hips', 'cheek', 'cheeks'}
RED_SET = {'red', 'crimson', 'scarlet'}
RED_NEAR = {'maroon', 'burgundy', 'pink', 'orange', 'magenta', 'rose'}
GRAY_SET = {'gray', 'grey', 'black', 'white', 'silver'}
GRAY_NEAR = {'beige', 'tan', 'cream', 'ivory', 'charcoal'}
COLOR_WORDS = (RED_SET | RED_NEAR | GRAY_SET | GRAY_NEAR |
               {'yellow', 'green', 'blue', 'purple', 'brown', 'gold', 'golden', 'teal', 'turquoise',
                'navy', 'violet', 'cyan', 'khaki', 'olive', 'lavender', 'aqua', 'copper', 'bronze',
                'lime', 'peach', 'salmon', 'indigo', 'tan', 'cyan', 'mauve', 'fuchsia', 'amber'})
BRIGHTNESS_WORDS = {'dark', 'bright', 'colorful', 'colourful', 'colored', 'coloured', 'multicolored',
                    'multicoloured', 'pale', 'light', 'color', 'colors', 'colour', 'colours'}
# "light" is an object, not a brightness word, after these ("traffic light").
LIGHT_OBJECT_PREFIXES = {'traffic', 'street', 'head', 'tail', 'stop', 'brake', 'flash', 'spot',
                         'sun', 'desk', 'signal'}
CONJ = {'and', 'both', 'either', 'neither'}
# Colors the fill_color edit is evaluated for: {color: (words meaning it, near-miss words)}.
# Only red is active; add an entry (e.g. 'blue') to evaluate another fill color.
FILL_COLORS = {'red': (RED_SET, RED_NEAR)}

SUPPORT_WORDS = {'on', 'under', 'beneath', 'underneath', 'inside', 'hanging', 'sitting', 'lying'}
MATW_WORDS = {'wooden', 'metal', 'plastic', 'glass', 'leather', 'striped', 'plaid', 'checkered',
              'shiny', 'brick', 'stone'}
TEXT_WORDS = {'say', 'says', 'written', 'read', 'letters', 'number', 'word', 'sign', 'logo',
              'text', 'label', 'brand', 'name'}
COMPARE_WORDS = {'same', 'different', 'than', 'similar', 'compared', 'bigger', 'larger', 'smaller',
                 'taller', 'shorter', 'older', 'younger'}
ON_SIDE = re.compile(r'\bon\s+(?:the\s+)?(?:left|right|top|bottom|side)\b')


def tokenize(question):
    """Lowercase words; hyphens split, contractions stay whole so "isn't" is detectable."""
    tokens = []
    for token in re.findall(r"[a-z0-9']+", question.lower()):
        token = token.strip("'")
        if token.endswith("'s"):
            token = token[:-2]
        if token:
            tokens.append(token)
    return tokens


def ish_base(token):
    """'reddish' -> 'red' (also handles doubled consonants); None if not a color -ish form."""
    if len(token) > 4 and token.endswith('ish'):
        base = token[:-3]
        for candidate in (base, base[:-1]):
            if candidate in COLOR_WORDS:
                return candidate
    return None


def derive_flags(question, prediction):
    """Merge the LLM flags with code-side flags (Part 2). Returns (flags set, derived dict)."""
    llm = set(prediction['flags'])
    tokens = tokenize(question)
    words = set(tokens)
    support_tokens = set(tokenize(ON_SIDE.sub(' ', question.lower())))

    color_tokens = [t for t in tokens if t in COLOR_WORDS or ish_base(t)]
    ish = [t for t in tokens if ish_base(t)]
    plain_colors = {t for t in tokens if t in COLOR_WORDS}
    brightness = False
    for i, token in enumerate(tokens):
        if token == 'light' and i and tokens[i - 1] in LIGHT_OBJECT_PREFIXES:
            continue
        if token in BRIGHTNESS_WORDS:
            brightness = True

    flags = set(llm)
    if support_tokens & SUPPORT_WORDS:
        flags.add('SUPPORT')
    for flag, lexicon in (('MATW', MATW_WORDS), ('TEXT', TEXT_WORDS), ('COMPARE', COMPARE_WORDS)):
        if words & lexicon:
            flags.add(flag)
    if words & CONJ:
        flags.add('MULTI_CONDITION')

    lr_word = bool(words & LR_WORDS)
    ud_word = bool(words & UD_WORDS)
    code = {
        'COLORW': bool(color_tokens) or brightness,
        'NEGATION': bool(words & NEGATION) or any(t.endswith("n't") for t in tokens),
        'LR_WORD': lr_word,
        'UD_WORD': ud_word,
        'BODY': lr_word and bool(words & BODY_WORDS),
    }
    flags.update(name for name, on in code.items() if on)

    # Direction veto, then subtype consistency.
    for prefix, present in (('LR', lr_word), ('UD', ud_word)):
        if not present:
            flags -= {f'{prefix}_{s}' for s in ('ID', 'ANSWER', 'SIDE', 'YN')}
        if f'{prefix}_SIDE' in flags:
            flags.discard(f'{prefix}_YN')
            flags.add(f'{prefix}_ANSWER')
        if f'{prefix}_YN' in flags:
            flags.add(f'{prefix}_ANSWER')

    # Color form must agree with the number of color words in the question.
    forms = {'COLOR_QUERY': len(color_tokens) == 0, 'COLOR_VERIFY': len(color_tokens) == 1,
             'COLOR_CHOOSE': len(color_tokens) >= 2}
    present = [f for f in forms if f in flags]
    for form in forms:
        flags.discard(form)
    if len(present) == 1 and forms[present[0]]:
        flags.add(present[0])

    derived = {
        'ASKS': {color: (bool(plain_colors & words), bool(plain_colors & near) or bool(ish))
                 for color, (words, near) in FILL_COLORS.items()},
        'ASKS_ACHROMATIC': bool(plain_colors & GRAY_SET),
        'ASKS_GRAY_NEAR': bool(plain_colors & GRAY_NEAR) or bool(ish),
        'EXIST': (bool(re.match(r'\s*(?:is|are)\s+there\b|\s*do\s+you\s+see\b', question.lower()))
                  and not words & (CONJ | {'or'}) and not code['NEGATION']),
    }
    return flags, derived


UNSAFE = 'unsafe'
INVARIANT = 'invariant'


def equivariant(mapping):
    return {'equivariant': mapping}


def judge(question, prediction):
    """Return the Part 6 record body for one question."""
    f, d = derive_flags(question, prediction)
    target = prediction['target']
    has = lambda name: name in f
    eq_gate = not has('MULTI_CONDITION') and not has('COMPARE') and target is not None

    def spatial(prefix, blockers, invariant, swap):
        _id, side, yn = has(prefix + '_ID'), has(prefix + '_SIDE'), has(prefix + '_YN')
        clear = not _id and not has('MULTI_CONDITION') and not any(has(b) for b in blockers)
        if side and clear:
            return equivariant(swap)
        if yn and clear and not has('NEGATION'):
            return equivariant({'yes': 'no', 'no': 'unchecked'})
        return INVARIANT if invariant else UNSAFE

    # The direction veto already cleared every LR_*/UD_* flag when no direction word exists.
    hflip = spatial('LR', ('BODY', 'TEXT'),
                    not has('LR_WORD') and not has('BODY') and not has('TEXT'),
                    {'left': 'right', 'right': 'left'})
    vflip = spatial('UD', ('SUPPORT', 'TEXT'),
                    not has('UD_WORD') and not any(has(x) for x in ('SUPPORT', 'GRAVITY', 'REF_SURFACE', 'TEXT')),
                    {'top': 'bottom', 'bottom': 'top', 'above': 'below', 'below': 'above'})

    plain = not has('MULTI_CONDITION') and not has('COMPARE')
    if has('COLOR_QUERY') and plain:
        grayscale = equivariant({'any': ['gray', 'black', 'white']})
    elif (has('COLOR_VERIFY') and not d['ASKS_ACHROMATIC'] and not d['ASKS_GRAY_NEAR']
          and not has('NEGATION') and plain):
        grayscale = equivariant({'yes': 'no', 'no': 'no'})
    elif not has('COLORW'):
        grayscale = INVARIANT
    else:
        grayscale = UNSAFE

    record = {'hflip': hflip, 'vflip': vflip, 'grayscale': grayscale, 'mr2': None}
    if target is None:
        return record

    neg = has('NEGATION')

    def fill(color):
        asks, near = d['ASKS'][color]
        if eq_gate and has('COLOR_QUERY'):
            return equivariant({'any': color})
        if eq_gate and has('COLOR_VERIFY') and asks and not neg:
            return equivariant({'yes': 'yes', 'no': 'yes'})
        if eq_gate and has('COLOR_VERIFY') and not asks and not near and not neg:
            return equivariant({'yes': 'no', 'no': 'no'})
        if eq_gate and has('COLOR_CHOOSE') and asks:
            return equivariant({'any': color})
        if not has('COLORW') and not has('MATW') and not has('TEXT'):
            return INVARIANT
        return UNSAFE

    occlude = equivariant({'any': 'no'}) if eq_gate and d['EXIST'] else UNSAFE
    record['mr2'] = {target: {'fill_color': {c: fill(c) for c in FILL_COLORS}, 'occlude_bbox': occlude}}
    return record


def build_labels(rows):
    """[[question_id, record], ...] for every row with a valid prediction, in input order."""
    labels = []
    for row in rows:
        prediction = row.get('prediction')
        if not row.get('valid') or prediction is None:
            continue
        labels.append([row['id'], judge(row['question'], validate_prediction(prediction))])
    return labels


def verdict_name(value):
    return value if isinstance(value, str) else 'equivariant'


def count_verdicts(labels):
    """{transformation: {verdict: count}} over global edits and MR2 edits."""
    counts = {name: {} for name in (*GLOBAL, *(f'fill_color:{c}' for c in FILL_COLORS), 'occlude_bbox')}
    for _, record in labels:
        for name in GLOBAL:
            verdict = verdict_name(record[name])
            counts[name][verdict] = counts[name].get(verdict, 0) + 1
        for edits in (record['mr2'] or {}).values():
            verdicts = {f'fill_color:{c}': v for c, v in edits['fill_color'].items()}
            verdicts['occlude_bbox'] = edits['occlude_bbox']
            for name, value in verdicts.items():
                verdict = verdict_name(value)
                counts.setdefault(name, {})
                counts[name][verdict] = counts[name].get(verdict, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='results.jsonl from run_inference.py')
    parser.add_argument('--output', type=Path, help='default: labels.json next to the input')
    args = parser.parse_args()
    try:
        rows = read_jsonl(args.input)
        labels = build_labels(rows)
        output = args.output or args.input.with_name('labels.json')
        save_json(output, labels)
        print(f'Saved {len(labels)} records to {output} ({len(rows) - len(labels)} questions skipped: no valid prediction)')
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
