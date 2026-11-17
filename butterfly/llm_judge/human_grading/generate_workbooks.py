#!/usr/bin/env python3
"""Split question-only human transformation grading into disjoint Excel workbooks."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from answer_rule_prompts import PROMPT_VERSION, TRANSFORMATIONS
from evaluate_rules import GLOBAL, LOCAL
from excel_io import write_workbook

DEFINITIONS = {
    **TRANSFORMATIONS,
    'grayscale': 'Convert the entire image to grayscale, preserving geometry.',
    'blur': 'Blur the entire image, reducing visible fine detail.',
    'noise': 'Add image noise throughout the image.',
    'fill_red': 'Segment one object referred to by the question and fill its mask uniformly red, preserving shape and location.',
    'fill_blue': 'Segment one object referred to by the question and fill its mask uniformly blue, preserving shape and location.',
    'occlude_bbox': 'Cover the selected object’s whole bounding box with an opaque black rectangle, including background inside the box.',
    'asymmetric_scale': 'Segment one object referred to by the question and scale its width and height by unequal positive factors around its center.',

}
DISPLAY_NAMES = {
    'hflip': 'Horizontal flip', 'vflip': 'Vertical flip',
    'grayscale': 'Grayscale', 'blur': 'Blur', 'noise': 'Noise',
    'fill_red': 'Fill Red', 'fill_blue': 'Fill Blue', 'occlude_bbox': 'Occlude BBox', 'asymmetric_scale': 'Asymmetric scale',
}
ALLOWED_LABELS = 'equivariant,invariant,unsafe'

# Plain-language summaries of the deterministic tables in ../PROMPTS.md.
# Human reviewers may choose any label, including equivariant, for every edit.
LOCAL_GUARDS = (
    'Edit only an object the question refers to, never an object offered solely as an answer choice. '
    'For a question with only two possible answers, editing any object other than the first object '
    'mentioned is unsafe. Every local edit is unsafe if the question asks whether something exists, '
    'involves readable text or left/right body parts, or concerns actions or interactions. '
)
FILL_RULE = (
    'After those checks, filling is unsafe if color, material, texture, pattern, or surface appearance '
    'identifies the selected object; if the question asks about material, object type, or identity; '
    'or if it mentions material or surface properties, compares properties of objects, refers to a '
    'category or collection, or combines multiple conditions with and/or. '
    'For a color question about the first object mentioned, the result is equivariant with a constant '
    '{color} answer only when a question without a two-answer limit has no explicit choices or includes '
    '{color}, or a two-answer question explicitly offers two distinct choices including {color}. '
    'Match color choices without regard to capitalization. Other color-question cases are unsafe. '
    'For other topics, any mention of color makes the edit unsafe; otherwise it is invariant.'
)
BLUR_RULE = (
    'Unsafe if the question involves readable text, material, texture, pattern, surface properties, '
    'actions or interactions, or comparisons of object properties, or asks for an object name '
    'without limiting the answer to two possibilities. Otherwise invariant. '
    'The automatic rules do not assign equivariant for this transformation.'
)
SUMMARIZED_RULES = {
    'hflip': (
        'Unsafe if the question involves readable text, left/right body parts, or a left/right '
        'relation used to identify an object. Otherwise, if a left/right relation is asked about '
        'or offered as an answer, it is equivariant when there are only two possible answers and '
        'unsafe for other answer formats. An otherwise equivariant result becomes unsafe when '
        'the question combines multiple conditions with and/or; a simple pair of answer choices '
        'does not count as multiple conditions. Otherwise invariant. Merely mentioning a horizontal '
        'direction does not by itself make the result equivariant.'
    ),
    'vflip': (
        'Unsafe if the question involves readable text, support, containment, physical orientation '
        '(such as on, under, inside, sitting, hanging, or lying), or an above/below relation used '
        'to identify an object. Otherwise, if an up/down relation is asked about or offered as an '
        'answer, it is equivariant when there are only two possible answers and unsafe for other '
        'answer formats. An otherwise equivariant result becomes unsafe when the question combines '
        'multiple conditions with and/or; a simple pair of answer choices does not count as multiple '
        'conditions. Otherwise invariant. Merely mentioning a vertical direction does not by itself '
        'make the result equivariant.'
    ),
    'grayscale': (
        'Unsafe if the question mentions color, asks about color, or compares properties of objects. '
        'Otherwise invariant. The automatic rules do not assign equivariant for this transformation.'
    ),
    'blur': BLUR_RULE,
    'noise': BLUR_RULE,
    'fill_red': LOCAL_GUARDS + FILL_RULE.format(color='red'),
    'fill_blue': LOCAL_GUARDS + FILL_RULE.format(color='blue'),
    'occlude_bbox': LOCAL_GUARDS + (
        'The automatic rules always label bounding-box occlusion unsafe because it removes the '
        'object and surrounding context. A reviewer may choose invariant or equivariant when '
        'a reliable answer relationship can be established.'
    ),
    'asymmetric_scale': LOCAL_GUARDS + (
        'After those checks, unsafe if size, proportions, or shape identifies the selected object; '
        'if the question asks about position, object type, or identity; or if it mentions size, '
        'proportions, shape, comparisons of object properties, categories or collections, left/right '
        'or up/down relations, support, containment, or physical orientation. Otherwise invariant. '
        'The automatic rules do not assign equivariant for this transformation.'
    ),
}


INSTRUCTIONS = [
    ('1. Grade each question', 'Open Grades and choose a label for each transformation. Keep the question unchanged; use only its wording, without guessing an image or original answer.'),
    ('Labels', 'invariant = answer stays the same. equivariant = answers follow a useful mapping. unsafe = no reliable relationship or the question may become unanswerable. Every transformation allows all three labels and has a mapping column.'),
    ('Rules and examples', 'Transformations summarizes the current automatic rules from PROMPTS.md in plain language. These are reference rules; human grading allows all three labels for every transformation. See the Examples sheet for worked grades and mappings.'),
    ('MR2 local edits', 'On Grades, specify one target object for local edits and unequal scale factors for asymmetric scale in Notes. The target must be an object referred to by the question, not an object offered solely as an answer choice. See Transformations for the current automatic rules.'),
    ('Target selection rule', 'Apply in order, and stop at the first match.\n\n1. Asked-about object. If the question asks the color of X, or whether X exists, X is the target. (“What color is the apple?”, “Is the shirt red?”, “Is there a cat?”)\n\n2. Grammatical subject. For other questions, take the head noun of the main clause: the thing the question is predicated on. (“Is the ball above the box?”, “What is the girl holding?”, “Is the man wearing a hat?”)\n\n3. Only named object. If the question names just one object, it is the target. (“What is on the table?”)\n\n4. Tie. If two objects both qualify, take the first in textual order.'),
    ('Never a target', 'Objects used only to identify another object (relative clauses, “to the left of the Z”).\n\nAnswer options (“an apple or a banana”).\n\nObjects whose existence the question doesn’t presuppose, such as the cup in “Is the man holding a cup?” The man is presupposed, so the man is the target.'),
    ('Blur / noise', 'Choose once in Blur / noise. The same label and mapping apply to both transformations.'),
    ('2. Map only if equivariant', 'Write original -> transformed, one line per original answer. Separate multiple possible outputs with commas. Cover every original answer. Only the mapping is required; leave it blank for invariant or unsafe.'),
    ('Tips', 'Use Alt+Enter (Option+Return on Mac) for new lines in a cell. Do not assume not-left means right. An empty equivariant mapping is highlighted red. Notes are optional except for local target and scale parameters.'),
]

# Complete worked rows in the same column order as the grading sheet.
EXAMPLES = [
    ('How many balls are there?',
     ('invariant', 'invariant', 'invariant', 'invariant', 'invariant', 'invariant', 'unsafe', 'invariant'),
     {},
     'Target: balls. Scale width by 1.5 and height by 0.75. Count is preserved by the other edits; occlusion hides objects.'),
    ('Is the car red or blue?',
     ('invariant', 'invariant', 'unsafe', 'invariant', 'equivariant', 'equivariant', 'unsafe', 'invariant'),
     {'fill_red': 'red -> red\nblue -> red', 'fill_blue': 'red -> blue\nblue -> blue'},
     'Target: car. Scale width by 1.5 and height by 0.75. Fills force their color; grayscale and occlusion lose the color answer.'),
    ('Is the ball above the box?',
     ('invariant', 'equivariant', 'invariant', 'invariant', 'invariant', 'invariant', 'unsafe', 'unsafe'),
     {'vflip': 'yes -> no\nno -> yes, no'},
     'Target: ball (subject), not box. Scale width by 1.5 and height by 0.75. A ball that was not above may be below or vertically aligned.'),
]


def example_rows():
    rows = []
    for index, (question, labels, mappings, notes) in enumerate(EXAMPLES, 1):
        row = {'Question ID': f'worked_{index:02d}', 'Question': question, 'Notes': notes}
        for name, label in zip(grading_names(), labels):
            row[label_header(name)] = label
            row[mapping_header(name)] = mappings.get(name, '')
        rows.append(row)
    return rows


def transformation_names():
    return [name for name in GLOBAL if name != 'rot180'] + [
        name for name in LOCAL if not name.startswith('segment_mr1_')]


def grading_names():
    return [name for name in transformation_names() if name != 'noise']


def label_header(name):
    return 'Blur / noise' if name == 'blur' else DISPLAY_NAMES[name]


def mapping_header(name):
    return label_header(name) + ' — mapping if equivariant'


def load_questions(path):
    """Accept the inference JSONL schema or the same records as a JSON array."""
    content = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.jsonl':
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    else:
        rows = json.loads(content)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Input must be a nonempty JSON array or JSONL of question records')
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each question must be an object')
        for field in ('id', 'question'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Each question needs a nonempty string {field}')
        if row['id'] in seen:
            raise ValueError(f'Duplicate question ID: {row["id"]}')
        seen.add(row['id'])
        if 'detailed_type' in row and not isinstance(row['detailed_type'], str):
            raise ValueError('detailed_type must be a string when supplied')
    return rows


def grading_rows(questions):
    return [{'Question ID': q['id'], 'Question': q['question']} for q in questions]


def create_workbook(path, questions):
    from openpyxl import load_workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    if len(questions) > 1_048_575:
        raise ValueError('Assignment exceeds the Excel row limit')
    sheets = {}
    groups = [('Grades', grading_names()), ('Examples', grading_names())]
    for title, names in groups:
        headers = ['Question ID', 'Question']
        for name in names:
            headers.append(label_header(name))
            headers.append(mapping_header(name))
        headers.append('Notes')
        sheets[title] = (headers, example_rows() if title == 'Examples' else grading_rows(questions))
    sheets['Quick guide'] = (['Step', 'What to do'], [{'Step': k, 'What to do': v} for k, v in INSTRUCTIONS])
    definitions = []
    for name in transformation_names():
        definition = DEFINITIONS[name]
        if name in {'blur', 'noise'}:
            definition += ' Graded together in the Blur / noise column; noise always uses the blur label.'
        definitions.append({'Transformation': DISPLAY_NAMES[name], 'Definition': definition,
                            'Summarized Rules': SUMMARIZED_RULES[name]})
    sheets['Transformations'] = (['Transformation', 'Definition', 'Summarized Rules'], definitions)
    sheets = {title: sheets[title] for title in ('Quick guide', 'Transformations', 'Examples', 'Grades')}
    write_workbook(path, sheets)
    book = load_workbook(path)
    try:
        for title, names in groups:
            sheet = book[title]
            first_label = 3
            sheet.freeze_panes = f'{get_column_letter(first_label)}2'
            sheet.sheet_view.zoomScale = 85
            sheet.sheet_view.showGridLines = False
            sheet.sheet_properties.tabColor = '2563EB'
            sheet.column_dimensions['A'].width = 18
            sheet.column_dimensions['B'].width = 62
            sheet.row_dimensions[1].height = 48
            for cell in sheet[1]:
                cell.alignment = Alignment(vertical='center', wrap_text=True)
            for row in range(2, sheet.max_row + 1):
                sheet.row_dimensions[row].height = 100
                for cell in sheet[row]:
                    cell.fill = PatternFill('solid', fgColor='F1F5F9' if row % 2 == 0 else 'FFFFFF')
                    cell.border = Border(bottom=Side(style='hair', color='CBD5E1'))
                sheet.cell(row, 2).font = Font(size=12, color='0F172A')
            next_column = first_label
            for name in names:
                label_col = get_column_letter(next_column)
                next_column += 1
                sheet.column_dimensions[label_col].width = 22
                choices = ALLOWED_LABELS
                validation = DataValidation(type='list', formula1=f'"{choices}"', allow_blank=True)
                validation.errorTitle = 'Choose a label'
                validation.error = 'Choose ' + choices.replace(',', ', ') + '.'
                validation.showErrorMessage = True
                validation.errorStyle = 'stop'
                validation.showDropDown = False
                validation.showInputMessage = True
                validation.promptTitle = label_header(name)
                validation.prompt = 'If equivariant, use original -> transformed lines in the next cell.'
                sheet.add_data_validation(validation)
                validation.add(f'{label_col}2:{label_col}{sheet.max_row}')
                for label, color in [('invariant', 'DCFCE7'), ('equivariant', 'FEF3C7'), ('unsafe', 'FEE2E2')]:
                    sheet.conditional_formatting.add(f'{label_col}2:{label_col}{sheet.max_row}', FormulaRule(
                        formula=[f'{label_col}2="{label}"'], fill=PatternFill('solid', fgColor=color)))
                map_col = get_column_letter(next_column)
                next_column += 1
                sheet.column_dimensions[map_col].width = 42
                sheet.conditional_formatting.add(f'{map_col}2:{map_col}{sheet.max_row}', FormulaRule(
                    formula=[f'AND({label_col}2="equivariant",{map_col}2="")'],
                    fill=PatternFill('solid', fgColor='FECACA'), stopIfTrue=True))
                sheet.conditional_formatting.add(f'{map_col}2:{map_col}{sheet.max_row}', FormulaRule(
                    formula=[f'{label_col}2="equivariant"'], fill=PatternFill('solid', fgColor='FEF3C7')))
            sheet.column_dimensions[get_column_letter(sheet.max_column)].width = 42
        guide = book['Quick guide']
        guide.column_dimensions['A'].width = 35
        guide.column_dimensions['B'].width = 105
        for row in range(2, guide.max_row + 1):
            guide.row_dimensions[row].height = {
                'Target selection rule': 260, 'Never a target': 150,
            }.get(guide.cell(row, 1).value, 65)
        for column, width in [('A', 27), ('B', 65), ('C', 110)]:
            book['Transformations'].column_dimensions[column].width = width
        for row in range(2, book['Transformations'].max_row + 1):
            name = transformation_names()[row - 2]
            book['Transformations'].row_dimensions[row].height = (
                230 if name in {'fill_red', 'fill_blue'} else 150)
        book.save(path)
    finally:
        book.close()


def generate(input_path, output_dir, num_workbooks, seed=None):
    questions = load_questions(input_path)
    if not 1 <= num_workbooks <= len(questions):
        raise ValueError('num-workbooks must be between 1 and the number of questions')
    if seed is not None:
        random.Random(seed).shuffle(questions)
    # mkdir is exclusive: never overwrite a grader's existing answers.
    output_dir.mkdir(parents=True, exist_ok=False)
    assignments = []
    for index in range(num_workbooks):
        subset = questions[index::num_workbooks]
        path = output_dir / f'grader_{index + 1:02d}.xlsx'
        create_workbook(path, subset)
        assignments.append({'workbook': path.name, 'question_ids': [q['id'] for q in subset],
                            'question_count': len(subset),
                            'grading_rows_per_sheet': len(subset),
                            'transformation_judgments': len(subset) * len(grading_names())})
    manifest = {'input': str(input_path), 'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
                'seed': seed,
                'question_count': len(questions), 'answer_rule_guidance_version': PROMPT_VERSION,
                'layout_version': 10, 'derived_labels': {'noise': 'blur'}, 'assignments': assignments}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Question JSON array or inference questions.jsonl')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing directories are refused')
    parser.add_argument('--num-workbooks', '-x', type=int, required=True)
    parser.add_argument('--seed', type=int, help='Optionally shuffle questions reproducibly before splitting')
    args = parser.parse_args()
    try:
        result = generate(args.input, args.output_dir, args.num_workbooks, args.seed)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    for assignment in result['assignments']:
        print(f'{assignment["workbook"]}: {assignment["question_count"]} questions, '
              f'{assignment["transformation_judgments"]} transformation judgments')


if __name__ == '__main__':
    main()
