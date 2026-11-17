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
NO_EQ_NOTE = (
    'The automatic rules do not assign equivariant for this transformation. This does not mean you '
    'should avoid it: choose equivariant whenever you believe the answer changes predictably.\n'
)
BLUR_RULE = NO_EQ_NOTE + (
    'Unsafe if the question involves readable text, material, texture, pattern, surface properties, '
    'actions or interactions, or comparisons of object properties, or asks for an object name '
    'without limiting the answer to two possibilities. Otherwise invariant.'
)
SUMMARIZED_RULES = {
    'hflip': (
        'Invariant: the question has no left/right wording, no left/right body parts, and no readable text.\n'
        'Equivariant: the question asks which side something is on (answer must be left or right, e.g. '
        '“Is X on the left or right?”), left/right is not used to identify an object, and there are no '
        'body parts, readable text, or multiple and/or conditions. Answer: swap left and right.\n'
        'Unsafe: everything else. This includes yes/no questions such as “Is X left of Y?” and any '
        'question that uses left/right to pick out an object.'
    ),
    'vflip': (
        'Invariant: the question has no up/down wording (top, bottom, above, below), no support or '
        'orientation words (on, under, inside, sitting, hanging, lying), and no readable text.\n'
        'Equivariant: the question asks which side something is on (answer must be top or bottom), '
        'up/down is not used to identify an object, and there is no support wording, readable text, or '
        'multiple and/or conditions. Answer: swap top and bottom, above and below.\n'
        'Unsafe: everything else, including yes/no questions such as “Is X above Y?”.'
    ),
    'grayscale': (
        'Invariant: the question contains no color or brightness wording (dark, bright, colorful).\n'
        'Equivariant: the question asks what color one object is, with no comparison or multiple '
        'conditions. Answer: gray, black, or white (black and white objects stay as they are). '
        'Also equivariant: “Is X <color>?” where the color is not gray, black, or white. Answer: no, '
        'because a chromatic color cannot survive grayscale.\n'
        'Unsafe: everything else, including “red or blue?” choice questions and yes/no questions '
        'that name gray, black, or white.'
    ),
    'blur': BLUR_RULE,
    'noise': BLUR_RULE,
    'fill_red': (
        'First pick one target object (see Target selection rule on the Quick guide); if there is none, '
        'there is no automatic label.\n'
        'Invariant: the question has no color, material, texture, pattern, surface, or readable-text wording.\n'
        'Equivariant: the target is clear, there is no comparison or multiple and/or conditions, and '
        'the question is about the target’s color. “What color is X?” answers red. “Is X <color>?” '
        'answers yes if the color is red, otherwise no. “Is X <color> or red?” answers red.\n'
        'Unsafe: everything else.'
    ),
    'fill_blue': LOCAL_GUARDS + FILL_RULE.format(color='blue'),
    'occlude_bbox': (
        'Never invariant: the object is named in the question, so hiding it cannot be shown irrelevant.\n'
        'Equivariant: the target is clear, there is no comparison or multiple and/or conditions, and '
        'one of these applies. “What color is X?” answers black. “Is X <color>?” answers yes if the color '
        'is black, otherwise no. “Is X <color> or black?” answers black. “Is there an X?” answers no '
        'when the wording implies only one X.\n'
        'Unsafe: everything else.\n'
        'Caveats: the color answers assume the box fully covers the object, and the existence answer '
        'is wrong if other instances of X remain outside the box.'
    ),
    'asymmetric_scale': NO_EQ_NOTE + LOCAL_GUARDS + (
        'After those checks, unsafe if size, proportions, or shape identifies the selected object; '
        'if the question asks about position, object type, or identity; or if it mentions size, '
        'proportions, shape, comparisons of object properties, categories or collections, left/right '
        'or up/down relations, support, containment, or physical orientation. Otherwise invariant.'
    ),
}


# (section, text, fill color). Section is bold in column A.
INSTRUCTIONS = [
    ('About this workbook',
     'You will judge how edits to an image would change the answer to a question. For each question, '
     'decide whether each edit (a flip, a recoloring, hiding an object, and so on) leaves the answer the same, '
     'changes it in a predictable way, or makes it unreliable. Use only the question wording: do not guess '
     'an image or the original answer.', 'EFF6FF'),
    ('Tabs',
     'Quick guide: these instructions.\n'
     'Transformations: what each edit does, plus a summary of the rules an automatic checker uses for it.\n'
     'Examples: four fully graded rows to copy the style from.\n'
     'Grades: where you work. One row per question.', 'EFF6FF'),
    ('You make the decisions',
     'The rules on the Transformations tab are a reference, not an answer key. They are deliberately '
     'oversimplified and may contain logic errors; finding those errors is exactly what this grading is for. '
     'Use them for ideas, but make every label and mapping on your own judgment. '
     'If a rule seems wrong for a question, ignore it and grade what you believe is correct.', 'FEE2E2'),
    ('1. Grade each question',
     'On Grades, choose a label for every transformation; all three labels are allowed for every transformation.', 'EFF6FF'),
    ('invariant',
     'The answer stays the same after the edit.', 'DCFCE7'),
    ('equivariant',
     'The answer changes in a predictable way (including always becoming one fixed answer). '
     'Write the mapping in the next column.', 'FEF3C7'),
    ('unsafe',
     'No reliable expectation: the answer could change unpredictably, or the question may become unanswerable.', 'FEE2E2'),
    ('2. Mapping (equivariant only)',
     'If you chose equivariant, say how the answer changes. Imagine the question had every possible original answer, '
     'and write one line for each: original answer -> answer after the edit.\n\n'
     'Example: horizontal flip of “Is the ball on the left or right of the box?” swaps the sides:\n'
     'left -> right\nright -> left\n\n'
     'Fixed result: if the edit always forces one answer, every line ends the same way '
     '(Fill Red on “Is the car red or blue?”: red -> red, blue -> red).\n'
     'Several possible results: separate them with commas (no -> yes, no).\n\n'
     'Cover every possible original answer, and don’t assume “not left” means right. '
     'For invariant or unsafe, leave the mapping blank. A blank mapping next to equivariant turns red.', 'EFF6FF'),
    ('3. Name the target (local edits)',
     'Fill Red, Fill Blue, Occlude BBox and Asymmetric scale edit one object. Enter it in the Target object '
     'column, chosen with the rule below.', 'EFF6FF'),
    ('No target object = unsafe',
     'If no object qualifies as the target (for example “Is it sunny?”), enter none in Target object and '
     'mark every local edit unsafe. Whole-image (Global Edit) transformations are still graded normally.', 'FEE2E2'),
    ('Target selection rule',
     'The target is the one object the question is really about. Ask: “If someone edited just one thing '
     'in the picture, which thing would the question depend on most?” Apply in order, and stop at the first match.\n\n'
     '1. Asked-about object. If the question asks about an object’s color, or whether it exists, that object is '
     'the target. (“What color is the apple?” → apple. “Is there a cat?” → cat.)\n'
     '2. Grammatical subject. Otherwise, take the thing doing or being described: the main noun the sentence is built around. '
     '(“Is the ball above the box?” → ball. “What is the girl holding?” → girl.)\n'
     '3. Only named object. If the question mentions just one object, it is the target. '
     '(“What is on the table?” → table.)\n'
     '4. Tie. If two objects still seem equally good, take the one mentioned first.', 'F8FAFC'),
    ('Never a target',
     'Objects used only to identify another object. They are just directions (“the girl to the left of the Z” → '
     'the girl, not Z).\n'
     'Answer options. They are possible answers, not things being asked about (“an apple or a banana”).\n'
     'Objects the question doesn’t assume are in the picture. “Is the man holding a cup?” may well have no cup, '
     'so the man is the target, not the cup. (Exception: in an existence question, the object asked about is the target.)', 'F8FAFC'),
    ('Blur / noise',
     'Graded once in the Blur / noise column; the same label and mapping apply to both.', 'F8FAFC'),
    ('Tips',
     'Use Alt+Enter (Option+Return on Mac) for a new line inside a cell.', 'F8FAFC'),
]

# Complete worked rows in the same column order as the grading sheet:
# Horizontal flip, Vertical flip, Grayscale, Blur / noise, Fill Red, Fill Blue, Occlude BBox, Asymmetric scale.
EXAMPLES = [
    ('Is the car red or blue?', 'car',
     ('invariant', 'invariant', 'unsafe', 'invariant', 'equivariant', 'equivariant', 'unsafe', 'invariant'),
     {'fill_red': 'red -> red\nblue -> red', 'fill_blue': 'red -> blue\nblue -> blue'},
     'COLOR question. Fills force their color; grayscale and occlusion remove the color, so the answer is unpredictable.'),
    ('Is the ball to the left or right of the box?', 'ball',
     ('equivariant', 'invariant', 'invariant', 'invariant', 'invariant', 'invariant', 'unsafe', 'unsafe'),
     {'hflip': 'left -> right\nright -> left'},
     'LEFT/RIGHT question. The ball is the subject; the box only helps locate it. '
     'Occlusion hides the ball being located.'),
    ('Is there a cat in the image?', 'cat',
     ('invariant', 'invariant', 'invariant', 'invariant', 'invariant', 'invariant', 'equivariant', 'invariant'),
     {'occlude_bbox': 'yes -> no\nno -> no'},
     'EXISTS question. A recolored cat still exists; if there is one cat, covering its box makes the answer no.'),
    ('Is it sunny?', 'none',
     ('invariant', 'invariant', 'invariant', 'invariant', 'unsafe', 'unsafe', 'unsafe', 'unsafe'),
     {},
     'NO TARGET question. It is about the whole scene, so no single object can be edited. '
     'Whole-image edits leave the answer alone; local edits have nothing to act on, so they are unsafe.'),
]


def wrapped_lines(text, width):
    """Approximate wrapped line count for sizing Excel rows."""
    return sum(max(1, -(-len(part) // width)) for part in text.split('\n'))


def example_rows():
    rows = []
    for index, (question, target, labels, mappings, notes) in enumerate(EXAMPLES, 1):
        row = {'Question ID': f'worked_{index:02d}', 'Question': question, 'Target object': target, 'Notes': notes}
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


def base_header(name):
    return 'Blur / noise' if name == 'blur' else DISPLAY_NAMES[name]


def label_header(name):
    base = base_header(name)
    return f'{base} ({"Global Edit" if name in GLOBAL else "Local Edit on Target Object"})'


def is_local(name):
    return name not in GLOBAL


def mapping_header(name):
    return base_header(name) + ' — mapping if equivariant'


def load_questions(path):
    """Read the inference questions.jsonl format: one JSON object per line."""
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path.name} line {number} is not valid JSON ({exc.msg}); expected questions.jsonl') from exc
    if not rows:
        raise ValueError('Input must be a nonempty questions.jsonl with one question record per line')
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
            if name == next(n for n in names if is_local(n)):
                headers.append('Target object')
            headers.append(label_header(name))
            headers.append(mapping_header(name))
        headers.append('Notes')
        sheets[title] = (headers, example_rows() if title == 'Examples' else grading_rows(questions))
    sheets['Quick guide'] = (['Topic', 'Guidance'], [{'Topic': k, 'Guidance': v} for k, v, _ in INSTRUCTIONS])
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
            target_column = 3 + 2 * sum(1 for n in names if not is_local(n))
            first_local = next(n for n in names if is_local(n))
            sheet.freeze_panes = 'C2'
            sheet.sheet_view.zoomScale = 85
            sheet.sheet_view.showGridLines = False
            sheet.sheet_properties.tabColor = '2563EB'
            sheet.column_dimensions['A'].width = 18
            sheet.column_dimensions['B'].width = 62
            sheet.column_dimensions[get_column_letter(target_column)].width = 22
            sheet.row_dimensions[1].height = 64
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
                if name == first_local:
                    next_column += 1  # Target object column sits between global and local edits
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
        guide.sheet_view.showGridLines = False
        guide.auto_filter.ref = None
        guide.sheet_properties.tabColor = 'DC2626'
        guide.column_dimensions['A'].width = 32
        guide.column_dimensions['B'].width = 110
        guide.row_dimensions[1].height = 28
        for cell in guide[1]:
            cell.alignment = Alignment(vertical='center', indent=1)
        thin = Side(style='thin', color='FFFFFF')
        for row, (_, text, color) in enumerate(INSTRUCTIONS, 2):
            guide.row_dimensions[row].height = max(34, 17 * wrapped_lines(text, 105) + 14)
            for column in (1, 2):
                cell = guide.cell(row, column)
                cell.fill = PatternFill('solid', fgColor=color)
                cell.border = Border(top=thin, bottom=thin)
                cell.alignment = Alignment(vertical='center', wrap_text=True, indent=1)
            guide.cell(row, 1).font = Font(bold=True, size=12, color='991B1B' if row == 2 else '0F172A')
            guide.cell(row, 2).font = Font(size=11, color='0F172A')
        for column, width in [('A', 27), ('B', 65), ('C', 110)]:
            book['Transformations'].column_dimensions[column].width = width
        for row in range(2, book['Transformations'].max_row + 1):
            name = transformation_names()[row - 2]
            book['Transformations'].row_dimensions[row].height = max(
                60, 16 * wrapped_lines(SUMMARIZED_RULES[name], 105) + 16)
            for cell in book['Transformations'][row]:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
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
                'layout_version': 13, 'derived_labels': {'noise': 'blur'}, 'assignments': assignments}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='questions.jsonl: one record per line with id, question, and optional detailed_type')
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
