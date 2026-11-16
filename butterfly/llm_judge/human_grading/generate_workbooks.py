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
from evaluate_rules import GLOBAL
from excel_io import write_workbook

DEFINITIONS = {
    **TRANSFORMATIONS,
    'grayscale': 'Convert the entire image to grayscale, preserving geometry.',
    'blur': 'Blur the entire image, reducing visible fine detail.',
    'noise': 'Add image noise throughout the image.',
    'fill_color': ('Planned color-fill transformation; not implemented yet. Specify the target, '
                   'fill color, and intended operation in Notes before grading. No default fill color is assumed.'),
}
DISPLAY_NAMES = {
    'hflip': 'Horizontal flip', 'vflip': 'Vertical flip', 'rot180': '180° rotation',
    'grayscale': 'Grayscale', 'blur': 'Blur', 'noise': 'Noise',
    'color_swap_red_blue': 'Red / blue swap',
    'fill_color': 'Fill color (planned)',
}
EQUIVARIANT = frozenset({'hflip', 'vflip', 'rot180', 'fill_color'})


def allowed_labels(name):
    return 'equivariant,invariant,unsafe' if name in EQUIVARIANT else 'invariant,unsafe'


INSTRUCTIONS = [
    ('1. Grade each question', 'Open Global edits and choose a label for each transformation. Keep the question unchanged; use only its wording, without guessing an image or original answer.'),
    ('Labels', 'invariant = answer stays the same. equivariant = answers follow a useful mapping. unsafe = no reliable relationship or the question may become unanswerable. Dropdowns offer only eligible labels; see Transformations.'),
    ('Blur / noise', 'Choose once in Blur / noise. The same label applies to both transformations.'),
    ('2. Map only if equivariant', 'Write original -> transformed, one line per original answer. Separate multiple possible outputs with commas. Cover every original answer. Only the mapping is required; leave it blank for invariant or unsafe.'),
    ('Mapping example', 'Horizontal flip: Is the ball left of the box?\n\nyes -> no\nno -> yes, no\n\nFor an edit that always makes the answer blue: Always -> blue'),
    ('Tips', 'Use Alt+Enter (Option+Return on Mac) for new lines in a cell. Do not assume not-left means right. An empty equivariant mapping is highlighted red. Notes are optional.'),
]


def transformation_names(include_color_swap=False):
    return list(GLOBAL) + ['fill_color'] + (['color_swap_red_blue'] if include_color_swap else [])


def grading_names(include_color_swap=False):
    return [name for name in transformation_names(include_color_swap) if name != 'noise']


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


def create_workbook(path, questions, include_color_swap=False):
    from openpyxl import load_workbook
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    if len(questions) > 1_048_575:
        raise ValueError('Assignment exceeds the Excel row limit')
    sheets = {}
    groups = [('Global edits', grading_names(include_color_swap))]
    for title, names in groups:
        headers = ['Question ID', 'Question']
        for name in names:
            headers.append(label_header(name))
            if name in EQUIVARIANT:
                headers.append(mapping_header(name))
        headers.append('Notes')
        sheets[title] = (headers, grading_rows(questions))
    sheets['Quick guide'] = (['Step', 'What to do'], [{'Step': k, 'What to do': v} for k, v in INSTRUCTIONS])
    definitions = []
    for title, names in groups:
        for name in transformation_names(include_color_swap):
            definition = DEFINITIONS[name]
            if name in {'blur', 'noise'}:
                definition += ' Graded together in the Blur / noise column; noise always uses the blur label.'
            definitions.append({'Transformation': DISPLAY_NAMES[name], 'Scope': title,
                                'Allowed labels': allowed_labels(name).replace(',', ', '),
                                'Can be equivariant?': 'Yes' if name in EQUIVARIANT else 'No',
                                'Definition': definition})
    sheets['Transformations'] = (['Transformation', 'Scope', 'Can be equivariant?', 'Allowed labels', 'Definition'], definitions)
    sheets = {title: sheets[title] for title in ('Quick guide', 'Transformations', 'Global edits')}
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
                choices = allowed_labels(name)
                validation = DataValidation(type='list', formula1=f'"{choices}"', allow_blank=True)
                validation.errorTitle = 'Choose a label'
                validation.error = 'Choose ' + choices.replace(',', ', ') + '.'
                validation.showErrorMessage = True
                validation.errorStyle = 'stop'
                validation.showDropDown = False
                validation.showInputMessage = True
                validation.promptTitle = label_header(name)
                validation.prompt = ('If equivariant, use original -> transformed lines in the next cell.'
                                     if name in EQUIVARIANT else 'This transformation allows only invariant or unsafe.')
                sheet.add_data_validation(validation)
                validation.add(f'{label_col}2:{label_col}{sheet.max_row}')
                for label, color in [('invariant', 'DCFCE7'), ('equivariant', 'FEF3C7'), ('unsafe', 'FEE2E2')]:
                    sheet.conditional_formatting.add(f'{label_col}2:{label_col}{sheet.max_row}', FormulaRule(
                        formula=[f'{label_col}2="{label}"'], fill=PatternFill('solid', fgColor=color)))
                if name not in EQUIVARIANT:
                    continue
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
            guide.row_dimensions[row].height = 115 if str(guide.cell(row, 1).value).startswith('Mapping example') else 55
        for column, width in [('A', 27), ('B', 20), ('C', 23), ('D', 39), ('E', 85)]:
            book['Transformations'].column_dimensions[column].width = width
        for row in range(2, book['Transformations'].max_row + 1):
            book['Transformations'].row_dimensions[row].height = 65
        book.save(path)
    finally:
        book.close()


def generate(input_path, output_dir, num_workbooks, seed=None, include_color_swap=False):
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
        create_workbook(path, subset, include_color_swap)
        assignments.append({'workbook': path.name, 'question_ids': [q['id'] for q in subset],
                            'question_count': len(subset),
                            'grading_rows_per_sheet': len(subset),
                            'transformation_judgments': len(subset) * len(grading_names(include_color_swap))})
    manifest = {'input': str(input_path), 'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
                'seed': seed, 'include_color_swap': include_color_swap,
                'question_count': len(questions), 'answer_rule_guidance_version': PROMPT_VERSION,
                'layout_version': 5, 'derived_labels': {'noise': 'blur'}, 'assignments': assignments}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Question JSON array or inference questions.jsonl')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing directories are refused')
    parser.add_argument('--num-workbooks', '-x', type=int, required=True)
    parser.add_argument('--seed', type=int, help='Optionally shuffle questions reproducibly before splitting')
    parser.add_argument('--include-color-swap', action='store_true', help='Also include experimental red/blue swap')
    args = parser.parse_args()
    try:
        result = generate(args.input, args.output_dir, args.num_workbooks, args.seed, args.include_color_swap)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    for assignment in result['assignments']:
        print(f'{assignment["workbook"]}: {assignment["question_count"]} questions, '
              f'{assignment["transformation_judgments"]} transformation judgments')


if __name__ == '__main__':
    main()
