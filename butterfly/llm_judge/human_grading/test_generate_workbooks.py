import json
from pathlib import Path
import sys
import tempfile
import unittest

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_workbooks import generate, load_questions


class WorkbookTests(unittest.TestCase):
    def test_example_split_and_grading_controls(self):
        source = Path(__file__).parent / 'examples/questions.json'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'assignments'
            manifest = generate(source, output, 4)
            ids = []
            for assignment in manifest['assignments']:
                ids.extend(assignment['question_ids'])
                self.assertEqual(assignment['question_count'], 5)
                book = load_workbook(output / assignment['workbook'])
                try:
                    self.assertEqual(book.sheetnames, ['Quick guide', 'Transformations', 'Examples', 'Grades'])
                    self.assertEqual(book.active.title, 'Quick guide')
                    for title, first_label in [('Grades', 3)]:
                        sheet = book[title]
                        self.assertEqual(sheet.max_row, 6)
                        self.assertEqual(len(sheet.data_validations.dataValidation), 8)
                        for validation in sheet.data_validations.dataValidation:
                            column = next(iter(validation.sqref.ranges)).min_col
                            self.assertEqual(validation.formula1, '"equivariant,invariant,unsafe"')
                            self.assertIn('mapping if equivariant', sheet.cell(1, column + 1).value)
                            mapping_cell = sheet.cell(2, column + 1).coordinate
                            formulas = [formula for area in sheet.conditional_formatting
                                        if mapping_cell in area
                                        for rule in sheet.conditional_formatting[area] for formula in rule.formula]
                            self.assertIn(f'AND({sheet.cell(2, column).coordinate}="equivariant",{mapping_cell}="")', formulas)
                            self.assertTrue(validation.showErrorMessage)
                            self.assertFalse(validation.showDropDown)
                        self.assertEqual([cell.value for cell in sheet['A'][1:]], assignment['question_ids'])
                        for row in sheet.iter_rows(min_row=2):
                            self.assertTrue(all(cell.value is None for cell in row[first_label - 1:]))
                        self.assertEqual(len(sheet.conditional_formatting), 16)
                    self.assertEqual(book['Grades'].freeze_panes, 'C2')
                    self.assertNotIn('Call3', book.sheetnames)
                    guide = ' '.join(str(cell.value) for row in book['Quick guide'] for cell in row)
                    self.assertIn('left -> right', guide)
                    self.assertIn('You make the decisions', guide)
                    self.assertIn('oversimplified', guide)
                    self.assertIn('all three labels', guide)
                    self.assertNotIn('JSON', guide)
                    self.assertNotIn('Reason:', guide)
                    self.assertNotIn('Assumptions:', guide)
                    self.assertEqual(book['Quick guide'].max_row, 15)
                    guide_rows = dict(book['Quick guide'].iter_rows(min_row=2, values_only=True))
                    target_rule = guide_rows['Target selection rule']
                    self.assertIn('Apply in order, and stop at the first match.', target_rule)
                    steps = ['1. Asked-about object.', '2. Grammatical subject.',
                             '3. Only named object.', '4. Tie.']
                    self.assertEqual([target_rule.index(step) for step in steps],
                                     sorted(target_rule.index(step) for step in steps))
                    self.assertIn('Is there a cat?', target_rule)
                    self.assertIn('What is on the table?', target_rule)
                    self.assertIn('Objects used only to identify another object', guide_rows['Never a target'])
                    self.assertIn('Answer options', guide_rows['Never a target'])
                    self.assertIn('the man is the target, not the cup.', guide_rows['Never a target'])
                    examples = book['Examples']
                    grades = book['Grades']
                    self.assertEqual([c.value for c in examples[1]], [c.value for c in grades[1]])
                    self.assertEqual(examples.freeze_panes, grades.freeze_panes)
                    self.assertEqual(examples.sheet_view, grades.sheet_view)
                    self.assertEqual(set(examples.column_dimensions), set(grades.column_dimensions))
                    for column in grades.column_dimensions:
                        self.assertEqual(dict(examples.column_dimensions[column]), dict(grades.column_dimensions[column]))
                    self.assertEqual(dict(examples.row_dimensions[1]), dict(grades.row_dimensions[1]))
                    for example_cell, grade_cell in zip(examples[2], grades[2]):
                        self.assertEqual(example_cell._style, grade_cell._style)
                    self.assertEqual(examples.row_dimensions[2].height, grades.row_dimensions[2].height)
                    self.assertEqual(len(examples.conditional_formatting), len(grades.conditional_formatting))
                    self.assertEqual(len(examples.data_validations.dataValidation), 8)
                    label_columns = [next(iter(v.sqref.ranges)).min_col
                                     for v in examples.data_validations.dataValidation]
                    example_headers = [c.value for c in examples[1]]
                    target_column = example_headers.index('Target object') + 1
                    self.assertTrue(all('(Global Edit)' in h for h in example_headers[2:target_column - 1] if 'mapping' not in h))
                    self.assertTrue(all('(Local Edit on Target Object)' in h for h in example_headers[target_column:-1] if 'mapping' not in h))
                    self.assertIn('no target object = unsafe', guide.lower())
                    seen_labels = set()
                    example_questions = ' '.join(examples.cell(r, 2).value.lower()
                                                 for r in range(2, examples.max_row + 1))
                    for needle in ('left or right', 'is there', 'red or blue'):
                        self.assertIn(needle, example_questions)
                    for row in range(2, examples.max_row + 1):
                        self.assertTrue(examples.cell(row, 1).value)
                        self.assertTrue(examples.cell(row, 2).value)
                        self.assertTrue(examples.cell(row, target_column).value)
                        for word in ('asymmetric', 'factor', 'width'):
                            self.assertNotIn(word, examples.cell(row, examples.max_column).value.lower())
                        for column in label_columns:
                            label = examples.cell(row, column).value
                            seen_labels.add(label)
                            self.assertIn(label, {'invariant', 'equivariant', 'unsafe'})
                            self.assertEqual(bool(examples.cell(row, column + 1).value), label == 'equivariant')
                    self.assertEqual(seen_labels, {'invariant', 'equivariant', 'unsafe'})
                    all_text = ' '.join(str(cell.value) for sheet in book for row in sheet for cell in row)
                    self.assertIn('local edits', all_text.lower())
                    for removed in ('Segment then MR1', '180° rotation', 'Global edits', 'Local edits'):
                        self.assertNotIn(removed, all_text)
                    self.assertEqual(assignment['transformation_judgments'], 40)
                    self.assertEqual(manifest['derived_labels'], {'noise': 'blur'})
                    self.assertEqual(manifest['layout_version'], 13)
                    headers = [cell.value for cell in book['Grades'][1]]
                    self.assertIn('Blur / noise (Global Edit)', headers)
                    self.assertIn('Fill Red (Local Edit on Target Object)', headers)
                    self.assertNotIn('Noise (Global Edit)', headers)
                    self.assertNotIn('Blur (Global Edit)', headers)
                    self.assertEqual(book['Transformations'].max_row, 10)
                    self.assertEqual([cell.value for cell in book['Transformations'][1]],
                                     ['Transformation', 'Definition', 'Summarized Rules'])
                    transformations = list(book['Transformations'].iter_rows(min_row=2, values_only=True))
                    for name, definition, rules in transformations:
                        self.assertTrue(definition and rules)
                        self.assertIn('unsafe', rules.lower())
                        # Rules must stay readable without internal question/identity flag names.
                        self.assertNotRegex(rules, r'\b[A-Z][A-Z_]+\b')
                        header = 'Blur / noise' if name in {'Blur', 'Noise'} else name
                        self.assertTrue(any(h.startswith(header + ' (') for h in headers))
                finally:
                    book.close()
            self.assertEqual(len(ids), 20)
            self.assertEqual(len(set(ids)), 20)
            with self.assertRaises(FileExistsError):
                generate(source, output, 4)

    def test_uneven_jsonl_literal_strings_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'questions.jsonl'
            rows = [{'id': str(i), 'question': '=HYPERLINK("example")'} for i in range(3)]
            source.write_text('\n'.join(json.dumps(r) for r in rows))
            manifest = generate(source, root / 'out', 2, seed=42)
            self.assertEqual([r['question_count'] for r in manifest['assignments']], [2, 1])
            book = load_workbook(root / 'out/grader_01.xlsx')
            try:
                self.assertEqual(book['Grades']['B2'].data_type, 's')
                self.assertEqual(book['Grades'].max_row, 3)
                self.assertEqual(len(book['Grades'].data_validations.dataValidation), 8)
                swap = book['Grades'].data_validations.dataValidation[-1]
                self.assertEqual(swap.formula1, '"equivariant,invariant,unsafe"')
            finally:
                book.close()
            for count in (0, 4):
                with self.assertRaises(ValueError):
                    generate(source, root / 'invalid', count)
            self.assertFalse((root / 'invalid').exists())
            source.write_text(json.dumps(rows[0]) + '\n' + json.dumps(rows[0]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_questions(source)


if __name__ == '__main__':
    unittest.main()
