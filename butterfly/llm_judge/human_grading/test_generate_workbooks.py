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
                    self.assertEqual(book.sheetnames, ['Quick guide', 'Transformations', 'Global edits'])
                    self.assertEqual(book.active.title, 'Quick guide')
                    for title, first_label in [('Global edits', 3)]:
                        sheet = book[title]
                        self.assertEqual(sheet.max_row, 6)
                        self.assertEqual(len(sheet.data_validations.dataValidation), 6)
                        for validation in sheet.data_validations.dataValidation:
                            column = next(iter(validation.sqref.ranges)).min_col
                            header = sheet.cell(1, column).value
                            can_map = title == 'Global edits' and header in {
                                'Horizontal flip', 'Vertical flip', '180° rotation', 'Fill color (planned)'}
                            self.assertEqual(validation.formula1, '"equivariant,invariant,unsafe"' if can_map else '"invariant,unsafe"')
                            if can_map:
                                self.assertIn('mapping if equivariant', sheet.cell(1, column + 1).value)
                            self.assertTrue(validation.showErrorMessage)
                            self.assertFalse(validation.showDropDown)
                        self.assertEqual([cell.value for cell in sheet['A'][1:]], assignment['question_ids'])
                        for row in sheet.iter_rows(min_row=2):
                            self.assertTrue(all(cell.value is None for cell in row[first_label - 1:]))
                        self.assertEqual(len(sheet.conditional_formatting), 10)
                    self.assertEqual(book['Global edits'].freeze_panes, 'C2')
                    self.assertNotIn('Call3', book.sheetnames)
                    guide = ' '.join(str(cell.value) for row in book['Quick guide'] for cell in row)
                    self.assertIn('yes -> no\nno -> yes, no', guide)
                    self.assertIn('Always -> blue', guide)
                    self.assertNotIn('JSON', guide)
                    self.assertNotIn('Reason:', guide)
                    self.assertNotIn('Assumptions:', guide)
                    self.assertEqual(book['Quick guide'].max_row, 7)
                    all_text = ' '.join(str(cell.value) for sheet in book for row in sheet for cell in row)
                    self.assertNotIn('local edit', all_text.lower())
                    self.assertNotIn('segment_mr1', all_text)
                    self.assertEqual(assignment['transformation_judgments'], 30)
                    self.assertEqual(manifest['derived_labels'], {'noise': 'blur'})
                    headers = [cell.value for cell in book['Global edits'][1]]
                    self.assertIn('Blur / noise', headers)
                    self.assertNotIn('Noise', headers)
                    self.assertNotIn('Blur', headers)
                    self.assertEqual(book['Transformations'].max_row, 8)
                    for name, scope, can_map, labels, definition in book['Transformations'].iter_rows(min_row=2, values_only=True):
                        sheet = book[scope]
                        header = 'Blur / noise' if name in {'Blur', 'Noise'} else name
                        column = next(cell.column for cell in sheet[1] if cell.value == header)
                        validation = next(v for v in sheet.data_validations.dataValidation if f'{sheet.cell(2, column).coordinate}' in v)
                        self.assertEqual(validation.formula1.strip('"'), labels.replace(', ', ','))
                        self.assertEqual(can_map, 'Yes' if 'equivariant' in labels else 'No')
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
            manifest = generate(source, root / 'out', 2, seed=42, include_color_swap=True)
            self.assertEqual([r['question_count'] for r in manifest['assignments']], [2, 1])
            book = load_workbook(root / 'out/grader_01.xlsx')
            try:
                self.assertEqual(book['Global edits']['B2'].data_type, 's')
                self.assertEqual(book['Global edits'].max_row, 3)
                self.assertEqual(len(book['Global edits'].data_validations.dataValidation), 7)
                swap = book['Global edits'].data_validations.dataValidation[-1]
                self.assertEqual(swap.formula1, '"invariant,unsafe"')
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
