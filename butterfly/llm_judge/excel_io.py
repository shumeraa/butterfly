"""Small openpyxl adapter. No model dependencies; literal strings stay literal."""
import json
from pathlib import Path


def cell_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_workbook(path, sheets, review=False):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.utils import get_column_letter
    book = Workbook()
    book.remove(book.active)
    for title, (headers, rows) in sheets.items():
        sheet = book.create_sheet(title)
        sheet.append(headers)
        for row in rows:
            values = [cell_value(row.get(key)) for key in headers]
            if any(isinstance(value, str) and len(value) > 32767 for value in values):
                raise ValueError(f"{title}: a cell exceeds Excel's 32767 character limit; refusing to truncate source data")
            sheet.append(values)
            for cell in sheet[sheet.max_row]:
                if isinstance(cell.value, str):
                    # Prevent formulas in questions, raw responses, or human input.
                    cell.data_type = 's'
        sheet.freeze_panes = 'A2'
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='304860')
        for idx, header in enumerate(headers, 1):
            sheet.column_dimensions[get_column_letter(idx)].width = (
                55 if any(s in header for s in ('question', 'summary', 'raw', 'warning', 'notes'))
                else 34 if header in {'objects', 'answer_options', 'question_flags', 'reason_code', 'object_name'} else 23)
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical='top', wrap_text=True)
        if review:
            colors = {'invariant': 'D9EAD3', 'equivariant': 'FFF2CC', 'unsafe': 'F4CCCC',
                      'evaluation_error': 'F4CCCC'}
            area = f'A2:{get_column_letter(len(headers))}{max(2, sheet.max_row)}'
            for word, color in colors.items():
                sheet.conditional_formatting.add(area, CellIsRule(
                    operator='equal', formula=['"' + word + '"'],
                    fill=PatternFill('solid', fgColor=color)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace a complete workbook, never expose a half-written zip file.
    import os
    import tempfile
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.xlsx')
    os.close(fd)
    try:
        book.save(temp)
        os.replace(temp, path)
    finally:
        book.close()
        if os.path.exists(temp):
            os.unlink(temp)


def read_workbook(path, sheet_name):
    from openpyxl import load_workbook
    book = load_workbook(path, read_only=True, data_only=False)
    try:
        sheet = book[sheet_name] if sheet_name else book.worksheets[0]
        iterator = sheet.iter_rows(values_only=True)
        headers = list(next(iterator, ()))
        if not headers or any(not isinstance(h, str) or not h for h in headers) or len(set(headers)) != len(headers):
            raise ValueError('Input headers must be nonempty, unique strings')
        return headers, [dict(zip(headers, row)) for row in iterator]
    finally:
        book.close()
