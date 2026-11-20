"""Selective HTTP ZIP extraction and exact question-ID joins, without internet."""
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.download_gqa_subset import HTTPRangeReader, download_images, select_records


class GQASubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as stream:
            stream.writestr('images/one.jpg', b'\xff\xd8selected-image\xff\xd9')
            stream.writestr('unneeded.bin', b'x' * (2 * 1024 * 1024))
            stream.writestr('images/two.jpg', b'\xff\xd8another-image\xff\xd9')
        cls.archive = archive.getvalue()

    def setUp(self):
        # Model the range protocol in memory so tests require no network/bind
        # permissions; a separate smoke run verifies the real GQA server.
        self.url = 'http://fixture.local/images.zip'
        def open_range(request, timeout):
            if 'unreachable.invalid' in request.full_url:
                raise AssertionError('Unexpected network request on resume')
            if request.full_url.endswith('/no-ranges'):
                response = io.BytesIO()
                response.status, response.headers = 200, {}
                return response
            start, end = map(int, re.fullmatch(r'bytes=(\d+)-(\d+)', request.get_header('Range')).groups())
            response = io.BytesIO(self.archive[start:end + 1])
            response.status = 206
            response.headers = {'Content-Range': f'bytes {start}-{end}/{len(self.archive)}',
                                'ETag': '"fixture"'}
            return response
        mock = patch('scripts.download_gqa_subset.urlopen', side_effect=open_range)
        mock.start()
        self.addCleanup(mock.stop)

    def test_grouped_and_official_annotations_preserve_ids(self):
        labels = [['001', {}], ['002', {}]]
        grouped = {'one': [{'questionId': '001', 'question': 'Q1', 'answer': 'yes'}],
                   'two': [{'questionId': '002', 'question': 'Q2', 'answer': 'no'}]}
        official = {'001': {'imageId': 'one', 'question': 'Q1', 'answer': 'yes'},
                    '002': {'imageId': 'two', 'question': 'Q2', 'answer': 'no'}}
        first = select_records(labels, grouped)
        self.assertEqual([row['questionId'] for row in first], ['001', '002'])
        self.assertEqual([row['imageId'] for row in first], ['one', 'two'])
        self.assertEqual(first[0]['answer'], 'yes')
        self.assertEqual(select_records(labels, official), first)
        self.assertEqual(len(select_records(labels, grouped, limit=1)), 1)
        with self.assertRaisesRegex(ValueError, 'not found'):
            select_records([['missing', {}]], grouped)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            select_records([['001', {}], ['001', {}]], grouped)

    def test_real_labels_all_match_validation_metadata(self):
        root = Path(__file__).resolve().parents[1]
        labels = json.loads((root / 'examples/metamorphic/labels.json').read_text())
        annotations = json.loads((root / 'my_datasets/annotation_files/GQA/balanced_val_annotation.json').read_text())
        records = select_records(labels, annotations)
        self.assertEqual(len(records), 200)
        self.assertEqual(len({row['imageId'] for row in records}), 198)
        self.assertEqual([row['questionId'] for row in records], [row[0] for row in labels])

    def test_extract_only_selected_image_and_resume_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = download_images([{'imageId': 'one'}, {'imageId': 'one'}], tmp, self.url)
            self.assertEqual(report['downloaded_images'], 1)
            self.assertLess(report['downloaded_bytes'], len(self.archive) // 4)
            self.assertEqual((Path(tmp) / 'one.jpg').read_bytes(), b'\xff\xd8selected-image\xff\xd9')
            self.assertFalse((Path(tmp) / 'two.jpg').exists())
            report = download_images([{'imageId': 'one'}], tmp, 'http://unreachable.invalid')
            self.assertEqual(report['downloaded_bytes'], 0)
            self.assertEqual(report['reused_images'], 1)

    def test_refuse_full_archive_and_enforce_transfer_limit(self):
        with self.assertRaisesRegex(ValueError, 'refusing'):
            HTTPRangeReader(self.url + '/no-ranges')
        with HTTPRangeReader(self.url, max_bytes=10) as remote:
            with self.assertRaisesRegex(ValueError, 'limit'):
                remote.read(100)

    def test_seek_and_buffered_reads(self):
        with HTTPRangeReader(self.url) as remote:
            self.assertEqual(remote.read(8), self.archive[:8])
            remote.seek(2)
            self.assertEqual(remote.read(20), self.archive[2:22])
            remote.seek(-22, 2)
            self.assertEqual(remote.read(), self.archive[-22:])
            self.assertEqual(remote.read(), b'')


if __name__ == '__main__':
    unittest.main()
