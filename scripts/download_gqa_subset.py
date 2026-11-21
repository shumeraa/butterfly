"""Recover labeled GQA tuples and download only their images using ZIP byte ranges."""
import argparse
from collections.abc import Mapping
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from urllib.request import Request, urlopen
import zipfile


ARCHIVE_URL = 'https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip'


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP file; never accepts a full-file response to a range request."""
    def __init__(self, url, *, max_bytes=512 * 1024 * 1024, timeout=30):
        super().__init__()
        self.url, self.max_bytes, self.timeout = url, max_bytes, timeout
        self.downloaded_bytes = 0
        self._position = 0
        self._buffer_start, self._buffer = 0, b''
        self._length = None
        self._version = None
        self._fetch(0, 0)

    def _fetch(self, start, end):
        requested = end - start + 1
        if self.downloaded_bytes + requested > self.max_bytes:
            raise ValueError('Download byte limit exceeded; raise --max-download-mb to continue')
        headers = {'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
                   'User-Agent': 'GQA-selective-downloader/1.0'}
        if self._version:
            headers['If-Range'] = self._version
        for attempt in range(3):
            try:
                with urlopen(Request(self.url, headers=headers), timeout=self.timeout) as response:
                    if response.status != 206:
                        raise ValueError(f'Server returned HTTP {response.status}; refusing a full archive download')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                         response.headers.get('Content-Range', ''))
                    if not match or tuple(map(int, match.groups()[:2])) != (start, end):
                        raise ValueError('Server returned an incorrect byte range')
                    length = int(match[3])
                    if self._length is not None and self._length != length:
                        raise ValueError('Remote archive changed during download')
                    self._length = length
                    self._version = self._version or response.headers.get('ETag') or response.headers.get('Last-Modified')
                    payload = response.read(requested + 1)
                    self.downloaded_bytes += len(payload)
                    if len(payload) != requested:
                        raise OSError('Incomplete or oversized byte-range response')
                    return payload
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(.5)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=os.SEEK_SET):
        origin = {os.SEEK_SET: 0, os.SEEK_CUR: self._position,
                  os.SEEK_END: self._length}.get(whence)
        if origin is None or origin + offset < 0:
            raise ValueError('Invalid seek')
        self._position = origin + offset
        return self._position

    def read(self, size=-1):
        if self.closed:
            raise ValueError('Read from closed HTTP archive')
        remaining = max(0, self._length - self._position)
        size = remaining if size is None or size < 0 else min(size, remaining)
        if not size:
            return b''
        parts = []
        buffer_end = self._buffer_start + len(self._buffer)
        if self._buffer_start <= self._position < buffer_end:
            count = min(size, buffer_end - self._position)
            offset = self._position - self._buffer_start
            parts.append(self._buffer[offset:offset + count])
            self._position += count
            size -= count
        if size:
            # Read ahead for ZIP's small header reads, but fetch large members
            # and the central directory in one request rather than scanning ZIP.
            count = min(max(size, 64 * 1024), self._length - self._position)
            self._buffer_start = self._position
            self._buffer = self._fetch(self._position, self._position + count - 1)
            parts.append(self._buffer[:size])
            self._position += size
        return b''.join(parts)


def select_records(labels, annotations, limit=None):
    """Join labeled IDs to either official GQA or repository grouped annotations."""
    if not isinstance(labels, list):
        raise ValueError('Expected [question_id, effects] labels')
    ids = []
    for row in labels:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError('Expected each label to contain [question_id, effects]')
        ids.append(str(row[0]))
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate question IDs in labels')
    if not ids:
        raise ValueError('Labels contain no question IDs')
    if limit is not None:
        if limit <= 0:
            raise ValueError('--limit must be positive')
        ids = ids[:limit]
    wanted, found = set(ids), {}
    groups = annotations.items() if isinstance(annotations, Mapping) else [(None, annotations)]
    for outer_id, values in groups:
        grouped = isinstance(values, list)
        for record in values if grouped else [values]:
            question_id = str(record.get('questionId', record.get('sample_id', outer_id)))
            if question_id not in wanted:
                continue
            if question_id in found:
                raise ValueError(f'Duplicate annotation for {question_id}')
            image_id = record.get('imageId', outer_id if grouped else None)
            if image_id is None or not re.fullmatch(r'[A-Za-z0-9_-]+', str(image_id)):
                raise ValueError(f'Invalid image ID for {question_id}')
            if not isinstance(record.get('question'), str) or not isinstance(record.get('answer'), str):
                raise ValueError(f'Question or answer missing for {question_id}')
            found[question_id] = dict(record, questionId=question_id,
                                      sample_id=question_id, imageId=str(image_id))
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f'{len(missing)} label IDs not found in annotations: {sorted(missing)[:10]}')
    return [found[question_id] for question_id in ids]


def download_images(records, output, archive_url=ARCHIVE_URL, max_bytes=512 * 1024 * 1024):
    """Read only ZIP metadata and the distinct image members still needed locally."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ids = sorted({record['imageId'] for record in records})
    missing = [image_id for image_id in ids if not (output / f'{image_id}.jpg').is_file()
               or (output / f'{image_id}.jpg').stat().st_size == 0]
    if not missing:
        return dict(downloaded_images=0, reused_images=len(ids), downloaded_bytes=0)
    with HTTPRangeReader(archive_url, max_bytes=max_bytes) as remote:
        with zipfile.ZipFile(remote) as archive:
            members = {}
            missing_ids = set(missing)
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.suffix.lower() == '.jpg' and path.stem in missing_ids:
                    if path.stem in members:
                        raise ValueError(f'Ambiguous image in ZIP: {path.stem}')
                    members[path.stem] = member
            absent = set(missing) - members.keys()
            if absent:
                raise ValueError(f'Image IDs absent from archive: {sorted(absent)[:10]}')
            for index, image_id in enumerate(missing, 1):
                # ZipFile checks decompression and CRC. Write atomically for resume.
                image = archive.read(members[image_id])
                if not image.startswith(b'\xff\xd8'):
                    raise ValueError(f'ZIP member is not a JPEG: {image_id}')
                fd, temporary = tempfile.mkstemp(dir=output, prefix='.download-')
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(image)
                    os.replace(temporary, output / f'{image_id}.jpg')
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                print(f'[{index}/{len(missing)}] {image_id}.jpg')
        return dict(downloaded_images=len(missing), reused_images=len(ids) - len(missing),
                    downloaded_bytes=remote.downloaded_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=Path('examples/metamorphic/labels.json'))
    parser.add_argument('--annotations', type=Path,
                        default=Path('my_datasets/annotation_files/GQA/balanced_val_annotation.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('output/gqa_subset'))
    parser.add_argument('--archive-url', default=ARCHIVE_URL)
    parser.add_argument('--max-download-mb', type=int, default=512)
    parser.add_argument('--limit', type=int, help='Optional number of labeled questions for a smoke run')
    parser.add_argument('--dry-run', action='store_true', help='Resolve IDs only; no network or files written')
    args = parser.parse_args()
    if args.max_download_mb <= 0:
        parser.error('--max-download-mb must be positive')
    try:
        records = select_records(json.loads(args.labels.read_text()),
                                 json.loads(args.annotations.read_text()), args.limit)
        summary = dict(questions=len(records), unique_images=len({r['imageId'] for r in records}),
                       labels=str(args.labels), annotations=str(args.annotations))
        print(json.dumps(summary, indent=2))
        if args.dry_run:
            return
        images = args.output_dir / 'images'
        summary.update(download_images(records, images, args.archive_url, args.max_download_mb * 1024 * 1024))
        tuples = [dict(question_id=r['questionId'], image_id=r['imageId'], question=r['question'],
                       answer=r['answer'], image_path=str((images / (r['imageId'] + '.jpg')).resolve()))
                  for r in records]
        for name, value in [('annotations.json', records), ('tuples.json', tuples), ('download_report.json', summary)]:
            (args.output_dir / name).write_text(json.dumps(value, indent=2) + '\n')
        print(f'Saved {len(records)} tuples; downloaded {summary["downloaded_bytes"]:,} archive bytes.')
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
