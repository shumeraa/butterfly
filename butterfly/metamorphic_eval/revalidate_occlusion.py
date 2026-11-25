#!/usr/bin/env python3
"""Re-render the occlusion tests with the improved occluder and check them independently.

Usage: python revalidate_occlusion.py <merged_dir> [device=cuda]
For each occlude_bbox test found in <merged>/metamorphic_eval/chunk_*/run/data.p it re-runs
occlude_objects on the ORIGINAL image with the current GroundedSAM defaults (grounding-dino-base,
lower threshold, dilated boxes, post-edit verification gate). Then it re-checks the result with the
tiny detector used earlier (so numbers compare with occlusion_validation_raw_v1.json), ignoring
detections that sit on the filled area.
Writes <merged>/metamorphic_eval/occlusion_revalidation.json and prints a summary.
"""
import glob
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
merged = Path(sys.argv[1])
device = sys.argv[2] if len(sys.argv) > 2 else 'cuda'

from PIL import Image  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: E402

from image_transforms.masking import GroundedSAM, occlude_objects  # noqa: E402

new = GroundedSAM.__new__(GroundedSAM)   # detector only: occlusion needs boxes, not SAM
det = 'IDEA-Research/grounding-dino-base'
new.device, new.box_threshold, new.text_threshold = device, .25, .25
new.detector_processor = AutoProcessor.from_pretrained(det)
new.detector = AutoModelForZeroShotObjectDetection.from_pretrained(det).to(device).eval()

tiny_id = 'IDEA-Research/grounding-dino-tiny'     # independent checker (same as validate_occlusion.py)
tiny = types.SimpleNamespace(
    detector_processor=AutoProcessor.from_pretrained(tiny_id), device=device, text_threshold=.25, box_threshold=.35,
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(tiny_id).to(device).eval())


def outside(boxes, changed):
    n = 0
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
        if changed[y0:y1, x0:x1].sum() / max(1, (x1 - x0) * (y1 - y0)) < .5:
            n += 1
    return n


rows = []
for f in sorted(glob.glob(str(merged / 'metamorphic_eval/chunk_*/run/data.p'))):
    for d in pickle.load(open(f, 'rb')):
        for t, m in enumerate(d.get('unit_test_metadata', [])):
            if m.get('transformation') != 'occlude_bbox':
                continue
            item = m['parameters']['item']
            original = Image.open(m['original_image_path']).convert('RGB')
            row = dict(questionId=d['questionId'], item=item, question=d['question'])
            try:
                out = occlude_objects(original, item, segmenter=new)
            except ValueError as error:
                row.update(status='rejected', reason=str(error))
                rows.append(row); print(row, flush=True)
                continue
            changed = (np.abs(np.asarray(original, dtype=np.int16) - np.asarray(out, dtype=np.int16)).sum(axis=2) > 0)
            row.update(status='kept', filled_fraction=round(float(changed.mean()), 3))
            for thr in (.35, .25):
                tiny.box_threshold = thr
                row[f'tiny_outside@{thr}'] = outside(GroundedSAM.detect_boxes(tiny, out, item), changed)
            rows.append(row); print(row, flush=True)
(merged / 'metamorphic_eval/occlusion_revalidation.json').write_text(json.dumps(rows, indent=1))
kept = [r for r in rows if r['status'] == 'kept']
print(f'\n{len(rows)} occlusion tests: {len(kept)} kept, {len(rows) - len(kept)} rejected by the gate')
for thr in ('0.35', '0.25'):
    print(f'kept tests still detected outside the fill (tiny @{thr}): '
          f'{sum(r[f"tiny_outside@{thr}"] > 0 for r in kept)}/{len(kept)}   (before: see occlusion_validation_raw_v1.json)')
print('mean filled fraction of kept tests:', round(sum(r['filled_fraction'] for r in kept) / max(1, len(kept)), 3))
