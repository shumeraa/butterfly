#!/usr/bin/env python3
"""Check that occlude_bbox tests really removed the object they claim is gone.

For every occlusion test, re-run Grounding DINO on the OCCLUDED image with the same object word.
If it still finds the object, the expected answer ("no such object") is unreliable.
Usage: python validate_occlusion.py <merged_dir> [device=auto|cuda|cpu]
Writes <merged>/metamorphic_eval/occlusion_validation.json
"""
import glob
import json
import pickle
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
merged = Path(sys.argv[1])
device = sys.argv[2] if len(sys.argv) > 2 else 'auto'   # auto = cuda when available

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: E402

from image_transforms.masking import GroundedSAM  # noqa: E402

if device == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, flush=True)
det_id = 'IDEA-Research/grounding-dino-tiny'
stub = types.SimpleNamespace(
    detector_processor=AutoProcessor.from_pretrained(det_id), device=device, text_threshold=.25, box_threshold=.35,
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(det_id).to(device).eval())

import numpy as np  # noqa: E402


def outside_count(boxes, changed):
    """Detections that are NOT (mostly) inside the region we filled: a box that sits on the white
    patch is the detector reacting to our edit, not a surviving object."""
    n = 0
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))
        area = max(1, (x1 - x0) * (y1 - y0))
        if changed[y0:y1, x0:x1].sum() / area < .5:
            n += 1
    return n


rows = []
for f in sorted(glob.glob(str(merged / 'metamorphic_eval/chunk_*/run/data.p'))):
    for d in pickle.load(open(f, 'rb')):
        for t, m in enumerate(d.get('unit_test_metadata', [])):
            if m.get('transformation') != 'occlude_bbox':
                continue
            item = m['parameters']['item']
            occluded = Image.open(d['unit_test_images'][t][0]).convert('RGB')
            original = Image.open(m['original_image_path']).convert('RGB')
            counts = {}
            changed = (np.abs(np.asarray(original, dtype=np.int16) - np.asarray(occluded.resize(original.size), dtype=np.int16))
                       .sum(axis=2) > 0)
            for name, img in (('original', original), ('occluded', occluded)):
                for thr in (.35, .25):
                    stub.box_threshold = thr
                    boxes = GroundedSAM.detect_boxes(stub, img, item)
                    counts[f'{name}@{thr}'] = len(boxes)
                    if name == 'occluded':
                        counts[f'outside@{thr}'] = outside_count(boxes, changed)
            rows.append(dict(questionId=d['questionId'], test=t, item=item, relation=m['relation'],
                             question=d['question'], **counts))
            print(rows[-1], flush=True)
(merged / 'metamorphic_eval/occlusion_validation.json').write_text(json.dumps(rows, indent=1))
print(len(rows), 'occlusion tests;', 'still detected @0.35:', sum(r['occluded@0.35'] > 0 for r in rows),
      '@0.25:', sum(r['occluded@0.25'] > 0 for r in rows))
print('still detected OUTSIDE the filled area @0.35:', sum(r['outside@0.35'] > 0 for r in rows),
      '@0.25:', sum(r['outside@0.25'] > 0 for r in rows))
