#!/usr/bin/env python3
"""Check that occlude_bbox tests really removed the object they claim is gone.

For every occlusion test, re-run Grounding DINO on the OCCLUDED image with the same object word.
If it still finds the object, the expected answer ("no such object") is unreliable.
Usage: python validate_occlusion.py <merged_dir> [device=cpu]
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
device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # noqa: E402

from image_transforms.masking import GroundedSAM  # noqa: E402

det_id = 'IDEA-Research/grounding-dino-tiny'
stub = types.SimpleNamespace(
    detector_processor=AutoProcessor.from_pretrained(det_id), device=device, text_threshold=.25, box_threshold=.35,
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(det_id).to(device).eval())

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
            for name, img in (('original', original), ('occluded', occluded)):
                for thr in (.35, .25):
                    stub.box_threshold = thr
                    counts[f'{name}@{thr}'] = len(GroundedSAM.detect_boxes(stub, img, item))
            rows.append(dict(questionId=d['questionId'], test=t, item=item, relation=m['relation'],
                             question=d['question'], **counts))
            print(rows[-1], flush=True)
(merged / 'metamorphic_eval/occlusion_validation.json').write_text(json.dumps(rows, indent=1))
print(len(rows), 'occlusion tests;', 'still detected @0.35:', sum(r['occluded@0.35'] > 0 for r in rows),
      '@0.25:', sum(r['occluded@0.25'] > 0 for r in rows))
