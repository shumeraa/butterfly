"""Independent, versioned prompt for answer-rule enrichment."""
import json

PROMPT_VERSION = 'gqa-answer-rules-v2'
TRANSFORMATIONS = {
    'hflip': 'Reflect the entire image horizontally.',
    'vflip': 'Reflect the entire image vertically.',
    'rot180': 'Rotate the entire image by 180 degrees.',
    'fill_red': 'Segment object index 0 (the target REFERENT) and fill its mask uniformly red, preserving shape and location.',
    'fill_blue': 'Segment object index 0 (the target REFERENT) and fill its mask uniformly blue, preserving shape and location.',
    'occlude_bbox': 'Cover the selected REFERENT object bounding box with an opaque black rectangle, including background inside the box.',

}

PROMPT = '''Determine how an image transformation changes the answer to a visual question.
Keep the question unchanged. You have no image or original answer; do not guess
them. The supplied analysis may be incorrect.

Question: {question}
Answer type: {answer_type}
Call 2 analysis: {call2_json}
Transformation and definition: {transformation_definition}

Choose:
- MAP: map each possible original answer to all allowed transformed answers.
  This covers swapping, preserving, and multiple possible outputs.
- CONSTANT: one transformed answer regardless of the original answer.
- UNRESOLVED: no useful relationship can be established, or transformed answers
  may fall outside the established answer domain.

Use yes/no for yes/no questions; preserve explicit answer-choice wording.
Do not assume "not left" means "right" or that offered choices are exhaustive.
Track changes to objects selected by identifying descriptions.
Prefer unconditional rules. State necessary assumptions; conditions needing
image evidence are unverified. If every input permits every answer, use UNRESOLVED.

Return only JSON:
{
  "mode": "MAP | CONSTANT | UNRESOLVED",
  "answer_domain": ["..."],
  "mapping": {"original answer": ["allowed transformed answers"]},
  "constant_answer": null,
  "conditions": [{"condition": "...", "verified_from_input": false}],
  "reason": "brief explanation"
}
For MAP, cover every answer in the domain.
Use mapping: {} for CONSTANT or UNRESOLVED.
Set constant_answer only for CONSTANT. Use conditions: [] if none are needed.

Example 1 input
Question: Is the ball left of the box?
Answer type: BINARY
Call 2 analysis: {"topic":"POSITION","objects":[{"name":"ball","role":"REFERENT","id_features":[]},{"name":"box","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["LR","LR_ANSWER"]}
Transformation and definition: hflip — Reflect the entire image horizontally.

Example 1 output
{"mode":"MAP","answer_domain":["yes","no"],"mapping":{"yes":["no"],"no":["yes","no"]},"constant_answer":null,"conditions":[],"reason":"Left becomes right. An original no may mean right or horizontally aligned, so it does not determine one transformed answer."}

Example 2 input
Question: Is the car red or blue?
Answer type: BINARY
Call 2 analysis: {"topic":"COLOR","objects":[{"name":"car","role":"REFERENT","id_features":[]}],"answer_options":["red","blue"],"question_flags":["COLORW"]}
Transformation and definition: fill_red — Segment object index 0 (the target REFERENT) and fill its mask uniformly red, preserving shape and location.

Example 2 output
{"mode":"CONSTANT","answer_domain":["red","blue"],"mapping":{},"constant_answer":"red","conditions":[],"reason":"The target car is filled uniformly red, so either original color yields red."}
'''


def render(question, answer_type, analysis, transformation):
    # Single formatting pass: question content cannot introduce substitutions.
    values = {'question': question, 'answer_type': answer_type,
              'call2_json': json.dumps(analysis, ensure_ascii=False),
              'transformation_definition': transformation + ' — ' + TRANSFORMATIONS[transformation]}
    # Only replace placeholders in the template, not in inserted values.
    import re
    return re.sub(r'\{(question|answer_type|call2_json|transformation_definition)\}',
                  lambda match: values[match[1]], PROMPT)
