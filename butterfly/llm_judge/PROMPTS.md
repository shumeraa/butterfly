# Inference prompt pipeline

Prompt version: `gqa-question-only-v4`. Both vLLM and Claude use these templates
from [prompts.py](prompts.py), substituting only the question text. Call 1 is
unchanged. Call 2 returns topic, ordered objects with REFERENT/ANSWER_OPTION roles
and id_features, answer_options, and question_flags. These classification templates are followed by an integrated answer-rule Call 3
in the vLLM runner for eligible transformations.
Images, gold answers, detailed types, and programs are excluded from model input.
Raw responses and validation diagnostics are retained.

## Call 1: answer format

```text
Question: {question}

Can the answer only be one of two things? This includes yes/no, a choice written in the question ("red or green", "an apple or a banana"), and left/right or top/bottom.

Examples:
Question: Is the bowl to the right of the green apple?
BINARY

Question: What color is the apple?
OPEN

Reply with one word: BINARY or OPEN.
```

## Call 2: BINARY branch

```text
Question: {question}

Analyze this visual question using only its wording, not an image or answer.

1. TOPIC: This question has a yes/no or two-choice answer. What does it check? Pick one:
POSITION: where something is (left/right, above/below, on, in front of)
COLOR: the color of something
MATERIAL: what something is made of, its pattern, or its surface (shiny, wet)
EXISTS: whether something is there at all
IDENTITY: what kind of thing something is
ACTION: what someone is doing, holding, wearing, or eating
SCENE: the weather, the place, or the time of day
OTHER: none of these

2. OBJECTS: List objects in the order they first appear.
- Keep identifying descriptions ("green apple", "man on the left").
- Exclude properties the question asks about rather than uses to identify an object.
- Exclude pronouns and generic words like image, picture, photo.
- Include object/person answer choices in objects, including alternatives in a final "X or Y?" clause.
- Give each object a role: REFERENT for an entity the question refers to or asks about; ANSWER_OPTION for an object/person offered as a possible answer. Keep objects in textual order, including answer options.
- Non-object choices (colors, materials, actions, directions, etc.) belong only in answer_options, not as separate objects.
- Use an empty list if none exist.
- For each object, include `id_features`: COLOR if its description uses color to identify it; MATERIAL for material, texture, pattern, or surface appearance; SIZE for size, width, height, length, or proportions; SHAPE for shape or geometric form. Include all applicable features. Properties being asked about are not identifying features. Otherwise use an empty list. Do not infer unmentioned properties.

3. ANSWER_OPTIONS: List explicit answer choices as strings in textual order, including both object and non-object "X or Y" choices. Use [] if no explicit choices are offered; do not invent yes/no options. Object/person choices must also appear in objects with role ANSWER_OPTION. An AND/OR condition about referents is not an answer-choice list ("Is there either a cat or a dog?" asks about existence; both are REFERENT objects and answer_options is []).

4. QUESTION_FLAGS: List all applicable features of the question wording:
LR: Any left/right spatial reference. Include this alongside LR_ID and/or LR_ANSWER when applicable.
UD: Any up/down spatial reference. Include this alongside UD_ID and/or UD_ANSWER when applicable.
LR_ID: A left/right relation used to identify or select an object, rather than the relation being asked about. For example, "What is the girl to the left of the frisbee doing?"
LR_ANSWER: A left/right relation being asked about or supplied as an answer choice. For example, "Is the girl to the left of the frisbee?"
UD_ID: An up/down relation used to identify or select an object, rather than the relation being asked about. For example, "What color is the ball above the box?"
UD_ANSWER: An up/down relation being asked about or supplied as an answer choice. For example, "Is the ball above the box?"
Include both _ID and _ANSWER flags when both roles occur, even for different objects. For example, "Is the girl to the left of the frisbee above the boy?" has LR, LR_ID, UD, and UD_ANSWER.
SUPPORT: Support, containment, or physical orientation (on, under, inside, sitting, hanging, lying).
TEXT: Readable text, letters, numbers, signs, or logos.
BODY: Left/right body parts.
COLORW: Color words or references to color.
MATW: Material, texture, pattern, or surface properties.
SIZEW: References size, width, height, length, or proportions, whether asked about or used to identify an object.
SHAPEW: References shape or geometric form, whether asked about or used to identify an object.
COUNTW: Asks about the number of objects, including count comparisons or a specified quantity. Do not flag a numeral used only to identify an object, such as "player number 3".
EXISTW: Asks whether something exists or is present.
CATW: References a category, group, or collection of objects.
COMPARE: Compares properties of multiple objects.
ACTW: Actions or interactions (wearing, holding, eating, riding).
MULTI_CONDITION: Requires evaluating multiple conditions joined by AND or OR, including "both"/"either" constructions. Include conjunctions or disjunctions in object-identifying clauses too. Do not flag simple alternative answers such as "red or blue" or "an apple or a banana". The words "both"/"either" count only when they express multiple conditions, not just because they appear.

Examples:
Question: Is the bowl to the right of the green apple?
{"topic":"POSITION","objects":[{"name":"bowl","role":"REFERENT","id_features":[]},{"name":"green apple","role":"REFERENT","id_features":["COLOR"]}],"answer_options":[],"question_flags":["LR","LR_ANSWER","COLORW"]}

Question: Is the man holding a blue umbrella?
{"topic":"ACTION","objects":[{"name":"man","role":"REFERENT","id_features":[]},{"name":"blue umbrella","role":"REFERENT","id_features":["COLOR"]}],"answer_options":[],"question_flags":["ACTW","COLORW"]}

Question: Is the fruit an apple or a banana?
{"topic":"IDENTITY","objects":[{"name":"fruit","role":"REFERENT","id_features":[]},{"name":"apple","role":"ANSWER_OPTION","id_features":[]},{"name":"banana","role":"ANSWER_OPTION","id_features":[]}],"answer_options":["apple","banana"],"question_flags":[]}

Question: Is the apple red or green?
{"topic":"COLOR","objects":[{"name":"apple","role":"REFERENT","id_features":[]}],"answer_options":["red","green"],"question_flags":["COLORW"]}

Question: Is the ball both red and above the box?
{"topic":"POSITION","objects":[{"name":"ball","role":"REFERENT","id_features":[]},{"name":"box","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["COLORW","UD","UD_ANSWER","MULTI_CONDITION"]}

Question: Is there either a cat or a dog?
{"topic":"EXISTS","objects":[{"name":"cat","role":"REFERENT","id_features":[]},{"name":"dog","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["EXISTW","MULTI_CONDITION"]}

Question: Is the tall bottle wider than the round jar?
{"topic":"OTHER","objects":[{"name":"tall bottle","role":"REFERENT","id_features":["SIZE"]},{"name":"round jar","role":"REFERENT","id_features":["SHAPE"]}],"answer_options":[],"question_flags":["SIZEW","SHAPEW","COMPARE"]}

Question: Is the plate round?
{"topic":"OTHER","objects":[{"name":"plate","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["SHAPEW"]}

Question: Are there three apples?
{"topic":"EXISTS","objects":[{"name":"apples","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["EXISTW","COUNTW"]}

Return ONLY valid JSON with keys "topic", "objects", "answer_options", and "question_flags". No explanation.
```

## Call 2: OPEN branch

```text
Question: {question}

Analyze this visual question using only its wording, not an image or answer.

1. TOPIC: What kind of word is the answer? Pick one:
COLOR: a color ("red")
MATERIAL: a material or pattern ("wood", "striped")
OBJECT: the name of a thing or person ("umbrella", "man")
ACTION: an activity ("playing", "sitting")
POSITION: a side or place in the image ("left", "top")
SCENE: weather or a place ("sunny", "kitchen")
OTHER: none of these

2. OBJECTS: List objects in the order they first appear.
- Keep identifying descriptions ("green apple", "man on the left").
- Exclude properties the question asks about rather than uses to identify an object.
- Exclude pronouns and generic words like image, picture, photo.
- Include object/person answer choices in objects, including alternatives in a final "X or Y?" clause.
- Give each object a role: REFERENT for an entity the question refers to or asks about; ANSWER_OPTION for an object/person offered as a possible answer. Keep objects in textual order, including answer options.
- Non-object choices (colors, materials, actions, directions, etc.) belong only in answer_options, not as separate objects.
- Use an empty list if none exist.
- For each object, include `id_features`: COLOR if its description uses color to identify it; MATERIAL for material, texture, pattern, or surface appearance; SIZE for size, width, height, length, or proportions; SHAPE for shape or geometric form. Include all applicable features. Properties being asked about are not identifying features. Otherwise use an empty list. Do not infer unmentioned properties.

3. ANSWER_OPTIONS: List explicit answer choices as strings in textual order, including both object and non-object "X or Y" choices. Use [] if no explicit choices are offered; do not invent yes/no options. Object/person choices must also appear in objects with role ANSWER_OPTION. An AND/OR condition about referents is not an answer-choice list ("Is there either a cat or a dog?" asks about existence; both are REFERENT objects and answer_options is []).

4. QUESTION_FLAGS: List all applicable features of the question wording:
LR: Any left/right spatial reference. Include this alongside LR_ID and/or LR_ANSWER when applicable.
UD: Any up/down spatial reference. Include this alongside UD_ID and/or UD_ANSWER when applicable.
LR_ID: A left/right relation used to identify or select an object, rather than the relation being asked about. For example, "What is the girl to the left of the frisbee doing?"
LR_ANSWER: A left/right relation being asked about or supplied as an answer choice. For example, "Is the girl to the left of the frisbee?"
UD_ID: An up/down relation used to identify or select an object, rather than the relation being asked about. For example, "What color is the ball above the box?"
UD_ANSWER: An up/down relation being asked about or supplied as an answer choice. For example, "Is the ball above the box?"
Include both _ID and _ANSWER flags when both roles occur, even for different objects. For example, "Is the girl to the left of the frisbee above the boy?" has LR, LR_ID, UD, and UD_ANSWER.
SUPPORT: Support, containment, or physical orientation (on, under, inside, sitting, hanging, lying).
TEXT: Readable text, letters, numbers, signs, or logos.
BODY: Left/right body parts.
COLORW: Color words or references to color.
MATW: Material, texture, pattern, or surface properties.
SIZEW: References size, width, height, length, or proportions, whether asked about or used to identify an object.
SHAPEW: References shape or geometric form, whether asked about or used to identify an object.
COUNTW: Asks about the number of objects, including count comparisons or a specified quantity. Do not flag a numeral used only to identify an object, such as "player number 3".
EXISTW: Asks whether something exists or is present.
CATW: References a category, group, or collection of objects.
COMPARE: Compares properties of multiple objects.
ACTW: Actions or interactions (wearing, holding, eating, riding).
MULTI_CONDITION: Requires evaluating multiple conditions joined by AND or OR, including "both"/"either" constructions. Include conjunctions or disjunctions in object-identifying clauses too. Do not flag simple alternative answers such as "red or blue" or "an apple or a banana". The words "both"/"either" count only when they express multiple conditions, not just because they appear.

Examples:
Question: What color is the apple?
{"topic":"COLOR","objects":[{"name":"apple","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["COLORW"]}

Question: What is the small girl wearing?
{"topic":"OBJECT","objects":[{"name":"small girl","role":"REFERENT","id_features":["SIZE"]}],"answer_options":[],"question_flags":["ACTW","SIZEW"]}

Question: What is beside the cup: an apple, a banana, or an orange?
{"topic":"OBJECT","objects":[{"name":"cup","role":"REFERENT","id_features":[]},{"name":"apple","role":"ANSWER_OPTION","id_features":[]},{"name":"banana","role":"ANSWER_OPTION","id_features":[]},{"name":"orange","role":"ANSWER_OPTION","id_features":[]}],"answer_options":["apple","banana","orange"],"question_flags":[]}

Question: What color is the apple: red, green, or blue?
{"topic":"COLOR","objects":[{"name":"apple","role":"REFERENT","id_features":[]}],"answer_options":["red","green","blue"],"question_flags":["COLORW"]}

Question: What is the man who is both wearing a hat and holding a cup doing?
{"topic":"ACTION","objects":[{"name":"man","role":"REFERENT","id_features":[]},{"name":"hat","role":"REFERENT","id_features":[]},{"name":"cup","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["ACTW","MULTI_CONDITION"]}

Question: What color is the object that is either above the box or below the shelf?
{"topic":"COLOR","objects":[{"name":"object","role":"REFERENT","id_features":[]},{"name":"box","role":"REFERENT","id_features":[]},{"name":"shelf","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["COLORW","UD","UD_ID","MULTI_CONDITION"]}

Question: What color is the tall bottle beside the round jar?
{"topic":"COLOR","objects":[{"name":"tall bottle","role":"REFERENT","id_features":["SIZE"]},{"name":"round jar","role":"REFERENT","id_features":["SHAPE"]}],"answer_options":[],"question_flags":["COLORW","SIZEW","SHAPEW"]}

Question: What shape is the plate?
{"topic":"OTHER","objects":[{"name":"plate","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["SHAPEW"]}

Question: How many apples are there?
{"topic":"OTHER","objects":[{"name":"apples","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["COUNTW"]}

Return ONLY valid JSON with keys "topic", "objects", "answer_options", and "question_flags". No explanation.
```

## Deterministic transformation rules

Rules use only the validated answer type, topic, objects with roles and
`id_features`, `answer_options`, and the 19 declared question flags. Gold detailed
types never affect labels. `I` means invariant, `C` equivariant, and `U` unsafe.
Invalid or missing fields produce evaluation errors, blank labels, and warnings;
older predictions need explicit object roles, answer_options, and the updated flags.
No schema fields or labels are guessed.

SIZEW, SHAPEW, and COUNTW capture question dependencies; SIZE and SHAPE in
id_features capture object identification. These additions are descriptive in
the current MR1 rules and do not independently change labels. No structured
dependency fields or MR2 rules are introduced here.

### Global MR1 transformations

Apply blockers before answer dependencies:

| Transformation | Rule                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------- |
| `hflip`      | U for TEXT, BODY, or LR_ID; otherwise C for BINARY with LR_ANSWER, U for OPEN with LR_ANSWER; otherwise I.    |
| `vflip`      | U for TEXT, SUPPORT, or UD_ID; otherwise C for BINARY with UD_ANSWER, U for OPEN with UD_ANSWER; otherwise I. |
| `grayscale`  | U for COLORW, COMPARE, or topic COLOR; otherwise I.                                                           |
| `blur`       | U for TEXT, MATW, ACTW, COMPARE, topic MATERIAL/ACTION, or OPEN topic OBJECT; otherwise I.                    |
| `noise`      | Same rule as blur.                                                                                            |

If MULTI_CONDITION is present, downgrade any otherwise C result to U. This flag
covers conjunctions and disjunctions of conditions, including both/either, but
not simple offered answers such as red or blue. Generic LR/UD flags remain
descriptive and do not trigger equivariance by themselves.

`rot180` has no independent blockers. Derive it from the final hflip/vflip labels:

| hflip / vflip combination     | rot180                    |
| ----------------------------- | ------------------------- |
| Either U                      | U                         |
| Both C                        | U (possible cancellation) |
| Exactly one C and the other I | C                         |
| Both I                        | I                         |

### MR2 local transformations

MR2 contains `segment_mr1` (six variants), `fill_red` (Fill Red), `fill_blue`
(Fill Blue), `occlude_bbox` (Occlude BBox), and `asymmetric_scale`.
There are ten local rows per REFERENT and six unchanged global MR1 rows per
question. `transformation_type` groups the segment variants under `segment_mr1`.

Apply shared guards in order:

1. Only REFERENT objects receive local rows. ANSWER_OPTION objects retain their
   original indices but cannot be edited.
2. BINARY permits only object index 0; other REFERENT edits are U.
3. EXISTW, TEXT, BODY, ACTW, or topic ACTION makes every MR2 edit U.

Then apply the transformation rule:

| Transformation                                                       | Deterministic rule                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `segment_mr1_grayscale`                                            | U for object COLOR identity; otherwise I only when global grayscale is I.                                                                                                                                                                                                                                                                                                                                                    |
| `segment_mr1_blur`, `segment_mr1_noise`                          | U for object MATERIAL identity; otherwise I only when the corresponding global rule is I.                                                                                                                                                                                                                                                                                                                                    |
| `segment_mr1_hflip`, `segment_mr1_vflip`, `segment_mr1_rot180` | I only when the corresponding global rule is I; C/U becomes U.                                                                                                                                                                                                                                                                                                                                                               |
| `fill_red`, `fill_blue`                                          | U for object COLOR/MATERIAL identity, MATERIAL/OBJECT/IDENTITY topic, or MATW/COMPARE/CATW/MULTI_CONDITION flags. For COLOR topic and target index 0, C means a constant red/blue answer: OPEN must have no explicit choices or include that color; BINARY must have two distinct explicit choices including that color (case insensitive). Other COLOR cases are U. For other topics, COLORW makes the fill U; otherwise I. |
| `occlude_bbox`                                                     | Cover the object's full bounding box, including background inside it, with opaque black. Deterministically U because it removes the object and context. Review permits I/C/U when a reviewer can establish an answer relationship.                                                                                                                                                                                           |
| `asymmetric_scale`                                                 | U for object SIZE/SHAPE identity; POSITION/OBJECT/IDENTITY topic; or SIZEW, SHAPEW, COMPARE, CATW, LR, UD, LR_ID, LR_ANSWER, UD_ID, UD_ANSWER, SUPPORT flags. Otherwise I.                                                                                                                                                                                                                                                   |

Fill Red and Fill Blue preserve object shape and location and allow constant-answer
equivariance; C does not assert that the original answer was different. Call 3
requests their answer rules automatically when the deterministic label is C.
Segment then MR1 and asymmetric scale retain I/U labels. Empty object lists have
no local rows. Labels are question-only predictions; masks and background plates
still need to correspond to the intended object.

### Exports and review

Inference review CSV includes answer_options as a JSON cell, alongside objects
(including role and id_features) and question_flags. CPU evaluation accepts JSONL,
CSV, or corrected Excel input and preserves source fields and inference diagnostics.

The raw workbook contains Questions, GlobalLabels, LocalEdits, and Warnings;
the review workbook contains Review, LocalEditReview, and Legend. Questions include
normalized_answer_options. Reviews display object roles and answer choices. Local
rows include the original object index, object_role, target restriction, specific
MR2 type, specific transformation, label, and reason. Invalid inputs have explicit errors;
identifiable non-option objects retain blank local error rows. Labels remain
predictions from wording, not visually verified answers.

## Call 3: answer rules

The vLLM runner uses the independent versioned prompt and two examples in
[answer_rule_prompts.py](answer_rule_prompts.py). It runs by default in the same
process, using the loaded model and the same result/checkpoint directory. Each
eligible transformation receives one fresh request with the question, validated
answer type, Call 2 analysis, and transformation definition. It returns MAP,
CONSTANT, or UNRESOLVED with answer domain, optional conditions, and a reason.
No Call 2 conversation history is replayed. Conditions remain unverified model
claims. Fill Red and Fill Blue are selected automatically for eligible local
equivariance, and the color example uses a constant red answer.

### Exact prompt

Prompt version: `gqa-answer-rules-v2`. The template below is copied exactly
from `answer_rule_prompts.py`. Only the four input placeholders are substituted;
`transformation_definition` contains the transformation name and definition.

```text
Determine how an image transformation changes the answer to a visual question.
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
```

### Transformation definitions

- `hflip`: Reflect the entire image horizontally.
- `vflip`: Reflect the entire image vertically.
- `rot180`: Rotate the entire image by 180 degrees.
- `fill_red`: Segment object index 0 (the target REFERENT) and fill its mask uniformly red, preserving shape and location.
- `fill_blue`: Segment object index 0 (the target REFERENT) and fill its mask uniformly blue, preserving shape and location.
- `occlude_bbox`: Cover the selected REFERENT object bounding box with an opaque black rectangle, including background inside the box.
