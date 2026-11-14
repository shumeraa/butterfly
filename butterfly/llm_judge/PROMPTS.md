# Inference prompt pipeline

Prompt version: `gqa-question-only-v3`. Both vLLM and Claude use these templates
from [prompts.py](prompts.py), substituting only the question text. Call 1 is
unchanged. Call 2 returns topic, ordered objects with REFERENT/ANSWER_OPTION roles
and id_features, answer_options, and question_flags. There are still two calls.
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
- For each object, include `id_features`: `COLOR` if its description uses color to identify it, MATERIAL if its description uses material, texture, pattern, or surface appearance. Otherwise use an empty list. Do not infer unmentioned properties.

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
- For each object, include `id_features`: `COLOR` if its description uses color to identify it, MATERIAL if its description uses material, texture, pattern, or surface appearance. Otherwise use an empty list. Do not infer unmentioned properties.

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
EXISTW: Asks whether something exists or is present.
CATW: References a category, group, or collection of objects.
COMPARE: Compares properties of multiple objects.
ACTW: Actions or interactions (wearing, holding, eating, riding).
MULTI_CONDITION: Requires evaluating multiple conditions joined by AND or OR, including "both"/"either" constructions. Include conjunctions or disjunctions in object-identifying clauses too. Do not flag simple alternative answers such as "red or blue" or "an apple or a banana". The words "both"/"either" count only when they express multiple conditions, not just because they appear.

Examples:
Question: What color is the apple?
{"topic":"COLOR","objects":[{"name":"apple","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["COLORW"]}

Question: What is the small girl wearing?
{"topic":"OBJECT","objects":[{"name":"small girl","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["ACTW"]}

Question: What is beside the cup: an apple, a banana, or an orange?
{"topic":"OBJECT","objects":[{"name":"cup","role":"REFERENT","id_features":[]},{"name":"apple","role":"ANSWER_OPTION","id_features":[]},{"name":"banana","role":"ANSWER_OPTION","id_features":[]},{"name":"orange","role":"ANSWER_OPTION","id_features":[]}],"answer_options":["apple","banana","orange"],"question_flags":[]}

Question: What color is the apple: red, green, or blue?
{"topic":"COLOR","objects":[{"name":"apple","role":"REFERENT","id_features":[]}],"answer_options":["red","green","blue"],"question_flags":["COLORW"]}

Question: What is the man who is both wearing a hat and holding a cup doing?
{"topic":"ACTION","objects":[{"name":"man","role":"REFERENT","id_features":[]},{"name":"hat","role":"REFERENT","id_features":[]},{"name":"cup","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["ACTW","MULTI_CONDITION"]}

Question: What color is the object that is either above the box or below the shelf?
{"topic":"COLOR","objects":[{"name":"object","role":"REFERENT","id_features":[]},{"name":"box","role":"REFERENT","id_features":[]},{"name":"shelf","role":"REFERENT","id_features":[]}],"answer_options":[],"question_flags":["COLORW","UD","UD_ID","MULTI_CONDITION"]}

Return ONLY valid JSON with keys "topic", "objects", "answer_options", and "question_flags". No explanation.
```

## Deterministic transformation rules

Rules use only the validated answer type, topic, objects with roles and
`id_features`, `answer_options`, and the 16 declared question flags. Gold detailed
types never affect labels. `I` means invariant, `C` equivariant, and `U` unsafe.
Invalid or missing fields produce evaluation errors, blank labels, and warnings;
older predictions need explicit object roles, answer_options, and the updated flags.
No schema fields or labels are guessed.

### Global MR1 transformations

Apply blockers before answer dependencies:

| Transformation | Rule |
| --- | --- |
| `hflip` | U for TEXT, BODY, or LR_ID; otherwise C for BINARY with LR_ANSWER, U for OPEN with LR_ANSWER; otherwise I. |
| `vflip` | U for TEXT, SUPPORT, or UD_ID; otherwise C for BINARY with UD_ANSWER, U for OPEN with UD_ANSWER; otherwise I. |
| `grayscale` | U for COLORW, COMPARE, or topic COLOR; otherwise I. |
| `blur` | U for TEXT, MATW, ACTW, COMPARE, topic MATERIAL/ACTION, or OPEN topic OBJECT; otherwise I. |
| `noise` | Same rule as blur. |

If MULTI_CONDITION is present, downgrade any otherwise C result to U. This flag
covers conjunctions and disjunctions of conditions, including both/either, but
not simple offered answers such as red or blue. Generic LR/UD flags remain
descriptive and do not trigger equivariance by themselves.

`rot180` has no independent blockers. Derive it from the final hflip/vflip labels:

| hflip / vflip combination | rot180 |
| --- | --- |
| Either U | U |
| Both C | U (possible cancellation) |
| Exactly one C and the other I | C |
| Both I | I |

### Local segment_mr1

Only REFERENT objects receive local segmentation results. ANSWER_OPTION objects
remain in the ordered objects list for provenance but receive no local edit rows.
The first listed object (index 0) remains the BINARY target: every local result
for a BINARY REFERENT at another index is U. OPEN evaluates every REFERENT.

First check the existing segmentation blockers: EXISTW, TEXT, BODY, ACTW, or
topic ACTION makes every local result U. Otherwise reuse global grayscale, blur,
and noise with the conservative mapping I -> I, C -> U, U -> U. Additionally,
COLOR in the edited object's own id_features blocks local grayscale; MATERIAL
blocks local blur/noise. These identity guards are object-specific.

The three local result names are `segment_mr1_grayscale`, `segment_mr1_blur`,
and `segment_mr1_noise`. They are the only local edit family. The evaluator emits
six global rows per question and three local rows per REFERENT, with no local
rows for an empty or answer-option-only object list.

### Exports and review

Inference review CSV includes answer_options as a JSON cell, alongside objects
(including role and id_features) and question_flags. CPU evaluation accepts JSONL,
CSV, or corrected Excel input and preserves source fields and inference diagnostics.

The raw workbook contains Questions, GlobalLabels, LocalEdits, and Warnings;
the review workbook contains Review, LocalEditReview, and Legend. Questions include
normalized_answer_options. Reviews display object roles and answer choices. Local
rows include the original object index, object_role, target restriction, specific
segment_mr1 transformation, label, and reason. Invalid inputs have explicit errors;
identifiable non-option objects retain blank local error rows. Labels remain
predictions from wording, not visually verified answers.
