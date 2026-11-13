# Inference prompt pipeline

Prompt version: `gqa-question-only-v2`. Both vLLM and Claude use these templates from
[prompts.py](prompts.py), replacing `{question}` with the question text only.

1. **Call 1:** Classify answer format as `BINARY` or `OPEN`.
2. **Call 2:** Use the matching BINARY or OPEN prompt to extract topic, ordered
   objects with `id_features`, and `question_flags`. Each call is independent;
   Call 1 selects the template, and Call 2 receives the original question.
3. **Validate and save:** Retain raw responses, parsed predictions, errors, and
   repair flags. An invalid Call 1 skips Call 2. Save JSONL, CSV, summary,
   and run metadata. Transformation rules and Excel exports run separately on CPU.

Images, answers, gold GQA types, and programs are excluded from model input.

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
- Do not list alternatives in a final "X or Y?" clause as objects.
- Use an empty list if none exist.
- For each object, include `id_features`: `COLOR` if its description uses color to identify it, MATERIAL if its description uses material, texture, pattern, or surface appearance. Otherwise use an empty list. Do not infer unmentioned properties.

3. QUESTION_FLAGS: List all applicable features of the question wording:
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
AND: Requires multiple conditions to both be true.

Examples:
Question: Is the bowl to the right of the green apple?
{"topic":"POSITION","objects":[{"name":"bowl","id_features":[]},{"name":"green apple","id_features":["COLOR"]}],"question_flags":["LR","LR_ANSWER","COLORW"]}

Question: Is the man holding a blue umbrella?
{"topic":"ACTION","objects":[{"name":"man","id_features":[]},{"name":"blue umbrella","id_features":["COLOR"]}],"question_flags":["ACTW","COLORW"]}

Return ONLY valid JSON with keys "topic", "objects", and "question_flags". No explanation.
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
- Do not list alternatives in a final "X or Y?" clause as objects.
- Use an empty list if none exist.
- For each object, include `id_features`: `COLOR` if its description uses color to identify it, MATERIAL if its description uses material, texture, pattern, or surface appearance. Otherwise use an empty list. Do not infer unmentioned properties.

3. QUESTION_FLAGS: List all applicable features of the question wording:
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
AND: Requires multiple conditions to both be true.

Examples:
Question: What color is the apple?
{"topic":"COLOR","objects":[{"name":"apple","id_features":[]}],"question_flags":["COLORW"]}

Question: What is the small girl wearing?
{"topic":"OBJECT","objects":[{"name":"small girl","id_features":[]}],"question_flags":["ACTW"]}

Return ONLY valid JSON with keys "topic", "objects", and "question_flags". No explanation.
```

## How transformation labels are calculated

[evaluate_rules.py](evaluate_rules.py) applies deterministic rules to the validated
answer type, topic, ordered objects, each object's `id_features`, and the question
flags above. It uses no images, gold GQA labels, or additional LLM calls.

- **Invariant (`I`):** The rule predicts that the answer stays the same.
- **Equivariant (`C`):** The rule predicts a change in the answer; it does not
  calculate the new answer.
- **Unsafe (`U`):** The rule does not establish a reliable answer relationship
  for the transformation.

These labels are **predicted by rules, not visually verified**. In the tables,
flag names mean that the flag is present; `topic` is the predicted topic.
Apply conditions in the listed order: the first matching condition wins.

### Global transformations

| Transformation | Ordered decision rule                                                                                                                                                           | English translation                                                                                                                                                                                                                               |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hflip`      | `U` if `TEXT`, `BODY`, or `LR_ID`; otherwise, if `LR_ANSWER`, assign `C` for BINARY and `U` for OPEN; otherwise `I`.                                            | Left/right object identification is unsafe. Only a left/right answer relation can trigger equivariance.                                                                                                                                           |
| `vflip`      | `U` if `TEXT`, `SUPPORT`, or `UD_ID`; otherwise, if `UD_ANSWER`, assign `C` for BINARY and `U` for OPEN; otherwise `I`.                                         | Up/down object identification is unsafe. Only an up/down answer relation can trigger equivariance.                                                                                                                                                |
| `rot180`     | `U` if `TEXT`, `SUPPORT`, `LR_ID`, or `UD_ID`; otherwise, if `LR_ANSWER` or `UD_ANSWER`, assign `C` for BINARY and `U` for OPEN; otherwise `I`.             | Object identification on either axis is unsafe and overrides answer relations on either axis.                                                                                                                                                     |
| `grayscale`  | `U` if `COLORW`, `COMPARE`, or `topic == COLOR`; otherwise `I`.                                                                                                       | Removing color is unsafe if the question mentions color, compares properties of multiple objects, or asks about color. Otherwise, predict the same answer.                                                                                        |
| `blur`       | `U` if any of `TEXT`, `MATW`, `ACTW`, `COMPARE` is present, if `topic` is MATERIAL or ACTION, or if the question is OPEN with `topic == OBJECT`; otherwise `I`. | Blurring is unsafe if the question involves readable text, material or surface properties, actions, or comparisons; asks about material or action; or asks for an object/person name in an open-ended answer. Otherwise, predict the same answer. |
| `noise`      | Uses the same rule as `blur`.                                                                                                                                                 | Adding noise uses the same checks as blurring: mark it unsafe for text, material or surface properties, actions, comparisons, material/action topics, or open-ended object/person naming. Otherwise, predict the same answer.                     |

For example, `TEXT` makes `hflip` unsafe even when a BINARY question also has
`LR_ANSWER`; the blocker takes precedence over the equivariant spatial rule.

Generic `LR` and `UD` flags remain descriptive; they do not trigger spatial
labels alone. Older predictions need the new role flags before reevaluation.

### Local transformations, evaluated separately for each object

The first listed object (index 0) is the target. **For BINARY questions, all three
local edits on every other object are always `U`**, overriding the rules below.
For OPEN questions, apply the rules to every mentioned object.

`color_id` means COLOR is in the edited object's own `id_features`;
`material_id` means MATERIAL is in that object's `id_features`. Global `COLORW`
and `MATW` flags do not substitute for these object identity features.

| Transformation  | Ordered decision rule                                                                                                                                                                                                            | English translation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `color_shift` | `U` if `EXISTW`, `topic == EXISTS`, `COMPARE`, or `color_id`; otherwise, if `topic == COLOR` and this is the first object, assign `C` only for BINARY without `AND`, and `U` otherwise; otherwise `I`.       | Subject to the per-object restriction above, changing an object's color is unsafe if the question asks about existence, compares properties, or uses color to identify that object. Otherwise, if the question asks about color and this is the first object, predict an answer change only for a two-choice question that does not require multiple conditions to both be true; mark other cases in that branch unsafe. Otherwise, predict the same answer.                                                  |
| `texture`     | `U` if `EXISTW`, `topic == EXISTS`, `COMPARE`, or `material_id`; otherwise, if `topic == MATERIAL` and this is the first object, assign `C` only for BINARY without `AND`, and `U` otherwise; otherwise `I`. | Subject to the per-object restriction above, changing an object's texture is unsafe if the question asks about existence, compares properties, or uses material, texture, pattern, or surface appearance to identify that object. Otherwise, if the question asks about material and this is the first object, predict an answer change only for a two-choice question that does not require multiple conditions to both be true; mark other cases in that branch unsafe. Otherwise, predict the same answer. |
| `segment_mr1` | `U` if any of `EXISTW`, `TEXT`, `BODY`, `ACTW` is present, or if `topic == ACTION`; otherwise `I`.                                                                                                                 | Subject to the per-object restriction above, this edit is unsafe if the question asks whether something exists, involves readable text or left/right body parts, or mentions or asks about actions. Otherwise, predict the same answer.                                                                                                                                                                                                                                                                       |

Equivariant `color_shift` and `texture` results carry
`requires_edit_value_check=true`. The actual edited value must map to the other
offered answer choice. A downstream executor must downgrade the label to unsafe
if that check fails. The rules do not invent edited values or answer options.

### Selective occlusion scenarios

These are question-level scenarios, separate from the three per-object edits.

| Transformation            | Decision rule                                                                                                                                                                           | English translation                                                                                                                                                                                                                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `occlusion_mentioned`   | `U` whenever at least one mentioned object exists. If there are none, record `not_applicable` with a blank label.                                                                   | Hiding an object mentioned in the question is always unsafe. If the question mentions no objects, this scenario does not apply and receives no label.                                                                                                                                      |
| `occlusion_unmentioned` | `U` if any of `EXISTW`, `CATW`, `COMPARE`, `ACTW` is present, if `topic` is EXISTS, SCENE, or ACTION, or if the question is OPEN with `topic == OBJECT`; otherwise `I`. | Hiding an unmentioned object is unsafe if the question asks about existence, references a category or collection, compares properties, mentions actions, asks about the scene or an action, or asks for an object/person name in an open-ended answer. Otherwise, predict the same answer. |

Unmentioned-object occlusion assumes an object could be selected later; the
pipeline has not located one. If there are no mentioned objects, there are no
per-object local edit rows.

### Invalid predictions and traceability

Missing fields, malformed JSON, invalid semantic labels, or an explicitly false
inference validity flag produce `evaluation_error` and blank transformation
labels, plus a diagnostic in `Warnings`. They never default to invariant or
unsafe. Each evaluated transformation includes a `reason_code` in the raw
workbook to identify the rule that produced its label.
