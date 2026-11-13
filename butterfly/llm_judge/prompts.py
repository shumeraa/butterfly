"""Versioned, exact question-only templates from plan.md."""

PROMPT_VERSION = "gqa-question-only-v1"

CALL1 = """Question: {question}

Can the answer only be one of two things? This includes yes/no, a choice written in the question ("red or green", "an apple or a banana"), and left/right or top/bottom.

Examples:
Question: Is the bowl to the right of the green apple?
BINARY

Question: What color is the apple?
OPEN

Reply with one word: BINARY or OPEN."""

CALL2_BINARY = """Question: {question}

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
LR: Left/right spatial references.
UD: Up/down spatial references.
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
{"topic":"POSITION","objects":[{"name":"bowl","id_features":[]},{"name":"green apple","id_features":["COLOR"]}],"question_flags":["LR","COLORW"]}

Question: Is the man holding a blue umbrella?
{"topic":"ACTION","objects":[{"name":"man","id_features":[]},{"name":"blue umbrella","id_features":["COLOR"]}],"question_flags":["ACTW","COLORW"]}

Return ONLY valid JSON with keys "topic", "objects", and "question_flags". No explanation."""

CALL2_OPEN = """Question: {question}

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
LR: Left/right spatial references.
UD: Up/down spatial references.
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

Return ONLY valid JSON with keys "topic", "objects", and "question_flags". No explanation."""

TEMPLATES = {"call1": CALL1, "BINARY": CALL2_BINARY, "OPEN": CALL2_OPEN}


def render(template, question):
    return template.replace("{question}", question)
