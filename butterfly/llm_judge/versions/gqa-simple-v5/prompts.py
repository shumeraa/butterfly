"""Versioned, exact question-only prompt from SIMPLE_PROMPTS.md (Part 1)."""

PROMPT_VERSION = "gqa-simple-v5"

PROMPT = """Question: {question}

Analyze this visual question using only its wording, not an image or answer.

1. TARGET: Choose the ONE object an image edit would focus on. Apply in order and stop at the first match:
   a. Asked-about object: if the question asks the color of X, or whether X exists, X is the target.
   b. Grammatical subject: otherwise, the head noun of the main clause.
   c. Only named object: if the question names just one object, it is the target.
   d. Tie: take the first qualifying object in textual order.
   Never choose: an object used only to identify another object ("to the left of the Z"), an answer option ("an apple or a banana"), or an object whose existence the question does not presuppose ("a cup" in "Is the man holding a cup?"; the man is the target). Exception: for existence questions ("Is there a cup?"), the object asked about is the target.
   Use null if no object qualifies or the question is about the whole scene.
   Keep the identifying description ("green apple", "man on the left").
   The edit applies to every instance of the target in the image, not just one.

2. FLAGS: List all that apply.

Color question form. Set one of these only if the question asks the color of the target and contains no other color words (for example, no color word inside an object-identifying clause like "the red car's door"). Never set more than one.
COLOR_QUERY: "What color is X?"
COLOR_VERIFY: "Is X <color>?"
COLOR_CHOOSE: "Is X <c1> or <c2>?"

Left/right:
LR_ID: Left/right used to identify or select an object. "What is the girl to the left of the frisbee doing?"
LR_ANSWER: Left/right being asked about or offered as a choice. "Is the girl to the left of the frisbee?"
LR_SIDE: Subtype of LR_ANSWER where the answer must be one of the two directions. "Is X on the left or right?", "Which side is X on?" A yes/no "Is X left of Y?" is NOT LR_SIDE.
LR_YN: Subtype of LR_ANSWER. A yes/no question whose only content is whether ONE left/right relation holds between two objects. "Is the ball to the left of the box?" Also set LR_ANSWER. Do NOT set LR_YN for "left or right" choices, for leftmost/rightmost, or when the question has any other condition or uses left/right to identify an object.

Up/down:
UD_ID: Up/down used to identify or select an object. "What color is the ball above the box?"
UD_ANSWER: Up/down being asked about or offered as a choice. "Is the ball above the box?"
UD_SIDE: Subtype of UD_ANSWER where the answer must be one of the two directions. "Is X on the top or bottom?"
UD_YN: Subtype of UD_ANSWER. A yes/no question whose only content is whether ONE up/down relation (above, below, over, under, on top of) holds between two objects. "Is the ball above the box?" Also set UD_ANSWER. Do NOT set UD_YN for "top or bottom" choices, for topmost/bottommost, or when the question has any other condition or uses up/down to identify an object.
Include both _ID and _ANSWER when both roles occur, even for different objects.

Gravity and orientation (these matter when the image is turned upside down):
SUPPORT: A support or containment relation between two objects (on, under, inside, resting on, standing on).
GRAVITY: A posture, motion, or orientation of a single object that only makes sense when the image is upright: sitting, standing, lying, hanging, leaning, kneeling, falling, jumping, flying, floating, pouring, upside down.
REF_SURFACE: The question mentions a natural reference surface or direction of the scene: sky, ground, floor, ceiling, roof, horizon, or a reflection. Set it even when that surface is the object being asked about.

Other:
TEXT: Readable text, letters, numbers, signs, logos, or an object that normally carries text (poster, license plate, book cover) when its text is relevant.
MATW: Material, texture, pattern, or surface (wooden, striped, shiny).
COMPARE: Compares properties of multiple objects ("same color?", "which is bigger?").
MULTI_CONDITION: Several conditions joined by AND/OR, including "both"/"either" constructions and conjunctions inside object-identifying clauses ("the man or the woman", "the girl who is smiling and wearing a hat"). Do not flag simple alternative answers like "red or blue".

Examples:
Question: What color is the apple?
{"target":"apple","flags":["COLOR_QUERY"]}

Question: Is the car red or blue?
{"target":"car","flags":["COLOR_CHOOSE"]}

Question: Is the shirt black?
{"target":"shirt","flags":["COLOR_VERIFY"]}

Question: Is the man wearing a red hat?
{"target":"man","flags":[]}

Question: What color is the cup on the left of the plate?
{"target":"cup","flags":["COLOR_QUERY","LR_ID"]}

Question: Is there a cat in the image?
{"target":"cat","flags":[]}

Question: Is the ball to the left or right of the box?
{"target":"ball","flags":["LR_ANSWER","LR_SIDE"]}

Question: Is the ball to the left of the box?
{"target":"ball","flags":["LR_ANSWER","LR_YN"]}

Question: Is the ball above the box?
{"target":"ball","flags":["UD_ANSWER","UD_YN"]}

Question: Is the lamp on the top or bottom shelf?
{"target":"lamp","flags":["UD_ANSWER","UD_SIDE"]}

Question: What is the girl to the left of the frisbee holding?
{"target":"girl","flags":["LR_ID"]}

Question: Is the girl to the left of the frisbee wearing a hat?
{"target":"girl","flags":["LR_ID"]}

Question: Is the bird sitting on the fence?
{"target":"bird","flags":["SUPPORT","GRAVITY"]}

Question: Is the man standing?
{"target":"man","flags":["GRAVITY"]}

Question: What color is the sky?
{"target":"sky","flags":["COLOR_QUERY","REF_SURFACE"]}

Question: What does the sign say?
{"target":"sign","flags":["TEXT"]}

Question: Is the wooden table round?
{"target":"table","flags":["MATW"]}

Question: Are the plate and the cup the same color?
{"target":"plate","flags":["COMPARE","MULTI_CONDITION"]}

Question: Is the man or the woman holding the umbrella?
{"target":"man","flags":["MULTI_CONDITION"]}

Question: Is it sunny?
{"target":null,"flags":[]}

Return ONLY valid JSON with keys "target" and "flags". No explanation."""


def render(question):
    """Substitute only the question; no image, answer, or dataset metadata is sent."""
    return PROMPT.replace("{question}", question)
