"""Versioned, exact question-only prompt from SIMPLE_PROMPTS.md (Part 1)."""

PROMPT_VERSION = "gqa-simple-v7"

PROMPT = """Question: {question}

Analyze this visual question using only its wording, not an image or answer. Prefer skipping an uncertain test to asserting an unreliable expected answer. Do not assume a unique object, a particular layout, or a visual cue not established by the wording.

1. TARGET: Choose the ONE object an image edit would focus on. Apply in order and stop at the first match:
   a. Asked-about object: if the question asks the color of X, or whether X exists, X is the target.
   b. Grammatical subject: otherwise, the head noun of the main clause.
   c. Only named object: if the question names just one object, it is the target.
   d. Tie: take the first qualifying object in textual order.
   Never choose: an object used only to identify another object ("to the left of the Z"), an answer option ("an apple or a banana"), or an object whose existence the question does not presuppose ("a cup" in "Is the man holding a cup?"; the man is the target). Exception: for existence questions ("Is there a cup?"), the object asked about is the target.
   Use null if no object qualifies or the question is about the whole scene.
   Keep the identifying description ("green apple", "man on the left"). Do not combine multiple object types into one target.
   The edit applies to every instance of the target in the image, not just one.

2. FLAGS: List all that apply.

Color question form. Set one of these only if the question asks the color of the target and contains no other color words (for example, no color word inside an object-identifying clause like "the red car's door"). Never set more than one.
COLOR_QUERY: "What color is X?"
COLOR_VERIFY: A yes/no question about the target's color ("Is X <color>?"). Do not set it for identity questions that use color to pick out an object ("What animal is black?").
COLOR_CHOOSE: "Is X <c1> or <c2>?"

Left/right:
LR_ID: Left/right used to identify or select an object. "What is the girl to the left of the frisbee doing?"
LR_ANSWER: Left/right being asked about or offered as a choice, even without the words left or right. Set this for "Which side of the photo is the woman on?" as well as "Is the girl to the left of the frisbee?"
LR_SIDE: Subtype of LR_ANSWER where the answer is left or right. Use it for an explicit left/right choice or an ordinary "Which side?" question that asks for the lateral side of the image. Do not set it for yes/no, for a vertical question, or if the wording leaves the axis or answer choices unclear.
LR_YN: Subtype of LR_ANSWER for a yes/no test of one left/right relation between objects or one object in the left/right half of the image ("Is the towel on the left?"). Do not set it for a direction choice, a superlative, or a relation used only to identify an object.

Up/down:
UD_ID: Up/down used to identify or select an object. "What color is the ball above the box?"
UD_ANSWER: Up/down being asked about or offered as a choice. "Is the ball above the box?"
UD_SIDE: Subtype of UD_ANSWER where the question explicitly offers top/bottom or above/below. Do not assume that "Which side?" means top/bottom.
UD_YN: Subtype of UD_ANSWER for a yes/no test of one above/below relation between objects or one object in the top/bottom half of the image ("Is the knife in the bottom part?"). Do not set it for a direction choice, a superlative, or a relation used only to identify an object.
Include both _ID and _ANSWER when both roles occur, even for different objects. For open image-position questions like "Which part of the image are the people in?", set both LR_ANSWER and UD_ANSWER, but neither _SIDE flag unless an answer pair is actually established.
FIXED_PAIR: Set with LR_YN or UD_YN for a single image-plane comparison between two fixed, singular referents, or between one fixed object and a named half of the image. Use the ordinary best-effort two-side reading: assume opposite sides, not a tie, for plain questions like "Is the ball to the left of the box?" or "Is the knife in the bottom part?" This lets the flip swap yes/no both ways; do not reject a clean case just because an exact tie is theoretically possible. Leave this flag unset if the wording suggests a center/boundary case, multiple conditions, changing referents, or existential, plural, group, or indefinite referents ("any mirrors", "a boy"). Mirrors may be on both sides of a lamp, so yes can stay yes.

Gravity and orientation (these matter when the image is turned upside down):
SUPPORT: A support or containment relation between two objects (on, under, inside, resting on, standing on).
GRAVITY: A posture, motion, or orientation of a single object that only makes sense when the image is upright: sitting, standing, lying, hanging, leaning, kneeling, falling, jumping, flying, floating, pouring, upside down.
REF_SURFACE: The question mentions a natural reference surface or direction of the scene: sky, ground, floor, ceiling, roof, horizon, or a reflection. Set it even when that surface is the object being asked about.

Fill color:
FILL_SAFE: Set only if the target remains identifiable after its whole mask is replaced with a flat color and the answer has a reliable relationship for the fill. A direct color question about a stable target qualifies, as can position, size, or a simple relation to an untouched object. The mask preserves outline and location, not clothing, accessories, markings, face or gaze, texture, transparency, or haze inside it. Leave FILL_SAFE unset if such details can determine the answer or pick out its referent (clothing a person is wearing, the type of food, a happy face, a clear or foggy sky). The absence of an explicit color or material word is not enough. Do not add FILL_SAFE without a target.

Other:
TEXT: Readable text, letters, numbers, signs, logos, or an object that normally carries text (poster, license plate, book cover) when its text is relevant.
MATW: Material, texture, pattern, or surface (wooden, striped, shiny).
COMPARE: Compares properties of multiple objects ("same color?", "which is bigger?").
MULTI_CONDITION: Several conditions joined by AND/OR, including "both"/"either" constructions and conjunctions inside object-identifying clauses ("the man or the woman", "the girl who is smiling and wearing a hat"). Do not flag simple alternative answers like "red or blue".

Examples:
Question: What color is the apple?
{"target":"apple","flags":["COLOR_QUERY","FILL_SAFE"]}

Question: Is the car red or blue?
{"target":"car","flags":["COLOR_CHOOSE","FILL_SAFE"]}

Question: Is the shirt black?
{"target":"shirt","flags":["COLOR_VERIFY","FILL_SAFE"]}

Question: What animal is black?
{"target":"black animal","flags":[]}

Question: Is the man wearing a red hat?
{"target":"man","flags":[]}

Question: What is the woman wearing?
{"target":"woman","flags":[]}

Question: Is the sky clear or foggy?
{"target":"sky","flags":["REF_SURFACE"]}

Question: What color is the cup on the left of the plate?
{"target":"cup on the left of the plate","flags":["COLOR_QUERY","LR_ID","FILL_SAFE"]}

Question: Is there a cat in the image?
{"target":"cat","flags":["FILL_SAFE"]}

Question: Is the ball to the left or right of the box?
{"target":"ball","flags":["LR_ANSWER","LR_SIDE","FILL_SAFE"]}

Question: On which side of the photo is the woman?
{"target":"woman","flags":["LR_ANSWER","LR_SIDE","FILL_SAFE"]}

Question: Which part of the image are the people in?
{"target":"people","flags":["LR_ANSWER","UD_ANSWER","FILL_SAFE"]}

Question: Is the ball to the left of the box?
{"target":"ball","flags":["LR_ANSWER","LR_YN","FIXED_PAIR","FILL_SAFE"]}

Question: Are there mirrors to the left of the lamp?
{"target":"mirrors to the left of the lamp","flags":["LR_ANSWER","LR_YN"]}

Question: Is the ball above the box?
{"target":"ball","flags":["UD_ANSWER","UD_YN","FIXED_PAIR","FILL_SAFE"]}

Question: Is the knife in the bottom part?
{"target":"knife","flags":["UD_ANSWER","UD_YN","FIXED_PAIR","FILL_SAFE"]}

Question: Is the lamp on the top or bottom shelf?
{"target":"lamp","flags":["UD_ANSWER","UD_SIDE","SUPPORT","FILL_SAFE"]}

Question: What is the girl to the left of the frisbee holding?
{"target":"girl to the left of the frisbee","flags":["LR_ID"]}

Question: Is the girl to the left of the frisbee wearing a hat?
{"target":"girl to the left of the frisbee","flags":["LR_ID"]}

Question: Is the bird sitting on the fence?
{"target":"bird","flags":["SUPPORT","GRAVITY"]}

Question: Is the man standing?
{"target":"man","flags":["GRAVITY","FILL_SAFE"]}

Question: What color is the sky?
{"target":"sky","flags":["COLOR_QUERY","REF_SURFACE","FILL_SAFE"]}

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
