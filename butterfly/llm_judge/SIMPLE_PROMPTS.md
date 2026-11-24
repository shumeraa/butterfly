# GQA metamorphic relations: question-only rules (v7)

One LLM call per question returns a small JSON object, `{"target": ..., "flags": [...]}`. Code adds its own lexicon-based flags, then computes all verdicts and expected answers deterministically and writes one JSON record per question (Part 6). There is no second answer-rule model call: expected answers are encoded in the Equivariant cells of the tables.

Verdicts: **I** = invariant (answer unchanged), **E** = equivariant (a predictable allowed answer or answer set for at least one input), **U** = unsafe (skip scoring, not a failure). Prefer precision over coverage: a false I/E verdict can create a false failure, whereas an abstention only reduces coverage.

Transformations covered:

- **MR1** (whole image, no models): horizontal flip, vertical flip, grayscale
- **MR2** (segment one target object, then edit it): occlude bbox (hides the object behind an opaque black box; it does not recolor the object), fill color (currently red only; the schema is keyed by color so others can be added)

MR2 uses a **single target per question**, and the edit is applied to **every instance** of that target in the image (the target "duck" means all ducks), so every question gets at most one verdict per transformation.

---

# Part 1: Call prompt

Only `{question}` is substituted. No image or answer is given to the model.

````text
Question: {question}

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

Return ONLY valid JSON with keys "target" and "flags". No explanation.
````

## Output schema

```json
{
  "target": "apple | null",
  "flags": ["COLOR_QUERY", "LR_ID", "..."]
}
```

---

# Part 2: Code-side derivation

Tokenize the lowercased question into words, splitting hyphens and contractions (`isn't`, `right-hand`). Lists are over-inclusive on purpose: for a negated gate (`¬X`), a missed word creates a fake verdict, while an extra word only costs coverage. Complete the lists with a one-time audit of the unique tokens in your question set.

## Lexicons (code only)

| Lexicon | Words | Used for |
| --- | --- | --- |
| `NEGATION` | not, no, never, none, neither, nor, without, nobody, nothing, any token ending in `n't` | gate on every yes/no mapping |
| `LR_WORDS` | left, right, leftmost, rightmost | `LR_WORD` |
| `UD_WORDS` | top, bottom, above, below, upper, lower, over, under, beneath, underneath, up, down | `UD_WORD` |
| `BODY_WORDS` | hand(s), arm(s), leg(s), foot, feet, eye(s), ear(s), shoulder(s), knee(s), elbow(s), wrist(s), ankle(s), hip(s), cheek(s) | `BODY`, only when `LR_WORD` is also present |
| `COLOR_WORDS` | all color names including shades (beige, tan, maroon, teal, ...) | `COLORW`, color token count |
| `BRIGHTNESS_WORDS` | dark, bright, colorful, colored, multicolored, pale, `-ish` forms | `COLORW` |
| `RED_SET` | red, crimson, scarlet | `ASKS_RED` (the fill color's own word set) |
| `RED_NEAR` | maroon, burgundy, pink, orange, magenta, rose | `ASKS_RED_NEAR` (fill-red verify becomes Unsafe) |
| `GRAY_SET` | gray, grey, black, white, silver | `ASKS_ACHROMATIC` |
| `GRAY_NEAR` | beige, tan, cream, ivory, charcoal | `ASKS_GRAY_NEAR` (grayscale verify becomes Unsafe) |
| `CONJ` | and, both, either, neither | `MULTI_CONDITION` |

## Tier B lexicons (unioned with the LLM flag)

| Flag | Code lexicon |
| --- | --- |
| `SUPPORT` | on, under, beneath, underneath, inside, hanging, sitting, lying |
| `MATW` | wooden, metal, plastic, glass, leather, striped, plaid, checkered, shiny, brick, stone |
| `TEXT` | say, says, written, read, letters, number, word, sign, logo, text, label, brand, name |
| `COMPARE` | same, different, than, similar, compared, bigger, larger, smaller, taller, shorter, older, younger |

Before matching `SUPPORT`, remove frame-position phrases like "on the left / right / top / bottom / side" and "on which side". Do not use an `-er` regex for `COMPARE` (it matches water, flower, player). Exclude "light" from `BRIGHTNESS_WORDS` in "traffic light", "street light", and similar.

## Merge rules

| Step | Rule |
| --- | --- |
| Gate flags | `SUPPORT`, `MATW`, `TEXT`, `COMPARE` are set if the LLM sets them or a Tier B word appears. |
| `MULTI_CONDITION` | Set if the LLM sets it or any `CONJ` word appears. |
| Code-only flags | `COLORW` (any color or brightness word), `NEGATION`, `LR_WORD`, `UD_WORD`, `BODY` (body word and `LR_WORD`). |
| Direction dependency | Keep semantic `_ID` and `_ANSWER` flags even if the corresponding axis word is absent. "Which side?" (without explicit vertical choices) blocks horizontal invariance; open "Which part of the image?" without a direction choice blocks invariance on both axes. |
| Subtype consistency | `_SIDE` and `_YN` imply `_ANSWER`, but conflicting subtypes on an axis are discarded. `_SIDE` needs supported axis choices or, for `LR_SIDE` only, an implicit lateral "which side" question without a vertical answer cue. `_YN` needs a yes/no form and a relation or frame-half phrase on its axis. |
| Positive guards | `FIXED_PAIR` must come from the LLM and is vetoed by open-ended or indefinite wording (e.g. `a`, `an`, `any`, `some`, `or`, `every`, or an existential prefix) and boundary or alignment cues (e.g. `centered`, `aligned`, `middle`). `FILL_SAFE` must come from the LLM. |
| Color consistency | `COLOR_QUERY` needs 0 color tokens, `COLOR_VERIFY` exactly 1 and a yes/no form, `COLOR_CHOOSE` 2 or more; otherwise the flag is dropped. More than one `COLOR_*` flag drops all of them. |

## Derived values

| Value | Definition |
| --- | --- |
| `LR`, `UD` | `LR_ID ∨ LR_ANSWER`, `UD_ID ∨ UD_ANSWER`. If both `_ID` and `_ANSWER` are set, treat as `_ID` (unsafe). |
| `EQ_GATE` | `¬MULTI_CONDITION ∧ ¬COMPARE ∧ target ≠ null` |
| `ASKS_RED` | a color token is in `RED_SET` |
| `ASKS_RED_NEAR` | a color token is in `RED_NEAR` |
| `ASKS_ACHROMATIC` | a color token is in `GRAY_SET` |
| `ASKS_GRAY_NEAR` | a color token is in `GRAY_NEAR` |
| `EXIST` | question starts with "is/are there" or "do you see", has no and/or/both/either/neither, and `¬NEGATION` |
| Answer normalization | lowercase; treat `gray` and `grey` as the same |

---

# Part 3: MR1 rule table (whole image)

| Transformation | Invariant | Equivariant | Unsafe | In English |
| --- | --- | --- | --- | --- |
| **Horizontal flip** | `¬LR_WORD ∧ ¬LR ∧ ¬BODY ∧ ¬TEXT` | `LR_SIDE ∧ ¬LR_ID ∧ ¬MULTI_CONDITION ∧ ¬BODY ∧ ¬TEXT` Answer: swap left↔right<br><br>`LR_YN ∧ FIXED_PAIR ∧ ¬LR_ID ∧ ¬MULTI_CONDITION ∧ ¬NEGATION ∧ ¬BODY ∧ ¬TEXT` Answer: swap yes↔no | Anything else | **I:** no lateral dependency (including implicit side questions), side words, left/right body part, or text.<br>**E:** a clear left/right answer swaps; a plain yes/no relation between fixed referents or a frame-half test swaps yes/no on the ordinary two-side reading, subject to the same identification, text, and compound blockers.<br>**U:** everything else, including ambiguous sides and existential or non-fixed pairs. |
| **Vertical flip** | `¬UD_WORD ∧ ¬UD ∧ ¬SUPPORT ∧ ¬GRAVITY ∧ ¬REF_SURFACE ∧ ¬TEXT` | `UD_SIDE ∧ ¬UD_ID ∧ ¬MULTI_CONDITION ∧ ¬SUPPORT ∧ ¬TEXT` Answer: swap top↔bottom, above↔below<br><br>`UD_YN ∧ FIXED_PAIR ∧ ¬UD_ID ∧ ¬MULTI_CONDITION ∧ ¬NEGATION ∧ ¬SUPPORT ∧ ¬TEXT` Answer: swap yes↔no | Anything else | **I:** no vertical dependency, up/down word, support, posture, natural reference surface, or text.<br>**E:** an explicit vertical choice swaps, or a plain yes/no relation between fixed referents or a frame-half test swaps yes/no on the ordinary two-side reading, subject to the identification, support, text, and compound blockers.<br>**U:** everything else. |
| **Grayscale** | `¬COLORW` | `COLOR_QUERY ∧ ¬MULTI_CONDITION ∧ ¬COMPARE` Answer: in `{gray, black, white}`<br><br>`COLOR_VERIFY ∧ ¬ASKS_ACHROMATIC ∧ ¬ASKS_GRAY_NEAR ∧ ¬NEGATION ∧ ¬MULTI_CONDITION ∧ ¬COMPARE` Answer: no | Anything else (includes `COLOR_CHOOSE`) | **I:** no color or brightness word in the question.<br>**E:** asks the color of the target (answer is gray, black, or white); or asks, without negation, whether the target is a clearly chromatic color (answer is `no`).<br>**U:** everything else, including two-color choice questions. |

Notes:

- **Yes/no swaps use a best-effort binary assumption.** For plain comparisons with fixed referents or named image halves, assume the original falls on one side or the other. A tie or an object straddling the dividing line is an exception, not a reason to abstain on every clean question. For existential or non-fixed pairs, even `yes` is not reliably mapped: different mirrors can witness the relation before and after a flip.
- **`GRAVITY` and `REF_SURFACE` apply only to the vertical invariant row.** "Is the sky at the top or bottom?" still has a clean swapped answer, so the equivariant rows keep only `SUPPORT`. Drop `REF_SURFACE` from the invariant row if coverage falls too far, since "What color is the sky?" has a well-defined answer and only the image is unnatural.

---

# Part 4: MR2 rule table (single target object)

If `target = null`, MR2 has no row.

| Transformation | Invariant | Equivariant | Unsafe | In English |
| --- | --- | --- | --- | --- |
| **Occlude bbox** (opaque black box) | Never (the target is named in the question, so hiding it cannot be shown irrelevant from the question alone) | `EQ_GATE ∧ EXIST` → no | Anything else (this includes every color question: the object is hidden, not recolored, so the visible color is not a reliable answer) | **I:** never.<br>**E:** a plain "is there X" question about one object, with no and/or and no negation (answer is `no`).<br>**U:** everything else, including every color question. |
| **Fill color** (mask only; shown for red, one row per color in `FILL_COLORS`) | `target ≠ null ∧ FILL_SAFE ∧ ¬COLORW ∧ ¬MATW ∧ ¬TEXT` | `EQ_GATE ∧ FILL_SAFE` plus one of:<br>`COLOR_QUERY` → red<br>`COLOR_VERIFY ∧ ASKS_RED ∧ ¬NEGATION` → yes<br>`COLOR_VERIFY ∧ ¬ASKS_RED ∧ ¬ASKS_RED_NEAR ∧ ¬NEGATION` → no<br>`COLOR_CHOOSE ∧ ASKS_RED` → red | Anything else | **I:** a specified target, affirmative evidence the answer survives the opaque fill, and no color, material, or text wording.<br>**E:** the target remains identifiable and the question asks its color, verifies one non-negated clear color, or offers two colors including red, without comparisons or extra conditions.<br>**U:** everything else, including erased or unverified answer evidence. |

---

# Part 5: Expected-answer mappings

The model never writes a mapping. Code derives it from the verdict and flags, in the notation human graders use: one line per original answer, `original -> allowed outputs`. `any` stands for every original answer not listed on another line. `unchecked` means the case is skipped. Because every instance of the target is edited, these hold however many instances exist.

| Case | Condition | In English | Mapping |
| --- | --- | --- | --- |
| Horizontal flip, left/right choice | `LR_SIDE ∧ ¬LR_ID ∧ ¬MULTI_CONDITION ∧ ¬BODY ∧ ¬TEXT` | Asks whether something is on the left or the right, with no left/right identification, body part, text, or extra conditions. | `left -> right`, `right -> left` |
| Horizontal flip, yes/no position | `LR_YN ∧ FIXED_PAIR ∧ ¬LR_ID ∧ ¬MULTI_CONDITION ∧ ¬NEGATION ∧ ¬BODY ∧ ¬TEXT` | Plain yes/no question about a fixed left/right comparison or image half. | `yes -> no`, `no -> yes` |
| Vertical flip, top/bottom choice | `UD_SIDE ∧ ¬UD_ID ∧ ¬MULTI_CONDITION ∧ ¬SUPPORT ∧ ¬TEXT` | Asks top or bottom, with no up/down identification, support relation, text, or extra conditions. | `top -> bottom`, `bottom -> top`, `above -> below`, `below -> above` |
| Vertical flip, yes/no position | `UD_YN ∧ FIXED_PAIR ∧ ¬UD_ID ∧ ¬MULTI_CONDITION ∧ ¬NEGATION ∧ ¬SUPPORT ∧ ¬TEXT` | Plain yes/no question about a fixed above/below comparison or image half. | `yes -> no`, `no -> yes` |
| Fill color (red), "What color is X?" | `EQ_GATE ∧ FILL_SAFE ∧ COLOR_QUERY` | Asks the color of the single identifiable target, with no comparison or extra conditions. | `any -> red` |
| Fill color (red), "Is X blue?" | `EQ_GATE ∧ FILL_SAFE ∧ COLOR_VERIFY ∧ ¬ASKS_RED ∧ ¬ASKS_RED_NEAR ∧ ¬NEGATION` | Asks if the target is a color that is not red or red-like, not negated. | `yes -> no`, `no -> no` |
| Fill color (red), "Is X red?" | `EQ_GATE ∧ FILL_SAFE ∧ COLOR_VERIFY ∧ ASKS_RED ∧ ¬NEGATION` | Asks if the target is red, not negated. | `yes -> yes`, `no -> yes` |
| Fill color (red), "Is X blue or red?" | `EQ_GATE ∧ FILL_SAFE ∧ COLOR_CHOOSE ∧ ASKS_RED` | Asks which of two colors the target is, and one of them is red. | `any -> red` |
| Occlude bbox, "Is there a cat?" | `EQ_GATE ∧ EXIST` | Plain "is there X" question about one object, with no and/or and no negation. | `any -> no` |
| Grayscale, color query | `COLOR_QUERY ∧ ¬MULTI_CONDITION ∧ ¬COMPARE` | Asks the color of the target, with no other color words. | `any -> gray, black, white` |
| Grayscale, "Is X red?" | `COLOR_VERIFY ∧ ¬ASKS_ACHROMATIC ∧ ¬ASKS_GRAY_NEAR ∧ ¬NEGATION ∧ ¬MULTI_CONDITION ∧ ¬COMPARE` | Asks if the target is a clearly chromatic color (not gray, black, white, or close to them), not negated. | `yes -> no`, `no -> no` |
| Any invariant verdict | Invariant condition for that transformation (Parts 3 and 4) | The question has nothing that depends on the edit, such as direction, color, material, or text. | the answer maps to itself |

Most mappings are constants. Flips can swap directions or swap yes/no. The grayscale rule is deliberately coarse (a human may refine it as `black -> black`, `white -> white`, `any -> gray`).

---

# Part 6: Output JSON

Code writes one record per question, as `[question_id, {...}]`, and the full output is a list of these records. The `question_id` comes from your dataset, not from the LLM. `fill_color` is keyed by fill color. Only `red` is evaluated now; adding a color to `FILL_COLORS` in `evaluate_rules.py` adds its key with the same rules (its own `ASKS`/`NEAR` word sets replace `RED_SET`/`RED_NEAR`, and "red" in the answers becomes that color).

```json
[
  [
    "006108",
    {
      "hflip": "invariant",
      "vflip": "invariant",
      "grayscale": {"equivariant": {"yes": "no", "no": "no"}},
      "mr2": {
        "bag": {
          "fill_color": {"red": {"equivariant": {"yes": "yes", "no": "yes"}}},
          "occlude_bbox": "unsafe"
        }
      }
    }
  ],
  [
    "006149",
    {
      "hflip": {"equivariant": {"left": "right", "right": "left"}},
      "vflip": "invariant",
      "grayscale": "invariant",
      "mr2": {
        "bag": {
          "fill_color": {"red": "invariant"},
          "occlude_bbox": "unsafe"
        }
      }
    }
  ],
  [
    "006201",
    {
      "hflip": "invariant",
      "vflip": "invariant",
      "grayscale": "invariant",
      "mr2": {
        "cat": {
          "fill_color": {"red": "invariant"},
          "occlude_bbox": {"equivariant": {"any": "no"}}
        }
      }
    }
  ]
]
```

The three records above come from "Is the bag red?", "Is the bag on the left or right?", and "Is there a cat in the image?".

| Field | Value |
| --- | --- |
| `hflip`, `vflip`, `grayscale` | `"invariant"`, `"unsafe"`, or `{"equivariant": mapping}` |
| `mr2` | `null` if `target = null`; otherwise `{target: {"fill_color": {color: verdict}, "occlude_bbox": verdict}}`, with the target text as returned by the LLM and each verdict in the same form as above |
| `mapping` | Object of `original answer: expected answer`, using the Part 5 notation |
| `"any"` key | Every original answer not listed in the mapping |
| List value | Several allowed outputs, e.g. `{"any": ["gray", "black", "white"]}` |
| `"unchecked"` value (legacy) | Skip this input before comparing answers. This sentinel may appear in older exports; it is never an expected answer. |
| `"invariant"` | The answer maps to itself, so there is no mapping object |

---

# Part 7: Evaluation notes

- **Check order:** for each transformation, check Equivariant, then Invariant, then default to Unsafe. For flips, `TEXT` (and `BODY` for the horizontal flip) force Unsafe first.
- **Yes/no swap cases:** score either original answer when the model's original answer matches gold. The swap is a best-effort assumption about ordinary non-tied comparisons, not an image-level proof. Skip legacy `unchecked` and absent mapping keys without a fallback. Report scored and skipped counts separately.
- **Segmentation:** the singular head noun can retrieve mask candidates, but resolve the full identifying description before editing. Skip if the intended targets cannot be verified or the masks are missing/inadequate; otherwise a failed or misdirected edit can look like a model violation.
- **Classifier quality:** measure precision of the I and E verdicts separately from coverage. Prioritize unsupported affirmative mappings or invariance, not conservative skips. Rerun classification with v7 before measuring coverage. Pre-v6 predictions lack `FIXED_PAIR` and `FILL_SAFE`; v6 did not use the yes/no subtypes for image-half questions.

---

# Caveats that change a label

- **Occlude existence → no** assumes the edit covers every instance of the target. If the segmenter misses an instance the answer stays yes. Drop this rule if you see false positives.
- **Occlude hides, it does not recolor,** so color questions are Unsafe for occlusion. Only existence has a reliable answer.
- **Grayscale query:** white and black objects stay white and black, so the accepted set is `{gray, black, white}`. Verify → no is safe because a chromatic color cannot survive grayscale.
- **Cases that need a map** ("same color?", "how many?", wooden/metal fill questions) stay Unsafe. If you later use the gold answer at evaluation time, encode them as small code rules rather than a model call.
