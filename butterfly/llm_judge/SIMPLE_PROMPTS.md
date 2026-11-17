# GQA metamorphic relations: question-only rules (v2)

One LLM call per question returns a small JSON object. All verdicts and expected answers are then computed deterministically from that JSON. There is no second answer-rule model call: expected answers are encoded in the Equivariant cells of the tables.

Verdicts: **I** = invariant (answer unchanged), **E** = equivariant (answer changes deterministically, including to a constant), **U** = unsafe (no reliable expectation).

Transformations covered:

- **MR1** (whole image, no models): horizontal flip, vertical flip, grayscale
- **MR2** (segment one target object, then edit it): occlude bbox (black), fill red

MR2 uses a **single target object per question**, so every question gets at most one verdict per transformation.

---

# Part 1: Call prompt

Only `{question}` is substituted. No image or answer is given to the model.

````text
Question: {question}

Analyze this visual question using only its wording, not an image or answer.

1. TARGET: Choose the ONE object an image edit would focus on. Apply in order and stop at the first match:
   a. Asked-about object: if the question asks the color of X, or whether X exists, X is the target.
   b. Grammatical subject: otherwise, the head noun of the main clause.
   c. Only named object: if the question names just one object, it is the target.
   d. Tie: take the first qualifying object in textual order.
   Never choose: an object used only to identify another object ("to the left of the Z"), an answer option ("an apple or a banana"), or an object whose existence the question does not presuppose ("a cup" in "Is the man holding a cup?"; the man is the target). Exception: for existence questions, the object asked about is the target.
   Use null if no object qualifies or the question is about the whole scene.
   Keep the identifying description ("green apple", "man on the left").

2. SOLE: true if the wording implies a single instance of the target (singular "the/a X", no plural, no count). false otherwise or if target is null.

3. COLOR_Q: Set only if the question asks the color of the target and contains no other color words.
   - "query": "What color is X?"
   - "verify": "Is X <color>?"
   - "choose": "Is X <c1> or <c2>?"
   - null otherwise.

4. ASKED_COLORS: Colors named in a verify or choose question, lowercase. [] otherwise.

5. FLAGS: List all that apply.
LR_ID: Left/right used to identify or select an object. "What is the girl to the left of the frisbee doing?"
LR_ANSWER: Left/right being asked about or offered as a choice. "Is the girl to the left of the frisbee?"
LR_SIDE: Subtype of LR_ANSWER where the answer must be one of the two directions. "Is X on the left or right?", "Which side is X on?" A yes/no "Is X left of Y?" is NOT LR_SIDE.
UD_ID: Up/down used to identify or select an object. "What color is the ball above the box?"
UD_ANSWER: Up/down being asked about or offered as a choice. "Is the ball above the box?"
UD_SIDE: Subtype of UD_ANSWER where the answer must be one of the two directions. "Is X on the top or bottom?"
Include both _ID and _ANSWER when both roles occur, even for different objects.
SUPPORT: on, under, inside, sitting, hanging, lying.
TEXT: Readable text, letters, numbers, signs, logos.
BODY: Left/right body parts.
COLORW: Any color word or reference to color or brightness (dark, bright, colorful).
MATW: Material, texture, pattern, surface (wooden, striped, shiny).
EXISTW: Asks whether something exists or is present.
COMPARE: Compares properties of multiple objects.
MULTI_CONDITION: Several conditions joined by AND/OR, including "both"/"either" constructions and conjunctions inside object-identifying clauses. Do not flag simple alternative answers like "red or blue".

Examples:
Question: What color is the apple?
{"target":"apple","sole":true,"color_q":"query","asked_colors":[],"flags":["COLORW"]}

Question: Is the car red or blue?
{"target":"car","sole":true,"color_q":"choose","asked_colors":["red","blue"],"flags":["COLORW"]}

Question: Is the shirt black?
{"target":"shirt","sole":true,"color_q":"verify","asked_colors":["black"],"flags":["COLORW"]}

Question: Is there a cat in the image?
{"target":"cat","sole":true,"color_q":null,"asked_colors":[],"flags":["EXISTW"]}

Question: Are there any dogs?
{"target":"dogs","sole":false,"color_q":null,"asked_colors":[],"flags":["EXISTW"]}

Question: Is the ball to the left or right of the box?
{"target":"ball","sole":true,"color_q":null,"asked_colors":[],"flags":["LR_ANSWER","LR_SIDE"]}

Question: Is the ball above the box?
{"target":"ball","sole":true,"color_q":null,"asked_colors":[],"flags":["UD_ANSWER"]}

Question: What is the girl to the left of the frisbee holding?
{"target":"girl","sole":true,"color_q":null,"asked_colors":[],"flags":["LR_ID"]}

Question: Is the man wearing a red hat?
{"target":"man","sole":true,"color_q":null,"asked_colors":[],"flags":["COLORW"]}

Question: Is the man holding a cup?
{"target":"man","sole":true,"color_q":null,"asked_colors":[],"flags":[]}

Question: What does the sign say?
{"target":"sign","sole":true,"color_q":null,"asked_colors":[],"flags":["TEXT"]}

Question: Is it sunny?
{"target":null,"sole":false,"color_q":null,"asked_colors":[],"flags":[]}

Return ONLY valid JSON with keys "target", "sole", "color_q", "asked_colors", and "flags". No explanation.
````

## Output schema

```json
{
  "target": "apple | null",
  "sole": true,
  "color_q": "query | verify | choose | null",
  "asked_colors": ["..."],
  "flags": ["..."]
}
```

## Derived values (computed in code)

- `LR = LR_ID ∨ LR_ANSWER`
- `UD = UD_ID ∨ UD_ANSWER`
- If both `_ID` and `_ANSWER` are set for a direction, treat the question as `_ID` (unsafe).
- `GRAY_SET = {gray, grey, black, white}`
- `EQ_GATE = ¬MULTI_CONDITION ∧ ¬COMPARE ∧ target ≠ null`
- Normalize answers before comparing: lowercase, and treat `gray` and `grey` as the same.

---

# Part 2: MR1 rule table (whole image)

| Transformation            | Invariant                       | Equivariant                                                                                                                             | Unsafe                                                                   |
| ------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Horizontal flip** | `¬LR ∧ ¬BODY ∧ ¬TEXT`    | `LR_SIDE ∧ ¬LR_ID ∧ ¬MULTI_CONDITION ∧ ¬BODY ∧ ¬TEXT`Answer: swap left↔right                                                 | Anything else                                                            |
| **Vertical flip**   | `¬UD ∧ ¬SUPPORT ∧ ¬TEXT` | `UD_SIDE ∧ ¬UD_ID ∧ ¬MULTI_CONDITION ∧ ¬SUPPORT ∧ ¬TEXT`Answer: swap top↔bottom, above↔below                                | Anything else                                                            |
| **Grayscale**       | `¬COLORW`                    | `color_q = query ∧ ¬MULTI_CONDITION ∧ ¬COMPARE`Answer: in `GRAY_SETcolor_q = verify ∧ asked_colors ∩ GRAY_SET = ∅`Answer: no | Anything else (includes `choose`, and verify with an achromatic color) |

---

# Part 3: MR2 rule table (single target object)

If `target = null`, MR2 has no row.

| Transformation                        | Invariant                                                                                                   | Equivariant                                                                                                                                                                                        | Unsafe           |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **Occlude bbox** (opaque black) | Never(the target is named in the question, so hiding it cannot be shown irrelevant from the question alone) | `EQ_GATE` plus one of:`color_q = query` → black`color_q = verify` → yes if the asked color is black, else no`color_q = choose ∧ black ∈ asked_colors` → black`EXISTW ∧ sole` → no | Anything else    |
| **Fill red** (mask only)        | `target ≠ null ∧ ¬COLORW ∧ ¬MATW ∧ ¬TEXT`                                                          | `EQ_GATE` plus one of:`color_q = query` → red`color_q = verify` → yes if the asked color is red, else no`color_q = choose ∧ red ∈ asked_colors` → red                                 | Anything else--- |

# Evaluation order

1. For each transformation, check **Equivariant**, then **Invariant**, then default to **Unsafe**.
2. For flips, `TEXT` (and `BODY` for the horizontal flip) force Unsafe before anything else.
3. For MR2, evaluate once per question using the single target.

---

# Caveats that change a label

- **Grayscale query:** white and black objects stay white and black, so the accepted answer set is `GRAY_SET`. The `verify → no` rule is safe because a chromatic color cannot survive grayscale.
- **Occlude existence → no:** depends on `sole`, which is a guess from the wording. If other instances exist outside the box the answer stays yes. Drop this rule if you see false positives.
- **Occlude color rules:** assume the box fully covers the target. A post-filter on mask/bbox area helps.
- **Flip yes/no questions:** these stay Unsafe because a yes→no flip needs the original answer. If you have the ground truth, add `one relation ∧ answer = yes → Equivariant (no)`.
- **Cases that would need a map:** "same color?", "how many?", and wooden/metal fill questions have several allowed outputs or depend on the original answer. They stay Unsafe. If you later use the gold answer at evaluation time, encode them as small code rules rather than a model call.
