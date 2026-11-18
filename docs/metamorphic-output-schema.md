# Metamorphic transformation output schema

This JSON contract connects question-only classification to image editing and
single-answer evaluation. It describes predicted metamorphic relations, not
visually verified answers. Color fills are parameterized; the schema does not
restrict them to red and blue.

## Output structure

The outer array contains `[questionId, transformations]` pairs. Question IDs
must be unique strings; preserve leading zeros. Join each ID to the dataset's
question text, image ID, and original answer outside this output.

For example, for "Does the bag appear to be red?":

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
          "fill_color": {
            "red": {"equivariant": {"yes": "yes", "no": "yes"}},
            "green": {"equivariant": {"yes": "no", "no": "no"}}
          },
          "occlude_bbox": "unsafe"
        }
      }
    }
  ]
]
```

Every question has `hflip`, `vflip`, `grayscale`, and `mr2` fields:

| Field | Meaning |
| --- | --- |
| `hflip` | Verdict for whole-image horizontal flip. |
| `vflip` | Verdict for whole-image vertical flip. |
| `grayscale` | Verdict for whole-image grayscale conversion. |
| `mr2` | Object-targeted edits, keyed by target description. |

`mr2` contains at most one target per question. Keep identifying descriptions,
such as `shirt the woman is wearing`. With no qualifying target, use `"mr2": {}`.
Each target has two transformation fields:

- `fill_color`: a map from configured canonical color names to verdicts. The
  lookup path is `mr2[target_description].fill_color[color]`.
- `occlude_bbox`: a verdict for hiding every matching target behind an opaque
  black bounding box. This is occlusion, not recoloring.

Each color entry is a separate edit of the original image, not a sequence of
fills. Each edit applies to every instance matching the target description.

## Verdicts and expected-answer functions

A verdict is exactly one of:

```json
"invariant"
```

```json
"unsafe"
```

```json
{"equivariant": {"left": "right", "right": "left"}}
```

- **Invariant:** the expected answer is the original answer.
- **Unsafe:** no reliable single expected answer is supplied. Skip scoring under
  this contract; generating the image for another purpose is not prohibited.
- **Equivariant:** a nonempty map from original answers to single expected
  answers. Multiple inputs may map to the same output, but one input cannot map
  to several outputs. Output values must be nonempty answer strings, never
  arrays or the wildcard `any`.

Normalize answer strings by trimming whitespace, lowercasing, and treating
`grey` as `gray`. Normalize mapping keys and values the same way. Do not apply
answer normalization to question IDs or target descriptions.

Look up the normalized original answer first. If it has no explicit entry, use
`any` as a reserved fallback input key. For example, `{"any": "purple"}` is a
constant function. An exact match takes precedence over `any`. If neither entry
exists, the case is not evaluable; skip it rather than inventing an expectation.
Compare the normalized observed answer to the resulting single expected answer.

Reject duplicate JSON object keys, duplicate question IDs, and mapping keys that
collide after normalization. Do not silently keep the last value.

A set-valued expectation can be a useful metamorphic relation, but it is outside
this single-answer contract. In particular, a grayscale color query whose
allowed result is gray, black, or white is `unsafe`, not a mapping to a list.

## Configurable colors and rule processing

The experiment supplies a palette independently of the LLM, for example:

```json
{"fill_colors": ["red", "green"]}
```

For each non-null target, the exporter emits one `fill_color` entry for every
configured color. Palette names must be unique after normalization. Use
canonical color names supported by the renderer, with a defined name-to-pixel
color mapping. An unsupported color is a configuration/execution error, not an
answer failure. Do not place raw RGB/hex values in answer mappings unless the
question's answer vocabulary explicitly uses them.

Generalize the existing fill-red rule to fill with color `c`. First require the
equivariance gate: a non-null target, no `MULTI_CONDITION`, and no `COMPARE`.
Then apply:

| Classified color question | Expected-answer mapping |
| --- | --- |
| `query` | `any -> c` |
| `verify` | `yes -> yes`, `no -> yes` if `c` matches the asked color; otherwise `yes -> no`, `no -> no`. |
| `choose`, with `c` among the asked colors | `any -> c` |
| `choose`, with `c` outside the asked colors | Unsafe. |

Compare canonical color names. If no equivariant case applies, fill is invariant
only when the target is non-null and none of `COLORW`, `MATW`, or `TEXT` is set;
otherwise it is unsafe. Check equivariance before invariance.

For `occlude_bbox`, the same equivariance gate plus `EXISTW` gives `any -> no`,
assuming all matching instances are successfully hidden. Otherwise occlusion
is unsafe; it is never invariant under these question-only rules.

Whole-image flip and grayscale verdicts retain their question-flag rules,
subject to the single-answer restriction above. No new color-specific flags
are necessary. The legacy `fill_red` and `fill_blue` operation keys are replaced
by `fill_color.red` and `fill_color.blue`; consumers must read the nested map.

## LLM classification specification

The LLM still returns only these four fields for each question:

```json
{
  "target": "bag",
  "color_q": "verify",
  "asked_colors": ["red"],
  "flags": ["COLORW"]
}
```

`target` is an identifying object description or null. `color_q` is `query`,
`verify`, `choose`, or null. `asked_colors` contains the colors named in a
verification/choice question and is empty otherwise. `flags` contains the
question-only semantic flags. The existing target-selection and flag rules
remain unchanged.

The model classifies question wording; it does not select the replacement
palette or write expected-answer functions. Deterministic rule processing takes
those classifications and the configured palette to produce the downstream
JSON. Adding a supported fill color therefore requires no new LLM output field,
color-specific flag, or model call.

## Execution requirements

Use the unchanged question on each independently transformed image. Resolve the
target against the original image and edit every matching instance before
scoring an MR2 expectation. Missing images, unresolved targets, missing masks,
or failed/incomplete edits are execution failures or skips, not answer failures.
The output supplies target descriptions, not segmentation masks or proof that
an edit succeeded.
