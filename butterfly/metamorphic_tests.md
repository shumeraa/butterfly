# Comparing metamorphic and ViUniT test suites

This extension lets us compare our own metamorphic image tests with ViUniT's
image tests when evaluating visual programs. Tests are generated **per question**
and shared across that question's candidate programs and improvement iterations.

The dispatcher, configuration, evaluation/training integration, and cache handling
are implemented. **The metamorphic generator is still a stub**: selecting
`metamorphic` or `hybrid` currently raises `NotImplementedError` until we implement
it. The default `viunit` mode uses the existing ViUniT generation, sampling, and
image generation/retrieval code.

## Selecting a suite

Add these settings to your experiment YAML, which is merged with
[`viunit_configs/base.yaml`](../viunit_configs/base.yaml):

```yaml
unit_test_generation:
  use_program: false

unit_test_suite:
  mode: viunit
  viunit_count: null
  metamorphic_count: 5
  metamorphic: {}
```

| Setting | Meaning |
| --- | --- |
| `mode: viunit` | Use the existing ViUniT tests only; this is the default. |
| `mode: metamorphic` | Use transformed versions of the original question image only. |
| `mode: hybrid` | Combine independently generated and budgeted tests from both sources. |
| `viunit_count` | Maximum ViUniT tests per question. `null` inherits `unit_test_sampling.num_unit_tests`. |
| `metamorphic_count` | Maximum custom tests per question; must be a positive integer. |
| `metamorphic` | Generator-specific settings passed to our custom implementation. |

`unit_test_sampling.strategy` continues to control how ViUniT caption/answer
candidates are sampled. Suite selection does not replace that strategy.
`image_generation` continues to control ViUniT's image source; choose diffusion
in that section for the diffusion comparison.

For a hybrid suite with a budget of six tests:

```yaml
unit_test_suite:
  mode: hybrid
  viunit_count: 3
  metamorphic_count: 3
  metamorphic:
    version: 1
```

These counts are upper limits, not guarantees of exactly that many generated
tests. Check the actual counts when comparing runs. An equal-budget experiment
could compare six ViUniT tests, six metamorphic tests, and three of each. Reuse
the same questions and candidate programs across conditions.

Both [`generate.py`](../generate.py) and [`train.py`](../train.py) use the new
per-question dispatcher. The standalone `butterfly/demo.py` and
`butterfly/demo_simple.py` scripts have not been wired to this toggle.
The legacy `use_program: true` ViUniT path remains available with its previous
cache behavior. Metamorphic and hybrid modes require `use_program: false`.

## Files and integration

| File | Responsibility |
| --- | --- |
| [`unit_test_generation/suites.py`](../unit_test_generation/suites.py) | Validate mode/counts, dispatch providers, combine tests, and manage cache identity. |
| [`unit_test_generation/metamorphic/generator.py`](../unit_test_generation/metamorphic/generator.py) | Define `MetamorphicTest` and the custom generator interface we need to implement. |
| [`unit_test_generation/metamorphic/__init__.py`](../unit_test_generation/metamorphic/__init__.py) | Export the generator and test record. |
| [`generate.py`](../generate.py), [`train.py`](../train.py) | Call the dispatcher and retain shared execution/scoring. Existing ViUniT orchestration is wrapped in `build_viunit_suite()`. |
| [`viunit_configs/base.yaml`](../viunit_configs/base.yaml) | Define default suite configuration. |
| [`tests/test_suites.py`](../tests/test_suites.py) | Exercise dispatch, alignment, cache reuse/invalidation, and invalid settings with fake providers. |

The flow is:

```text
Experiment YAML
    -> per-question suite dispatcher
        -> ViUniT: generate caption/answer candidates -> sample -> generate/retrieve images
        -> metamorphic: transform the original question image
        -> hybrid: combine the two sources after their separate selection steps
    -> shared program execution and scoring
```

The dispatcher fills the existing flat fields on each dataset record:

```python
record['selected_unit_tests']  # (description/caption, expected_answer) pairs
record['unit_test_images']     # Corresponding image groups
record['unit_test_metadata']   # Corresponding source labels and custom metadata
record['unit_test_suite_key']  # Cache fingerprint
```

The lists share the same ordering. ViUniT tests precede metamorphic tests in a
hybrid suite. Every candidate program for a question receives that same suite.
The existing scorer computes the combined suite score; separate source-level
score reporting is not implemented. Source labels support later analysis.

## Implementing our generator

Implement `generate_tests()` in
[`unit_test_generation/metamorphic/generator.py`](../unit_test_generation/metamorphic/generator.py).
Its keyword-only interface is:

```python
def generate_tests(
    *,
    question_id,
    question,
    original_image_path,
    original_answer,
    num_tests,
    output_dir,
    seed,
    config,
) -> list[MetamorphicTest]:
    ...
```

The dispatcher supplies `config` from `unit_test_suite.metamorphic`. No program
or iteration is passed because generation is per question. It creates
`output_dir` before calling the generator.

The implementation should select transformations, apply them to the original
image, save the transformed images, and return records such as:

```python
MetamorphicTest(
    image_path=transformed_path,  # Existing absolute Path to a transformed image
    expected_answer=original_answer,
    description='Horizontal reflection',
    metadata={
        'transformation': 'flip',
        'parameters': {'direction': 'horizontal'},
    },
)
```

Return between one and `num_tests` records. Use the seed for random choices and
leave the original image intact. Reuse the original answer only for
answer-preserving transformations; answer-changing tests require an explicit
new expected answer. Every expected answer must be a string. The dispatcher
adapts these records to the existing executor's test/image structures.

Reusable transformation code can go in a new
`unit_test_generation/metamorphic/transformations.py`. The existing
`llm_experiments/` decision outputs are not automatically imported or applied;
connect them inside the generator if we want to use them.

## Cache and artifacts

Custom transformed images are written by the generator under:

```text
<output_dir>/metamorphic/<cache-key>/
```

Suite fields and results are persisted in the existing `<output_dir>/data.p`.
The fingerprint includes question ID/text/answer, original image path and file
size/modification time, seed, suite settings, provider settings, prompt file
contents, and execution settings.

If the fingerprint changes, or selected tests/images are absent, cached test
candidates, prompts, selected tests, images, metadata, and unit-test results are
invalidated together before cached feedback can influence program generation.
Existing generated programs are retained. Old caches without fingerprints are
rebuilt once. A mismatch for any question currently rebuilds the whole batch to
preserve the existing ViUniT batching assumptions.

Generator source-code changes are not automatically fingerprinted. Increment a
setting such as `unit_test_suite.metamorphic.version` after changing the generator
to force regeneration. Cache validation also does not inspect whether saved
transformed-image files have subsequently been deleted; keep those artifacts
with the run or force regeneration by changing the version.

## Verification

From the repository root:

```bash
python -m unittest discover -s tests -v
python -m py_compile generate.py train.py unit_test_generation/suites.py
```


