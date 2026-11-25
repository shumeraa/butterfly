# Metamorphic transformations vs. ViUniT for selecting visual programs (gqa-simple-v5)

## 1. Overview

**Task.** For each GQA question a code LLM (CodeLlama, ViperGPT-style prompt) writes 5 candidate programs. A *selector* must pick one program without seeing the gold answer. We compare selectors by the accuracy of the program they pick.

**Data.** 601 questions on 581 images, covering 102 GQA question types (about 6 each). Every program was executed on the real image in two separate ways. (1) *Scoring:* its output is compared with the GQA gold answer to give a correct/wrong label. No selector sees this label. (2) *Metamorphic reference:* its output on the unedited image becomes the reference that its answers on edited images are checked against. The gold answer is never used for this, so the metamorphic score measures self-consistency, not correctness.

**Selectors.**
- *random*: pick uniformly.
- *majority vote*: pick the most common program output. Crashed programs never agree with each other.
- *ViUniT*: score each program on 5 unit tests built from text descriptions of imagined images.
- *metamorphic*: run each program on edited versions of the real image (hflip, vflip, grayscale, color fill, object occlusion). A program passes an edit if its new answer matches what its own answer on the original image implies. Each question gets a different set of edits: flips and grayscale apply to any image, but color fill and occlusion need an object in the question that can be found and edited. Questions have 0 to 5 edits (mostly 2 to 4), so scores are shares of different-sized sets.
  - *Usually the answer should not change (invariant).* Flipping an image top-to-bottom should not change the answer to "Is there a car?", so the rule expects the same answer the program gave on the original image.
  - *Sometimes the answer should change, in a predictable way (equivariant).* Flipping an image left-to-right swaps left and right. If a program answered "left" to "Is the cat on the left or right?", it should answer "right" on the flipped image. So the rule says: original "left" → expected "right", and the reverse.
  - *Sometimes the answer should change, but we can't say to what.* Take "Is the cat on the left of the dog?" If the program said "yes", the flipped image should give "no". If it said "no", the flipped answer could be "yes" or still "no" (for example, if the cat is not beside the dog at all). The rule has nothing to predict, so it is marked `unchecked` and the transformation is skipped. A skipped transformation is left out of the program's score, as if it was never run.
  - *Checking occlusion with Grounding DINO.* To occlude an object (say "vase"), Grounding DINO, a text-prompted object detector, first finds it and its box is painted over. We then run DINO a second time on the edited image, with a lower detection threshold, to confirm the object is really gone. If it still finds the object outside the painted area, that area is painted too, for up to two extra rounds. If the object is still detected after that, the occlusion transformation is dropped for that question. This matters because the expected answer after occlusion is "the object is not there". If part of the object is still visible, a program that correctly sees it would be marked wrong.
- *oracle*: pick a correct program whenever one exists.
- *combinations*: ViUniT and metamorphic combined as a tie-break, or pooled tests.

**Figure 1. Edits applied to one image.** Question: "Is there a truck in the photograph?" (gold answer: yes). A program that answered "yes" on the original should still answer "yes" after grayscale, flips, or turning the truck red (invariant). After the truck is occluded it should answer "no" (equivariant). The occlusion box is rectangular, so it also hides part of the boat.

![Original image and five edits, each with its expected answer](images/fig1_transformations.png)

**Metric.** Accuracy of the chosen program. When programs tie for best, the score is the fraction of the tied programs that are correct (equal to picking one at random). 95% CIs are paired bootstrap over questions (5000 resamples).

**Key structure.** The dataset has 601 questions, but the results that matter come from the **344 "mixed" questions**, where at least one of the 5 programs is right and at least one is wrong. Only there can a selector make a difference. On the other 257 (208 where all 5 programs are wrong, 49 where all 5 are right) every selector gets the same score, so they say nothing about selection quality and only dilute the comparison. Accuracy is also higher on the mixed subset than on all questions (random: 0.455 vs. 0.342), because the 208 all-wrong questions, where everyone scores 0, are removed. Read mixed-subset numbers against random on the same rows, not against the all-question numbers.

## 2. Results

### Table 1. Baselines, ViUniT, metamorphic (accuracy of selected program)

| selector | all (n=601) | mixed (n=344) |
|---|---|---|
| random | 0.342 | 0.455 |
| majority vote | 0.438 | 0.624 |
| ViUniT | 0.448 | 0.640 |
| metamorphic | 0.421 | 0.593 |
| oracle | 0.654 | 1.000 |

Metamorphic beats random by +0.079 on all questions and +0.138 on mixed ones, but it trails ViUniT (−0.027 all, CI [−0.054, +0.001]; −0.047 mixed, CI [−0.095, −0.000]). ViUniT and majority vote are statistically indistinguishable (−0.010, CI [−0.037, +0.018] for majority vs. ViUniT on all questions).

### Table 2. Combining ViUniT and metamorphic (mixed questions, n=344)

| selector | accuracy | Δ vs ViUniT (95% CI) |
|---|---|---|
| ViUniT | 0.640 | – |
| ViUniT, ties broken by metamorphic | 0.677 | +0.037 [+0.017, +0.056] |
| metamorphic, ties broken by ViUniT | **0.686** | +0.046 [+0.006, +0.082] |
| pooled tests (5 ViUniT + n metamorphic) | 0.684 | +0.044 [+0.011, +0.074] |

*Tie-break.* When several programs get the same top score from one selector, that selector cannot choose between them. A tie-break lets the other selector decide among just those programs. For example, "ViUniT, ties broken by metamorphic" keeps the programs with the best ViUniT score, then picks the one with the best metamorphic score. If they are still tied, one is picked at random.

Every combination beats ViUniT alone, by +0.037 to +0.046, with CIs that exclude 0. The combinations are close to each other, so no one of them is clearly the best. The oracle (1.000) leaves a large gap that none of them closes.

## 3. Analysis

### 3.1 Why metamorphic alone trails ViUniT
A metamorphic transformation checks that the answer is *stable*, not that it is *right*. A program that ignores the image and always answers "no" passes every flip and grayscale transformation.
- Programs with a perfect metamorphic score are correct only 47% of the time. Programs scoring below 0.5 are correct 12% of the time.
- The score is good at rejecting bad programs and weak at identifying good ones.
- Within a question, the chance that a correct program outscores a wrong one is 0.68 for metamorphic against 0.71 for ViUniT and 0.73 for majority vote.

### 3.2 Why combining helps
The two signals are partly independent, and the clearest gain is breaking ties. ViUniT leaves all 5 programs tied on 100 questions. On the 24 of those that are mixed, ViUniT scores 0.48 and metamorphic scores 0.70.

### 3.3 Performance by question type (mixed questions)
Each row is a slice of the 344 mixed questions, and each number is the accuracy of the program that selector picks on that slice. A selector is useful to the extent that it beats the random column on the same row. Rows in the first block split by the GQA gold answer. Rows in the second block split by whether the question text mentions "left" or "right". These blocks overlap, because a question can appear in both.

| slice | n | random | ViUniT | metamorphic |
|---|---|---|---|---|
| gold answer = yes | 110 | 0.47 | 0.60 | 0.55 |
| gold answer = no | 129 | 0.48 | 0.66 | 0.62 |
| gold answer is open-ended (a word) | 105 | 0.41 | 0.66 | 0.60 |
| question mentions left/right | 41 | 0.47 | 0.63 | 0.63 |
| question does not mention left/right | 303 | 0.45 | 0.64 | 0.59 |

- Metamorphic trails ViUniT on every gold-answer type, most on "yes".
- Left/right is the one slice where metamorphic matches ViUniT. Likely reason: horizontal-flip transformations are often skipped on these questions (see `unchecked` in Section 1), so the score rests mostly on the other edits. Only 43% of left/right questions keep a flip transformation, against 97% of the rest. This is an explanation I have not tested, and with n=41 the 0.04 gap to other questions is within noise.

### 3.4 Part of the signal is crash detection
About a third of wrong programs (307 of 937 on mixed questions) crash. A changed image changes whether or not a program crashes, so metamorphic scores can separate those programs easily. On mixed questions where no program crashed (n=147), metamorphic falls to 0.52, close to random (0.50), while ViUniT stays at 0.65.

### 3.5 Performance by transform present in the question (mixed questions)

| transform | n | ViUniT | metamorphic |
|---|---|---|---|
| occlude_bbox | 12 | 0.44 | **0.82** |
| fill_color | 176 | 0.63 | 0.60 |
| hflip / vflip / grayscale | 262–311 | 0.65–0.66 | 0.59–0.60 |

Each row groups the questions that were given that edit. Occlusion is the only edit where metamorphic clearly beats ViUniT (0.82 vs. 0.44). Occlusion blanks out the object the question asks about, so a program that really looks at the image should change its answer to "the object is gone", while a program that guesses will not. Color fill, flips and grayscale all look about the same.
