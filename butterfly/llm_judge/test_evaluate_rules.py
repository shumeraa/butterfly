import json
import re
import unittest
from pathlib import Path

from common import validate_prediction
from evaluate_rules import INVARIANT, UNSAFE, judge
from prompts import PROMPT, PROMPT_VERSION


class QuestionOnlyRulesTests(unittest.TestCase):
    def verdicts(self, question, target, *flags):
        return judge(question, validate_prediction({'target': target, 'flags': list(flags)}))

    def fill(self, question, target, *flags):
        return self.verdicts(question, target, *flags)['mr2'][target]['fill_color']['red']

    def test_documented_prompt_matches_runtime_and_examples_validate(self):
        docs = Path(__file__).with_name('SIMPLE_PROMPTS.md').read_text(encoding='utf-8')
        self.assertIn(f'({PROMPT_VERSION.removeprefix("gqa-simple-")})', docs.splitlines()[0])
        self.assertEqual(docs.split('````text\n', 1)[1].split('\n````', 1)[0], PROMPT)
        for question, response in re.findall(r'^Question: (.+)\n(\{.+\})$', PROMPT, re.MULTILINE):
            with self.subTest(question=question):
                validate_prediction(json.loads(response))

    def test_fixed_pair_swaps_both_yes_no_answers(self):
        for direction, question in (
            ('hflip', 'Is the ball to the left of the box?'),
            ('vflip', 'Is the ball above the box?'),
        ):
            yn_flag = 'LR_YN' if direction == 'hflip' else 'UD_YN'
            with self.subTest(direction=direction):
                self.assertEqual(self.verdicts(question, 'ball', yn_flag)[direction], UNSAFE)
                self.assertEqual(self.verdicts(question, 'ball', yn_flag, 'FIXED_PAIR')[direction],
                                 {'equivariant': {'yes': 'no', 'no': 'yes'}})

    def test_existential_or_indefinite_relation_cannot_map_yes_to_no(self):
        for question, target in (
            ('Are there mirrors to the left of the lamp?', 'mirrors'),
            ('Do you see glasses to the right of the device?', 'glasses'),
            ('Is the man to the left of a boy?', 'man'),
            ('Is the cup to the left of any plate?', 'cup'),
        ):
            with self.subTest(question=question):
                result = self.verdicts(question, target, 'LR_YN', 'FIXED_PAIR')
                self.assertEqual(result['hflip'], UNSAFE)
        result = self.verdicts('Are there flags above the roof?', 'flags', 'UD_YN', 'FIXED_PAIR')
        self.assertEqual(result['vflip'], UNSAFE)

    def test_frame_half_uses_yes_no_not_direction_answers(self):
        for direction, subtype, question, target in (
            ('hflip', 'LR', 'Is the towel on the left?', 'towel'),
            ('hflip', 'LR', 'Is the fence on the right of the photo?', 'fence'),
            ('vflip', 'UD', 'Is the knife in the bottom part?', 'knife'),
        ):
            with self.subTest(question=question):
                self.assertEqual(self.verdicts(question, target, subtype + '_SIDE')[direction], UNSAFE)
                self.assertEqual(self.verdicts(question, target, subtype + '_YN', 'FIXED_PAIR')[direction],
                                 {'equivariant': {'yes': 'no', 'no': 'yes'}})

    def test_boundary_wording_vetoes_the_two_side_assumption(self):
        question = 'Is the ball centered to the left of the box?'
        self.assertEqual(self.verdicts(question, 'ball', 'LR_YN', 'FIXED_PAIR')['hflip'], UNSAFE)

    def test_implicit_lateral_side_does_not_require_direction_word(self):
        question = 'On which side of the photo is the woman?'
        recognized = self.verdicts(question, 'woman', 'LR_SIDE')
        self.assertEqual(recognized['hflip'], {'equivariant': {'left': 'right', 'right': 'left'}})
        self.assertEqual(recognized['vflip'], INVARIANT)
        self.assertEqual(self.verdicts(question, 'woman')['hflip'], UNSAFE)

    def test_open_image_region_has_no_guaranteed_axis(self):
        for flags in ((), ('LR_SIDE', 'UD_SIDE')):
            result = self.verdicts('Which part of the image are the people in?', 'people', *flags)
            self.assertEqual((result['hflip'], result['vflip']), (UNSAFE, UNSAFE))

    def test_explicit_choices_keep_the_correct_axis(self):
        result = self.verdicts('Is the paddle to the right or to the left of the man?', 'paddle', 'LR_SIDE')
        self.assertEqual(result['hflip'], {'equivariant': {'left': 'right', 'right': 'left'}})
        result = self.verdicts('Which part of the picture is the knife in, the bottom or the top?',
                               'knife', 'UD_SIDE')
        self.assertEqual(result['hflip'], INVARIANT)
        self.assertEqual(result['vflip']['equivariant']['bottom'], 'top')

    def test_fill_requires_affirmative_evidence(self):
        for question, target in (
            ('What is the woman wearing?', 'woman'),
            ('Is the person wearing a bracelet?', 'person'),
            ('Is the sky clear or foggy?', 'sky'),
            ('What type of food is near the knife?', 'food'),
        ):
            with self.subTest(question=question):
                self.assertEqual(self.fill(question, target), UNSAFE)
        self.assertEqual(self.fill('What color is the apple?', 'apple', 'COLOR_QUERY'), UNSAFE)

    def test_reliable_fill_cases_remain_scorable(self):
        self.assertEqual(self.fill('How large is the bench?', 'bench', 'FILL_SAFE'), INVARIANT)
        self.assertEqual(self.fill('What color is the apple?', 'apple', 'COLOR_QUERY', 'FILL_SAFE'),
                         {'equivariant': {'any': 'red'}})
        self.assertEqual(self.fill('Is the car red?', 'car', 'COLOR_VERIFY', 'FILL_SAFE'),
                         {'equivariant': {'yes': 'yes', 'no': 'yes'}})
        self.assertEqual(self.fill('Is the bag pink?', 'bag', 'COLOR_VERIFY', 'FILL_SAFE'), UNSAFE)

    def test_color_selector_does_not_get_a_yes_no_mapping(self):
        self.assertEqual(self.fill('What animal is black?', 'black animal', 'COLOR_VERIFY', 'FILL_SAFE'), UNSAFE)

    def test_fill_guard_does_not_change_plain_occlusion(self):
        result = self.verdicts('Is there a cat in the image?', 'cat')
        self.assertEqual(result['mr2']['cat']['occlude_bbox'], {'equivariant': {'any': 'no'}})


if __name__ == '__main__':
    unittest.main()
