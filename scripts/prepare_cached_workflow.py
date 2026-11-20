"""Create photo-backed dummy fixtures from a generated custom-test JSON file."""
import argparse
import json
import pickle
import shutil
from pathlib import Path


def prepare(image, tests_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    photo = destination / 'dummy.jpg'
    if Path(image).resolve() != photo.resolve():
        shutil.copy2(image, photo)
    generated = json.loads(Path(tests_path).read_text())
    annotations, programs, tests = [], [], {}
    for question_id, outputs, question in [
        ('006149', ['left', 'right', 'above'], 'Which side?'),
        ('006108', ['yes', 'no', 'maybe'], 'Is the dummy answer yes?'),
    ]:
        annotations.append(dict(questionId=question_id, imageId='dummy',
                                question=question, answer=outputs[0]))
        codes = [f'def execute_command(image):\n    return {answer!r}\n' for answer in outputs]
        codes.append('def execute_command(image):\n    raise RuntimeError("intentional baseline failure")\n')
        programs.append(dict(generated_code={0: codes}))
        tests[question_id] = [dict(case, image_paths=[str(photo)])
                              for case in generated[question_id]]
    (destination / 'annotations.json').write_text(json.dumps(annotations, indent=2) + '\n')
    (destination / 'custom_tests.json').write_text(json.dumps(tests, indent=2) + '\n')
    with (destination / 'cached_programs.p').open('wb') as stream:
        pickle.dump(programs, stream)
    print(f'Prepared {len(annotations)} questions, 8 candidates, '
          f'{sum(map(len, tests.values()))} custom tests using {photo}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--tests', required=True, type=Path)
    parser.add_argument('--destination', type=Path, default=Path('fixtures/cached_workflow'))
    args = parser.parse_args()
    prepare(args.image, args.tests, args.destination)
