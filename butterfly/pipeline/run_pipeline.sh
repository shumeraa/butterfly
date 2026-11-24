#!/usr/bin/env bash
# Login-node driver for the full metamorphic pipeline on the merged question set.
#   bash butterfly/pipeline/run_pipeline.sh [PROMPT_VERSION] [--test10] [--nodes N]
#     PROMPT_VERSION  default gqa-simple-v5 (the previous prompt). Newest is gqa-simple-v7.
#     --test10        run only the first 10 questions into <version>_test10 (dry run, ~20 min)
#     --nodes N       1 or 2 (default 2; the dev QOS allows 1 running job / 2 nodes / 3 submitted)
# Output: $ROOT/<version>/{judge_run,metamorphic_eval}; $ROOT/merged holds the shared inputs.
# Re-running the same command resumes; finished stages are skipped (DONE markers).
# The merged inputs ($ROOT/merged) must exist: run butterfly/metamorphic_eval/merge_dataset.py with
#   SOURCES=<v1 root>:<v2 root> OUT=$ROOT/merged   once the v2 chunks are all DONE.
set -euo pipefail
VERSION=gqa-simple-v5; TEST=0; NODES=2
while [ $# -gt 0 ]; do
    case $1 in
        --test10) TEST=1;;
        --nodes) NODES=$2; shift;;
        gqa-simple-*) VERSION=$1;;
        *) echo "unknown arg $1" >&2; exit 2;;
    esac; shift
done
D=/work/11808/shumera/ls6/butterfly_shared_resources/datasets
export ROOT=${ROOT:-$D/viz_program_eval_600}
PROJECT=/work/11808/shumera/ls6/projects/butterfly
[ -d "$PROJECT/butterfly/llm_judge/versions/$VERSION" ] || [ "$VERSION" = gqa-simple-v7 ] \
    || { echo "unknown prompt version $VERSION" >&2; exit 1; }
M=$ROOT/merged
[ -f "$M/MANIFEST.json" ] || { echo "Missing $M/MANIFEST.json: run merge_dataset.py first (see header)" >&2; exit 1; }

NAME=$VERSION; [ $TEST = 1 ] && NAME=${VERSION}_test10
V=$ROOT/$NAME
mkdir -p "$V" "$ROOT/logs"
if [ $TEST = 1 ]; then
    head -10 "$M/questions.jsonl" > "$V/questions.jsonl"
    python3 -c "
import json,sys
ids={json.loads(l)['id'] for l in open('$V/questions.jsonl')}
open('$V/test_ids.txt','w').write('\n'.join(sorted(ids))+'\n')
with open('$V/dataset.jsonl','w') as f:
    for l in open('$M/dataset.jsonl'):
        if json.loads(l)['questionId'] in ids: f.write(l)"
else
    ln -sfn "$M/questions.jsonl" "$V/questions.jsonl"
    ln -sfn "$M/dataset.jsonl" "$V/dataset.jsonl"
fi
ln -sfn "$M/data.p" "$V/data.p"; ln -sfn "$M/real_pass" "$V/real_pass"; ln -sfn "$M/MANIFEST.json" "$V/MANIFEST.json"
echo "$VERSION" > "$V/prompt_version.txt"

jid=$(sbatch --parsable -N "$NODES" -J "pipe-$NAME" \
      -o "$ROOT/logs/pipe-$NAME.%j.out" -e "$ROOT/logs/pipe-$NAME.%j.out" \
      --export=ALL,ROOT="$ROOT",VERSION="$VERSION",NAME="$NAME",TEST="$TEST" \
      "$PROJECT/butterfly/pipeline/pipeline_job.sh" | tail -1)
echo "submitted job $jid ($NAME, $NODES node(s)); log: $ROOT/logs/pipe-$NAME.$jid.out"
