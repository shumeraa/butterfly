
#!/usr/bin/env python3
"""Download Qwen weights or run text inference smoke tests."""

import argparse
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download:
        from huggingface_hub import snapshot_download

        model = os.getenv("MODEL_ID", "Qwen/Qwen3.8-27B")
        revision = os.getenv("MODEL_REVISION", "main")

        path = snapshot_download(model, revision=revision)
        print(f"Downloaded model to: {path}")
        return

    run_tests()


def run_tests():
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3.8-27B")
    revision = os.getenv("MODEL_REVISION", "main")
    tp_size = int(os.getenv("TP_SIZE", "2"))
    dtype = os.getenv("DTYPE", "bfloat16")
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "2048"))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "128"))
    gpu_utilization = float(
        os.getenv("GPU_MEMORY_UTILIZATION", "0.90")
    )

    result_dir = Path(
        os.getenv(
            "RESULTS_DIR",
            "qwen38-results/manual-" + time.strftime("%Y%m%d-%H%M%S"),
        )
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    # Check configuration, not GPU model or arbitrary VRAM limits.
    if tp_size < 1:
        raise ValueError("TP_SIZE must be positive")
    if max_model_len < 1 or max_new_tokens < 1:
        raise ValueError("Token limits must be positive")
    if not 0 < gpu_utilization <= 1:
        raise ValueError("GPU_MEMORY_UTILIZATION must be in (0, 1]")

    # Avoid manual CUDA initialization before vLLM.
    # vLLM validates hardware, device count and available memory.
    metadata = {
        "model": model_id,
        "requested_revision": revision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "cuda": torch.version.cuda,
        "visible_gpus": os.getenv("CUDA_VISIBLE_DEVICES"),
        "tensor_parallel_size": tp_size,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "max_new_tokens": max_new_tokens,
        "gpu_memory_utilization": gpu_utilization,
        "thinking": False,
    }

    metadata_path = result_dir / "metadata.json"

    def save_metadata():
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n"
        )

    save_metadata()
    print(json.dumps(metadata, indent=2), flush=True)

    print("Loading model...", flush=True)
    start = time.perf_counter()

    llm = LLM(
        model=model_id,
        revision=revision,
        dtype=dtype,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=gpu_utilization,
        seed=0,
    )

    metadata["load_seconds"] = round(
        time.perf_counter() - start, 3
    )
    save_metadata()

    tokenizer = llm.get_tokenizer()

    def generate(prompt, limit):
        messages = [{"role": "user", "content": prompt}]

        # Disable thinking when the tokenizer supports this option.
        # Otherwise use its normal chat template.
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        start = time.perf_counter()

        output = llm.generate(
            [text],
            SamplingParams(
                temperature=0.0,
                max_tokens=limit,
            ),
            use_tqdm=False,
        )[0].outputs[0]

        seconds = time.perf_counter() - start

        raw = output.text.strip()
        answer = raw.rsplit("</think>", 1)[-1].strip()
        count = len(output.token_ids)

        return {
            "raw_response": raw,
            "answer": answer,
            "generated_tokens": count,
            "seconds": round(seconds, 3),
            "tokens_per_second": round(
                count / seconds, 2
            ) if seconds > 0 else 0,
        }

    print("Warming up...", flush=True)
    generate("Reply with OK.", 8)

    tests = [
        (
            "arithmetic",
            "What is 17 * 23? Reply with only the integer.",
            "391",
        ),
        (
            "fact",
            "What is the capital of France? "
            "Reply with only the city name.",
            "Paris",
        ),
        (
            "instruction",
            "Reply with exactly this word and nothing else: butterfly",
            "butterfly",
        ),
    ]

    passed = 0

    with (result_dir / "responses.jsonl").open("w") as handle:
        for name, prompt, expected in tests:
            result = generate(prompt, max_new_tokens)

            # A simple answer-quality check, not an inference
            # success/failure condition.
            normalized = re.sub(
                r"[\s.!]+$", "", result["answer"]
            ).casefold()

            ok = normalized == expected.casefold()
            passed += int(ok)

            record = {
                "test": name,
                "prompt": prompt,
                "expected": expected,
                **result,
                "passed": ok,
            }

            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record, indent=2), flush=True)

    # Snapshot GPU usage without initializing CUDA in the parent.
    memory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ) if subprocess.run(
        ["which", "nvidia-smi"],
        capture_output=True,
        check=False,
    ).returncode == 0 else None

    summary = {
        "inference_completed": True,
        "passed": passed,
        "total": len(tests),
        "gpu_memory_snapshot_csv": (
            memory.stdout.strip()
            if memory is not None and memory.returncode == 0
            else None
        ),
    }

    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2), flush=True)

    # Successful inference is a successful smoke test.
    # Answer correctness is recorded separately.


if __name__ == "__main__":
    main()
