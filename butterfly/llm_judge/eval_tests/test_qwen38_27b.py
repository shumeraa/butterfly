#!/usr/bin/env python3
"""Download Qwen weights or run the text inference smoke tests."""

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
    parser.add_argument("--download", action="store_true", help="Download weights and exit")
    args = parser.parse_args()
    os.environ.setdefault("MODEL_ID", "Qwen/Qwen3.8-27B")
    os.environ.setdefault("MODEL_REVISION", "main")
    os.environ.setdefault("MAX_NEW_TOKENS", "128")
    os.environ.setdefault("RESULTS_DIR", "qwen38-results/manual-" + time.strftime("%Y%m%d-%H%M%S"))
    if args.download:
        from huggingface_hub import snapshot_download
        print(snapshot_download(os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"]))
        return
    run_tests()


def run_tests():
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    Path(os.environ["RESULTS_DIR"]).mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Check the GPU allocation and PyTorch installation.")
    if torch.cuda.device_count() != 2:
        raise SystemExit(f"Expected exactly 2 visible A100 GPUs; found {torch.cuda.device_count()}.")
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        if "A100" not in name.upper():
            raise SystemExit(f"Expected an A100 at GPU {index}; found {name}.")
    max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
    if max_new_tokens < 1:
        raise SystemExit("MAX_NEW_TOKENS must be positive.")

    devices = []
    usable_memory = 0
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        devices.append({"index": index, "name": torch.cuda.get_device_name(index),
                        "total_gib": round(total / 2**30, 2)})
        usable_memory += int(free * 0.90)
    if usable_memory < 60 * 2**30:
        raise SystemExit("Need at least 60 GiB usable GPU memory across the two A100s.")

    model_id = os.environ["MODEL_ID"]
    revision = os.environ["MODEL_REVISION"]
    result_dir = Path(os.environ["RESULTS_DIR"])
    metadata = {
        "model": model_id, "requested_revision": revision,
        "python": platform.python_version(), "torch": torch.__version__,
        "vllm": vllm.__version__, "cuda": torch.version.cuda,
        "devices": devices, "max_new_tokens": max_new_tokens,
        "dtype": "bfloat16", "thinking": False,
        "tensor_parallel_size": 2, "pipeline_parallel_size": 1,
        "max_model_len": 2048, "gpu_memory_utilization": 0.90,
    }
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    start = time.perf_counter()
    # Shard the model across the two allocated GPUs using BF16 weights.
    llm = LLM(
        model=model_id, revision=revision, tokenizer_revision=revision,
        dtype="bfloat16", tensor_parallel_size=2, pipeline_parallel_size=1,
        distributed_executor_backend="mp", gpu_memory_utilization=0.90,
        max_model_len=2048, max_num_seqs=1, enforce_eager=True,
        language_model_only=True, cpu_offload_gb=0, seed=0,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    metadata["load_seconds"] = round(time.perf_counter() - start, 3)
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    def generate(prompt, limit):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        start = time.perf_counter()
        output = llm.generate(
            [text], SamplingParams(temperature=0.0, max_tokens=limit),
            use_tqdm=False,
        )[0].outputs[0]
        seconds = time.perf_counter() - start
        raw = output.text.strip()
        tokens = output.token_ids
        # Retain raw output; score the final answer if a reasoning delimiter appears.
        answer = raw.rsplit("</think>", 1)[-1].strip()
        return raw, answer, len(tokens), seconds

    print("Warming up...", flush=True)
    generate("Reply with OK.", 8)

    tests = [
        ("arithmetic", "What is 17 * 23? Reply with only the integer.", "391"),
        ("fact", "What is the capital of France? Reply with only the city name.", "Paris"),
        ("instruction", "Reply with exactly this word and nothing else: butterfly", "butterfly"),
    ]
    passed = 0
    with (result_dir / "responses.jsonl").open("w") as handle:
        for name, prompt, expected in tests:
            raw, answer, count, seconds = generate(prompt, max_new_tokens)
            ok = re.sub(r"[\s.!]+$", "", answer).casefold() == expected.casefold()
            passed += int(ok)
            record = {"test": name, "prompt": prompt, "expected": expected,
                      "raw_response": raw, "answer": answer, "passed": ok,
                      "generated_tokens": count, "seconds": round(seconds, 3),
                      "tokens_per_second": round(count / seconds, 2)}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record, indent=2), flush=True)

    # vLLM owns worker processes, so query GPU memory through nvidia-smi
    # rather than reporting the parent process's PyTorch allocator statistics.
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    summary = {"passed": passed, "total": len(tests),
               "gpu_memory_snapshot_csv": memory.stdout.strip() if memory.returncode == 0 else None}
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    raise SystemExit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
