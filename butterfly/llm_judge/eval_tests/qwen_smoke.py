
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="Qwen/Qwen3.8-27B",
        dtype="bfloat16",
        tensor_parallel_size=2,
        max_model_len=2048,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        seed=0,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
    )

    outputs = llm.generate(
        ["Explain what a GPU is in one sentence."],
        sampling_params,
    )

    for output in outputs:
        print("OUTPUT:", output.outputs[0].text)


if __name__ == "__main__":
    main()

