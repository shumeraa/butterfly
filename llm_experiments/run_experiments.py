from pathlib import Path
from datasets import load_dataset
from itertools import islice
import json
from openai import OpenAI
import pandas as pd
import base64
import mimetypes
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv




PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
QA_FILE = DATASET_DIR / "qa_pairs.json"
IMAGE_DIR = DATASET_DIR / "images"
OUTPUT_DIR = DATASET_DIR / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "outputs.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORMATIONS = {
    "brightness": {
        "description": "Adjust global brightness",
        "parameters": {"factor": [0.8, 1.2]}
    },
    "contrast": {
        "description": "Adjust global contrast",
        "parameters": {"factor": [0.8, 1.2]}
    },
    "gaussian_noise": {
        "description": "Add mild Gaussian pixel noise",
        "parameters": {"sigma": [2, 8]}
    },
    "horizontal_flip": {
        "description": "Mirror image horizontally",
        "parameters": {}
    },
    "rotation": {
        "description": "Rotate entire image slightly",
        "parameters": {"degrees": [-10, 10]}
    },
    "background_blur": {
        "description": "Blur background without modifying "
                       "the question-relevant foreground",
        "parameters": {"radius": [2, 5]}
    },
    "color_filter": {
        "description": "Apply a color filter to the whole image",
        "parameters": {}
    },
    "background_color": {
        "description": "Change background colors while "
                       "preserving foreground objects",
        "parameters": {}
    },
    "object_removal": {
        "description": "Remove a question-irrelevant object with masking",
        "parameters": {}
    },
    "none": {
        "description": "No sufficiently safe transformation",
        "parameters": {}
    }
}

VARIANCE_MODE="invariant" # Change to variant for variant test experimentation

SYSTEM_PROMPT = f"""
You are a metamorphic testing decision module for
visual question answering (VQA).

Given an image (when available), a question, and its
ground-truth answer, choose ONE answer-{VARIANCE_MODE} image transformation.

Your goal is to generate answer-preserving metamorphic
transformations for evaluating VQA systems.

Reason about:
1. What visual evidence supports the answer?
2. Which objects, attributes, and spatial relationships
   must remain unchanged?
3. Which regions or attributes can be modified?
4. Which available transformation is most suitable?

Return only a JSON object with these fields:
{{
  "transformation": "transformation name",
  "parameters": {{}},
  "target_object": "description or null",
  "preserve": ["visual invariant"],
  "reason": "short explanation",
  "confidence": 0.0
}}

Confidence is a heuristic judgment, not a calibrated
probability of answer preservation.
"""
MODEL = "gpt-6-luna"

MAX_EXAMPLES = 25

def check_dirs_exist():
    print("Dataset exists:", DATASET_DIR.exists())
    print("QA file exists:", QA_FILE.exists())
    print("Images exist:", IMAGE_DIR.exists())
    return DATASET_DIR.exists() and QA_FILE.exists() and IMAGE_DIR.exists()

def load_dataset_srcs():
    dataset = load_dataset(
        "Mineru/GQA",
        split="testdev_balanced",
        streaming=True
    )

    qa_pairs = []

    for i, example in enumerate(islice(dataset, 100)):
        image = example["image"]
        image_id = f"image_{i:04d}"
        filename = f"{image_id}.jpg"

        image.convert("RGB").save(IMAGE_DIR / filename)

        qa_pairs.append({
            "image_id": image_id,
            "question_id": example["question_id"],
            "image": f"images/{filename}",
            "question": example["question"],
            "answer": example["answer"]
        })

    with open(QA_FILE, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Saved {len(qa_pairs)} QA pairs")

def init_openai_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")

    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized successfully!")
    return client

def load_qa_pairs(file_path):
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", data.get("qa_pairs", []))

    elif file_path.suffix.lower() == ".csv":
        data = pd.read_csv(file_path).fillna("").to_dict(
            orient="records"
        )
    else:
        raise ValueError("Supported formats: JSON, CSV")

    if not isinstance(data, list):
        raise ValueError("Expected a list of QA records")

    for i, item in enumerate(data):
        if "question" not in item or "answer" not in item:
            raise ValueError(f"Missing QA fields at index {i}")

    return data
def encode_image(image_path):
    image_path = Path(image_path)

    mime_type = mimetypes.guess_type(
        str(image_path)
    )[0] or "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"



def decide_transformation(qa_pair):
    question = str(qa_pair["question"])
    answer = str(qa_pair["answer"])

    prompt = f"""Question: {question} Ground-truth answer: {answer}"""

    content = [
        {"type": "input_text", "text": prompt}
    ]

    # Optional: include image for visual reasoning
    '''image_path = qa_pair.get("image")

    if INCLUDE_IMAGES and image_path:
        full_path = DATASET_DIR / image_path

        if not full_path.is_file():
            raise FileNotFoundError(full_path)

        content.append({
            "type": "input_image",
            "image_url": encode_image(full_path),
            "detail": "auto"
        })'''

    # Convert transformation dictionary to choices
    choices = [
        {
            "value": name,
            "description": (
                info["description"]
                if isinstance(info, dict)
                else str(info)
            )
        }
        for name, info in TRANSFORMATIONS.items()
    ]

    response = client.decisions.create(
        model=MODEL,
        input=[
            {
                "role": "user",
                "content": content
            }
        ],
        questions=[
            {
                "type": "choice",
                "name": "transformation",
                "instructions": (SYSTEM_PROMPT),
                "choices": choices
            }
        ]
    )

    # Extract the choice answer
    choice_answer = response.answers[0]

    name = choice_answer.choice

    if name not in TRANSFORMATIONS:
        raise ValueError(
            f"Unknown transformation: {name}"
        )

    probabilities = [
        {
            "transformation": p.value,
            "probability": p.probability
        }
        for p in (choice_answer.probabilities or [])
    ]

    return {
        "transformation": name,
        "parameters": {},
        "target_object": None,
        "preserve": [],
        "reason": None,
        "confidence": choice_answer.confidence,
        "probabilities": probabilities
    }

def save_results_chart():
    # Load results
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Extract successful transformation decisions
    transformations = []

    for record in results:
        if record.get("status") != "success":
            continue

        decision = record.get("decision", {})
        transformation = decision.get("transformation")

        if transformation:
            transformations.append(transformation)

    # Count selections
    counts = pd.Series(transformations).value_counts()

    print(f"Total records: {len(results)}")
    print(f"Successful decisions: {len(transformations)}")
    print("\nTransformation frequencies:")
    print(counts.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        counts.index,
        counts.values,
        color="#4472C4",
        edgecolor="#2B2B2B",
        linewidth=0.7
    )

    # Display count above each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            str(int(height)),
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            xytext=(0, 5),
            textcoords="offset points"
        )

    ax.set_title(
        "OpenAI Decisions: Selected Image Transformations",
        fontsize=17,
        fontweight="bold",
        pad=18
    )
    ax.set_xlabel("Transformation", fontsize=13)
    ax.set_ylabel("Number of Selections", fontsize=13)

    ax.tick_params(axis="x", labelrotation=35, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    plt.setp(ax.get_xticklabels(), ha="right")

    ax.set_ylim(0, max(counts.values) * 1.2)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save graph
    CHART_FILE = OUTPUT_DIR / "transformation_distribution.png"
    plt.savefig(CHART_FILE, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nChart saved to: {CHART_FILE}")

if __name__ == "__main__":
    check_dirs_exist()
    load_dataset_srcs()

    client = init_openai_client()

    qa_pairs = load_qa_pairs(QA_FILE)
    print(f"Loaded {len(qa_pairs)} QA pairs")

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            results = json.load(f)
    else:
        results = []

    completed = {
        result["record_index"]
        for result in results
        if result.get("status") == "success"
    }

    for index, qa in enumerate(
        tqdm(qa_pairs[:MAX_EXAMPLES])
    ):
        if index in completed:
            continue

        try:
            decision = decide_transformation(qa)

            result = {
                "record_index": index,
                "image_id": qa.get("image_id"),
                "image": qa.get("image"),
                "question": qa["question"],
                "answer": qa["answer"],
                "status": "success",
                "decision": decision
            }

        except Exception as e:
            result = {
                "record_index": index,
                "image_id": qa.get("image_id"),
                "status": "error",
                "error": str(e)
            }

        results = [
            r for r in results
            if r.get("record_index") != index
        ]
        results.append(result)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Saved {len(results)} records to {OUTPUT_FILE}")


