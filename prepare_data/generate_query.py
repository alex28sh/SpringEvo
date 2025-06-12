import os
import json
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List
import random

from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Update these credentials / model settings before running the script
api_key = os.getenv("OPENAI_API_KEY", "")  # Prefer environment variable for safety
model = os.getenv("OPENAI_API_MODEL", "o1-mini")
max_entries = 10

# Directories & files
DIFF_DIR = Path(__file__).resolve().parent.parent / "data_filtered" / "diffs_meaningful"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data_filtered" / "queries"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "queries_spring_mini.json"

# ---------------------------------------------------------------------------
# Prompt templates (mirrors the ones used for Rust, but adapted for Spring/Java)
# ---------------------------------------------------------------------------

prompt_templates: Dict[str, str] = {
    "stabilized": (
        "As a Spring Framework expert, generate a concise Java programming task and "
        "corresponding Spring-based method signature that implicitly requires the use "
        "of a recently added Spring Framework API. ### Core Principles ###  Follow "
        "these rules: Do not directly mention the API's name or signature. Clearly "
        "describe a real-world scenario solved uniquely or efficiently by this newly "
        "added feature. The generated method signature should strongly hint at the "
        "use of the new API. Be unpredictable in your task format (e.g., don't always "
        "begin with 'You ...'. Instead, vary between interrogative, imperative, "
        "conditional, declarative, challenge-based, and scenario-based styles)."
    ),
    "signature": (
        "As a Spring Framework expert, generate a concise Java programming task and "
        "corresponding method signature that implicitly requires using an API whose "
        "signature was recently updated. ### Core Principles ### Follow these rules: "
        "Do not directly reveal the name or signature changes of the API. Provide a "
        "realistic context emphasizing limitations or inconveniences solved by the "
        "updated API. The generated method signature should implicitly lead to "
        "choosing the newly updated API over older versions. Be unpredictable in your "
        "task format (e.g., don't always begin with 'You ...')."
    ),
    "implicit": (
        "As a Spring Framework expert, generate a concise Java programming task and "
        "corresponding method signature that implicitly requires an API whose internal "
        "implementation or functionality recently changed without altering its "
        "signature. ### Core Principles ### Follow these rules: Do not directly mention "
        "the API name or its internal changes explicitly. Clearly describe an "
        "observable behavior improvement (e.g., performance, memory usage, correctness). "
        "The generated method signature and context should implicitly prompt the use of "
        "this API, relying on documentation or inferred behavior differences. Be "
        "unpredictable in your task format."
    ),
    "deprecated": (
        "As a Spring Framework expert, generate a concise Java programming task and "
        "corresponding method signature that implicitly encourages replacing a "
        "deprecated or removed API with a newly recommended one. ### Core Principles ### "
        "Follow these rules: Do not explicitly mention the deprecated API's name. "
        "Provide context that strongly suggests the disadvantages (performance, safety, "
        "usability) of the old method. The generated method signature should implicitly "
        "guide users towards the recommended alternative. Be unpredictable in your task "
        "format (e.g., don't always begin with 'You ...')."
    ),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def classify_change(change_value: str) -> str:
    """Map raw diff 'change' value to a generic changetype."""
    mapping = {
        "added": "stabilized",
        "removed": "deprecated",
        "signature_changed": "signature",
        "modifier_changed": "signature",
    }
    return mapping.get(change_value, "implicit")


def parse_versions_from_filename(filename: str):
    """Given a diff filename like 'v6.1.20__v6.2.7.json', return (from_v, to_v)."""
    base = filename.rsplit("/", 1)[-1]
    if base.endswith(".json"):
        base = base[:-5]
    # Expect pattern old__new
    parts = base.split("__")
    if len(parts) == 2:
        return parts[0], parts[1]
    # fallback if pattern unexpected
    return None, None


def collect_api_changes(diff_dir: Path) -> List[Dict]:
    """Traverse all diff JSON files and flatten API changes into a list of dicts."""
    api_entries: List[Dict] = []
    for json_file in diff_dir.glob("*.json"):
        from_v, to_v = parse_versions_from_filename(json_file.name)
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                diff_data = json.load(f)
        except Exception as exc:
            print(f"Failed to load {json_file}: {exc}")
            continue

        if not isinstance(diff_data, dict):
            # Unexpected format – skip
            continue

        # random.sample(diff_data.items(), k=max_entries)

        data_shrinked = random.sample(diff_data.items(), min(max_entries, len(diff_data.items())))

        for api_name, info in data_shrinked:
            change_raw = info.get("change", "implicit")
            changetype = classify_change(change_raw)

            entry = {
                "library": "spring-framework",
                "name": api_name,
                "from_version": from_v,
                "to_version": to_v,
                "type": info.get("kind"),
                "signature": info.get("signature"),
                "documentation": info.get("doc"),
                "changetype": changetype,
                "source_code": info.get("source_code"),
            }
            api_entries.append(entry)
    return api_entries

# ---------------------------------------------------------------------------
# Query generation using OpenAI
# ---------------------------------------------------------------------------

client = None
if api_key:
    client = OpenAI(api_key=api_key)
else:
    raise ValueError("No API key provided")


def generate_task(api_entry: Dict, max_retries: int = 3) -> Dict:
    """Generate a query + signature pair for a single API entry via LLM."""
    changetype = api_entry["changetype"]
    prompt = prompt_templates.get(changetype, prompt_templates["implicit"])

    supplement = (
        f"""
    ### API Characteristics ###
    {api_entry}
    ### Generation Format ###
    <query>
    [Your generated query here]
    </query>
    <signature>
    [Your generated signature here]
    </signature>
    """
    )

    for attempt in range(max_retries):
        try:
            temp = min(0.7 + attempt * 0.1, 0.9)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": supplement},
                ],
                temperature=temp,
            )

            response = completion.choices[0].message.content
            query = ""
            signature = ""

            if "<query>" in response and "</query>" in response:
                query = response.split("<query>")[1].split("</query>")[0].strip()
            if "<signature>" in response and "</signature>" in response:
                signature = response.split("<signature>")[1].split("</signature>")[0].strip()

            if query and signature:
                api_entry["query"] = query
                api_entry["function_signature"] = signature
                return api_entry
            else:
                print(f"Attempt {attempt+1}: Failed to parse LLM response. Retrying…")
        except Exception as exc:
            print(f"Attempt {attempt+1}: Error – {exc}. Retrying…")
        time.sleep(2)

    # All retries failed
    api_entry["query"] = "ERROR: Failed to generate query"
    api_entry["function_signature"] = "ERROR: Failed to generate signature"
    return api_entry

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_and_generate():
    api_changes = collect_api_changes(DIFF_DIR)
    updated_entries: List[Dict] = []

    progress_bar = tqdm(total=len(api_changes), desc="Generating queries")

    batch_size = 50
    for idx in range(0, len(api_changes), batch_size):
        batch = api_changes[idx : idx + batch_size]
        batch_results = [None] * len(batch)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_idx = {executor.submit(generate_task, entry): i for i, entry in enumerate(batch)}
            for future in concurrent.futures.as_completed(future_to_idx):
                original_idx = future_to_idx[future]
                batch_results[original_idx] = future.result()
                progress_bar.update(1)

        updated_entries.extend(batch_results)
        if idx % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(updated_entries, f, indent=2, ensure_ascii=False)

    progress_bar.close()


if __name__ == "__main__":
    process_and_generate()
