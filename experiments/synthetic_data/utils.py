import json
import os
import sys
import threading
import time
from typing import Any, Dict, List

import yaml

GENERATION_RPM = 250
GENERATION_MAX_WORKERS = 15

rate_limit_lock = threading.Lock()
last_request_time = 0.0

def get_synthetic_data_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def get_project_root() -> str:
    return os.path.abspath(os.path.join(get_synthetic_data_dir(), "..", ".."))

def get_generated_dir() -> str:
    return os.path.join(get_synthetic_data_dir(), "generated")

def get_prompts_dir() -> str:
    return os.path.join(get_synthetic_data_dir(), "prompts")

def add_project_paths() -> None:
    root = get_project_root()
    if root not in sys.path:
        sys.path.append(root)
    src_dir = os.path.join(root, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)

def load_prompt(prompt_filename: str) -> str:
    with open(os.path.join(get_prompts_dir(), prompt_filename), "r") as f:
        return f.read()

def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Fallback: attempt to extract the first JSON object
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx+1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse JSON. Bad response snippet: {text[:200]}...") from e

def rate_limited_call_gemini(*args, **kwargs) -> str:
    add_project_paths()
    from src.llm_gemini import call_gemini
    
    global last_request_time
    delay = 60.0 / GENERATION_RPM
    with rate_limit_lock:
        now = time.time()
        elapsed = now - last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)
        last_request_time = time.time()
    return call_gemini(*args, **kwargs)

def load_domain_schemas() -> dict:
    path = os.path.join(get_synthetic_data_dir(), "domain_schemas.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_allowed_domains(domain_schemas: dict) -> list[str]:
    return domain_schemas.get("domains", [])

def get_allowed_query_types(domain_schemas: dict) -> list[str]:
    return domain_schemas.get("query_types", [])

def get_domain_schema_text(domain_schemas: dict, domain: str) -> str:
    domain_data = domain_schemas.get(domain, {})
    return yaml.dump(domain_data, sort_keys=False)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def read_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
