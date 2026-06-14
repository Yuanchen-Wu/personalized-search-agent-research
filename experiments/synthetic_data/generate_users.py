import argparse
import csv
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from utils import (
    GENERATION_MAX_WORKERS,
    ensure_dir,
    get_generated_dir,
    get_project_root,
    load_prompt,
    parse_json_response,
    rate_limited_call_gemini,
    write_jsonl,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=20)
    parser.add_argument("--model", type=str, default="gemini-flash-latest")
    args = parser.parse_args()

    generated_dir = get_generated_dir()
    ensure_dir(generated_dir)
    prompt_template = load_prompt("generate_user_prompt.txt")

    users = []
    print(f"Generating {args.num_users} users using model {args.model}...")
    
    def generate_single_user(index):
        prompt = prompt_template + f"\n\nGenerate user number {index+1} with a unique persona."
        response = rate_limited_call_gemini(
            prompt=prompt,
            model=args.model,
            temperature=1.0,
            response_mime_type="application/json"
        )
        user_data = parse_json_response(response)
        
        if "persona_id" not in user_data or not user_data["persona_id"]:
            user_data["persona_id"] = f"user_{str(uuid.uuid4())[:8]}"
        return user_data
        
    with ThreadPoolExecutor(max_workers=GENERATION_MAX_WORKERS) as executor:
        futures = {executor.submit(generate_single_user, i): i for i in range(args.num_users)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating users"):
            try:
                user_data = future.result()
                users.append(user_data)
            except Exception as e:
                print(f"\nFailed to generate a user: {e}")

    # Write output
    write_jsonl(os.path.join(generated_dir, "users.jsonl"), users)
            
    simplified_personas = []
    for u in users:
        simplified = {
            "persona_id": u["persona_id"],
            "description": u.get("short_name", ""),
            "attributes": {
                "demographics": u.get("demographics", {}),
                "latent_profile": u.get("latent_profile", {})
            },
            "observable_history": u.get("observable_history", []),
            "distractor_history": u.get("distractor_history", [])
        }
        simplified_personas.append(simplified)
        
    write_jsonl(os.path.join(get_project_root(), "experiments", "sample_personas.generated.jsonl"), simplified_personas)

    # Save to CSV
    csv_path = os.path.join(generated_dir, "users.csv")
    if users:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["persona_id", "short_name", "location", "observable_history_count"])
            for u in users:
                loc = u.get("demographics", {}).get("location", "")
                history_count = len(u.get("observable_history", []))
                writer.writerow([u.get("persona_id"), u.get("short_name"), loc, history_count])

    # Save to MD
    md_path = os.path.join(generated_dir, "users_preview.md")
    with open(md_path, "w") as f:
        f.write("# Generated Users Preview\n\n")
        for u in users:
            f.write(f"## {u.get('short_name', 'Unnamed')} ({u.get('persona_id')})\n")
            f.write(f"**Location:** {u.get('demographics', {}).get('location', 'Unknown')}\n\n")
            f.write("### Latent Profile\n")
            for domain, prof in u.get("latent_profile", {}).items():
                f.write(f"- **{domain}:** {json.dumps(prof)}\n")
            f.write("\n### Observable History (Sample)\n")
            for h in u.get("observable_history", [])[:5]:
                f.write(f"- [{h.get('domain')}] {h.get('type')}: {h.get('content')}\n")
            f.write("\n### Distractor History (Sample)\n")
            for h in u.get("distractor_history", [])[:2]:
                f.write(f"- [{h.get('domain')}] {h.get('type')}: {h.get('content')}\n")
            f.write("\n---\n\n")

    print(f"Done! {len(users)} users saved to generated directory and experiments/sample_personas.generated.jsonl")

if __name__ == "__main__":
    main()
