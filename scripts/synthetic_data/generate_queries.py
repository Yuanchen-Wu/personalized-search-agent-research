import argparse
import csv
import json
import os
import sys
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
    get_allowed_domains,
    get_allowed_query_types,
    get_domain_schema_text,
    get_generated_dir,
    get_project_root,
    load_domain_schemas,
    load_prompt,
    parse_json_response,
    rate_limited_call_gemini,
    read_jsonl,
    write_jsonl,
)

def main():
    parser = argparse.ArgumentParser()
    generated_dir = get_generated_dir()
    parser.add_argument("--users_file", type=str, default=os.path.join(generated_dir, "users.jsonl"))
    parser.add_argument("--queries_per_user_per_domain", type=int, default=2)
    parser.add_argument("--model", type=str, default="gemini-flash-latest")
    args = parser.parse_args()

    ensure_dir(generated_dir)
    prompt_template = load_prompt("generate_query_prompt.txt")
    
    domain_schemas = load_domain_schemas()
    domains = get_allowed_domains(domain_schemas)
    query_types = get_allowed_query_types(domain_schemas)

    users = read_jsonl(args.users_file)
    if not users:
        print(f"Error: Users file {args.users_file} not found or empty. Generate users first.")
        sys.exit(1)

    print(f"Loaded {len(users)} users.")
    queries = []
    total_expected = len(users) * len(domains) * args.queries_per_user_per_domain

    print(f"Generating queries (Total expected: {total_expected}) using model {args.model}...")
    
    def generate_single_query(user, domain, q_type, domain_schema_info):
        persona_id = user.get("persona_id")
        user_profile_json = json.dumps(user, indent=2)
        
        prompt = prompt_template.replace("{user_profile}", user_profile_json)
        prompt = prompt.replace("{domain}", domain)
        prompt += f"\n\nConstraint: Ensure the query type is {q_type}."
        prompt += f"\nHere is the domain schema for {domain} for your reference:\n{domain_schema_info}"
        prompt += f"\nGenerate a unique example."

        response = rate_limited_call_gemini(
            prompt=prompt,
            model=args.model,
            temperature=1.0,
            response_mime_type="application/json"
        )
        query_data = parse_json_response(response)
        
        if "example_id" not in query_data or not query_data["example_id"]:
            query_data["example_id"] = f"{domain}_{persona_id}_{str(uuid.uuid4())[:6]}"
        
        query_data["persona_id"] = persona_id
        return query_data

    tasks = []
    query_type_idx = 0
    for user in users:
        for domain in domains:
            domain_schema_info = get_domain_schema_text(domain_schemas, domain)
            for _ in range(args.queries_per_user_per_domain):
                q_type = query_types[query_type_idx % len(query_types)]
                query_type_idx += 1
                tasks.append((user, domain, q_type, domain_schema_info))

    with ThreadPoolExecutor(max_workers=GENERATION_MAX_WORKERS) as executor:
        futures = {executor.submit(generate_single_query, *task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating queries"):
            try:
                query_data = future.result()
                queries.append(query_data)
            except Exception as e:
                print(f"\nFailed to generate a query: {e}")

    write_jsonl(os.path.join(generated_dir, "queries.jsonl"), queries)
            
    simplified_queries = []
    for q in queries:
        simplified = {
            "query_id": q.get("example_id", ""),
            "query": q.get("ambiguous_query", ""),
            "domain": q.get("domain", ""),
            "query_type": q.get("query_type", ""),
            "persona_id": q.get("persona_id", ""),
            "metadata": {
                "clear_hidden_intent": q.get("clear_hidden_intent", ""),
                "must_use": q.get("personalization_targets", {}).get("must_use", []),
                "should_not_use": q.get("personalization_targets", {}).get("should_not_use", []),
                "desired_fanout_keywords": q.get("personalization_targets", {}).get("desired_fanout_keywords", [])
            }
        }
        simplified_queries.append(simplified)
        
    write_jsonl(os.path.join(get_project_root(), "experiments", "sample_queries.generated.jsonl"), simplified_queries)

    csv_path = os.path.join(generated_dir, "queries.csv")
    if queries:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["example_id", "persona_id", "domain", "query_type", "ambiguous_query", "clear_hidden_intent"])
            for q in queries:
                writer.writerow([
                    q.get("example_id"),
                    q.get("persona_id"),
                    q.get("domain"),
                    q.get("query_type"),
                    q.get("ambiguous_query"),
                    q.get("clear_hidden_intent")
                ])

    md_path = os.path.join(generated_dir, "queries_preview.md")
    with open(md_path, "w") as f:
        f.write("# Generated Queries Preview\n\n")
        queries_by_domain = {}
        for q in queries:
            d = q.get("domain", "unknown")
            if d not in queries_by_domain:
                queries_by_domain[d] = []
            queries_by_domain[d].append(q)
            
        for domain, d_queries in queries_by_domain.items():
            f.write(f"## Domain: {domain}\n\n")
            for q in d_queries:
                f.write(f"### {q.get('example_id')} (User: {q.get('persona_id')})\n")
                f.write(f"**Type:** {q.get('query_type')}\n\n")
                f.write(f"**Ambiguous Query:** `{q.get('ambiguous_query')}`\n\n")
                f.write(f"**Clear Hidden Intent:** {q.get('clear_hidden_intent')}\n\n")
                
                targets = q.get("personalization_targets", {})
                f.write(f"- **Must Use:** {', '.join(targets.get('must_use', []))}\n")
                f.write(f"- **Should Not Use:** {', '.join(targets.get('should_not_use', []))}\n")
                f.write(f"- **Desired Fanout Keywords:** {', '.join(targets.get('desired_fanout_keywords', []))}\n\n")
            f.write("---\n\n")

    print(f"Done! {len(queries)} queries saved to generated directory and experiments/sample_queries.generated.jsonl")

if __name__ == "__main__":
    main()
