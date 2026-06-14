import json
import os
import sys

from utils import (
    get_allowed_domains,
    get_allowed_query_types,
    get_generated_dir,
    load_domain_schemas,
    read_jsonl,
)

def main():
    generated_dir = get_generated_dir()
    users_file = os.path.join(generated_dir, "users.jsonl")
    queries_file = os.path.join(generated_dir, "queries.jsonl")

    domain_schemas = load_domain_schemas()
    valid_domains = set(get_allowed_domains(domain_schemas))
    valid_query_types = set(get_allowed_query_types(domain_schemas))

    users = []
    queries = []
    errors = []

    # 1. Parse Users
    user_ids = set()
    if not os.path.exists(users_file):
        errors.append(f"Users file missing: {users_file}")
    else:
        # Re-read line by line to get line numbers for parse errors if any,
        # but read_jsonl is safer. Let's just use read_jsonl and manually check.
        # However, to preserve exact error output line numbers, we should read normally
        # but since the prompt says "Use read_jsonl(...)", we will adapt.
        raw_users = read_jsonl(users_file)
        for line_idx, user in enumerate(raw_users, 1):
            if "persona_id" not in user:
                errors.append(f"User line {line_idx}: missing persona_id")
                continue
                
            pid = user["persona_id"]
            if pid in user_ids:
                errors.append(f"Duplicate persona_id: {pid}")
            user_ids.add(pid)
            users.append(user)
            
            for req in ["short_name", "demographics", "latent_profile", "observable_history", "distractor_history"]:
                if req not in user:
                    errors.append(f"User {pid}: missing {req}")

    # 2. Parse Queries
    example_ids = set()
    queries_by_domain = {}
    queries_by_type = {}
    
    if not os.path.exists(queries_file):
        errors.append(f"Queries file missing: {queries_file}")
    else:
        raw_queries = read_jsonl(queries_file)
        for line_idx, query in enumerate(raw_queries, 1):
            if "example_id" not in query:
                errors.append(f"Query line {line_idx}: missing example_id")
                continue
                
            eid = query["example_id"]
            if eid in example_ids:
                errors.append(f"Duplicate example_id: {eid}")
            example_ids.add(eid)
            queries.append(query)
            
            d = query.get("domain")
            qt = query.get("query_type")
            queries_by_domain[d] = queries_by_domain.get(d, 0) + 1
            queries_by_type[qt] = queries_by_type.get(qt, 0) + 1
            
            pid = query.get("persona_id")
            if pid not in user_ids:
                errors.append(f"Query {eid}: Invalid persona_id reference: {pid}")
                
            if d not in valid_domains:
                errors.append(f"Query {eid}: Invalid domain: {d}")
                
            if qt not in valid_query_types:
                errors.append(f"Query {eid}: Invalid query_type: {qt}")
                
            if not query.get("ambiguous_query"):
                errors.append(f"Query {eid}: ambiguous_query is empty")
                
            if not query.get("clear_hidden_intent"):
                errors.append(f"Query {eid}: clear_hidden_intent is empty")
                
            targets = query.get("personalization_targets", {})
            must_use = targets.get("must_use", [])
            should_not_use = targets.get("should_not_use", [])
            keywords = targets.get("desired_fanout_keywords", [])
            
            if qt != "overpersonalization_trap" and not must_use:
                errors.append(f"Query {eid}: missing must_use items for {qt}")
                
            if not should_not_use:
                errors.append(f"Query {eid}: missing should_not_use items")
                
            if not keywords:
                errors.append(f"Query {eid}: missing desired_fanout_keywords")

    summary = {
        "num_users": len(users),
        "num_queries": len(queries),
        "queries_by_domain": queries_by_domain,
        "queries_by_type": queries_by_type,
        "errors": errors
    }
    
    print(json.dumps(summary, indent=2))
    
    if errors:
        sys.exit(1)
    else:
        print("\nAll validations passed!")

if __name__ == "__main__":
    main()
