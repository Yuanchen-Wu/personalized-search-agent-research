import argparse
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path so we can import from search_agent
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from search_agent.llm_gemini import call_gemini

def write_jsonl(path, data):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def generate_personas(num_personas):
    personas = []
    for i in range(num_personas):
        pid = f"user_{str(uuid.uuid4())[:8]}"
        personas.append({
            "persona_id": pid,
            "description": f"Synthetic User {i}",
            "attributes": {
                "demographics": {"age": "25-34", "location": "Urban"},
                "latent_profile": {"financial_background": "budget-conscious", "stem_background": "high"}
            },
            "observable_history": [
                {"timestamp": "2023-10-01T10:00:00Z", "content": "best budget laptops for coding"},
                {"timestamp": "2023-10-02T10:00:00Z", "content": "how to learn python"}
            ],
            "distractor_history": []
        })
    return personas

def generate_queries(personas):
    queries = []
    task_categories = {
        "search_native": [
            "shopping_commerce", 
            "tech_product_comparison"
        ],
        "synthesis_native": [
            "technical_explanation",
            "professional_career_strategy"
        ]
    }
    
    # Just generating static realistic examples for the sake of the ablation
    for p in personas:
        for t_type, categories in task_categories.items():
            for cat in categories:
                qid = f"q_{t_type[:6]}_{str(uuid.uuid4())[:6]}"
                
                # Mock queries based on category
                if cat == "shopping_commerce":
                    q = "What laptop should I buy for school?"
                elif cat == "tech_product_comparison":
                    q = "Macbook Air vs Pro for computer science student"
                elif cat == "technical_explanation":
                    q = "Explain gradient descent simply"
                else:
                    q = "How do I transition from software engineering to product management?"
                    
                queries.append({
                    "query_id": qid,
                    "task_type": t_type,
                    "task_category": cat,
                    "query": q,
                    "persona_relevant_dimensions": ["financial_background", "stem_background"]
                })
    return queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=2)
    args = parser.parse_args()

    personas_path = os.path.join(PROJECT_ROOT, "data", "generated", "synthetic_personas_v1.jsonl")
    queries_path = os.path.join(PROJECT_ROOT, "data", "generated", "synthetic_queries_v1.jsonl")
    
    os.makedirs(os.path.dirname(personas_path), exist_ok=True)
    
    personas = generate_personas(args.num_users)
    queries = generate_queries(personas)
    
    write_jsonl(personas_path, personas)
    write_jsonl(queries_path, queries)
    
    print(f"Saved {len(personas)} personas to {personas_path}")
    print(f"Saved {len(queries)} queries to {queries_path}")

if __name__ == "__main__":
    main()
