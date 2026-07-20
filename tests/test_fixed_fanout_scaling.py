"""Unit tests for fixed_fanout_scaling_v1 experiment implementation.

All Gemini and Tavily API calls are mocked. No network access or API keys required.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
from unittest.mock import MagicMock, patch

from search_agent.evidence import (
    deduplicate_search_results,
    filter_unique_documents,
    sample_retrieval_evidence_for_evaluator,
    select_evidence_for_synthesis,
)
from search_agent.fixed_fanout import (
    _is_near_duplicate,
    _normalize_query,
    compute_plan_cache_key,
    compute_search_cache_key,
    generate_ordered_fanout_plan,
    get_or_create_shared_plan,
    search_tavily_cached,
)
from search_agent.logging_utils import append_run_log, build_run_log, new_run_id
from search_agent.run_agent import run_agent
from search_agent.schemas import (
    CostProxy,
    FanoutBranch,
    Persona,
    QueryRecord,
    SearchResult,
)
from search_agent.synthesize import synthesize_answer
from search_agent.adaptive_loop import AssessDecision, run_adaptive_retrieval


class TestFixedFanoutScaling(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.sample_query = QueryRecord(
            query="What is the best laptop for machine learning research?",
            query_id="q_ml_laptop",
            task_type="retrieval_sensitive",
            task_category="shopping_product_recommendation",
            macro_domain="education",
        )
        self.sample_persona = Persona(
            persona_id="ml_phd",
            description="PhD student on a budget interested in Linux compatibility",
            macro_domain="education",
            attributes={"demographics": {"budget": "$1500", "os": "Linux"}},
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch("search_agent.fixed_fanout.call_gemini")
    def test_generating_exactly_eight_valid_candidate_branches(self, mock_gemini):
        mock_branches = [
            {
                "priority_rank": i,
                "branch_type": "generic" if i == 1 else ("personalized" if i == 2 else "supplementary"),
                "query": f"best ml research laptop query {i}",
                "information_need": f"need {i}",
                "rationale": f"rationale {i}",
                "used_persona_fields": [],
            }
            for i in range(1, 9)
        ]
        mock_gemini.return_value = json.dumps(mock_branches)

        branches, events = generate_ordered_fanout_plan(
            user_query=self.sample_query.query,
            persona=self.sample_persona,
            candidate_pool_size=8,
        )

        self.assertEqual(len(branches), 8)
        self.assertEqual([b.priority_rank for b in branches], list(range(1, 9)))

    @patch("search_agent.fixed_fanout.call_gemini")
    def test_enforcing_sequential_priority_ranks(self, mock_gemini):
        # Model returns unordered ranks
        mock_branches = [
            {"priority_rank": 99, "branch_type": "generic", "query": f"unique laptop query {i}"}
            for i in range(1, 9)
        ]
        mock_gemini.return_value = json.dumps(mock_branches)

        branches, _ = generate_ordered_fanout_plan(
            user_query=self.sample_query.query,
            persona=self.sample_persona,
            candidate_pool_size=8,
        )

        self.assertEqual([b.priority_rank for b in branches], [1, 2, 3, 4, 5, 6, 7, 8])

    @patch("search_agent.fixed_fanout.call_gemini")
    def test_exact_duplicate_removal(self, mock_gemini):
        # Model returns exact duplicates
        mock_branches = [
            {"priority_rank": 1, "branch_type": "generic", "query": "ml laptop GPU benchmark"},
            {"priority_rank": 2, "branch_type": "generic", "query": "ml laptop GPU benchmark"},
            {"priority_rank": 3, "branch_type": "generic", "query": "ML Laptop GPU Benchmark"},
        ] + [
            {"priority_rank": i, "branch_type": "supplementary", "query": f"ml laptop topic {i}"}
            for i in range(4, 10)
        ]
        mock_gemini.return_value = json.dumps(mock_branches)

        branches, _ = generate_ordered_fanout_plan(
            user_query=self.sample_query.query,
            persona=self.sample_persona,
            candidate_pool_size=8,
        )

        queries = [b.query.lower().strip() for b in branches]
        self.assertEqual(len(set(queries)), 8)

    def test_simple_near_duplicate_detection(self):
        q1 = "best laptops for deep learning research"
        q2 = "best laptops for deep learning research!"
        q3 = "top rated laptops deep learning research"
        self.assertTrue(_is_near_duplicate(q2, [q1]))
        self.assertTrue(_is_near_duplicate(q3, [q1]))
        self.assertFalse(_is_near_duplicate("budget gpu recommendations for tensorflow", [q1]))

    @patch("search_agent.fixed_fanout.call_gemini")
    def test_repair_behavior_when_fewer_than_eight_returned(self, mock_gemini):
        # First call returns 5 valid branches
        resp1 = json.dumps([
            {"priority_rank": i, "branch_type": "generic", "query": f"query {i}"}
            for i in range(1, 6)
        ])
        # Repair call returns 3 remaining valid branches
        resp2 = json.dumps([
            {"priority_rank": i, "branch_type": "supplementary", "query": f"repaired query {i}"}
            for i in range(6, 9)
        ])
        mock_gemini.side_effect = [resp1, resp2]

        branches, events = generate_ordered_fanout_plan(
            user_query=self.sample_query.query,
            persona=self.sample_persona,
            candidate_pool_size=8,
        )

        self.assertEqual(len(branches), 8)
        self.assertTrue(any(e.get("event_type") == "repair_attempt" for e in events))

    @patch("search_agent.fixed_fanout.call_gemini")
    def test_deterministic_fallback_behavior(self, mock_gemini):
        # Gemini fails completely or returns only 2 branches on both initial and repair
        mock_gemini.side_effect = [
            json.dumps([{"priority_rank": 1, "branch_type": "generic", "query": "ml laptop basic"}]),
            json.dumps([]),
        ]

        branches, events = generate_ordered_fanout_plan(
            user_query=self.sample_query.query,
            persona=self.sample_persona,
            candidate_pool_size=8,
        )

        self.assertEqual(len(branches), 8)
        self.assertTrue(any(e.get("event_type") == "deterministic_fallback" for e in events))

    def test_nested_prefix_consistency(self):
        plan = [
            FanoutBranch(branch_type="generic", query=f"query_{i}", priority_rank=i)
            for i in range(1, 9)
        ]
        k1 = plan[:1]
        k2 = plan[:2]
        k4 = plan[:4]
        k8 = plan[:8]

        self.assertEqual([b.query for b in k1], ["query_1"])
        self.assertEqual([b.query for b in k2], ["query_1", "query_2"])
        self.assertEqual([b.query for b in k4], ["query_1", "query_2", "query_3", "query_4"])
        self.assertEqual(len(k8), 8)

    @patch("search_agent.fixed_fanout.call_gemini")
    def test_identical_shared_plan_ids_across_k_conditions(self, mock_gemini):
        mock_gemini.return_value = json.dumps([
            {"priority_rank": i, "branch_type": "generic", "query": f"query {i}"}
            for i in range(1, 9)
        ])
        cache_file = os.path.join(self.tmp_dir, "fanout_plans.jsonl")

        id1, plan1, _, _ = get_or_create_shared_plan(
            query_id="q1",
            user_query=self.sample_query.query,
            persona=self.sample_persona,
            cache_path=cache_file,
        )
        id2, plan2, _, is_hit = get_or_create_shared_plan(
            query_id="q1",
            user_query=self.sample_query.query,
            persona=self.sample_persona,
            cache_path=cache_file,
        )

        self.assertEqual(id1, id2)
        self.assertTrue(is_hit)
        self.assertEqual(len(plan1), len(plan2))

    @patch("search_agent.fixed_fanout.call_gemini")
    def test_only_one_planner_call_per_query_persona_pair(self, mock_gemini):
        mock_gemini.return_value = json.dumps([
            {"priority_rank": i, "branch_type": "generic", "query": f"query {i}"}
            for i in range(1, 9)
        ])
        cache_file = os.path.join(self.tmp_dir, "fanout_plans.jsonl")

        # Call get_or_create_shared_plan 4 times
        for _ in range(4):
            get_or_create_shared_plan(
                query_id="q1",
                user_query=self.sample_query.query,
                persona=self.sample_persona,
                cache_path=cache_file,
            )

        self.assertEqual(mock_gemini.call_count, 1)

    def test_realized_branch_counts_per_method(self):
        plan = [
            FanoutBranch(branch_type="generic", query=f"query_{i}", priority_rank=i)
            for i in range(1, 9)
        ]
        self.assertEqual(len(plan[:1]), 1)
        self.assertEqual(len(plan[:2]), 2)
        self.assertEqual(len(plan[:4]), 4)
        self.assertEqual(len(plan[:8]), 8)

    def test_fixed_k2_cannot_access_evidence_from_branches_3_to_8(self):
        b1_results = [SearchResult(title="T1", url="http://1.com", content="C1", score=0.9, rank=1, branch_type="generic", branch_query="q1")]
        b2_results = [SearchResult(title="T2", url="http://2.com", content="C2", score=0.8, rank=1, branch_type="personalized", branch_query="q2")]
        b3_results = [SearchResult(title="T3", url="http://3.com", content="C3", score=0.7, rank=1, branch_type="constraint", branch_query="q3")]

        all_branch_results = {1: b1_results, 2: b2_results, 3: b3_results}

        # fixed_k2 prefix only gets branches 1 and 2
        k2_evidence = []
        for rank in [1, 2]:
            k2_evidence.extend(all_branch_results[rank])

        self.assertEqual(len(k2_evidence), 2)
        self.assertNotIn("http://3.com", [r.url for r in k2_evidence])

    def test_stable_fanout_plan_cache_keys(self):
        k1 = compute_plan_cache_key("q1", "test query", self.sample_persona, "gemini-flash", seed=42)
        k2 = compute_plan_cache_key("q1", "test query", self.sample_persona, "gemini-flash", seed=42)
        k3 = compute_plan_cache_key("q1", "different query", self.sample_persona, "gemini-flash", seed=42)
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)

    def test_stable_search_cache_keys(self):
        s1 = compute_search_cache_key("ml laptop gpu", "tavily", "basic", 5)
        s2 = compute_search_cache_key("ML Laptop GPU ", "tavily", "basic", 5)
        s3 = compute_search_cache_key("ml laptop gpu", "tavily", "advanced", 5)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)

    @patch("search_agent.fixed_fanout.search_tavily")
    def test_search_result_reuse_across_prefix_conditions(self, mock_tavily):
        mock_tavily.return_value = [
            SearchResult(title="Res1", url="http://res.com", content="c", score=0.9, rank=1, branch_type="generic", branch_query="q1")
        ]
        cache_file = os.path.join(self.tmp_dir, "search_cache.jsonl")

        res1, hit1 = search_tavily_cached("q1", cache_path=cache_file)
        res2, hit2 = search_tavily_cached("q1", cache_path=cache_file)

        self.assertFalse(hit1)
        self.assertTrue(hit2)
        self.assertEqual(mock_tavily.call_count, 1)

    @patch("search_agent.fixed_fanout.search_tavily")
    def test_cache_invalidation_when_search_parameters_change(self, mock_tavily):
        mock_tavily.return_value = []
        cache_file = os.path.join(self.tmp_dir, "search_cache.jsonl")

        search_tavily_cached("q1", search_depth="basic", cache_path=cache_file)
        search_tavily_cached("q1", search_depth="advanced", cache_path=cache_file)

        self.assertEqual(mock_tavily.call_count, 2)

    def test_requested_versus_realized_count_validation(self):
        log = build_run_log(
            variant="fixed_k4",
            query_record=self.sample_query,
            persona=self.sample_persona,
            fanout_branches=[FanoutBranch(branch_type="generic", query="q")],
            raw_search_results=[],
            final_answer="ans",
            cost_proxy=CostProxy(),
            requested_fanout_count=4,
            realized_fanout_count=4,
        )
        self.assertEqual(log.requested_fanout_count, log.realized_fanout_count)

    def test_evidence_deduplication(self):
        raw = [
            SearchResult(title="T1", url="http://dup.com", content="c1", score=0.9, rank=1, branch_type="g", branch_query="q1"),
            SearchResult(title="T2", url="http://dup.com", content="c2", score=0.8, rank=1, branch_type="p", branch_query="q2"),
            SearchResult(title="T3", url="http://unique.com", content="c3", score=0.7, rank=1, branch_type="c", branch_query="q3"),
        ]
        deduped = deduplicate_search_results(raw)
        self.assertFalse(deduped[0].is_duplicate_url)
        self.assertTrue(deduped[1].is_duplicate_url)

        unique_only = filter_unique_documents(raw)
        self.assertEqual(len(unique_only), 2)

    def test_fixed_document_budget_selection(self):
        raw = [
            SearchResult(title=f"T{i}", url=f"http://site{i}.com", content=f"c{i}", score=0.9 - 0.1 * i, rank=i, branch_type="g", branch_query=f"q{i}")
            for i in range(1, 10)
        ]
        selected = select_evidence_for_synthesis(raw, evidence_budget_mode="fixed_document_budget", max_documents=4)
        self.assertEqual(len(selected), 4)

    def test_retrieval_evaluation_samples_later_branches(self):
        results = [
            SearchResult(title=f"T{b}_{r}", url=f"http://site_{b}_{r}.com", content="c", score=0.9, rank=r, branch_type="g", branch_query=f"query_branch_{b}")
            for b in range(1, 9)
            for r in range(1, 4)
        ]
        sampled = sample_retrieval_evidence_for_evaluator(results, mode="top_m_per_branch", top_m_per_branch=2)
        # 8 branches x 2 per branch = 16 sampled results
        self.assertEqual(len(sampled), 16)
        branch_queries = set(r.branch_query for r in sampled)
        self.assertEqual(len(branch_queries), 8)

    def test_cost_accounting_correctness(self):
        cost = CostProxy(num_gemini_calls=2, num_tavily_calls=4, num_fanout_branches=4, num_raw_results=20)
        self.assertEqual(cost.num_gemini_calls, 2)
        self.assertEqual(cost.num_tavily_calls, 4)

    def test_resume_behavior(self):
        runs_file = os.path.join(self.tmp_dir, "runs.jsonl")
        log = build_run_log(
            variant="fixed_k2",
            query_record=self.sample_query,
            persona=self.sample_persona,
            fanout_branches=[],
            raw_search_results=[],
            final_answer="ans",
            cost_proxy=CostProxy(),
            seed=42,
        )
        append_run_log(log, path=runs_file)

        from scripts.run_fixed_fanout_benchmark import load_completed_run_keys
        keys = load_completed_run_keys(runs_file)
        self.assertIn(("q_ml_laptop", "ml_phd", "fixed_k2", 42), keys)

    @patch("scripts.run_fixed_fanout_benchmark.get_or_create_shared_plan")
    def test_dry_run_makes_no_external_api_calls(self, mock_plan):
        from scripts.run_fixed_fanout_benchmark import main as benchmark_main
        benchmark_main(["--config", "configs/fixed_fanout_scaling_v1.yaml", "--dry_run", "--limit", "1"])
        self.assertEqual(mock_plan.call_count, 0)

    @patch("search_agent.run_agent.collect_search_results")
    @patch("search_agent.run_agent.synthesize_answer")
    @patch("search_agent.run_agent.generate_fanout_queries")
    def test_backward_compatibility_with_placement_ablation_v1(
        self, mock_fanout, mock_synth, mock_search
    ):
        mock_fanout.return_value = [FanoutBranch(branch_type="generic", query="q1")]
        mock_search.return_value = []
        mock_synth.return_value = "Answer"

        log = run_agent(
            query_record=self.sample_query,
            persona=self.sample_persona,
            variant="V1_generic_fanout",
            experiment_name="placement_ablation_v1",
        )

        self.assertEqual(log.variant, "V1_generic_fanout")
        self.assertEqual(log.experiment_name, "placement_ablation_v1")


class TestAdaptiveBudgetModes(unittest.TestCase):
    """Fixed-budget (adaptive_kN) vs variable-budget (adaptive_bN) controller behavior.

    All plan/search/assessor calls are stubbed -- no network or API keys. The core
    invariant under test: fixed-budget mode spends EXACTLY budget_cap searches
    (cost-matched to fixed_k), backfilling from the shared plan when the assessor
    proposes too few new queries.
    """

    def setUp(self):
        self.persona = Persona(
            persona_id="p1", description="d", macro_domain="education", attributes={}
        )
        # 8-branch shared plan with distinct queries (matches SEED_PLAN_POOL_SIZE).
        self.plan = [
            FanoutBranch(branch_type="generic", query=f"plan query {i}", priority_rank=i)
            for i in range(1, 9)
        ]

    @staticmethod
    def _stub_search():
        return (
            [SearchResult(title="T", url="http://x.com", content="c", score=0.5,
                          rank=1, branch_type="generic", branch_query="bq")],
            False,
        )

    def _plan_return(self):
        return ("plan_1", list(self.plan), [], True)

    @patch("search_agent.adaptive_loop.assess_and_propose")
    @patch("search_agent.adaptive_loop.search_tavily_cached")
    @patch("search_agent.adaptive_loop.get_or_create_shared_plan")
    def test_fixed_budget_reaches_exact_k_via_proposals(self, mock_plan, mock_search, mock_assess):
        mock_plan.return_value = self._plan_return()
        mock_search.return_value = self._stub_search()
        # seed=2 + two accepted proposals (per_round_cap=1) => realized 4, no backfill.
        mock_assess.side_effect = [
            AssessDecision(sufficient=False, proposed_branches=[
                FanoutBranch(branch_type="supplementary", query="novel alpha")]),
            AssessDecision(sufficient=False, proposed_branches=[
                FanoutBranch(branch_type="supplementary", query="novel beta")]),
        ]
        result = run_adaptive_retrieval(
            user_query="q", persona=self.persona, query_id="q1",
            budget_cap=4, seed_size=2, max_rounds=5, per_round_cap=1,
            fill_to_budget=True, use_cache=False,
        )
        self.assertEqual(result.cost.realized_fanout_count, 4)
        self.assertEqual(result.cost.num_backfilled, 0)
        self.assertEqual(len(result.branches), 4)
        self.assertEqual([b.priority_rank for b in result.branches], [1, 2, 3, 4])

    @patch("search_agent.adaptive_loop.assess_and_propose")
    @patch("search_agent.adaptive_loop.search_tavily_cached")
    @patch("search_agent.adaptive_loop.get_or_create_shared_plan")
    def test_fixed_budget_backfills_when_assessor_says_sufficient(self, mock_plan, mock_search, mock_assess):
        mock_plan.return_value = self._plan_return()
        mock_search.return_value = self._stub_search()
        # Assessor immediately calls it sufficient and proposes nothing -> must still
        # reach k by backfilling from the plan; the "sufficient" is only a side-signal.
        mock_assess.return_value = AssessDecision(sufficient=True, proposed_branches=[])
        result = run_adaptive_retrieval(
            user_query="q", persona=self.persona, query_id="q1",
            budget_cap=4, seed_size=2, max_rounds=5, per_round_cap=1,
            fill_to_budget=True, use_cache=False,
        )
        self.assertEqual(result.cost.realized_fanout_count, 4)
        self.assertEqual(result.cost.num_backfilled, 2)
        self.assertTrue(result.cost.sufficient_before_budget)
        self.assertEqual(result.cost.stop_reason, "filled_to_budget")

    @patch("search_agent.adaptive_loop.assess_and_propose")
    @patch("search_agent.adaptive_loop.search_tavily_cached")
    @patch("search_agent.adaptive_loop.get_or_create_shared_plan")
    def test_fixed_budget_backfills_on_duplicate_proposals(self, mock_plan, mock_search, mock_assess):
        mock_plan.return_value = self._plan_return()
        mock_search.return_value = self._stub_search()
        # Proposes only a near-duplicate of an already-searched seed query -> filtered,
        # so the loop can't advance on its own and backfill must complete the budget.
        mock_assess.return_value = AssessDecision(
            sufficient=False,
            proposed_branches=[FanoutBranch(branch_type="generic", query="plan query 1")],
        )
        result = run_adaptive_retrieval(
            user_query="q", persona=self.persona, query_id="q1",
            budget_cap=4, seed_size=2, max_rounds=5, per_round_cap=1,
            fill_to_budget=True, use_cache=False,
        )
        self.assertEqual(result.cost.realized_fanout_count, 4)
        self.assertEqual(result.cost.num_backfilled, 2)

    @patch("search_agent.adaptive_loop.assess_and_propose")
    @patch("search_agent.adaptive_loop.search_tavily_cached")
    @patch("search_agent.adaptive_loop.get_or_create_shared_plan")
    def test_variable_budget_still_stops_on_sufficient(self, mock_plan, mock_search, mock_assess):
        mock_plan.return_value = self._plan_return()
        mock_search.return_value = self._stub_search()
        mock_assess.return_value = AssessDecision(sufficient=True, proposed_branches=[])
        result = run_adaptive_retrieval(
            user_query="q", persona=self.persona, query_id="q1",
            budget_cap=8, seed_size=2, max_rounds=5, per_round_cap=4,
            fill_to_budget=False, use_cache=False,
        )
        # Early stop at the seed; no backfill in variable mode.
        self.assertEqual(result.cost.realized_fanout_count, 2)
        self.assertEqual(result.cost.stop_reason, "sufficient")
        self.assertEqual(result.cost.num_backfilled, 0)

    def test_parse_adaptive_method_encodes_mode(self):
        from scripts.run_fixed_fanout_benchmark import parse_adaptive_method
        self.assertEqual(
            parse_adaptive_method("adaptive_k4"),
            {"budget_cap": 4, "strictness": None, "fill_to_budget": True},
        )
        self.assertEqual(
            parse_adaptive_method("adaptive_b8"),
            {"budget_cap": 8, "strictness": None, "fill_to_budget": False},
        )
        self.assertEqual(
            parse_adaptive_method("adaptive_b8_strict"),
            {"budget_cap": 8, "strictness": "strict", "fill_to_budget": False},
        )
        self.assertTrue(parse_adaptive_method("adaptive_k4_lenient")["fill_to_budget"])
        self.assertIsNone(parse_adaptive_method("fixed_k4"))


if __name__ == "__main__":
    unittest.main()
