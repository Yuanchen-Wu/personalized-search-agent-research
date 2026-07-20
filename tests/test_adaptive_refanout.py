"""Unit tests for the re-fanout adaptive loop (adaptive_refanout).

All Gemini and Tavily calls are mocked. No network access or API keys required.
Core invariants: the controller approves a round when coverage_score >=
approval_threshold, rejected rounds are discarded (synthesis sees only the approved
round), judge feedback propagates into the next fan-out, and cost counts every
round's searches (variable cost = rounds x k).
"""

from __future__ import annotations

import json
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
from unittest.mock import patch

from search_agent.adaptive_refanout import (
    RetrievalJudgeDecision,
    generate_fanout,
    judge_retrieval,
    run_refanout_retrieval,
)
from search_agent.schemas import FanoutBranch, Persona, SearchResult


def _branch(q, bt="generic"):
    return FanoutBranch(branch_type=bt, query=q)


def _search_echo(**kw):
    """Return one SearchResult tagged with its query, so we can trace which round."""
    q = kw.get("query", "")
    return (
        [SearchResult(title=f"T:{q}", url=f"http://x/{q}", content="c", score=0.5,
                      rank=1, branch_type=kw.get("branch_type", "generic"), branch_query=q)],
        False,
    )


class TestRefanoutControlFlow(unittest.TestCase):
    def setUp(self):
        self.persona = Persona(persona_id="p1", description="d", macro_domain="education", attributes={})

    @patch("search_agent.adaptive_refanout.judge_retrieval")
    @patch("search_agent.adaptive_refanout.search_tavily_cached")
    @patch("search_agent.adaptive_refanout.generate_fanout")
    def test_round1_approved(self, mock_gen, mock_search, mock_judge):
        mock_gen.return_value = ([_branch("r1a"), _branch("r1b")], 0.1, 1)
        mock_search.side_effect = _search_echo
        mock_judge.return_value = RetrievalJudgeDecision(coverage_score=5)
        res = run_refanout_retrieval(
            user_query="q", persona=self.persona, query_id="q1",
            fanout_size=2, max_rounds=3, approval_threshold=4, use_cache=False,
        )
        self.assertEqual(res.cost.num_rounds, 1)
        self.assertEqual(res.cost.approved_round, 1)
        self.assertEqual(res.cost.approved_score, 5)
        self.assertEqual(res.cost.stop_reason, "approved")
        self.assertEqual(res.cost.num_tavily_calls, 2)          # 1 round x k=2
        self.assertEqual({r.branch_query for r in res.approved_results}, {"r1a", "r1b"})

    @patch("search_agent.adaptive_refanout.judge_retrieval")
    @patch("search_agent.adaptive_refanout.search_tavily_cached")
    @patch("search_agent.adaptive_refanout.generate_fanout")
    def test_low_then_high_score_discards_first_round(self, mock_gen, mock_search, mock_judge):
        mock_gen.side_effect = [
            ([_branch("r1a"), _branch("r1b")], 0.1, 1),
            ([_branch("r2a"), _branch("r2b")], 0.1, 1),
        ]
        mock_search.side_effect = _search_echo
        mock_judge.side_effect = [
            RetrievalJudgeDecision(coverage_score=2, coverage_gaps=["gap"], feedback="need more"),
            RetrievalJudgeDecision(coverage_score=5),
        ]
        res = run_refanout_retrieval(
            user_query="q", persona=self.persona, query_id="q1",
            fanout_size=2, max_rounds=3, approval_threshold=4, use_cache=False,
        )
        self.assertEqual(res.cost.num_rounds, 2)
        self.assertEqual(res.cost.approved_round, 2)
        self.assertEqual(res.cost.num_tavily_calls, 4)          # 2 rounds x k=2
        # Approved evidence is ONLY round 2's -- round 1 was discarded.
        self.assertEqual({r.branch_query for r in res.approved_results}, {"r2a", "r2b"})
        # Round 1's judge feedback was propagated into round 2's fan-out generation.
        round2 = mock_gen.call_args_list[1].kwargs
        self.assertEqual(round2["feedback"], "need more")
        self.assertEqual(round2["coverage_gaps"], ["gap"])
        self.assertEqual(round2["prior_queries"], ["r1a", "r1b"])

    @patch("search_agent.adaptive_refanout.judge_retrieval")
    @patch("search_agent.adaptive_refanout.search_tavily_cached")
    @patch("search_agent.adaptive_refanout.generate_fanout")
    def test_threshold_boundary_controls_approval(self, mock_gen, mock_search, mock_judge):
        # A round scoring exactly 3 is approved at tau=3 but retried (never approved) at tau=4.
        mock_gen.return_value = ([_branch("a"), _branch("b")], 0.1, 1)
        mock_search.side_effect = _search_echo
        mock_judge.return_value = RetrievalJudgeDecision(coverage_score=3)
        approved = run_refanout_retrieval(
            user_query="q", persona=self.persona, query_id="q1",
            fanout_size=2, max_rounds=2, approval_threshold=3, use_cache=False,
        )
        self.assertEqual(approved.cost.approved_round, 1)       # 3 >= 3 -> approved
        retried = run_refanout_retrieval(
            user_query="q", persona=self.persona, query_id="q1",
            fanout_size=2, max_rounds=2, approval_threshold=4, use_cache=False,
        )
        self.assertIsNone(retried.cost.approved_round)          # 3 < 4 -> never approved
        self.assertEqual(retried.cost.stop_reason, "max_rounds_exhausted")

    @patch("search_agent.adaptive_refanout.judge_retrieval")
    @patch("search_agent.adaptive_refanout.search_tavily_cached")
    @patch("search_agent.adaptive_refanout.generate_fanout")
    def test_never_approved_falls_back_to_best_round(self, mock_gen, mock_search, mock_judge):
        mock_gen.side_effect = [
            ([_branch("r1a"), _branch("r1b")], 0.1, 1),
            ([_branch("r2a"), _branch("r2b")], 0.1, 1),
        ]
        mock_search.side_effect = _search_echo
        # Neither approves at tau=4, and round 2 is WORSE than round 1 (drift).
        mock_judge.side_effect = [
            RetrievalJudgeDecision(coverage_score=3, coverage_gaps=["x"], feedback="f"),
            RetrievalJudgeDecision(coverage_score=2, coverage_gaps=["y"], feedback="g"),
        ]
        res = run_refanout_retrieval(
            user_query="q", persona=self.persona, query_id="q1",
            fanout_size=2, max_rounds=2, approval_threshold=4, use_cache=False,
        )
        self.assertEqual(res.cost.num_rounds, 2)
        self.assertIsNone(res.cost.approved_round)
        self.assertEqual(res.cost.stop_reason, "max_rounds_exhausted")
        self.assertEqual(res.cost.num_tavily_calls, 4)
        # Fallback uses the BEST round (round 1, score 3), NOT the last (round 2, score 2).
        self.assertEqual(res.cost.fallback_round, 1)
        self.assertEqual(res.cost.approved_score, 3)
        self.assertEqual({r.branch_query for r in res.approved_results}, {"r1a", "r1b"})

    @patch("search_agent.adaptive_refanout.judge_retrieval")
    @patch("search_agent.adaptive_refanout.search_tavily_cached")
    @patch("search_agent.adaptive_refanout.generate_fanout")
    def test_events_carry_per_round_evidence(self, mock_gen, mock_search, mock_judge):
        # Each round's OWN evidence is logged (even discarded rounds) so max_rounds
        # can be reconstructed post-hoc from a single high-cap run.
        mock_gen.side_effect = [
            ([_branch("r1a"), _branch("r1b")], 0.1, 1),
            ([_branch("r2a"), _branch("r2b")], 0.1, 1),
        ]
        mock_search.side_effect = _search_echo
        mock_judge.side_effect = [
            RetrievalJudgeDecision(coverage_score=2, coverage_gaps=["g"], feedback="f"),
            RetrievalJudgeDecision(coverage_score=5),
        ]
        res = run_refanout_retrieval(
            user_query="q", persona=self.persona, query_id="q1",
            fanout_size=2, max_rounds=3, approval_threshold=4, use_cache=False,
        )
        round_events = [e for e in res.events if e["event_type"] == "refanout_round"]
        self.assertEqual(len(round_events), 2)
        # Round 1 kept its own evidence in the event even though synthesis discarded it.
        self.assertEqual({d["branch_query"] for d in round_events[0]["results"]}, {"r1a", "r1b"})
        self.assertEqual({d["branch_query"] for d in round_events[1]["results"]}, {"r2a", "r2b"})


class TestRefanoutParsing(unittest.TestCase):
    def setUp(self):
        self.persona = Persona(persona_id="p1", description="d", macro_domain="education", attributes={})

    @patch("search_agent.adaptive_refanout.call_gemini")
    def test_generate_fanout_parses_and_trims(self, mock_gemini):
        mock_gemini.return_value = json.dumps([
            {"branch_type": "generic", "query": "alpha"},
            {"branch_type": "personalized", "query": "beta"},
            {"branch_type": "constraint", "query": "gamma"},
        ])
        branches, latency, attempts = generate_fanout(
            user_query="q", persona=self.persona, fanout_size=2,
            round_idx=1, prior_queries=[], coverage_gaps=[], feedback="",
        )
        self.assertEqual(len(branches), 2)                      # trimmed to fanout_size
        self.assertEqual([b.priority_rank for b in branches], [1, 2])

    @patch("search_agent.adaptive_refanout.call_gemini")
    def test_judge_retrieval_parses_score(self, mock_gemini):
        mock_gemini.return_value = json.dumps({
            "coverage_score": 2,
            "coverage_gaps": ["missing budget options"],
            "feedback": "search for specific budget models",
            "rationale": "results are too generic",
        })
        decision = judge_retrieval(
            user_query="q", persona=self.persona, evidence=[],
            fanout_branches=[_branch("alpha")],
        )
        self.assertEqual(decision.coverage_score, 2)
        self.assertEqual(decision.coverage_gaps, ["missing budget options"])
        self.assertEqual(decision.feedback, "search for specific budget models")
        self.assertTrue(decision.parse_ok)

    @patch("search_agent.adaptive_refanout.call_gemini")
    def test_judge_score_is_clamped(self, mock_gemini):
        mock_gemini.return_value = json.dumps({"coverage_score": 9})
        decision = judge_retrieval(user_query="q", persona=self.persona, evidence=[], fanout_branches=[])
        self.assertEqual(decision.coverage_score, 5)            # clamped to the 1-5 scale

    @patch("search_agent.adaptive_refanout.call_gemini")
    def test_judge_unparseable_defaults_to_top_score(self, mock_gemini):
        mock_gemini.return_value = "not json at all"
        decision = judge_retrieval(user_query="q", persona=self.persona, evidence=[], fanout_branches=[])
        self.assertEqual(decision.coverage_score, 5)            # clean stop, not a runaway
        self.assertFalse(decision.parse_ok)

    @patch("search_agent.adaptive_refanout.call_gemini")
    def test_judge_averages_k_samples(self, mock_gemini):
        # K=3 votes on the SAME evidence -> mean score; raw votes kept; gaps unioned;
        # feedback taken from the lowest-scoring (most critical) sample.
        mock_gemini.side_effect = [
            json.dumps({"coverage_score": 3, "coverage_gaps": ["a"], "feedback": "fb_low", "rationale": "r_low"}),
            json.dumps({"coverage_score": 4, "coverage_gaps": ["b"], "feedback": "fb_mid", "rationale": "r_mid"}),
            json.dumps({"coverage_score": 4, "coverage_gaps": ["a"], "feedback": "fb_hi", "rationale": "r_hi"}),
        ]
        decision = judge_retrieval(
            user_query="q", persona=self.persona, evidence=[], fanout_branches=[],
            judge_samples=3,
        )
        self.assertAlmostEqual(decision.coverage_score, 3.667, places=2)  # mean of 3,4,4
        self.assertEqual(decision.sample_scores, [3, 4, 4])
        self.assertEqual(decision.num_samples, 3)
        self.assertEqual(decision.coverage_gaps, ["a", "b"])              # deduped union
        self.assertEqual(decision.feedback, "fb_low")                    # lowest-scoring sample
        self.assertTrue(decision.parse_ok)


class TestRefanoutMethodParsing(unittest.TestCase):
    def test_parse_refanout_method(self):
        from scripts.run_fixed_fanout_benchmark import parse_refanout_method
        self.assertEqual(parse_refanout_method("refanout_k4"), {"fanout_size": 4, "threshold": None})
        self.assertEqual(parse_refanout_method("refanout_k4_t3"), {"fanout_size": 4, "threshold": 3})
        self.assertEqual(parse_refanout_method("refanout_k4_t4p5"), {"fanout_size": 4, "threshold": 4.5})
        self.assertIsNone(parse_refanout_method("fixed_k4"))
        self.assertIsNone(parse_refanout_method("adaptive_k4"))


if __name__ == "__main__":
    unittest.main()
