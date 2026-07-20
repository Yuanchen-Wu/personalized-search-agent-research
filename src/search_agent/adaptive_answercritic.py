"""Leak-free draft-answer-critic adaptive loop (prototype).

A variant of the re-fanout loop that judges the DRAFT ANSWER instead of the retrieved
evidence, so re-fanning is driven by what the ANSWER still fails to do -- not by raw
coverage. Each round:

  1. FAN-OUT k persona-conditioned queries (round 1 from scratch, later rounds from the
     critic's feedback about what the answer needs).
  2. SEARCH all k -> this round's evidence.
  3. SYNTHESIZE a DRAFT answer from that evidence (hardened v2 synthesizer).
  4. CRITIQUE the draft with a LEAK-FREE answer critic (1-5 answer_score + gaps + whether
     more evidence would even help).
       - score >= threshold -> approve; that round's draft IS the final answer.
       - else               -> re-fan for what the ANSWER needs and retry.
  5. Loop until approved or ``max_rounds``; on exhaustion use the BEST-scoring round's draft.

LEAK-FREE (load-bearing): the critic sees only agent-visible inputs -- the query,
``persona.render_for_agent()``, this round's evidence, and the agent's OWN draft answer.
It NEVER sees the frozen evaluation rubric / gold intent. This module imports nothing
from ``rubrics``. Using the rubric-aware final judge here would be leakage.

Cost note: unlike the re-fanout loop (one synthesis at the end), this synthesizes a draft
EVERY round, so it is more expensive -- that is the price of judging the answer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_RESULTS_PER_BRANCH,
    DEFAULT_SEARCH_DEPTH,
)
from .adaptive_refanout import _call_judge_once, _render_evidence_digest, generate_fanout
from .evidence import (
    compute_context_character_count,
    deduplicate_search_results,
    filter_unique_documents,
    select_evidence_for_synthesis,
)
from .fanout import _persona_block
from .fixed_fanout import search_tavily_cached
from .llm_gemini import call_gemini
from .meta_prompt import ANSWER_CRITIC_PROMPT_V1, SYNTHESIS_PROMPT_HARDENED_V2
from .synthesize import _format_evidence
from .schemas import FanoutBranch, Persona, SearchResult

# NOTE: do NOT `import rubrics` here -- see the leak-free invariant in the docstring.


@dataclass
class AnswerCriticDecision:
    """Aggregated leak-free critique of one round's draft answer (K-sampled, mean score)."""

    answer_score: float                       # mean of sample_scores (1-5); approve when >= threshold
    answer_gaps: List[str] = field(default_factory=list)
    needs_more_evidence: bool = True          # False => the gap is synthesis (evidence was there, unused)
    feedback: str = ""                        # guidance for the next fan-out
    rationale: str = ""
    latency: float = 0.0
    sample_scores: List[int] = field(default_factory=list)
    attempts: int = 1
    parse_ok: bool = True


@dataclass
class AnswerCriticResult:
    final_answer: str
    approved_branches: List[FanoutBranch]
    approved_evidence: List[SearchResult]
    approved_round: Optional[int]
    best_round: Optional[int]
    approved_score: Optional[float]
    stop_reason: str
    num_rounds: int
    num_tavily_calls: int
    num_synthesis_calls: int
    num_critic_calls: int
    events: List[Dict[str, Any]]


def _synthesize_draft(user_query: str, persona: Optional[Persona],
                      evidence: List[SearchResult], model: str, seed: int) -> str:
    """Draft answer via the hardened-v2 synthesizer (same rendering as the pipeline)."""
    prompt = SYNTHESIS_PROMPT_HARDENED_V2.format(
        user_query=user_query,
        persona_block=_persona_block(persona) or "(no user context provided)",
        evidence_block=_format_evidence(evidence),
    )
    return call_gemini(prompt, model=model, temperature=0.4, seed=seed)


def judge_answer(*, user_query: str, persona: Optional[Persona], evidence: List[SearchResult],
                 draft_answer: str, model: str = DEFAULT_GEMINI_MODEL, seed: int = 42,
                 judge_samples: int = 3, judge_temperature: float = 0.2,
                 max_llm_retries: int = 2) -> AnswerCriticDecision:
    """Leak-free K-sampled critique of the draft answer. Mean answer_score; gaps unioned;
    feedback/rationale from the harshest sample (most actionable)."""
    digest, _ = _render_evidence_digest(evidence)
    prompt = ANSWER_CRITIC_PROMPT_V1.format(
        user_query=user_query,
        persona_block=_persona_block(persona) or "(no user context provided)",
        num_evidence=len(evidence),
        evidence_digest=digest,
        draft_answer=draft_answer,
    )
    scores: List[int] = []
    recs: List[tuple] = []   # (score, gaps, needs_more, feedback, rationale)
    latency = 0.0
    attempts = 0
    for s in range(max(1, judge_samples)):
        parsed, lat, att = _call_judge_once(
            prompt, model=model, seed=seed + 1000 * s,
            temperature=judge_temperature, max_llm_retries=max_llm_retries)
        latency += lat
        attempts += att
        if isinstance(parsed, dict):
            try:
                sc = int(round(float(parsed.get("answer_score", 3))))
            except (ValueError, TypeError):
                sc = 3
            sc = max(1, min(5, sc))
            gaps = parsed.get("answer_gaps") or []
            if not isinstance(gaps, list):
                gaps = [str(gaps)]
            recs.append((sc, [str(g) for g in gaps], bool(parsed.get("needs_more_evidence", True)),
                         str(parsed.get("feedback", "")), str(parsed.get("rationale", ""))))
            scores.append(sc)

    if not scores:
        return AnswerCriticDecision(answer_score=5.0, rationale="(critic unparseable; accepting round)",
                                    latency=latency, attempts=attempts, parse_ok=False)
    mean = round(sum(scores) / len(scores), 3)
    seen: set = set()
    gaps_union: List[str] = []
    for _sc, gaps, _nm, _fb, _rt in recs:
        for g in gaps:
            if g not in seen:
                seen.add(g)
                gaps_union.append(g)
    worst = min(recs, key=lambda r: r[0])
    return AnswerCriticDecision(
        answer_score=mean, answer_gaps=gaps_union, needs_more_evidence=worst[2],
        feedback=worst[3], rationale=worst[4], latency=latency,
        sample_scores=scores, attempts=attempts,
    )


def run_answercritic_loop(*, user_query: str, persona: Optional[Persona], query_id: str,
                          fanout_size: int = 4, max_rounds: int = 3, approval_threshold: float = 4.0,
                          planner_model: str = DEFAULT_GEMINI_MODEL,
                          synthesizer_model: str = DEFAULT_GEMINI_MODEL,
                          critic_model: str = DEFAULT_GEMINI_MODEL,
                          judge_samples: int = 3, judge_temperature: float = 0.2, seed: int = 42,
                          search_depth: str = DEFAULT_SEARCH_DEPTH,
                          max_results_per_branch: int = DEFAULT_MAX_RESULTS_PER_BRANCH,
                          search_cache_path: Optional[str] = None, use_cache: bool = True
                          ) -> AnswerCriticResult:
    """Run the fan-out -> search -> DRAFT -> critique -> re-fan loop for one (query, persona)."""
    events: List[Dict[str, Any]] = []
    prior_queries: List[str] = []
    gaps: List[str] = []
    feedback: str = ""
    n_tavily = n_synth = n_critic = 0

    best = None       # (score, draft, evidence, branches, round)
    approved = None

    for rnd in range(1, max_rounds + 1):
        branches, _gen_lat, _gen_att = generate_fanout(
            user_query=user_query, persona=persona, fanout_size=fanout_size, round_idx=rnd,
            prior_queries=prior_queries, coverage_gaps=gaps, feedback=feedback,
            model=planner_model, seed=seed)

        raw: List[SearchResult] = []
        for b in branches:
            res, _hit = search_tavily_cached(
                query=b.query, branch_type=b.branch_type, max_results=max_results_per_branch,
                search_depth=search_depth, cache_path=search_cache_path, use_cache=use_cache)
            raw.extend(res)
            n_tavily += 1
        evidence = select_evidence_for_synthesis(deduplicate_search_results(raw), "all", None, None)

        draft = _synthesize_draft(user_query, persona, evidence, synthesizer_model, seed)
        n_synth += 1
        dec = judge_answer(user_query=user_query, persona=persona, evidence=evidence,
                           draft_answer=draft, model=critic_model, seed=seed,
                           judge_samples=judge_samples, judge_temperature=judge_temperature)
        n_critic += 1

        if best is None or dec.answer_score > best[0]:
            best = (dec.answer_score, draft, evidence, branches, rnd)

        is_ok = dec.answer_score >= approval_threshold
        events.append({
            "event_type": "answercritic_round", "round": rnd, "fanout_size": len(branches),
            "queries": [b.query for b in branches], "num_results": len(evidence),
            "answer_score": dec.answer_score, "sample_scores": dec.sample_scores,
            "approval_threshold": approval_threshold, "approved": is_ok,
            "needs_more_evidence": dec.needs_more_evidence, "answer_gaps": dec.answer_gaps,
            "feedback": dec.feedback, "rationale": dec.rationale, "draft_len": len(draft or ""),
            # Full per-round draft + evidence + branches: makes each round self-contained so
            # lower answer-score thresholds are derivable post-hoc (the "t=5 trick") and any
            # round can be re-scored later. (Large, but that is the price of derivability.)
            "draft": draft,
            "evidence": [r.as_dict() for r in evidence],
            "branches": [b.as_dict() for b in branches],
        })

        if is_ok:
            approved = (dec.answer_score, draft, evidence, branches, rnd)
            break
        prior_queries = [b.query for b in branches]
        gaps = dec.answer_gaps
        feedback = dec.feedback

    chosen = approved or best
    for i, b in enumerate(chosen[3], start=1):
        b.priority_rank = i
    events.append({
        "event_type": "answercritic_stop",
        "stop_reason": "approved" if approved else "max_rounds_exhausted",
        "num_rounds": len(events) if not events or events[-1]["event_type"] == "answercritic_round" else len(events),
        "approved_round": approved[4] if approved else None,
        "best_round": best[4], "approved_score": chosen[0],
    })
    return AnswerCriticResult(
        final_answer=chosen[1], approved_branches=chosen[3], approved_evidence=chosen[2],
        approved_round=approved[4] if approved else None, best_round=best[4],
        approved_score=chosen[0], stop_reason="approved" if approved else "max_rounds_exhausted",
        num_rounds=sum(1 for e in events if e["event_type"] == "answercritic_round"),
        num_tavily_calls=n_tavily, num_synthesis_calls=n_synth, num_critic_calls=n_critic,
        events=events,
    )
