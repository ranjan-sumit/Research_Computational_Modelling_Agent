"""
LLM Council — Multi-model deliberation system.
Each agent runs on a different NVIDIA NIM model, giving genuinely
different reasoning styles rather than just different prompts.

Council composition:
  Agent 1 — Dr. Statistician      → openai/gpt-oss-20b
  Agent 2 — Dr. Epidemiologist    → meta/llama-3.2-3b-instruct
  Agent 3 — Dr. Analyst           → nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
  Chair   — Consensus Synthesiser → openai/gpt-oss-120b  (master)

Used in:
  Stage 9:  Council selects computational model
  Stage 11: Council synthesises simulation insights
"""
import json
import re
from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── Per-model configs ────────────────────────────────────────────────────────

COUNCIL_MODELS = {
    "gpt_20b": {
        "model":       "openai/gpt-oss-20b",
        "temperature": 1.0,
        "top_p":       1.0,
        "max_tokens":  4096,
        "extra_body":  None,
        "has_reasoning": True,
    },
    "llama_3b": {
        "model":       "meta/llama-3.2-3b-instruct",
        "temperature": 0.2,
        "top_p":       0.7,
        "max_tokens":  1024,
        "extra_body":  None,
        "has_reasoning": False,
    },
    "nemotron": {
        "model":       "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "temperature": 0.6,
        "top_p":       0.95,
        "max_tokens":  8192,
        "extra_body":  {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 4096,
        },
        "has_reasoning": True,
    },
    "gpt_120b": {
        "model":       "openai/gpt-oss-120b",
        "temperature": 1.0,
        "top_p":       1.0,
        "max_tokens":  4096,
        "extra_body":  None,
        "has_reasoning": True,
    },
}

# ── Agent definitions — who uses which model ─────────────────────────────────

MODEL_SELECTION_AGENTS = [
    {
        "name":       "Dr. Statistician",
        "role":       "Biostatistician",
        "model_key":  "gpt_20b",
        "lens": (
            "You are a biostatistician. Assess the extracted parameters and recommend "
            "the most statistically appropriate computational model. "
            "Consider: data types, CI coverage, parameter completeness, sample sizes. "
            "Be precise about which model fits the data best and why."
        ),
    },
    {
        "name":       "Dr. Epidemiologist",
        "role":       "Clinical Epidemiologist",
        "model_key":  "llama_3b",
        "lens": (
            "You are a clinical epidemiologist. Assess the parameters from a "
            "population health perspective. Consider: study designs, population "
            "heterogeneity, generalisability, compatible populations. "
            "Flag population mismatch issues that affect model validity."
        ),
    },
    {
        "name":       "Dr. Analyst",
        "role":       "Computational Research Analyst",
        "model_key":  "nemotron",
        "lens": (
            "You are a computational research analyst. Assess whether extracted "
            "parameters are practically sufficient for simulation. "
            "Consider: missing parameters, wide CIs, whether multiple scenarios "
            "(optimistic/conservative) are needed. Be pragmatic."
        ),
    },
]

INSIGHT_SYNTHESIS_AGENTS = [
    {
        "name":       "Dr. Clinician",
        "role":       "Clinical Practitioner",
        "model_key":  "gpt_20b",
        "lens": (
            "You are a senior clinician reviewing simulation results. "
            "Interpret findings from a patient care perspective. "
            "What do these numbers mean for clinical decision-making? "
            "What should a practitioner act on? What is still uncertain?"
        ),
    },
    {
        "name":       "Dr. Researcher",
        "role":       "Research Scientist",
        "model_key":  "llama_3b",
        "lens": (
            "You are a research scientist reviewing simulation results. "
            "Interpret findings from a research gap perspective. "
            "What do outputs reveal about what we still don't know? "
            "Which parameters drove the most uncertainty?"
        ),
    },
    {
        "name":       "Dr. Critic",
        "role":       "Methodological Critic",
        "model_key":  "nemotron",
        "lens": (
            "You are a methodological critic reviewing simulation results. "
            "Challenge the findings constructively. What assumptions were made? "
            "Where could the model be wrong? What limitations must be disclosed? "
            "What would change the conclusions?"
        ),
    },
]

# ── Consensus + final decision prompts ──────────────────────────────────────

CONSENSUS_PROMPT = """You are the Council Chair. Three expert agents have given assessments.
Synthesise their views into a single consensus recommendation.

Return ONLY JSON:
{
  "consensus": "clear summary of the agreed recommendation",
  "key_points": ["3-5 bullet points the council agrees on"],
  "disagreements": ["points where agents differed and how resolved"],
  "confidence": "high|medium|low",
  "caveats": ["important limitations or conditions"]
}"""

MODEL_COUNCIL_FINAL_PROMPT = """Based on the council debate, produce the final model selection.

Return ONLY JSON:
{
  "selected_model": "monte_carlo|markov_chain|bayesian_network|survival_model|hybrid_monte_carlo_markov",
  "rationale": "2-3 sentence explanation",
  "scenarios": ["list of scenarios e.g. Base Case, Optimistic, Conservative"],
  "population_notes": "any population weighting needed",
  "key_parameters_to_use": ["most important parameter names"],
  "parameters_to_flag": ["parameters with issues"],
  "confidence": "high|medium|low",
  "alternative_if_insufficient": "fallback model"
}"""

INSIGHT_COUNCIL_FINAL_PROMPT = """Based on the council debate, synthesise actionable insights.

Return ONLY JSON:
{
  "headline_finding": "single most important finding in one sentence",
  "clinical_insights": ["3-5 insights for clinical practice"],
  "research_insights": ["3-5 insights about what field still needs"],
  "key_uncertainties": ["2-3 areas of highest simulation uncertainty"],
  "actionable_recommendations": ["2-4 specific next steps"],
  "limitations": ["2-4 honest limitations of this simulation"],
  "publishability": "assessment of whether findings could support a methodology paper",
  "confidence": "high|medium|low"
}"""


# ── Core model caller ────────────────────────────────────────────────────────

def _call_model(api_key: str, model_key: str, system: str, user: str) -> str:
    """
    Call a specific NVIDIA NIM model with its own config.
    Handles streaming + reasoning_content skipping.
    """
    cfg = COUNCIL_MODELS[model_key]
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

    kwargs = dict(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_tokens"],
        stream=True,
    )
    if cfg.get("extra_body"):
        kwargs["extra_body"] = cfg["extra_body"]

    stream = client.chat.completions.create(**kwargs)

    parts = []
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            parts.append(content)

    return "".join(parts).strip()


def _parse_json(raw: str) -> dict | list:
    """Parse JSON from model response, handling common wrapping issues."""
    raw = raw.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object or array
        for pattern in (r'\{.*\}', r'\[.*\]'):
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
    return {}


# ── Core council runner ──────────────────────────────────────────────────────

def run_council(agents: list, context: str, final_prompt: str,
                api_key: str, master_model_key: str = "gpt_120b") -> dict:
    """
    Full council deliberation:
    1. Each agent (own model) gives independent assessment
    2. Master model (gpt-oss-120b) synthesises consensus
    3. Master model produces final structured decision

    Returns: {agents, consensus, final_decision, transcript}
    """
    agent_responses = []
    transcript_lines = []

    # Step 1 — Each agent deliberates on their own model
    for agent in agents:
        system = (
            f"You are {agent['name']}, a {agent['role']} on a medical research council.\n\n"
            f"{agent['lens']}\n\n"
            "Give your assessment in 3-5 key points. Be direct and specific. "
            "Give your expert opinion and recommendation — do not just summarise data."
        )
        user = f"Council context:\n\n{context}\n\nGive your expert assessment."

        model_info = COUNCIL_MODELS[agent["model_key"]]
        try:
            response = _call_model(api_key, agent["model_key"], system, user)
        except Exception as e:
            response = f"[{agent['name']} ({model_info['model']}) failed: {e}]"

        agent_responses.append({
            "agent":    agent["name"],
            "role":     agent["role"],
            "model":    model_info["model"],
            "response": response,
        })
        transcript_lines.append(
            f"**{agent['name']} ({agent['role']}) — {model_info['model']}**\n{response}"
        )

    # Step 2 — Master model synthesises consensus
    debate_text = "\n\n---\n\n".join(
        f"{r['agent']} ({r['role']}) [{r['model']}]:\n{r['response']}"
        for r in agent_responses
    )
    consensus_context = f"Original context:\n{context}\n\nAgent assessments:\n{debate_text}"

    raw_consensus = _call_model(
        api_key, master_model_key,
        CONSENSUS_PROMPT + "\n\nCRITICAL: return ONLY JSON, no fences.",
        consensus_context,
    )
    consensus = _parse_json(raw_consensus)
    if not isinstance(consensus, dict):
        consensus = {"consensus": raw_consensus, "key_points": [], "confidence": "medium"}

    # Step 3 — Master model produces final structured decision
    final_context = (
        f"{consensus_context}\n\n"
        f"Council consensus:\n{json.dumps(consensus, indent=2)}"
    )
    raw_final = _call_model(
        api_key, master_model_key,
        final_prompt + "\n\nCRITICAL: return ONLY JSON, no fences.",
        final_context,
    )
    final_decision = _parse_json(raw_final)
    if not isinstance(final_decision, dict):
        final_decision = {"decision": raw_final}

    transcript = "\n\n".join(transcript_lines)
    transcript += f"\n\n---\n\n**Chair Consensus:**\n{consensus.get('consensus', '')}"

    return {
        "agents":         agent_responses,
        "consensus":      consensus,
        "final_decision": final_decision,
        "transcript":     transcript,
    }


# ── Stage 9: Model Selection Council ────────────────────────────────────────

def council_select_model(params: list, sufficiency: dict,
                         context: dict, client) -> dict:
    """Council deliberates on which computational model to use."""
    api_key = client.api_key

    param_summary = {
        "total_parameters":   len(params),
        "categories_present": list(set(p.get("category", "") for p in params)),
        "params_with_ci":     sum(1 for p in params if p.get("ci_lower") is not None),
        "conditions_covered": list(set(p.get("condition", "") for p in params))[:8],
        "sample_params": [
            {k: p.get(k) for k in ["name", "category", "value", "ci_lower", "ci_upper", "confidence"]}
            for p in params[:8]
        ],
        "sufficiency": sufficiency,
    }

    council_context = (
        f"Research domain: {context.get('domain', 'Healthcare AI')}\n"
        f"Research interest: {context.get('interest', 'Disease modelling')}\n\n"
        f"Extracted parameters from "
        f"{len(set(p.get('source_paper','') for p in params))} papers:\n"
        f"{json.dumps(param_summary, indent=2)}"
    )

    return run_council(
        agents        = MODEL_SELECTION_AGENTS,
        context       = council_context,
        final_prompt  = MODEL_COUNCIL_FINAL_PROMPT,
        api_key       = api_key,
    )


# ── Stage 11: Insight Synthesis Council ─────────────────────────────────────

def council_synthesise_insights(sim_results: dict, params: list,
                                 gaps: list, context: dict, client) -> dict:
    """Council synthesises insights from simulation results."""
    api_key = client.api_key

    gap_titles = [g.get("title", "") for g in gaps[:5]]

    council_context = (
        f"Research domain: {context.get('domain', 'Healthcare AI')}\n\n"
        f"Model used: {sim_results.get('model_used', 'Unknown')}\n"
        f"Scenarios run: {sim_results.get('scenarios_run', [])}\n"
        f"Headline numbers: {json.dumps(sim_results.get('headline_numbers', {}), indent=2)}\n"
        f"Sensitivity rankings: {json.dumps(sim_results.get('sensitivity_ranking', [])[:5], indent=2)}\n\n"
        f"Research gaps previously identified:\n{json.dumps(gap_titles, indent=2)}\n\n"
        f"Parameter count used: {len(params)}"
    )

    return run_council(
        agents        = INSIGHT_SYNTHESIS_AGENTS,
        context       = council_context,
        final_prompt  = INSIGHT_COUNCIL_FINAL_PROMPT,
        api_key       = api_key,
    )
