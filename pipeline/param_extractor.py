"""
Stage 8: Quantitative Parameter Extractor
Reads wiki pages and paper sections, pulls every numerical parameter
with uncertainty, population tag, and source citation.
Output feeds the simulation engine and is saved as CSV.
"""
import json
import re
import csv
import io

PARAM_EXTRACTION_PROMPT = """You are a clinical biostatistician extracting quantitative parameters
from healthcare research paper summaries.

Extract EVERY numerical value that could be used in a mathematical model.
Be exhaustive — include even tentative or approximate values.

Return a JSON array where each item is:
{
  "name": "human_readable_parameter_name",
  "category": "prevalence|incidence|odds_ratio|relative_risk|hazard_ratio|transition_probability|efficacy|survival|sensitivity|specificity|mean|proportion|rate|other",
  "value": <central estimate as float>,
  "ci_lower": <lower CI as float or null>,
  "ci_upper": <upper CI as float or null>,
  "unit": "percent|per_1000|per_year|absolute|ratio|dimensionless",
  "population": "description of population this applies to",
  "condition": "disease or clinical condition",
  "time_horizon": "timeframe if applicable e.g. 5-year or annual or null",
  "source_paper": "paper title or filename",
  "source_section": "Results or Table 2 or Abstract etc",
  "confidence": "high|medium|low",
  "notes": "any important caveats about this value"
}

Include values even if CIs are missing. Flag confidence as low if value is approximate."""

SUFFICIENCY_CHECK_PROMPT = """You are assessing whether extracted quantitative parameters
are sufficient to run a meaningful computational model.

Return JSON:
{
  "sufficient": true or false,
  "minimum_viable_model": "description or null",
  "parameter_count": <int>,
  "coverage_score": <0 to 10>,
  "missing_critical": ["list of missing parameter types"],
  "recommendation": "proceed|warn_and_proceed|insufficient"
}"""


def extract_parameters(wiki: dict, papers: list, client) -> list:
    """Extract all quantitative parameters from wiki pages and paper sections."""
    all_params = []
    wiki_pages = wiki.get("pages", [])

    for i, (page, paper) in enumerate(zip(wiki_pages, papers)):
        title = page.get("title", page.get("source_file", f"Paper {i+1}"))

        context_parts = [f"Paper: {title}"]

        for field in ["key_findings", "limitations", "methods", "datasets"]:
            items = page.get(field, [])
            if items:
                context_parts.append(field.upper() + ":\n" + "\n".join(f"- {x}" for x in items))

        sections = paper.get("sections", {})
        for sec_name, sec_text in sections.items():
            if any(k in sec_name.lower() for k in ["result", "statistic", "table", "finding", "outcome", "survival", "efficacy"]):
                context_parts.append(f"[{sec_name}]\n{sec_text[:3000]}")

        for tbl in paper.get("tables", [])[:5]:
            context_parts.append(f"[Table p.{tbl['page']}]\n{tbl['content'][:1000]}")

        context = "\n\n".join(context_parts)
        user_prompt = f"Extract ALL quantitative parameters from this paper.\n\nSource: \"{title}\"\n\n{context}"

        raw = client.chat_json(PARAM_EXTRACTION_PROMPT, user_prompt, max_tokens=6000)

        try:
            params = json.loads(raw)
            if not isinstance(params, list):
                params = []
        except Exception:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            params = []
            if match:
                try:
                    params = json.loads(match.group())
                except Exception:
                    pass

        for p in params:
            p["paper_index"] = i
            p["source_paper"] = p.get("source_paper") or title
        all_params.extend(params)

    return all_params


def check_sufficiency(params: list, client) -> dict:
    """Check if extracted parameters are sufficient for modelling."""
    if not params:
        return {
            "sufficient": False,
            "recommendation": "insufficient",
            "parameter_count": 0,
            "coverage_score": 0,
            "missing_critical": ["No parameters extracted"],
            "minimum_viable_model": None,
        }

    summary = {
        "total_params": len(params),
        "categories": list(set(p.get("category", "") for p in params)),
        "conditions": list(set(p.get("condition", "") for p in params))[:10],
        "params_with_ci": sum(1 for p in params if p.get("ci_lower") is not None),
        "sample": params[:6],
    }

    raw = client.chat_json(
        SUFFICIENCY_CHECK_PROMPT,
        f"Parameters summary:\n{json.dumps(summary, indent=2)}",
        max_tokens=800,
    )

    try:
        result = json.loads(raw)
    except Exception:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(match.group()) if match else {
            "sufficient": True,
            "recommendation": "warn_and_proceed",
            "parameter_count": len(params),
            "coverage_score": 5,
            "missing_critical": [],
            "minimum_viable_model": "Monte Carlo simulation",
        }

    result["parameter_count"] = len(params)
    return result


def params_to_csv(params: list) -> str:
    """Convert parameter list to CSV string."""
    if not params:
        return "name,category,value\nNo parameters extracted,,,\n"

    fieldnames = [
        "name", "category", "value", "ci_lower", "ci_upper",
        "unit", "population", "condition", "time_horizon",
        "source_paper", "source_section", "confidence", "notes",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for p in params:
        writer.writerow({k: p.get(k, "") for k in fieldnames})
    return output.getvalue()


def group_params_by_category(params: list) -> dict:
    """Group parameters by category for display and modelling."""
    groups = {}
    for p in params:
        cat = p.get("category", "other")
        groups.setdefault(cat, []).append(p)
    return groups
