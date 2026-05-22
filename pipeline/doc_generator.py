"""
Stage 12: Document Generator
Packages everything into a downloadable ZIP:
  - parameters.csv
  - simulation_results.json
  - council_debate.md
  - insights.md
  - full_report.md
"""
import json
import io
import zipfile
from datetime import datetime


def _fmt_md_section(title: str, content: str) -> str:
    return f"## {title}\n\n{content}\n\n---\n\n"


def build_council_debate_md(model_council: dict, insight_council: dict) -> str:
    """Format both council debates as readable Markdown."""
    lines = ["# LLM Council Debates\n"]

    lines.append("## Council 1 — Model Selection\n")
    for agent in model_council.get("agents", []):
        lines.append(f"### {agent['agent']} ({agent['role']})\n")
        lines.append(agent["response"])
        lines.append("\n")

    consensus = model_council.get("consensus", {})
    if consensus:
        lines.append("### Council Consensus — Model Selection\n")
        lines.append(consensus.get("consensus", ""))
        lines.append("\n")
        for pt in consensus.get("key_points", []):
            lines.append(f"- {pt}")
        lines.append("\n")

    final = model_council.get("final_decision", {})
    if final:
        lines.append("### Final Decision\n")
        lines.append(f"**Selected Model:** {final.get('selected_model', '')}\n")
        lines.append(f"**Rationale:** {final.get('rationale', '')}\n")
        lines.append(f"**Scenarios:** {', '.join(final.get('scenarios', []))}\n")

    lines.append("\n---\n\n## Council 2 — Insight Synthesis\n")
    for agent in insight_council.get("agents", []):
        lines.append(f"### {agent['agent']} ({agent['role']})\n")
        lines.append(agent["response"])
        lines.append("\n")

    final_insight = insight_council.get("final_decision", {})
    if final_insight:
        lines.append("### Synthesised Insights\n")
        lines.append(f"**Headline:** {final_insight.get('headline_finding', '')}\n\n")
        lines.append("**Clinical Insights:**\n")
        for ci in final_insight.get("clinical_insights", []):
            lines.append(f"- {ci}")
        lines.append("\n\n**Research Insights:**\n")
        for ri in final_insight.get("research_insights", []):
            lines.append(f"- {ri}")
        lines.append("\n\n**Limitations:**\n")
        for lim in final_insight.get("limitations", []):
            lines.append(f"- {lim}")

    return "\n".join(lines)


def build_insights_md(insight_council: dict) -> str:
    """Standalone insights document."""
    final = insight_council.get("final_decision", {})
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Computational Modelling Insights\n*Generated: {now}*\n",
        f"## Headline Finding\n{final.get('headline_finding', 'N/A')}\n",
    ]

    for section, key in [
        ("Clinical Insights", "clinical_insights"),
        ("Research Insights", "research_insights"),
        ("Key Uncertainties", "key_uncertainties"),
        ("Actionable Recommendations", "actionable_recommendations"),
        ("Limitations", "limitations"),
    ]:
        items = final.get(key, [])
        if items:
            lines.append(f"## {section}\n")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

    pub = final.get("publishability", "")
    if pub:
        lines.append(f"## Publishability Assessment\n{pub}\n")

    return "\n".join(lines)


def build_full_report_md(
    context: dict,
    params: list,
    sufficiency: dict,
    model_council: dict,
    sim_results: dict,
    insight_council: dict,
) -> str:
    """Full end-to-end report in one Markdown document."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    final_model = model_council.get("final_decision", {})
    final_insight = insight_council.get("final_decision", {})

    lines = [
        f"# Computational Modelling Report\n*Generated: {now}*\n",
        f"**Domain:** {context.get('domain', 'Healthcare AI')}",
        f"**Interest:** {context.get('interest', 'N/A')}",
        f"**Parameters Extracted:** {len(params)}",
        f"**Model Used:** {sim_results.get('model_used', 'N/A')}",
        f"**Scenarios:** {', '.join(sim_results.get('scenarios_run', []))}",
        "\n---\n",
        "## Parameter Sufficiency Assessment\n",
        f"- **Sufficient for modelling:** {sufficiency.get('sufficient', False)}",
        f"- **Coverage score:** {sufficiency.get('coverage_score', 0)}/10",
        f"- **Recommendation:** {sufficiency.get('recommendation', 'N/A')}",
    ]

    missing = sufficiency.get("missing_critical", [])
    if missing:
        lines.append("\n**Missing parameters:**")
        for m in missing:
            lines.append(f"- {m}")

    lines.append("\n---\n## Model Selection (Council Decision)\n")
    lines.append(f"**Selected:** {final_model.get('selected_model', 'N/A')}")
    lines.append(f"\n{final_model.get('rationale', '')}\n")

    lines.append("\n---\n## Simulation Results\n")
    headline = sim_results.get("headline_numbers", {})
    for k, v in list(headline.items())[:10]:
        lines.append(f"- **{k}:** {v}")

    sens = sim_results.get("sensitivity_ranking", [])
    if sens:
        lines.append("\n### Most Influential Parameters (Sensitivity Analysis)\n")
        for i, s in enumerate(sens[:5], 1):
            lines.append(f"{i}. {s['parameter']} — importance score: {s['importance']}")

    lines.append("\n---\n## Synthesised Insights\n")
    lines.append(f"**Headline:** {final_insight.get('headline_finding', 'N/A')}\n")

    for section, key in [
        ("Clinical Insights", "clinical_insights"),
        ("Research Insights", "research_insights"),
        ("Actionable Recommendations", "actionable_recommendations"),
        ("Limitations", "limitations"),
    ]:
        items = final_insight.get(key, [])
        if items:
            lines.append(f"\n### {section}\n")
            for item in items:
                lines.append(f"- {item}")

    lines.append("\n---\n*Report generated by Research Gap Analyzer — Computational Lab*")
    return "\n".join(lines)


def build_zip(
    params_csv: str,
    sim_results: dict,
    model_council: dict,
    insight_council: dict,
    context: dict,
    params: list,
    sufficiency: dict,
) -> bytes:
    """
    Build the complete ZIP package with all output files.
    Returns bytes ready for st.download_button.
    """
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. parameters.csv
        zf.writestr("parameters.csv", params_csv)

        # 2. simulation_results.json
        zf.writestr(
            "simulation_results.json",
            json.dumps(sim_results, indent=2, default=str),
        )

        # 3. council_debate.md
        debate_md = build_council_debate_md(model_council, insight_council)
        zf.writestr("council_debate.md", debate_md)

        # 4. insights.md
        insights_md = build_insights_md(insight_council)
        zf.writestr("insights.md", insights_md)

        # 5. full_report.md
        report_md = build_full_report_md(
            context, params, sufficiency, model_council, sim_results, insight_council
        )
        zf.writestr("full_report.md", report_md)

    buf.seek(0)
    return buf.read()
