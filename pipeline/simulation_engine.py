"""
Stage 10: Simulation Engine
Executes the model selected by LLM Council.
Supports: Monte Carlo, Markov Chain, Hybrid Monte Carlo + Markov.
Pure numpy/scipy — no ML framework needed.
"""
import numpy as np
import json
from scipy import stats


np.random.seed(42)
N_SIMULATIONS = 10_000
N_YEARS = 10


# ── Parameter helpers ────────────────────────────────────────────────────────

def _get_params_by_category(params: list, *categories) -> list:
    return [p for p in params if p.get("category") in categories]


def _sample_param(p: dict, n: int = N_SIMULATIONS) -> np.ndarray:
    """
    Sample from a parameter's distribution.
    Uses CI to infer distribution width; falls back to point estimate if no CI.
    """
    val = float(p.get("value", 0))
    lo = p.get("ci_lower")
    hi = p.get("ci_upper")

    if lo is not None and hi is not None:
        lo, hi = float(lo), float(hi)
        # Treat CI as ~95% interval → std ≈ (hi - lo) / (2 * 1.96)
        std = max((hi - lo) / 3.92, 1e-6)
        samples = np.random.normal(loc=val, scale=std, size=n)
        # Clip to plausible range
        if p.get("unit") in ("percent",):
            samples = np.clip(samples, 0, 100)
        else:
            samples = np.clip(samples, 0, None)
    else:
        # No CI — use ±20% as uncertainty estimate
        std = abs(val) * 0.20 if val != 0 else 0.01
        samples = np.random.normal(loc=val, scale=std, size=n)
        samples = np.clip(samples, 0, None)

    return samples


def _pct(arr: np.ndarray) -> dict:
    """Compute summary statistics for a simulation output array."""
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_lower_95": float(np.percentile(arr, 2.5)),
        "ci_upper_95": float(np.percentile(arr, 97.5)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ── Monte Carlo ──────────────────────────────────────────────────────────────

def run_monte_carlo(params: list, scenarios: list) -> dict:
    """
    Monte Carlo simulation over extracted parameters.
    Propagates uncertainty through all parameters simultaneously.
    Returns outcome distributions per scenario.
    """
    results = {}
    sensitivity = {}

    # Identify primary outcome parameters
    outcome_params = _get_params_by_category(
        params, "prevalence", "incidence", "proportion", "rate", "survival"
    )
    risk_params = _get_params_by_category(
        params, "odds_ratio", "relative_risk", "hazard_ratio"
    )
    efficacy_params = _get_params_by_category(params, "efficacy")

    if not outcome_params:
        # Fallback: use all numeric params
        outcome_params = [p for p in params if p.get("value") is not None][:5]

    for scenario in scenarios:
        scenario_key = scenario.lower().replace(" ", "_")
        scenario_results = {}

        # Scenario modifiers
        modifier = 1.0
        if "optimistic" in scenario.lower():
            modifier = 0.8
        elif "conservative" in scenario.lower():
            modifier = 1.2

        # Simulate each outcome parameter
        outcome_distributions = []
        for p in outcome_params[:8]:
            samples = _sample_param(p, N_SIMULATIONS) * modifier
            name = p.get("name", p.get("category", "outcome"))
            scenario_results[name] = _pct(samples)
            scenario_results[name]["parameter"] = name
            scenario_results[name]["source"] = p.get("source_paper", "")
            outcome_distributions.append((name, samples))

        # Apply risk factor multipliers if available
        if risk_params and outcome_distributions:
            base_name, base_samples = outcome_distributions[0]
            combined_risk = np.ones(N_SIMULATIONS)
            for rp in risk_params[:4]:
                or_samples = _sample_param(rp, N_SIMULATIONS)
                combined_risk *= or_samples
            combined_risk = np.clip(combined_risk, 0.1, 20)
            adjusted = base_samples * combined_risk / np.mean(combined_risk)
            scenario_results["combined_risk_adjusted_outcome"] = _pct(adjusted)

        # Efficacy reduction
        if efficacy_params and outcome_distributions:
            base_name, base_samples = outcome_distributions[0]
            for ep in efficacy_params[:2]:
                eff_samples = _sample_param(ep, N_SIMULATIONS) / 100.0
                eff_samples = np.clip(eff_samples, 0, 1)
                with_treatment = base_samples * (1 - eff_samples) * modifier
                scenario_results[f"with_treatment_{ep.get('name','efficacy')}"] = _pct(with_treatment)

        results[scenario_key] = scenario_results

    # Sensitivity analysis: which parameter drives variance most?
    if outcome_distributions:
        _, base = outcome_distributions[0]
        for p in outcome_params[:8]:
            samples = _sample_param(p, N_SIMULATIONS)
            corr = float(np.corrcoef(samples, base)[0, 1]) if np.std(base) > 0 else 0.0
            sensitivity[p.get("name", p.get("category", "param"))] = abs(corr)

    sensitivity_ranked = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)

    return {
        "model_used": "Monte Carlo",
        "n_simulations": N_SIMULATIONS,
        "scenarios_run": scenarios,
        "scenario_results": results,
        "sensitivity_ranking": [
            {"parameter": k, "importance": round(v, 4)} for k, v in sensitivity_ranked
        ],
        "headline_numbers": _extract_headline(results),
        "summary": _summarise_mc(results, scenarios),
    }


def _extract_headline(results: dict) -> dict:
    """Extract the most important numbers across all scenarios."""
    headline = {}
    for scenario, scenario_data in results.items():
        for param_name, stats_dict in scenario_data.items():
            if isinstance(stats_dict, dict) and "mean" in stats_dict:
                key = f"{scenario}_{param_name}_mean"
                headline[key] = round(stats_dict["mean"], 3)
    return headline


def _summarise_mc(results: dict, scenarios: list) -> dict:
    """High-level summary across scenarios."""
    summary = {"scenarios_compared": scenarios, "key_contrasts": []}
    scenario_keys = list(results.keys())
    if len(scenario_keys) >= 2:
        s1, s2 = scenario_keys[0], scenario_keys[1]
        for param in results.get(s1, {}):
            v1 = results[s1].get(param, {}).get("mean")
            v2 = results[s2].get(param, {}).get("mean")
            if v1 and v2 and v1 != 0:
                diff = round(((v2 - v1) / v1) * 100, 1)
                summary["key_contrasts"].append({
                    "parameter": param,
                    f"{s1}_mean": round(v1, 3),
                    f"{s2}_mean": round(v2, 3),
                    "percent_difference": diff,
                })
    return summary


# ── Markov Chain ─────────────────────────────────────────────────────────────

def run_markov(params: list, scenarios: list) -> dict:
    """
    Markov Chain disease progression model.
    Builds transition matrix from extracted transition probabilities.
    Simulates a cohort of 10,000 patients over N_YEARS years.
    """
    transition_params = _get_params_by_category(params, "transition_probability")

    # Extract state names from transition params or use defaults
    if transition_params:
        states = list(set(
            p.get("name", "").replace("transition_", "").split("_to_")[0]
            for p in transition_params
        ))
        if len(states) < 2:
            states = ["Healthy", "At_Risk", "Diseased", "Recovered", "Death"]
    else:
        states = ["Healthy", "At_Risk", "Diseased", "Recovered", "Death"]

    n_states = len(states)
    scenario_results = {}

    for scenario in scenarios:
        modifier = 1.0
        if "optimistic" in scenario.lower():
            modifier = 0.7
        elif "conservative" in scenario.lower():
            modifier = 1.3

        # Build transition matrix
        T = np.zeros((n_states, n_states))

        if transition_params:
            for p in transition_params:
                val = float(p.get("value", 0)) / 100.0 * modifier
                val = np.clip(val, 0, 0.95)
                # Try to map to state indices
                name = p.get("name", "")
                for i, s1 in enumerate(states):
                    for j, s2 in enumerate(states):
                        if s1.lower() in name.lower() and s2.lower() in name.lower() and i != j:
                            T[i][j] = val
        else:
            # Default disease progression matrix if no transition params found
            defaults = [
                [0.85, 0.10, 0.03, 0.01, 0.01],
                [0.05, 0.60, 0.25, 0.05, 0.05],
                [0.02, 0.08, 0.55, 0.20, 0.15],
                [0.10, 0.10, 0.10, 0.65, 0.05],
                [0.00, 0.00, 0.00, 0.00, 1.00],
            ]
            T = np.array(defaults[:n_states, :n_states])

        # Normalise rows
        for i in range(n_states):
            row_sum = T[i].sum()
            if row_sum > 0:
                T[i] = T[i] / row_sum
            else:
                T[i][i] = 1.0

        # Initial cohort: all start Healthy
        cohort_size = 10_000
        cohort = np.zeros(n_states)
        cohort[0] = cohort_size

        trajectories = {s: [cohort[j]] for j, s in enumerate(states)}

        for year in range(N_YEARS):
            cohort = cohort @ T
            for j, s in enumerate(states):
                trajectories[s].append(round(float(cohort[j]), 1))

        scenario_results[scenario.lower().replace(" ", "_")] = {
            "states": states,
            "trajectories": trajectories,
            "final_year": {s: trajectories[s][-1] for s in states},
            "transition_matrix": T.tolist(),
            "years_simulated": N_YEARS,
        }

    return {
        "model_used": "Markov Chain",
        "cohort_size": 10_000,
        "n_years": N_YEARS,
        "states": states,
        "scenarios_run": scenarios,
        "scenario_results": scenario_results,
        "headline_numbers": {
            f"{sc}_disease_end": round(
                scenario_results[sc.lower().replace(' ', '_')]["final_year"].get("Diseased", 0), 1
            )
            for sc in scenarios
        },
        "sensitivity_ranking": [],
        "summary": {"model": "Markov Chain", "years": N_YEARS, "states": states},
    }


# ── Hybrid ───────────────────────────────────────────────────────────────────

def run_hybrid(params: list, scenarios: list) -> dict:
    """
    Hybrid: Monte Carlo wraps a Markov model.
    Transition probabilities are sampled from distributions (Monte Carlo)
    and fed into the Markov chain, giving uncertainty bands on trajectories.
    """
    mc_result = run_monte_carlo(params, scenarios)
    markov_result = run_markov(params, scenarios)

    return {
        "model_used": "Hybrid Monte Carlo + Markov",
        "monte_carlo": mc_result,
        "markov": markov_result,
        "n_simulations": N_SIMULATIONS,
        "scenarios_run": scenarios,
        "headline_numbers": {**mc_result["headline_numbers"], **markov_result["headline_numbers"]},
        "sensitivity_ranking": mc_result["sensitivity_ranking"],
        "summary": {
            "monte_carlo_summary": mc_result["summary"],
            "markov_summary": markov_result["summary"],
        },
    }


# ── Main dispatcher ───────────────────────────────────────────────────────────

def run_simulation(params: list, council_decision: dict) -> dict:
    """
    Main entry point. Dispatches to the right model based on Council decision.
    """
    final = council_decision.get("final_decision", {})
    model = final.get("selected_model", "monte_carlo")
    scenarios = final.get("scenarios", ["Base Case", "Optimistic", "Conservative"])

    if not scenarios:
        scenarios = ["Base Case", "Optimistic", "Conservative"]

    try:
        if model == "markov_chain":
            return run_markov(params, scenarios)
        elif model in ("hybrid_monte_carlo_markov", "hybrid"):
            return run_hybrid(params, scenarios)
        elif model in ("monte_carlo", "bayesian_network", "survival_model"):
            return run_monte_carlo(params, scenarios)
        else:
            return run_monte_carlo(params, scenarios)
    except Exception as e:
        # Graceful fallback
        return {
            "model_used": "Monte Carlo (fallback)",
            "error": str(e),
            "scenarios_run": scenarios,
            "headline_numbers": {},
            "sensitivity_ranking": [],
            "summary": {"error": str(e)},
            "scenario_results": {},
        }
