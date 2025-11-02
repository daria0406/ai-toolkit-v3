# streamlit_app.py
# Run with: streamlit run streamlit_app.py

import textwrap
from typing import Callable, Optional
from urllib.parse import urlencode

import streamlit as st

# -------------------------------
# Config & page setup
# -------------------------------
st.set_page_config(
    page_title="Goal Hijacking — Agent Autonomy Demo",
    page_icon="🧭",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Interactive module to explore agent autonomy, internal drift, and mitigations."
    },
)

# -------------------------------
# Labels & global mappings
# -------------------------------
LABEL_BY_SLUG = {
    "intro": "Intro",
    "financial": "Financial",
    "recruiting": "Recruiting",
    "medical": "Medical",
    "baseline": "Biased Baseline (Agent Autonomy)",
    "mitigations": "Mitigations",
    "exercises": "Exercises",
    "conclusions": "Conclusions",
    "resources": "Resources",
}

ALL_SECTIONS = [
    "intro",
    "financial",
    "recruiting",
    "medical",
    "baseline",
    "mitigations",
    "exercises",
    "conclusions",
    "resources",
]

# -------------------------------
# Optional integration hooks
# Replace with your real function.
# -------------------------------
def show_agent_autonomy(key_prefix: str = "goal_baseline"):
    """
    Safe demo stub. Replace with your real UI logic that exercises agent autonomy.
    Keep it side-effect-free unless you have proper approvals/guardrails.
    """
    with st.container():
        st.subheader("Autonomy Sandbox (Demo Stub)")
        st.write(
            "This is a no-op sandbox placeholder. Plug in your simulation here to "
            "show internal drift or risky actions, **without** making external API calls."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk score (simulated)", "0.12", delta="+0.02")
            st.caption("Toy score based on heuristic drift features.")
        with col2:
            st.json(
                {
                    "actions": ["propose_trade", "summarize_market_news"],
                    "constraints": ["leverage<=2x", "no_crypto", "approval_required>threshold"],
                    "approvals": {"propose_trade": "pending"},
                }
            )

# -------------------------------
# Renderers
# -------------------------------
def _render_intro():
    st.title("Goal Hijacking")
    st.header("Intro")
    st.write(
        "Displays the UI for the Agent Autonomy & Internal Drift module and runs the "
        "LangChain scenarios with the Anthropic API."
    )
    st.info("We’ll demonstrate risky actions, constraints, and approvals workflow.")

def _render_use_case(slug: str):
    st.title("Use cases")
    st.header(LABEL_BY_SLUG.get(slug, slug.capitalize()))
    st.write(
        "Portfolio trading, hiring agents, clinical triage: ensure guardrails, "
        "constraints, and approvals."
    )
    with st.expander("Typical risks to highlight"):
        if slug == "financial":
            st.markdown(
                "- Leverage escalation beyond limits\n"
                "- Unauthorized asset classes (e.g., crypto)\n"
                "- Order-splitting to evade controls"
            )
        elif slug == "recruiting":
            st.markdown(
                "- Proxy discrimination in candidate scoring\n"
                "- Unvetted outreach at scale\n"
                "- Sensitive data leakage"
            )
        elif slug == "medical":
            st.markdown(
                "- Off-label advice or scope creep\n"
                "- Hallucinated contraindications\n"
                "- Incomplete triage escalation"
            )

def _render_baseline(hook: Optional[Callable] = None):
    st.title("How it works")
    st.header("Biased Baseline (Agent Autonomy)")
    st.write(
        "Baseline agent with minimal constraints; observe risky actions and internal drift."
    )
    st.caption(
        "Safe demo stub provided by your `show_agent_autonomy` function (no external API calls here)."
    )
    hook = hook or (lambda **_: st.warning("`show_agent_autonomy` not provided. Using stub."))
    try:
        hook(key_prefix="goal_baseline")
    except TypeError:
        # Fallback if signature differs
        hook()

def _render_mitigations():
    st.title("How it works")
    st.header("Mitigations")
    st.write(
        """
- **Hard constraints** (rules/limits)
- **Human-in-the-loop approvals** for high-risk actions
- **Multi-objective optimization** to balance risk/reward
        """
    )
    with st.expander("Example policy snippet"):
        st.code(
            """\
policy:
  trading:
    max_leverage: 2.0
    banned_assets: ["crypto"]
    approval_required:
      - any_order_notional > 10000
      - new_asset_class == true
enforcement:
  block_on_violation: true
  log_all_decisions: true
  reviewers: ["risk@company.com"]
            """,
            language="yaml",
        )

def _render_exercises():
    st.title("Exercises")
    st.write(
        "Sketch an approval policy and constraints for your agent. "
        "(Optional) paste a scenario prompt:"
    )
    scenario = st.text_area(
        "Scenario",
        value="An agent tries to maximize returns; prevent leverage above 2x and block crypto trading.",
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        draft_ok = st.button("Validate Policy Draft")
    with col2:
        strict_mode = st.toggle("Strict enforcement (block on violation)", value=True)

    if draft_ok:
        st.success(
            "Looks good — bring this to the agent lab to test against simulated orders."
        )
        st.write(
            {
                "strict_mode": strict_mode,
                "scenario_len": len(scenario.strip()),
            }
        )

def _render_summary(slug: str):
    st.title("Summary")
    st.header(LABEL_BY_SLUG.get(slug, slug.capitalize()))
    if slug == "resources":
        st.markdown(
            "- ISO 42001 AI MS • NIST AI RMF • EU AI Act (high-risk systems)"
        )
        with st.expander("Practical reading order"):
            st.markdown(
                "1) NIST AI RMF overview → 2) EU AI Act scope & risk tiers → "
                "3) ISO 42001 processes for sustained compliance."
            )
    else:
        st.write(
            "You examined baseline autonomy, identified risks, and drafted mitigations with approvals."
        )

# -------------------------------
# Registry-driven dispatcher
# -------------------------------
def render_goal(sub: str, *, show_autonomy: Optional[Callable] = None):
    SECTIONS = {
        "intro": _render_intro,
        "financial": lambda: _render_use_case("financial"),
        "recruiting": lambda: _render_use_case("recruiting"),
        "medical": lambda: _render_use_case("medical"),
        "baseline": lambda: _render_baseline(show_autonomy or show_agent_autonomy),
        "mitigations": _render_mitigations,
        "exercises": _render_exercises,
        "conclusions": lambda: _render_summary("conclusions"),
        "resources": lambda: _render_summary("resources"),
    }

    if sub not in SECTIONS:
        st.title("Section not found")
        st.warning(
            f"Unknown section: '{sub}'. Choose one of: {', '.join(SECTIONS.keys())}."
        )
        return

    SECTIONS[sub]()

# -------------------------------
# URL helpers
# -------------------------------
def get_sub_from_query(default: str = "intro") -> str:
    qp = st.query_params.get("sub", default)
    # st.query_params returns str or list[str]; normalize
    if isinstance(qp, list):
        qp = qp[0] if qp else default
    return qp if qp in ALL_SECTIONS else default

def set_query_param_sub(sub: str):
    st.query_params.update({"sub": sub})

# -------------------------------
# Main app
# -------------------------------
def main():
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        current_sub = get_sub_from_query()

        # Radio keeps state; updating also syncs URL
        chosen = st.radio(
            "Pick a section",
            options=ALL_SECTIONS,
            format_func=lambda s: LABEL_BY_SLUG.get(s, s.capitalize()),
            index=ALL_SECTIONS.index(current_sub),
        )
        if chosen != current_sub:
            set_query_param_sub(chosen)

        st.markdown("---")
        st.caption(
            "Tip: Share a deep link with `?sub=baseline` (or any section slug)."
        )

    # Content surface
    render_goal(st.query_params.get("sub", "intro"))

    # Footer note
    st.markdown("---")
    st.caption(
        "Demo UI for exploring agent autonomy, internal drift, and guardrails. "
        "Replace stubs with your real simulations."
    )

if __name__ == "__main__":
    main()
