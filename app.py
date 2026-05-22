"""
Research Gap Analyzer
An AI-powered tool for identifying research gaps across academic papers.
Architecture: LLM Wiki + PageIndex + LazyGraphRAG + Academic Validation
"""
import streamlit as st
import tempfile
import os
import json
import time
from pathlib import Path

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Gap Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark editorial theme ────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }
  .stApp { background-color: #0d1117; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #58a6ff;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  /* ── Main header ── */
  .main-header {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1.1;
    background: linear-gradient(135deg, #58a6ff 0%, #79c0ff 50%, #a5d6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
  }
  .sub-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #8b949e;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-bottom: 2rem;
  }

  /* ── Stage cards ── */
  .stage-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .stage-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #58a6ff;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
  }
  .stage-body { font-size: 0.9rem; color: #c9d1d9; }

  /* ── Gap cards ── */
  .gap-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
  }
  .gap-card.open { border-left-color: #3fb950; }
  .gap-card.partial { border-left-color: #d29922; }
  .gap-card.solved { border-left-color: #f85149; }
  .gap-card.pending { border-left-color: #8b949e; }
  .gap-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #e6edf3;
    margin-bottom: 0.5rem;
  }
  .gap-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    margin-bottom: 0.6rem;
  }
  .gap-desc { font-size: 0.88rem; color: #c9d1d9; line-height: 1.6; }

  /* ── Proposal cards ── */
  .proposal-card {
    background: #0d2136;
    border: 1px solid #1f4f7a;
    border-radius: 8px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.2rem;
  }
  .proposal-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #79c0ff;
    margin-bottom: 0.8rem;
  }
  .proposal-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.8rem;
    margin-bottom: 0.3rem;
  }
  .proposal-body { font-size: 0.88rem; color: #c9d1d9; line-height: 1.6; }

  /* ── Badges ── */
  .badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    margin-right: 0.4rem;
  }
  .badge-open { background: #0f3d1a; color: #3fb950; border: 1px solid #238636; }
  .badge-partial { background: #2d1f00; color: #d29922; border: 1px solid #9e6a03; }
  .badge-solved { background: #3d0f0f; color: #f85149; border: 1px solid #da3633; }
  .badge-high { background: #0f2d3d; color: #79c0ff; border: 1px solid #1f6feb; }
  .badge-medium { background: #2d2500; color: #e3b341; border: 1px solid #9e6a03; }
  .badge-low { background: #1c1c1c; color: #8b949e; border: 1px solid #30363d; }

  /* ── Wiki card ── */
  .wiki-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
  }
  .wiki-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #e6edf3;
    margin-bottom: 0.6rem;
  }
  .wiki-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .wiki-item { font-size: 0.83rem; color: #c9d1d9; margin-left: 0.8rem; }

  /* ── Graph stats ── */
  .stat-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
  }
  .stat-number {
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    color: #58a6ff;
    display: block;
  }
  .stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3) !important;
  }

  /* ── Inputs ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stSelectbox > div > div {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 8px;
    padding: 1rem;
  }
  [data-testid="stFileUploader"]:hover { border-color: #58a6ff; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #161b22;
    border-radius: 8px 8px 0 0;
    border-bottom: 1px solid #30363d;
    padding: 0 0.5rem;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    color: #8b949e !important;
    padding: 0.7rem 1.2rem !important;
    text-transform: uppercase !important;
  }
  .stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
  }
  .stTabs [data-baseweb="tab-panel"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 1.5rem;
  }

  /* ── Expanders ── */
  .streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #c9d1d9 !important;
  }
  .streamlit-expanderContent {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-top: none !important;
  }

  /* ── Progress ── */
  .stProgress > div > div { background: linear-gradient(90deg, #1f6feb, #58a6ff) !important; }

  /* ── Divider ── */
  hr { border-color: #30363d !important; }

  /* ── Code ── */
  code { font-family: 'DM Mono', monospace !important; color: #79c0ff !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── Imports ─────────────────────────────────────────────────────────────────
from utils.azure_client import AzureOpenAIClient
from utils.export import to_json, to_markdown_report
from pipeline.pdf_parser import parse_pdf
from pipeline.page_index import build_tree
from pipeline.wiki_compiler import build_wiki
from pipeline.graph_builder import build_knowledge_graph
from pipeline.gap_detector import detect_gaps
from pipeline.academic_search import validate_gaps
from pipeline.proposal_generator import generate_proposals
from pipeline.param_extractor import (
    extract_parameters, check_sufficiency,
    params_to_csv, group_params_by_category,
)
from pipeline.llm_council import council_select_model, council_synthesise_insights
from pipeline.simulation_engine import run_simulation
from pipeline.doc_generator import build_zip
from domain_config import (
    get_domain_names, get_domain_display_options,
    parse_domain_selection, get_domain_config, DOMAINS
)

if "comp_results" not in st.session_state:
    st.session_state.comp_results = None
if "results" not in st.session_state:
    st.session_state.results = None
if "running" not in st.session_state:
    st.session_state.running = False


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Research Gap Analyzer")
    st.markdown("---")

    st.markdown("##### API Configuration")
    st.markdown("""
    <div style="background:#0d2136; border:1px solid #1f4f7a; border-radius:6px;
                padding:0.8rem 1rem; margin-bottom:0.8rem;">
      <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#58a6ff;
                  text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">
        Model Stack
      </div>
      <div style="font-size:0.78rem; color:#e6edf3; font-weight:500; margin-bottom:0.2rem;">
        🧠 Master — openai/gpt-oss-120b
      </div>
      <div style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#484f58; line-height:1.9;">
        Council Agent 1 · gpt-oss-20b<br>
        Council Agent 2 · llama-3.2-3b-instruct<br>
        Council Agent 3 · nemotron-nano-omni-30b<br>
        Chair · gpt-oss-120b
      </div>
      <div style="font-family:'DM Mono',monospace; font-size:0.6rem; color:#484f58; margin-top:0.4rem;">
        via NVIDIA NIM
      </div>
    </div>
    """, unsafe_allow_html=True)

    api_key = st.text_input(
        "NVIDIA API Key",
        type="password",
        placeholder="nvapi-••••••••••••••••",
        help="Single key used for all models. Get yours at build.nvidia.com",
    )
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace; font-size:0.62rem; color:#484f58;">'
        '🔗 <a href="https://build.nvidia.com" target="_blank" style="color:#58a6ff;">'
        'Get API key → build.nvidia.com</a></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### Research Domain")

    domain_options = get_domain_display_options()
    # Default to Healthcare AI since that's Sumit's focus
    default_idx = next((i for i, d in enumerate(domain_options) if "Healthcare" in d), 0)
    selected_display = st.selectbox("Select Domain", domain_options, index=default_idx)
    selected_domain = parse_domain_selection(selected_display)
    domain_cfg = get_domain_config(selected_domain)

    # Show domain description + what gets tuned
    st.markdown(f"""
    <div style="background:#0d2136; border:1px solid #1f4f7a; border-radius:6px; padding:0.7rem 0.9rem; margin-top:0.3rem;">
      <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#58a6ff; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem;">Domain Active</div>
      <div style="font-size:0.78rem; color:#c9d1d9;">{domain_cfg['description']}</div>
      <div style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#484f58; margin-top:0.4rem; line-height:1.7;">
        ✦ Wiki prompts tuned<br>
        ✦ Graph entity types tuned<br>
        ✦ Gap priorities tuned<br>
        ✦ Proposal framing tuned<br>
        ✦ Academic search boosted
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Research Context")
    interest = st.text_area(
        "Your Specific Interest",
        placeholder=f"e.g., {domain_cfg['gap_examples'][:80]}...",
        height=75
    )
    gap_type = st.selectbox(
        "Gap Type Focus",
        ["Any", "Methodology", "Application", "Dataset", "Evaluation", "Theory", "Benchmark"],
    )

    st.markdown("---")
    st.markdown("##### Pipeline Options")
    use_vision = st.checkbox("🖼️ Vision Analysis (figures/tables)", value=False,
                              help="Sends figure pages to GPT-4o Vision. Increases accuracy but uses more API calls.")

    st.markdown("---")
    st.markdown("##### 🏛️ Computational Lab (NVIDIA Council)")
    nvidia_api_key = st.text_input(
        "NVIDIA NIM API Key",
        type="password",
        placeholder="nvapi-••••••••••••••••",
        help="Used only for the LLM Council in the Computational Lab tab. Get key at build.nvidia.com",
    )
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace; font-size:0.62rem; color:#484f58;">'
        '🔗 <a href="https://build.nvidia.com" target="_blank" style="color:#58a6ff;">'
        'build.nvidia.com → free tier available</a></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #484f58; line-height: 1.8;">
    PIPELINE<br>
    ① Multimodal PDF Parse<br>
    ② PageIndex Tree Build<br>
    ③ LLM Wiki Compile<br>
    ④ LazyGraphRAG Build<br>
    ⑤ Gap Detection<br>
    ⑥ Academic Validation<br>
    ⑦ Proposal Generation
    </div>
    """, unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Research Gap Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload research papers → AI discovers what the field hasn\'t done yet</div>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload up to 5 research papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload 2–5 related research papers. The more focused the set, the better the gap analysis.",
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Please upload a maximum of 5 papers.")
        uploaded_files = uploaded_files[:5]
    
    n = len(uploaded_files)
    cols = st.columns(min(n, 5))
    for i, (col, f) in enumerate(zip(cols, uploaded_files)):
        with col:
            size_kb = len(f.getvalue()) // 1024
            st.markdown(f"""
            <div class="stage-card" style="text-align:center; padding:0.8rem;">
              <div class="stage-title">Paper {i+1}</div>
              <div class="stage-body" style="font-size:0.78rem; word-break:break-word;">{f.name}</div>
              <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#8b949e; margin-top:0.3rem;">{size_kb} KB</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("")

run_col, _ = st.columns([1, 3])
with run_col:
    run_btn = st.button("🚀 Run Analysis", disabled=st.session_state.running)

# ── Pipeline Execution ────────────────────────────────────────────────────────
if run_btn and uploaded_files:
    # Validate inputs
    if not api_key:
        st.error("Please enter your NVIDIA API key in the sidebar.")
        st.stop()
    if len(uploaded_files) < 2:
        st.warning("Please upload at least 2 papers for meaningful gap analysis.")
        st.stop()

    st.session_state.running = True

    client = AzureOpenAIClient(api_key=api_key.strip())

    context = {
        "domain": selected_domain,
        "interest": interest or "General research improvements",
        "gap_type": gap_type,
    }

    progress_bar = st.progress(0)
    status_area = st.empty()

    def update_status(msg: str, pct: int):
        progress_bar.progress(pct)
        status_area.markdown(f"""
        <div class="stage-card">
          <div class="stage-title">Pipeline Running</div>
          <div class="stage-body">⟳ {msg}</div>
        </div>
        """, unsafe_allow_html=True)

    try:
        all_papers = []
        all_trees = []

        # ── Stage 1+2: PDF Parsing + PageIndex ──────────────────────────────
        update_status("Stage 1 of 7 — Parsing PDFs & building PageIndex trees...", 5)
        for i, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            update_status(f"Stage 1 — Parsing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...", 5 + i * 5)

            paper = parse_pdf(
                pdf_path=tmp_path,
                filename=uploaded_file.name,
                client=client if use_vision else None,
                use_vision=use_vision,
            )
            tree = build_tree(paper)

            all_papers.append(paper)
            all_trees.append(tree)
            os.unlink(tmp_path)

        progress_bar.progress(20)

        # ── Stage 3: LLM Wiki Compiler ───────────────────────────────────────
        update_status("Stage 3 of 7 — Compiling LLM Wiki pages (Karpathy pattern)...", 22)
        wiki = build_wiki(all_papers, all_trees, client, domain_config=domain_cfg)
        progress_bar.progress(38)

        # ── Stage 4: Knowledge Graph ─────────────────────────────────────────
        update_status("Stage 4 of 7 — Building knowledge graph (LazyGraphRAG)...", 40)
        graph = build_knowledge_graph(wiki, client, domain_config=domain_cfg)
        progress_bar.progress(55)

        # ── Stage 5: Gap Detection ────────────────────────────────────────────
        update_status("Stage 5 of 7 — Running gap detection agent...", 57)
        gaps = detect_gaps(wiki, graph, context, client, domain_config=domain_cfg)
        progress_bar.progress(70)

        # ── Stage 6: Academic Validation ─────────────────────────────────────
        update_status("Stage 6 of 7 — Validating against Semantic Scholar & arXiv...", 72)
        validated_gaps = validate_gaps(gaps, client, domain_config=domain_cfg)
        progress_bar.progress(85)

        # ── Stage 7: Proposal Generation ─────────────────────────────────────
        update_status("Stage 7 of 7 — Generating research proposals...", 87)
        proposals = generate_proposals(validated_gaps, wiki, context, client, domain_config=domain_cfg)
        progress_bar.progress(100)

        # ── Store results ──────────────────────────────────────────────────────
        st.session_state.results = {
            "papers": [
                {"filename": p["filename"], "wiki": w, "char_count": p.get("char_count", 0)}
                for p, w in zip(all_papers, wiki["pages"])
            ],
            "wiki": wiki,
            "graph": graph,
            "gaps": validated_gaps,
            "proposals": proposals,
            "communities": graph.get("communities", []),
            "context": context,
        }

        progress_bar.empty()
        status_area.empty()
        st.success(f"✅ Analysis complete — {len(validated_gaps)} gaps found, {len(proposals)} proposals generated.")

    except Exception as e:
        progress_bar.empty()
        status_area.empty()
        st.error(f"Pipeline error: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())

    finally:
        st.session_state.running = False


# ── Results Display ───────────────────────────────────────────────────────────
if st.session_state.results:
    res = st.session_state.results
    gaps = res.get("gaps", [])
    proposals = res.get("proposals", [])
    graph = res.get("graph", {})
    wiki = res.get("wiki", {})
    wiki_pages = wiki.get("pages", [])
    communities = res.get("communities", [])
    context = res.get("context", {})

    st.markdown("---")

    # ── Domain banner ──────────────────────────────────────────────────────
    res_domain = context.get("domain", "General AI/ML")
    res_cfg = get_domain_config(res_domain)
    st.markdown(f"""
    <div style="background:#0d2136; border:1px solid #1f4f7a; border-radius:8px;
                padding:0.8rem 1.2rem; margin-bottom:1.2rem; display:flex; align-items:center; gap:1rem;">
      <div style="font-size:1.6rem;">{res_cfg['icon']}</div>
      <div>
        <div style="font-family:'DM Mono',monospace; font-size:0.68rem; color:#58a6ff;
                    text-transform:uppercase; letter-spacing:0.12em;">Domain-Tuned Analysis</div>
        <div style="font-size:0.9rem; color:#e6edf3; font-weight:500;">{res_domain}</div>
        <div style="font-size:0.75rem; color:#8b949e;">{res_cfg['description']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary metrics ────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        (len(res.get("papers", [])), "Papers"),
        (graph.get("stats", {}).get("entity_count", 0), "Entities"),
        (graph.get("stats", {}).get("community_count", 0), "Communities"),
        (len(gaps), "Gaps Found"),
        (len(proposals), "Proposals"),
    ]
    for col, (val, label) in zip([m1, m2, m3, m4, m5], metrics):
        with col:
            st.markdown(f"""
            <div class="stat-box">
              <span class="stat-number">{val}</span>
              <span class="stat-label">{label}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Wiki Pages",
        "🕸️ Knowledge Graph",
        "🔍 Research Gaps",
        "💡 Proposals",
        "📦 Export",
        "📊 Computational Lab",
    ])

    # ── Tab 1: Wiki Pages ────────────────────────────────────────────────
    with tab1:
        st.markdown("### Compiled Wiki Pages")
        st.markdown(
            '<div style="font-size:0.82rem; color:#8b949e; margin-bottom:1rem;">Each paper compiled into structured knowledge using the LLM Wiki pattern. Click to expand.</div>',
            unsafe_allow_html=True,
        )
        for i, page in enumerate(wiki_pages):
            title = page.get("title", page.get("source_file", f"Paper {i+1}"))
            with st.expander(f"📄 {title}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="wiki-label">Contributions</div>', unsafe_allow_html=True)
                    for c in page.get("contributions", [])[:5]:
                        st.markdown(f'<div class="wiki-item">• {c}</div>', unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown('<div class="wiki-label">Methods</div>', unsafe_allow_html=True)
                    for m in page.get("methods", [])[:5]:
                        st.markdown(f'<div class="wiki-item">• {m}</div>', unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown('<div class="wiki-label">Datasets</div>', unsafe_allow_html=True)
                    for d in page.get("datasets", [])[:4]:
                        st.markdown(f'<div class="wiki-item">• {d}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="wiki-label">Key Findings</div>', unsafe_allow_html=True)
                    for f in page.get("key_findings", [])[:4]:
                        st.markdown(f'<div class="wiki-item">• {f}</div>', unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown('<div class="wiki-label">Limitations ⚠️</div>', unsafe_allow_html=True)
                    for lim in page.get("limitations", [])[:4]:
                        st.markdown(f'<div class="wiki-item" style="color:#f85149;">• {lim}</div>', unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown('<div class="wiki-label">Future Work 🔭</div>', unsafe_allow_html=True)
                    for fw in page.get("future_work", [])[:4]:
                        st.markdown(f'<div class="wiki-item" style="color:#3fb950;">• {fw}</div>', unsafe_allow_html=True)

        # Cross-links
        cross_links = wiki.get("cross_links", {})
        if cross_links.get("shared_concepts"):
            st.markdown("### Cross-Paper Links")
            for link in cross_links["shared_concepts"][:6]:
                papers_str = ", ".join(link.get("papers", []))
                st.markdown(f"""
                <div class="stage-card" style="margin-bottom:0.5rem;">
                  <div class="stage-title">Shared Concept</div>
                  <div class="stage-body"><strong>{link.get('concept','')}</strong> — {link.get('context','')} <br>
                  <span style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#58a6ff;">{papers_str}</span></div>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 2: Knowledge Graph ───────────────────────────────────────────
    with tab2:
        st.markdown("### Knowledge Graph (LazyGraphRAG)")
        stats = graph.get("stats", {})
        gc1, gc2, gc3, gc4 = st.columns(4)
        for col, (label, val) in zip(
            [gc1, gc2, gc3, gc4],
            [
                ("Entities", stats.get("entity_count", 0)),
                ("Relations", stats.get("relationship_count", 0)),
                ("Communities", stats.get("community_count", 0)),
                ("Orphan Signals", stats.get("orphan_count", 0)),
            ],
        ):
            with col:
                st.markdown(f"""
                <div class="stat-box">
                  <span class="stat-number" style="font-size:1.5rem;">{val}</span>
                  <span class="stat-label">{label}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### Communities")
        for comm in communities:
            with st.expander(f"🏘️ {comm.get('theme', 'Community')}"):
                st.markdown(f'<div class="stage-body">{comm.get("summary", "")}</div>', unsafe_allow_html=True)
                if comm.get("gap_signal"):
                    st.markdown(f"""
                    <div style="background:#2d1f00; border:1px solid #9e6a03; border-radius:4px; padding:0.6rem 0.8rem; margin-top:0.6rem; font-size:0.82rem; color:#d29922;">
                      ⚠️ Gap Signal: {comm['gap_signal']}
                    </div>
                    """, unsafe_allow_html=True)

        # Orphan concepts
        orphans = graph.get("orphan_concepts", [])
        if orphans:
            st.markdown("#### Orphan Concepts")
            st.markdown('<div style="font-size:0.82rem; color:#8b949e; margin-bottom:0.8rem;">Concepts mentioned in only one paper — unexplored territory.</div>', unsafe_allow_html=True)
            for o in orphans[:8]:
                entity = o.get("entity", {})
                st.markdown(f"""
                <div class="stage-card" style="padding:0.7rem 1rem; margin-bottom:0.4rem;">
                  <span style="font-family:'DM Mono',monospace; color:#79c0ff; font-size:0.82rem;">{entity.get('name','')}</span>
                  <span style="font-family:'DM Mono',monospace; color:#484f58; font-size:0.7rem; margin-left:0.8rem;">[{entity.get('type','')}]</span>
                  <div style="font-size:0.78rem; color:#8b949e; margin-top:0.2rem;">{o.get('signal','')}</div>
                </div>
                """, unsafe_allow_html=True)

        # Entities table
        with st.expander("View all entities"):
            entities = graph.get("entities", [])
            if entities:
                rows = []
                for e in entities:
                    rows.append({
                        "Name": e.get("name", ""),
                        "Type": e.get("type", ""),
                        "Papers": ", ".join(e.get("papers", []))[:60],
                        "Importance": e.get("importance", ""),
                    })
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── Tab 3: Research Gaps ──────────────────────────────────────────────
    with tab3:
        st.markdown("### Identified Research Gaps")

        open_gaps = [g for g in gaps if g.get("validation_status") == "open"]
        partial_gaps = [g for g in gaps if g.get("validation_status") == "partial"]
        solved_gaps = [g for g in gaps if g.get("validation_status") == "solved"]

        f1, f2, f3 = st.columns(3)
        for col, label, cnt, color in [
            (f1, "🟢 Open", len(open_gaps), "#3fb950"),
            (f2, "🟡 Partial", len(partial_gaps), "#d29922"),
            (f3, "🔴 Solved", len(solved_gaps), "#f85149"),
        ]:
            with col:
                st.markdown(f"""
                <div class="stat-box">
                  <span class="stat-number" style="color:{color}; font-size:1.5rem;">{cnt}</span>
                  <span class="stat-label">{label}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        for gap in gaps:
            status = gap.get("validation_status", "pending")
            confidence = gap.get("confidence", "medium")

            status_badges = {
                "open": '<span class="badge badge-open">Open</span>',
                "partial": '<span class="badge badge-partial">Partial</span>',
                "solved": '<span class="badge badge-solved">Solved</span>',
                "pending": '<span class="badge" style="background:#1c1c1c;color:#8b949e;border:1px solid #30363d;">Pending</span>',
            }
            conf_badges = {
                "high": '<span class="badge badge-high">High Confidence</span>',
                "medium": '<span class="badge badge-medium">Medium Confidence</span>',
                "low": '<span class="badge badge-low">Low Confidence</span>',
            }

            st.markdown(f"""
            <div class="gap-card {status}">
              <div class="gap-title">{gap.get('title', 'Untitled Gap')}</div>
              <div class="gap-meta">
                {status_badges.get(status, '')} {conf_badges.get(confidence, '')}
                <span class="badge" style="background:#1a1a2e;color:#8b949e;border:1px solid #30363d;">{gap.get('gap_type','')}</span>
              </div>
              <div class="gap-desc">{gap.get('description', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Evidence, validation & related work"):
                ev_col, rel_col = st.columns([1, 1])
                with ev_col:
                    st.markdown('<div class="wiki-label">Evidence from Papers</div>', unsafe_allow_html=True)
                    for ev in gap.get("evidence", []):
                        st.markdown(f'<div class="wiki-item" style="margin-bottom:0.3rem;">📎 {ev}</div>', unsafe_allow_html=True)
                    if gap.get("missing_connection"):
                        st.markdown("")
                        st.markdown('<div class="wiki-label">Missing Connection</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="wiki-item">{gap["missing_connection"]}</div>', unsafe_allow_html=True)
                    if gap.get("validation_reasoning"):
                        st.markdown("")
                        st.markdown('<div class="wiki-label">Validation Note</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="wiki-item" style="color:#8b949e;">{gap["validation_reasoning"]}</div>', unsafe_allow_html=True)

                with rel_col:
                    existing = gap.get("existing_papers", [])
                    if existing:
                        st.markdown('<div class="wiki-label">Related Existing Work</div>', unsafe_allow_html=True)
                        for ep in existing[:4]:
                            yr = ep.get("year", "")
                            src = ep.get("source", "")
                            st.markdown(f'<div class="wiki-item" style="margin-bottom:0.4rem;">📄 {ep.get("title","N/A")} ({yr}) <span style="color:#58a6ff; font-size:0.65rem;">[{src}]</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="wiki-item" style="color:#3fb950;">No directly relevant papers found — gap appears open.</div>', unsafe_allow_html=True)

    # ── Tab 4: Proposals ─────────────────────────────────────────────────
    with tab4:
        st.markdown("### Research Proposals")
        st.markdown('<div style="font-size:0.82rem; color:#8b949e; margin-bottom:1.5rem;">Concrete, actionable research paper ideas generated from validated open gaps.</div>', unsafe_allow_html=True)

        for i, proposal in enumerate(proposals, 1):
            confidence = proposal.get("confidence", "medium")
            effort = proposal.get("effort_estimate", "")
            risk = proposal.get("risk", "")

            effort_color = {"short": "#3fb950", "medium": "#d29922", "long": "#f85149"}.get(
                effort.split()[0].lower() if effort else "", "#8b949e"
            )

            st.markdown(f"""
            <div class="proposal-card">
              <div class="proposal-title">#{i}  {proposal.get('title','Untitled Proposal')}</div>
              <div>
                <span class="badge badge-{confidence}">{confidence.title()} Confidence</span>
                <span class="badge" style="background:#1a2a1a;color:{effort_color};border:1px solid {effort_color};">{effort}</span>
                <span class="badge" style="background:#1c1c1c;color:#8b949e;border:1px solid #30363d;">Risk: {risk}</span>
              </div>
              <div class="proposal-section">Problem Statement</div>
              <div class="proposal-body">{proposal.get('problem_statement','')}</div>
              <div class="proposal-section">Proposed Methodology</div>
              <div class="proposal-body">{proposal.get('methodology','')}</div>
              <div class="proposal-section">Novelty</div>
              <div class="proposal-body">{proposal.get('novelty','')}</div>
              <div class="proposal-section">Expected Contribution</div>
              <div class="proposal-body">{proposal.get('expected_contribution','')}</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Experiments, datasets & source papers"):
                e1, e2 = st.columns(2)
                with e1:
                    st.markdown('<div class="wiki-label">Suggested Experiments</div>', unsafe_allow_html=True)
                    for exp in proposal.get("suggested_experiments", []):
                        st.markdown(f'<div class="wiki-item">🧪 {exp}</div>', unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown('<div class="wiki-label">Potential Datasets</div>', unsafe_allow_html=True)
                    for ds in proposal.get("potential_datasets", []):
                        st.markdown(f'<div class="wiki-item">📊 {ds}</div>', unsafe_allow_html=True)
                with e2:
                    st.markdown('<div class="wiki-label">Builds On</div>', unsafe_allow_html=True)
                    for src in proposal.get("builds_on", []):
                        st.markdown(f'<div class="wiki-item">📎 {src}</div>', unsafe_allow_html=True)
                    st.markdown("")
                    if proposal.get("addresses_gap"):
                        st.markdown('<div class="wiki-label">Addresses Gap</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="wiki-item" style="color:#79c0ff;">{proposal["addresses_gap"]}</div>', unsafe_allow_html=True)

    # ── Tab 5: Export ─────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Export Results")
        st.markdown('<div style="font-size:0.82rem; color:#8b949e; margin-bottom:1.5rem;">Download your analysis in different formats.</div>', unsafe_allow_html=True)

        ex1, ex2 = st.columns(2)

        with ex1:
            st.markdown("""
            <div class="stage-card">
              <div class="stage-title">Markdown Report</div>
              <div class="stage-body">Full analysis as a readable Markdown document — suitable for sharing with collaborators.</div>
            </div>
            """, unsafe_allow_html=True)
            md_report = to_markdown_report(res, context)
            st.download_button(
                label="⬇ Download Markdown Report",
                data=md_report,
                file_name="research_gap_report.md",
                mime="text/markdown",
            )

        with ex2:
            st.markdown("""
            <div class="stage-card">
              <div class="stage-title">JSON Export</div>
              <div class="stage-body">Complete structured data — wiki pages, knowledge graph, gaps, proposals. For programmatic use.</div>
            </div>
            """, unsafe_allow_html=True)
            json_export = to_json(res)
            st.download_button(
                label="⬇ Download JSON",
                data=json_export,
                file_name="research_gap_analysis.json",
                mime="application/json",
            )

        st.markdown("---")
        st.markdown("#### Raw JSON Preview")
        with st.expander("View JSON"):
            st.json(res)


    # ── Tab 6: Computational Lab ──────────────────────────────────────────
    with tab6:
        st.markdown("### 📊 Computational Lab")
        st.markdown(
            '<div style="font-size:0.82rem; color:#8b949e; margin-bottom:1.5rem;">'
            'Extract quantitative parameters → LLM Council selects model → '
            'Simulation runs → Council synthesises insights → Download full package.'
            '</div>', unsafe_allow_html=True,
        )

        col_run, col_info = st.columns([1, 3])
        with col_run:
            run_comp = st.button("🧬 Run Computational Lab", key="run_comp")
        with col_info:
            st.markdown(
                '<div style="font-size:0.78rem; color:#8b949e; padding-top:0.5rem;">'
                'Stages 8–12 · ~3–5 min · Uses same NVIDIA API key</div>',
                unsafe_allow_html=True,
            )

        if run_comp:
            comp_progress = st.progress(0)
            comp_status = st.empty()

            def comp_update(msg, pct):
                comp_progress.progress(pct)
                comp_status.markdown(
                    f'<div class="stage-card"><div class="stage-title">Running</div>'
                    f'<div class="stage-body">⟳ {msg}</div></div>',
                    unsafe_allow_html=True,
                )

            try:
                # Recreate client here — api_key is in scope from sidebar
                client = AzureOpenAIClient(api_key=api_key.strip())

                papers_raw = [
                    {"sections": {}, "tables": [], "filename": p.get("filename", "")}
                    for p in res.get("papers", [])
                ]

                comp_update("Stage 8 — Extracting quantitative parameters from papers...", 10)
                params = extract_parameters(wiki, papers_raw, client)
                sufficiency = check_sufficiency(params, client)
                csv_data = params_to_csv(params)
                comp_progress.progress(28)

                comp_update(f"Stage 9 — LLM Council debating model ({len(params)} params)...", 30)
                model_council = council_select_model(params, sufficiency, context, client)
                comp_progress.progress(50)

                model_name = model_council.get("final_decision", {}).get("selected_model", "monte_carlo")
                comp_update(f"Stage 10 — Running {model_name.replace('_',' ').title()}...", 52)
                sim_results = run_simulation(params, model_council)
                comp_progress.progress(70)

                comp_update("Stage 11 — Council synthesising clinical insights...", 73)
                insight_council = council_synthesise_insights(sim_results, params, gaps, context, client)
                comp_progress.progress(88)

                comp_update("Stage 12 — Building report package...", 90)
                zip_bytes = build_zip(
                    params_csv=csv_data, sim_results=sim_results,
                    model_council=model_council, insight_council=insight_council,
                    context=context, params=params, sufficiency=sufficiency,
                )
                comp_progress.progress(100)

                st.session_state.comp_results = {
                    "params": params, "sufficiency": sufficiency,
                    "csv_data": csv_data, "model_council": model_council,
                    "sim_results": sim_results, "insight_council": insight_council,
                    "zip_bytes": zip_bytes,
                }
                comp_progress.empty()
                comp_status.empty()
                st.success(f"✅ Done — {len(params)} parameters, {sim_results.get('model_used','model')} run.")

            except Exception as e:
                comp_progress.empty()
                comp_status.empty()
                st.error(f"Error: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

        if st.session_state.comp_results:
            cr = st.session_state.comp_results
            params          = cr["params"]
            sufficiency     = cr["sufficiency"]
            model_council   = cr["model_council"]
            sim_results     = cr["sim_results"]
            insight_council = cr["insight_council"]

            st.markdown("---")

            suff_color = "#3fb950" if sufficiency.get("sufficient") else "#d29922"
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid {suff_color}; border-radius:8px;
                        padding:0.8rem 1.2rem; margin-bottom:1.2rem;">
              <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:{suff_color};
                          text-transform:uppercase; letter-spacing:0.1em;">Parameter Sufficiency</div>
              <div style="font-size:0.88rem; color:#e6edf3; margin-top:0.3rem;">
                {len(params)} parameters extracted · Score: {sufficiency.get('coverage_score',0)}/10 ·
                {sufficiency.get('recommendation','').replace('_',' ').title()}
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Stage 8 — Extracted Parameters")
            grouped = group_params_by_category(params)
            for cat, cat_params in list(grouped.items())[:8]:
                with st.expander(f"📊 {cat.replace('_',' ').title()} ({len(cat_params)})"):
                    for p in cat_params:
                        ci_str = f" (CI: {p['ci_lower']}–{p['ci_upper']})" if p.get("ci_lower") and p.get("ci_upper") else ""
                        st.markdown(
                            f'<div class="wiki-item" style="margin-bottom:0.4rem;">'
                            f'<span style="color:#79c0ff;">{p.get("name","")}</span> = '
                            f'<span style="color:#3fb950;font-family:\'DM Mono\',monospace;">'
                            f'{p.get("value","")}{ci_str}</span> {p.get("unit","")}'
                            f'<span style="color:#484f58;font-size:0.7rem;"> [{p.get("source_paper","")[:45]}]</span>'
                            f'</div>', unsafe_allow_html=True,
                        )

            st.download_button("⬇ Download parameters.csv", data=cr["csv_data"],
                               file_name="parameters.csv", mime="text/csv")

            st.markdown("#### Stage 9 — LLM Council: Model Selection")
            fd = model_council.get("final_decision", {})
            st.markdown(f"""
            <div style="background:#0d2136;border:1px solid #1f4f7a;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;">
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#58a6ff;text-transform:uppercase;letter-spacing:0.1em;">Council Decision</div>
              <div style="font-size:1.05rem;color:#e6edf3;font-weight:500;margin-top:0.3rem;">{fd.get('selected_model','').replace('_',' ').title()}</div>
              <div style="font-size:0.83rem;color:#c9d1d9;margin-top:0.4rem;">{fd.get('rationale','')}</div>
              <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#8b949e;margin-top:0.4rem;">Confidence: {fd.get('confidence','')} · Scenarios: {', '.join(fd.get('scenarios',[]))}</div>
            </div>
            """, unsafe_allow_html=True)

            for agent in model_council.get("agents", []):
                with st.expander(f"🤖 {agent['agent']} — {agent['role']}"):
                    st.markdown(f'<div style="font-size:0.85rem;color:#c9d1d9;line-height:1.7;">{agent["response"]}</div>', unsafe_allow_html=True)

            consensus = model_council.get("consensus", {})
            for pt in consensus.get("key_points", []):
                st.markdown(f'<div class="wiki-item">✦ {pt}</div>', unsafe_allow_html=True)

            st.markdown(f"#### Stage 10 — Simulation: {sim_results.get('model_used','')}")
            scenarios_run = sim_results.get("scenarios_run", [])
            if scenarios_run:
                sc_cols = st.columns(len(scenarios_run))
                headline = sim_results.get("headline_numbers", {})
                for col, sc in zip(sc_cols, scenarios_run):
                    sc_key = sc.lower().replace(" ", "_")
                    vals = [v for k, v in headline.items() if sc_key in k]
                    val = round(vals[0], 3) if vals else "N/A"
                    with col:
                        st.markdown(f'<div class="stat-box"><span class="stat-number" style="font-size:1.4rem;">{val}</span><span class="stat-label">{sc}</span></div>', unsafe_allow_html=True)

            for i, s in enumerate(sim_results.get("sensitivity_ranking", [])[:6], 1):
                bar = int(s["importance"] * 100)
                st.markdown(
                    f'<div style="margin-bottom:0.4rem;"><div style="font-family:\'DM Mono\',monospace;font-size:0.75rem;color:#c9d1d9;">{i}. {s["parameter"]}</div>'
                    f'<div style="background:#30363d;border-radius:3px;height:6px;margin-top:2px;">'
                    f'<div style="background:#58a6ff;width:{bar}%;height:6px;border-radius:3px;"></div></div></div>',
                    unsafe_allow_html=True,
                )

            markov_data = sim_results if sim_results.get("model_used") == "Markov Chain" else sim_results.get("markov", {})
            sc_results = markov_data.get("scenario_results", {})
            if sc_results:
                first_sc = list(sc_results.values())[0]
                trajectories = first_sc.get("trajectories", {})
                states = first_sc.get("states", [])
                if trajectories:
                    try:
                        import plotly.graph_objects as go
                        colors_st = ["#3fb950","#d29922","#f85149","#58a6ff","#8b949e"]
                        fig = go.Figure()
                        for state, color in zip(states, colors_st):
                            vals = trajectories.get(state, [])
                            fig.add_trace(go.Scatter(x=list(range(len(vals))), y=vals, name=state, line=dict(color=color), mode="lines+markers"))
                        fig.update_layout(title="Markov Chain — State Trajectories", paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                            font=dict(color="#c9d1d9", size=11), height=360,
                            xaxis=dict(title="Year", gridcolor="#30363d"),
                            yaxis=dict(title="Cohort (10,000)", gridcolor="#30363d"),
                            margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.info("pip install plotly for charts")

            st.markdown("#### Stage 11 — LLM Council: Clinical Insights")
            fi = insight_council.get("final_decision", {})
            headline = fi.get("headline_finding", "")
            if headline:
                st.markdown(f'<div style="background:#0f3d1a;border:1px solid #238636;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;"><div style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#3fb950;text-transform:uppercase;letter-spacing:0.1em;">Headline Finding</div><div style="font-size:1rem;color:#e6edf3;margin-top:0.4rem;font-style:italic;">"{headline}"</div></div>', unsafe_allow_html=True)

            ic1, ic2 = st.columns(2)
            with ic1:
                st.markdown('<div class="wiki-label">Clinical Insights</div>', unsafe_allow_html=True)
                for ci in fi.get("clinical_insights", []):
                    st.markdown(f'<div class="wiki-item">🏥 {ci}</div>', unsafe_allow_html=True)
                st.markdown("")
                st.markdown('<div class="wiki-label">Recommendations</div>', unsafe_allow_html=True)
                for rec in fi.get("actionable_recommendations", []):
                    st.markdown(f'<div class="wiki-item" style="color:#3fb950;">→ {rec}</div>', unsafe_allow_html=True)
            with ic2:
                st.markdown('<div class="wiki-label">Research Insights</div>', unsafe_allow_html=True)
                for ri in fi.get("research_insights", []):
                    st.markdown(f'<div class="wiki-item">🔬 {ri}</div>', unsafe_allow_html=True)
                st.markdown("")
                st.markdown('<div class="wiki-label">Limitations</div>', unsafe_allow_html=True)
                for lim in fi.get("limitations", []):
                    st.markdown(f'<div class="wiki-item" style="color:#d29922;">⚠️ {lim}</div>', unsafe_allow_html=True)

            st.markdown("**Agent Debate:**")
            for agent in insight_council.get("agents", []):
                with st.expander(f"🤖 {agent['agent']} — {agent['role']}"):
                    st.markdown(f'<div style="font-size:0.85rem;color:#c9d1d9;line-height:1.7;">{agent["response"]}</div>', unsafe_allow_html=True)

            pub = fi.get("publishability", "")
            if pub:
                st.markdown(f'<div style="background:#161b22;border:1px solid #58a6ff;border-radius:6px;padding:0.7rem 1rem;margin-top:1rem;"><div class="wiki-label">Publishability</div><div style="font-size:0.85rem;color:#c9d1d9;margin-top:0.3rem;">{pub}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Stage 12 — Download Report Package")
            st.markdown('<div class="stage-card"><div class="stage-title">Package Contents</div><div class="stage-body">📄 parameters.csv · 📊 simulation_results.json · 🏛️ council_debate.md · 💡 insights.md · 📋 full_report.md</div></div>', unsafe_allow_html=True)
            st.download_button("⬇ Download ZIP", data=cr["zip_bytes"],
                               file_name="computational_lab_output.zip", mime="application/zip")

