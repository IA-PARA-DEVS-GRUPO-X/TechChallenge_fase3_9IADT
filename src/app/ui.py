"""Interface de demonstracao (Streamlit) do assistente medico.

Executar:
    streamlit run src/app/ui.py --server.enableCORS false \
        --server.enableXsrfProtection false

Mostra, numa unica tela, os quatro pontos exigidos no video de entrega:
funcionamento da LLM personalizada, execucao do fluxo automatizado,
resposta clinica contextualizada e os logs/validacao das respostas.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st  # noqa: E402

from src.app.llm import adapter_available  # noqa: E402
from src.config import ADAPTER_DIR, BASE_LLM, RETRIEVER_TOP_K  # noqa: E402
from src.db.patients import list_patients  # noqa: E402
from src.security import audit  # noqa: E402
from src.workflow.graph import run_flow  # noqa: E402

LEVEL_UI = {
    "critical": ("error", "CRITICAL"),
    "warning": ("warning", "WARNING"),
    "info": ("info", "INFO"),
}

EXAMPLES = [
    "Is early diagnosis associated with better outcomes in sepsis?",
    "What does the evidence say about corticosteroid use in COPD exacerbation?",
    "Does early mobilization reduce length of stay in critically ill patients?",
]

st.set_page_config(page_title="Medical Assistant - Phase 3", layout="wide")


@st.cache_data(show_spinner=False)
def load_patients():
    return list_patients()


# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")

    if adapter_available():
        st.success("Fine-tuned LLM loaded")
        st.caption(f"{BASE_LLM}\n+ LoRA: {ADAPTER_DIR.name}")
    else:
        st.warning("LoRA adapter missing: using the base model")

    patients = load_patients()
    options = ["(no patient)"] + [
        f"{p['id']} - {p['iniciais']}, {p['idade']}y, {p['diagnostico']}"
        for p in patients
    ]
    choice = st.selectbox("Patient", options)
    patient_id = None if choice.startswith("(") else choice.split(" - ")[0]

    k = st.slider("Evidence passages (k)", 1, 8, RETRIEVER_TOP_K)

    st.divider()
    st.caption(
        "Decision support system. It does not prescribe: every suggestion "
        "goes through programmatic guardrails and requires human validation."
    )

# -------------------------------------------------------------- corpo
st.title("Medical Assistant - Tech Challenge Phase 3")

example = st.selectbox("Example questions", EXAMPLES)
question = st.text_area(
    "Clinical question (in English)",
    value=example,
    height=90,
    help="The model and the evidence base are in English. "
         "The question is sanitized (PII) before reaching the model.",
)

if st.button("Run flow", type="primary"):
    with st.spinner("Running the clinical flow..."):
        st.session_state["state"] = run_flow(question, patient_id=patient_id, k=k)

state = st.session_state.get("state")

if state:
    st.caption(f"trace_id: `{state['trace_id']}`")

    st.subheader("Executed flow")
    st.code(" -> ".join(state.get("path", [])), language="text")

    alerts = state.get("alerts", [])
    if alerts:
        st.subheader("Alerts for the medical team")
        for a in alerts:
            fn_name, label = LEVEL_UI.get(a["level"], ("info", "INFO"))
            getattr(st, fn_name)(f"**{label} - {a['type']}**  \n{a['message']}")

    st.subheader("Answer")
    st.markdown(state.get("answer", ""))

    violations = state.get("violations", [])
    if violations:
        st.warning("Guardrails triggered: " + ", ".join(sorted(set(violations))))
    if state.get("requires_human_validation"):
        st.info("Content pending validation by a healthcare professional.")

    sources = state.get("sources", [])
    if sources:
        st.subheader("Sources used (explainability)")
        for f in sources:
            if f["type"] == "scientific_evidence":
                with st.expander(f"PMID {f['pmid']} - conclusion: {f['conclusion']}"):
                    st.write(f["excerpt"] + "...")
                    st.markdown(f"[Open in PubMed]({f['url']})")
            else:
                st.caption(f"Medical record of patient {f['patient_id']} "
                           f"(source: {f['origin']})")

    st.subheader("Audit trail")
    events = audit.read_events(trace_id=state["trace_id"])
    st.caption(f"{len(events)} event(s) recorded in this execution")
    st.json([{"timestamp": e["timestamp"], "event": e["event"], **e["detail"]}
             for e in events])
else:
    st.info("Select a patient, write the question and run the flow.")
