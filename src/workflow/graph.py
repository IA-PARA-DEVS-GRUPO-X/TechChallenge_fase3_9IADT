"""Fluxo de decisao automatizado do assistente medico (LangGraph).

Requisito: "organizar fluxos de decisao automatizados e seguros, onde, 
ao receber informacoes sobre um paciente, o sistema possa acionar diferentes 
etapas, como verificar exames pendentes,sugerir tratamentos e emitir alertas 
para a equipe medica".

Nomes de nos e mensagens em ingles, alinhados ao idioma do modelo e da
interface.
"""

from typing import Dict, List, Optional

from typing_extensions import TypedDict

from src.app.chain import (
    ASSISTANT_PROMPT,
    NO_PATIENT,
    build_sources,
    format_evidence,
)
from src.config import RETRIEVER_TOP_K
from src.db.patients import format_patient, get_patient, get_pending_exams
from src.security import audit, guardrails


class AssistantState(TypedDict, total=False):
    """Estado compartilhado entre os nos do grafo."""

    question: str
    patient_id: Optional[str]
    k: int
    trace_id: str

    sanitized_question: str
    patient_context: str
    pending_exams: List[Dict]
    docs: List
    draft_answer: str

    answer: str
    sources: List[Dict]
    alerts: List[Dict]
    violations: List[str]
    requires_human_validation: bool
    blocked: bool
    path: List[str]


def _trace(state: AssistantState, node: str) -> AssistantState:
    """Registra a passagem pelo no, tanto no log quanto no estado."""
    audit.log_event(
        audit.EVENT_NODE, state["trace_id"], {"node": node},
        patient_id=state.get("patient_id"),
    )
    return {"path": state.get("path", []) + [node]}


# ------------------------------------------------------------------ nos
def triage(state: AssistantState) -> AssistantState:
    """Guardrail de entrada: sanitiza PII e barra pedidos fora de escopo."""
    update = _trace(state, "triage")
    gate = guardrails.check_input(state["question"])

    audit.log_event(
        audit.EVENT_INPUT, state["trace_id"],
        {"question": gate.text, "violations": gate.violations},
        patient_id=state.get("patient_id"),
    )

    update.update({
        "sanitized_question": gate.text,
        "violations": list(gate.violations),
        "blocked": not gate.allowed,
    })
    if not gate.allowed:
        update.update({"answer": gate.text, "sources": [], "alerts": [],
                       "requires_human_validation": False})
    return update


def load_record(state: AssistantState) -> AssistantState:
    """Consulta a base estruturada (SQLite) do paciente, se houver."""
    update = _trace(state, "load_record")
    patient_id = state.get("patient_id")

    if not patient_id:
        update["patient_context"] = NO_PATIENT
        return update

    patient = get_patient(patient_id)
    if patient is None:
        audit.log_event(audit.EVENT_ERROR, state["trace_id"],
                        {"reason": "patient_not_found"},
                        patient_id=patient_id)
        update.update({
            "blocked": True,
            "answer": f"Patient {patient_id} not found in the database.",
            "sources": [], "alerts": [], "patient_context": NO_PATIENT,
        })
        return update

    audit.log_event(
        audit.EVENT_PATIENT, state["trace_id"],
        {"diagnosis": patient["diagnostico"], "allergies": patient["alergias"]},
        patient_id=patient_id,
    )
    update["patient_context"] = format_patient(patient)
    return update


def check_exams(state: AssistantState) -> AssistantState:
    """Levanta os exames pendentes: define o desvio condicional do fluxo."""
    update = _trace(state, "check_exams")
    patient_id = state.get("patient_id")
    update["pending_exams"] = get_pending_exams(patient_id) if patient_id else []
    return update


def exams_alert(state: AssistantState) -> AssistantState:
    """Emite alerta: ha exames que podem mudar a conduta sugerida."""
    update = _trace(state, "exams_alert")
    names = [e["nome"] for e in state.get("pending_exams", [])]

    alert = {
        "level": "warning",
        "type": "pending_exams",
        "message": ("Advice produced with incomplete information: "
                    f"{len(names)} pending exam(s) - {', '.join(names)}."),
    }
    audit.log_event(audit.EVENT_ALERT, state["trace_id"], alert,
                    patient_id=state.get("patient_id"))
    update["alerts"] = state.get("alerts", []) + [alert]
    return update


def retrieve_evidence(state: AssistantState) -> AssistantState:
    """Recupera a evidencia cientifica no banco vetorial (RAG)."""
    from src.rag.vectorstore import get_retriever

    update = _trace(state, "retrieve_evidence")
    k = state.get("k", RETRIEVER_TOP_K)
    docs = get_retriever(k).invoke(state["sanitized_question"])

    audit.log_event(
        audit.EVENT_RETRIEVAL, state["trace_id"],
        {"k": k, "pmids": [d.metadata.get("pmid") for d in docs]},
        patient_id=state.get("patient_id"),
    )
    update["docs"] = docs
    return update


def suggest_approach(state: AssistantState) -> AssistantState:
    """Gera a sugestao com a LLM customizada, ancorada em evidencia."""
    from langchain_core.output_parsers import StrOutputParser

    from src.app.llm import get_llm

    update = _trace(state, "suggest_approach")
    docs = state.get("docs", [])

    chain = ASSISTANT_PROMPT | get_llm() | StrOutputParser()
    draft = chain.invoke({
        "question": state["sanitized_question"],
        "patient_context": state.get("patient_context", NO_PATIENT),
        "evidence": format_evidence(docs),
    })

    audit.log_event(audit.EVENT_LLM, state["trace_id"],
                    {"characters": len(draft)},
                    patient_id=state.get("patient_id"))
    update["draft_answer"] = draft.strip()
    return update


def guardrail(state: AssistantState) -> AssistantState:
    """Valida a sugestao fora do modelo antes de qualquer exibicao."""
    update = _trace(state, "guardrail")

    checked = guardrails.check_output(
        state.get("draft_answer", ""), has_evidence=bool(state.get("docs"))
    )
    audit.log_event(
        audit.EVENT_GUARDRAIL, state["trace_id"],
        {"violations": checked.violations,
         "requires_human_validation": checked.requires_human_validation},
        patient_id=state.get("patient_id"),
    )

    update.update({
        "answer": checked.text,
        "sources": build_sources(state.get("docs", []), state.get("patient_id")),
        "violations": state.get("violations", []) + checked.violations,
        "requires_human_validation": checked.requires_human_validation,
    })
    return update


def alert_team(state: AssistantState) -> AssistantState:
    """Consolida os alertas destinados a equipe medica."""
    update = _trace(state, "alert_team")
    alerts = list(state.get("alerts", []))
    answer = state.get("answer", "")
    patient_id = state.get("patient_id")
    previous = len(alerts)

    # Alerta critico: so e possivel cruzando a saida da LLM com a base
    # estruturada; nenhum prompt garantiria esta verificacao.
    if patient_id:
        patient = get_patient(patient_id)
        for substance in (patient or {}).get("alergias", []):
            if substance.lower() in answer.lower():
                alerts.append({
                    "level": "critical", "type": "allergy",
                    "message": (f"The answer mentions '{substance}', a "
                                f"substance to which patient {patient_id} "
                                "has a registered allergy."),
                })

    violations = state.get("violations", [])
    if "dose_in_output" in violations or "frequency_in_output" in violations:
        alerts.append({
            "level": "critical", "type": "prescription_attempt",
            "message": ("The model produced dosing information; it was "
                        "automatically redacted and requires a prescription "
                        "from a licensed professional."),
        })

    if "missing_citation" in violations:
        alerts.append({
            "level": "warning", "type": "missing_source",
            "message": ("Answer without a PMID citation: the source of the "
                        "claim could not be traced."),
        })

    alerts.append({
        "level": "info", "type": "human_validation",
        "message": ("Advice pending validation by the responsible healthcare "
                    "professional."),
    })

    for a in alerts[previous:]:
        audit.log_event(audit.EVENT_ALERT, state["trace_id"], a,
                        patient_id=patient_id)

    update["alerts"] = alerts
    return update


# --------------------------------------------------------- roteamento
def route_triage(state: AssistantState) -> str:
    return "blocked" if state.get("blocked") else "ok"


def route_record(state: AssistantState) -> str:
    return "blocked" if state.get("blocked") else "ok"


def route_exams(state: AssistantState) -> str:
    return "pending" if state.get("pending_exams") else "none"


# ------------------------------------------------------------- grafo
def build_graph():
    """Monta e compila o grafo do fluxo clinico."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(AssistantState)

    g.add_node("triage", triage)
    g.add_node("load_record", load_record)
    g.add_node("check_exams", check_exams)
    g.add_node("exams_alert", exams_alert)
    g.add_node("retrieve_evidence", retrieve_evidence)
    g.add_node("suggest_approach", suggest_approach)
    g.add_node("guardrail", guardrail)
    g.add_node("alert_team", alert_team)

    g.add_edge(START, "triage")
    g.add_conditional_edges("triage", route_triage,
                            {"ok": "load_record", "blocked": END})
    g.add_conditional_edges("load_record", route_record,
                            {"ok": "check_exams", "blocked": END})
    g.add_conditional_edges("check_exams", route_exams,
                            {"pending": "exams_alert",
                             "none": "retrieve_evidence"})
    g.add_edge("exams_alert", "retrieve_evidence")
    g.add_edge("retrieve_evidence", "suggest_approach")
    g.add_edge("suggest_approach", "guardrail")
    g.add_edge("guardrail", "alert_team")
    g.add_edge("alert_team", END)

    return g.compile()


_GRAPH = None


def get_graph():
    """Compila o grafo uma unica vez por processo."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_flow(
    question: str,
    patient_id: Optional[str] = None,
    k: int = RETRIEVER_TOP_K,
    trace_id: Optional[str] = None,
) -> AssistantState:
    """Executa o atendimento completo e devolve o estado final."""
    state: AssistantState = {
        "question": question,
        "patient_id": patient_id,
        "k": k,
        "trace_id": trace_id or audit.new_trace_id(),
        "alerts": [], "violations": [], "path": [], "blocked": False,
    }
    return get_graph().invoke(state)


def export_diagram(path=None) -> str:
    """Exporta o diagrama do fluxo em Mermaid (usado no relatorio)."""
    from src.config import DOCS_DIR

    path = path or DOCS_DIR / "fluxo_langgraph.mmd"
    mermaid = get_graph().get_graph().draw_mermaid()
    path.write_text(mermaid, encoding="utf-8")
    print(f"[ok] diagrama -> {path}")
    return mermaid
