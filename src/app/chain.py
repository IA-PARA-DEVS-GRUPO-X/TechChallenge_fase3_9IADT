"""Pipeline LangChain do assistente medico.

Combina tres fontes de informacao numa unica resposta rastreavel:
  1. LLM customizada (fine-tunada em PubMedQA)
  2. Evidencia cientifica recuperada do banco vetorial (RAG)
  3. Prontuario estruturado do paciente (SQLite)

A resposta sempre carrega as fontes utilizadas (PMIDs e id do paciente),
atendendo ao requisito de explainability.
"""

from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import RETRIEVER_TOP_K, SYSTEM_PROMPT
from src.db.patients import format_patient, get_patient
from src.security import audit, guardrails

# O formato de saida replica o usado no fine-tuning (Verdict/Rationale).
# Com formato livre o modelo ignora as regras, inclusive a citacao de PMID.
ASSISTANT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "Clinical question: {question}\n\n"
     "Patient record:\n{patient_context}\n\n"
     "Scientific evidence:\n{evidence}\n\n"
     "Answer in this exact format:\n"
     "Verdict: <yes/no/maybe>\n"
     "Rationale: <one sentence, based only on the evidence above>\n"
     "Sources: <PMID numbers used>\n"
     "Patient: <one sentence relating the answer to this patient>"),
])

NO_PATIENT = "No patient specified; answer in general terms."


def format_evidence(docs: List) -> str:
    """Prefixa cada trecho com o PMID, para que o modelo possa citar."""
    if not docs:
        return "No relevant evidence found in the database."

    blocos = []
    for doc in docs:
        pmid = doc.metadata.get("pmid", "unknown")
        blocos.append(f"PMID {pmid}\n{doc.page_content}")
    return "\n\n".join(blocos)


def build_sources(docs: List, patient_id: Optional[str]) -> List[Dict]:
    """Monta a lista de fontes devolvida junto da resposta."""
    sources = [
        {
            "type": "scientific_evidence",
            "pmid": doc.metadata.get("pmid"),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{doc.metadata.get('pmid')}/",
            "conclusion": doc.metadata.get("label"),
            "excerpt": doc.page_content[:300],
        }
        for doc in docs
    ]
    if patient_id:
        sources.append({
            "type": "medical_record",
            "patient_id": patient_id,
            "origin": "data/patients.db",
        })
    return sources


def ask(
    question: str,
    patient_id: Optional[str] = None,
    k: int = RETRIEVER_TOP_K,
    trace_id: Optional[str] = None,
) -> Dict:
    """Executa o pipeline completo e devolve resposta + fontes."""
    from src.app.llm import get_llm
    from src.rag.vectorstore import get_retriever

    trace_id = trace_id or audit.new_trace_id()

    gate = guardrails.check_input(question)
    audit.log_event(
        audit.EVENT_INPUT, trace_id,
        {"question": gate.text, "violations": gate.violations},
        patient_id=patient_id,
    )
    if not gate.allowed:
        return {"answer": gate.text, "sources": [], "patient_id": patient_id,
                "trace_id": trace_id, "guardrail": gate}
    question = gate.text

    patient_context = NO_PATIENT
    if patient_id:
        patient = get_patient(patient_id)
        if patient is None:
            audit.log_event(audit.EVENT_ERROR, trace_id,
                            {"reason": "patient_not_found"},
                            patient_id=patient_id)
            return {"answer": f"Patient {patient_id} not found in the database.",
                    "sources": [], "patient_id": patient_id,
                    "trace_id": trace_id, "guardrail": None}
        patient_context = format_patient(patient)
        audit.log_event(
            audit.EVENT_PATIENT, trace_id,
            {"diagnosis": patient["diagnostico"],
             "pending_exams": sum(1 for e in patient["exames"]
                                  if e["status"] == "pendente")},
            patient_id=patient_id,
        )

    docs = get_retriever(k).invoke(question)
    audit.log_event(
        audit.EVENT_RETRIEVAL, trace_id,
        {"k": k, "pmids": [d.metadata.get("pmid") for d in docs]},
        patient_id=patient_id,
    )

    chain = ASSISTANT_PROMPT | get_llm() | StrOutputParser()
    raw_answer = chain.invoke({
        "question": question,
        "patient_context": patient_context,
        "evidence": format_evidence(docs),
    })
    audit.log_event(audit.EVENT_LLM, trace_id,
                    {"characters": len(raw_answer)}, patient_id=patient_id)

    checked = guardrails.check_output(raw_answer.strip(),
                                      has_evidence=bool(docs))
    audit.log_event(
        audit.EVENT_GUARDRAIL, trace_id,
        {"violations": checked.violations,
         "requires_human_validation": checked.requires_human_validation},
        patient_id=patient_id,
    )

    return {
        "answer": checked.text,
        "sources": build_sources(docs, patient_id),
        "patient_id": patient_id,
        "trace_id": trace_id,
        "guardrail": checked,
    }
