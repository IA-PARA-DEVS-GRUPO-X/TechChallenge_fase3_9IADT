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

# O formato de saida replica o usado no fine-tuning (Verdict/Rationale).
# Testes mostraram que pedir um formato livre faz o modelo ignorar as
# regras, inclusive a citacao de PMID: modelos pequenos aderem melhor ao
# formato que viram no treino.
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
    """Numera os trechos recuperados e prefixa o PMID para citacao."""
    if not docs:
        return "No relevant evidence found in the database."

    # Sem numeracao ("[1]", "[2]"): o modelo tende a citar o primeiro
    # identificador que ve, e o indice da lista nao e uma fonte rastreavel.
    blocos = []
    for doc in docs:
        pmid = doc.metadata.get("pmid", "unknown")
        blocos.append(f"PMID {pmid}\n{doc.page_content}")
    return "\n\n".join(blocos)


def build_sources(docs: List, patient_id: Optional[str]) -> List[Dict]:
    """Monta a lista de fontes devolvida junto da resposta."""
    sources = [
        {
            "tipo": "evidencia_cientifica",
            "pmid": doc.metadata.get("pmid"),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{doc.metadata.get('pmid')}/",
            "conclusao": doc.metadata.get("label"),
            "trecho": doc.page_content[:300],
        }
        for doc in docs
    ]
    if patient_id:
        sources.append({
            "tipo": "prontuario",
            "paciente_id": patient_id,
            "origem": "data/patients.db",
        })
    return sources


def ask(
    question: str,
    patient_id: Optional[str] = None,
    k: int = RETRIEVER_TOP_K,
) -> Dict:
    """Executa o pipeline completo e devolve resposta + fontes."""
    from src.app.llm import get_llm
    from src.rag.vectorstore import get_retriever

    patient_context = NO_PATIENT
    if patient_id:
        patient = get_patient(patient_id)
        if patient is None:
            return {
                "answer": f"Paciente {patient_id} nao encontrado na base.",
                "sources": [],
                "patient_id": patient_id,
            }
        patient_context = format_patient(patient)

    docs = get_retriever(k).invoke(question)

    chain = ASSISTANT_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({
        "question": question,
        "patient_context": patient_context,
        "evidence": format_evidence(docs),
    })

    return {
        "answer": answer.strip(),
        "sources": build_sources(docs, patient_id),
        "patient_id": patient_id,
    }
