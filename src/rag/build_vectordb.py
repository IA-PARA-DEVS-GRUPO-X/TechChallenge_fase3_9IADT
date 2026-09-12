"""Indexa o dataset curado no banco vetorial Chroma.

Cada documento guarda o PMID em metadata: e esse identificador que
permite ao assistente citar a fonte de cada afirmacao (explainability).
"""

import json
import shutil
from typing import List

from src.config import (
    PROCESSED_FILE,
    VECTORDB_COLLECTION,
    VECTORDB_DIR,
)
from src.rag.vectorstore import get_embeddings


def load_documents() -> List:
    from langchain_core.documents import Document

    with open(PROCESSED_FILE, encoding="utf-8") as f:
        dados = json.load(f)

    documentos = []
    for item in dados:
        # O texto indexado reune pergunta, conclusao e evidencia: e sobre
        # ele que a busca por significado sera feita.
        conteudo = (
            f"Question: {item['question']}\n"
            f"Conclusion: {item['long_answer']}\n"
            f"Evidence: {item['context']}"
        )
        documentos.append(
            Document(
                page_content=conteudo,
                metadata={
                    "pmid": item["id"],
                    "label": item["label"],
                    "source": f"PubMed PMID {item['id']}",
                    "question": item["question"],
                },
            )
        )
    return documentos


def run() -> None:
    from langchain_chroma import Chroma

    if VECTORDB_DIR.exists():
        shutil.rmtree(VECTORDB_DIR)
        print(f"[info] indice anterior removido de {VECTORDB_DIR}")

    documentos = load_documents()
    print(f"[info] {len(documentos)} documentos carregados.")

    Chroma.from_documents(
        documents=documentos,
        embedding=get_embeddings(),
        collection_name=VECTORDB_COLLECTION,
        persist_directory=str(VECTORDB_DIR),
    )
    print(f"[ok] banco vetorial criado em {VECTORDB_DIR}")


if __name__ == "__main__":
    run()
