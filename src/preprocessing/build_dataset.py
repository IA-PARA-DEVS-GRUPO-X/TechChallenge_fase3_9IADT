"""Preprocessing, anonimizacao e curadoria do PubMedQA.

Pipeline:
  1. Normalizacao Unicode NFKC e remocao de caracteres de controle
  2. Anonimizacao (anonymizer.py)
  3. Curadoria: tamanho minimo/maximo e validade do rotulo
  4. Deduplicacao por SHA-256 do par pergunta+contexto
"""

import hashlib
import json
import re
import unicodedata
from typing import Dict, List

from src.config import (
    MAX_CONTEXT_CHARS,
    MIN_ANSWER_CHARS,
    MIN_CONTEXT_CHARS,
    PROCESSED_FILE,
    PUBMEDQA_RAW_FILE,
    STATS_FILE,
    VALID_LABELS,
)
from src.preprocessing.anonymizer import anonymize

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t]+")


def normalize(text: str) -> str:
    """Normalizacao Unicode e limpeza de caracteres invisiveis."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def build_context(item: dict) -> str:
    """Junta os trechos do artigo, prefixando cada um com sua secao."""
    contextos = item.get("CONTEXTS", [])
    secoes = item.get("LABELS", [])

    partes = []
    for i, trecho in enumerate(contextos):
        secao = secoes[i] if i < len(secoes) else "CONTEXT"
        partes.append(f"{secao}: {trecho}")
    return "\n".join(partes)


def run() -> None:
    with open(PUBMEDQA_RAW_FILE, encoding="utf-8") as f:
        bruto = json.load(f)

    total_bruto = len(bruto)
    pii_total: Dict[str, int] = {}
    descartes: Dict[str, int] = {}
    vistos = set()
    final: List[dict] = []

    def descartar(motivo: str) -> None:
        descartes[motivo] = descartes.get(motivo, 0) + 1

    for pmid, item in bruto.items():
        pergunta = normalize(item.get("QUESTION", ""))
        contexto = normalize(build_context(item))
        conclusao = normalize(item.get("LONG_ANSWER", ""))
        rotulo = item.get("final_decision", "").strip().lower()

        # --- anonimizacao (conta os achados por categoria) ---
        for campo in (pergunta, contexto, conclusao):
            _, achados = anonymize(campo)
            for k, v in achados.items():
                pii_total[k] = pii_total.get(k, 0) + v

        pergunta, _ = anonymize(pergunta)
        contexto, _ = anonymize(contexto)
        conclusao, _ = anonymize(conclusao)

        # --- curadoria ---
        if rotulo not in VALID_LABELS:
            descartar("rotulo_invalido")
            continue
        if len(conclusao) < MIN_ANSWER_CHARS:
            descartar("resposta_curta")
            continue
        if len(contexto) < MIN_CONTEXT_CHARS:
            descartar("contexto_curto")
            continue
        if len(contexto) > MAX_CONTEXT_CHARS:
            contexto = contexto[:MAX_CONTEXT_CHARS]

        # --- deduplicacao ---
        chave = hashlib.sha256(
            (pergunta + contexto).encode("utf-8")
        ).hexdigest()
        if chave in vistos:
            descartar("duplicado")
            continue
        vistos.add(chave)

        final.append({
            "id": pmid,
            "question": pergunta,
            "context": contexto,
            "long_answer": conclusao,
            "label": rotulo,
        })

    # --- estatisticas ---
    distribuicao: Dict[str, int] = {}
    for ex in final:
        distribuicao[ex["label"]] = distribuicao.get(ex["label"], 0) + 1

    media_chars = (
        sum(len(ex["context"]) for ex in final) / len(final) if final else 0
    )

    stats = {
        "arquivo_origem": PUBMEDQA_RAW_FILE.name,
        "curadoria": {
            "total_bruto": total_bruto,
            "total_final": len(final),
            "descartes": descartes,
        },
        "distribuicao_labels": distribuicao,
        "pii_substituida": pii_total,
        "media_chars_contexto": round(media_chars, 1),
    }

    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[ok] {len(final)} exemplos curados -> {PROCESSED_FILE}")
    print(f"[ok] estatisticas -> {STATS_FILE}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
