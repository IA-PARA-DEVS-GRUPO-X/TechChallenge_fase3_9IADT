"""Divide o dataset curado em treino/validacao/teste.

A divisao e estratificada por rotulo e usa seed fixa, garantindo que
qualquer pessoa que rode o projeto obtenha exatamente os mesmos grupos.

Saida em JSONL no formato de chat esperado pelo modelo.
"""

import json
import random
from typing import Dict, List

from src.config import (
    INSTRUCTION_TEMPLATE,
    PROCESSED_FILE,
    RANDOM_SEED,
    RATIONALE_PREFIX,
    SPLIT_RATIOS,
    SYSTEM_PROMPT,
    TEST_FILE,
    TRAIN_FILE,
    VAL_FILE,
    VERDICT_PREFIX,
)


def to_chat(exemplo: dict) -> dict:
    """Converte um exemplo curado para o formato de conversa."""
    pergunta_formatada = INSTRUCTION_TEMPLATE.format(
        question=exemplo["question"],
        context=exemplo["context"],
    )
    resposta = (
        f"{VERDICT_PREFIX} {exemplo['label']}\n"
        f"{RATIONALE_PREFIX} {exemplo['long_answer']}"
    )

    return {
        "id": exemplo["id"],
        "label": exemplo["label"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pergunta_formatada},
            {"role": "assistant", "content": resposta},
        ],
    }


def split_estratificado(exemplos: List[dict]) -> Dict[str, List[dict]]:
    """Divide mantendo a proporcao de rotulos em cada grupo."""
    rng = random.Random(RANDOM_SEED)

    # Agrupa por rotulo
    por_rotulo: Dict[str, List[dict]] = {}
    for ex in exemplos:
        por_rotulo.setdefault(ex["label"], []).append(ex)

    grupos: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    for rotulo, itens in sorted(por_rotulo.items()):
        itens = itens[:]           # copia, para nao alterar a lista original
        rng.shuffle(itens)

        n = len(itens)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        grupos["train"].extend(itens[:n_train])
        grupos["val"].extend(itens[n_train:n_train + n_val])
        grupos["test"].extend(itens[n_train + n_val:])

    # Embaralha cada grupo para nao ficar ordenado por rotulo
    for g in grupos.values():
        rng.shuffle(g)

    return grupos


def salvar_jsonl(exemplos: List[dict], caminho) -> None:
    """Grava um objeto JSON por linha (formato JSONL)."""
    with open(caminho, "w", encoding="utf-8") as f:
        for ex in exemplos:
            f.write(json.dumps(to_chat(ex), ensure_ascii=False) + "\n")


def run() -> None:
    with open(PROCESSED_FILE, encoding="utf-8") as f:
        exemplos = json.load(f)

    grupos = split_estratificado(exemplos)

    for nome, caminho in (("train", TRAIN_FILE), ("val", VAL_FILE),
                          ("test", TEST_FILE)):
        salvar_jsonl(grupos[nome], caminho)

        distribuicao: Dict[str, int] = {}
        for ex in grupos[nome]:
            distribuicao[ex["label"]] = distribuicao.get(ex["label"], 0) + 1

        proporcoes = " | ".join(
            f"{r} {n} ({n/len(grupos[nome]):.0%})"
            for r, n in sorted(distribuicao.items())
        )
        print(f"{nome:6s} {len(grupos[nome]):4d} exemplos  ->  {proporcoes}")

    print(f"\n[ok] arquivos gravados em {TRAIN_FILE.parent}")


if __name__ == "__main__":
    run()
