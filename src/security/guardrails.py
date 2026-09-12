"""Guardrails do assistente medico

Ponto central: as regras que existem no prompt do sistema sao apenas uma
recomendacao ao modelo e podem ser contornadas (prompt injection,
alucinacao, deriva do fine-tuning). Este modulo aplica a validacao de
forma **programatica**, fora do modelo, em duas fronteiras:

  - entrada  -> `check_input`: remove PII e bloqueia pedidos fora do escopo
  - saida    -> `check_output`: redige posologia, exige citacao de fonte e
                anexa o aviso de validacao humana obrigatoria

Toda decisao e devolvida de forma estruturada para que o fluxo LangGraph
possa ramificar e a trilha de auditoria possa registrar o motivo.
"""

import re
from dataclasses import dataclass, field
from typing import List

from src.preprocessing.anonymizer import anonymize

# --------------------------------------------------------------------
# Mensagens fixas
# --------------------------------------------------------------------
DISCLAIMER = (
    "[AVISO] Conteudo de apoio a decisao clinica. Nao constitui prescricao. "
    "Toda conduta exige validacao de um profissional de saude responsavel."
)

REDACTION = "[POSOLOGIA REMOVIDA - requer prescricao medica]"

OUT_OF_SCOPE_MSG = (
    "Pergunta fora do escopo do assistente. Este sistema responde apenas a "
    "duvidas clinicas de apoio a decisao, com base na evidencia indexada e "
    "no prontuario do paciente."
)

# --------------------------------------------------------------------
# Padroes de deteccao
# --------------------------------------------------------------------
# Dose explicita: valor numerico seguido de unidade farmacologica.
_DOSE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|mcg|[µu]g|g|ml|mL|L|UI|IU|mEq|mmol|"
    r"comprimidos?|capsulas?|gotas?|ampolas?|frascos?)\b"
    r"(?:\s?/\s?(?:kg|m2|dia|h))?",
    re.IGNORECASE,
)

# Frequencia posologica: 8/8h, 12/12 horas, 1x/dia, de 6 em 6 horas, BID/TID.
_FREQUENCY = re.compile(
    r"\b(?:\d{1,2}\s?/\s?\d{1,2}\s?h(?:oras?)?"
    r"|\d{1,2}\s?x\s?/?\s?(?:ao\s+)?dia"
    r"|de\s+\d{1,2}\s+em\s+\d{1,2}\s+horas"
    r"|a\s+cada\s+\d{1,2}\s+horas"
    r"|BID|TID|QID|QD|SOS|ACM)\b",
    re.IGNORECASE,
)

# Verbo prescritivo dirigido ao paciente (imperativo ou 1a pessoa).
_PRESCRIPTIVE = re.compile(
    r"\b(?:prescrev[oa]|receit[oa]|administre|administrar[- ]se[- ]a|"
    r"inicie|iniciar\s+imediatamente|suspend[ao]|aumente\s+a\s+dose|"
    r"reduza\s+a\s+dose|tome|prescribe|administer)\b",
    re.IGNORECASE,
)

# Pedido explicito de prescricao na pergunta do medico.
_PRESCRIPTION_REQUEST = re.compile(
    r"\b(?:qual\s+(?:a\s+)?dose|que\s+dose|posologia|quantos?\s+mg|"
    r"prescreva|receite|me\s+de\s+a\s+receita|what\s+dose|how\s+many\s+mg)\b",
    re.IGNORECASE,
)

# Assuntos claramente fora do dominio clinico.
_OUT_OF_SCOPE = re.compile(
    r"\b(?:senha|cartao\s+de\s+credito|piada|futebol|invista|"
    r"ignore\s+(?:as\s+)?instruc|ignore\s+(?:all\s+)?previous|"
    r"esqueca\s+as\s+regras|aja\s+como\s+se\s+nao|disregard\s+previous)\b",
    re.IGNORECASE,
)

# Citacao de fonte no formato usado pelo pipeline.
_CITATION = re.compile(r"\bPMID\s*:?\s*\d+|\b\d{7,8}\b")


@dataclass
class GuardrailResult:
    """Resultado estruturado de uma verificacao.

    - `allowed`: False interrompe o fluxo (usado na entrada)
    - `text`: conteudo ja sanitizado/redigido, pronto para uso
    - `violations`: codigos das regras acionadas, para auditoria
    - `requires_human_validation`: sempre True na saida clinica
    """

    allowed: bool
    text: str
    violations: List[str] = field(default_factory=list)
    requires_human_validation: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.violations)


# --------------------------------------------------------------- entrada
def check_input(question: str) -> GuardrailResult:
    """Sanitiza a pergunta do medico antes de ela chegar a LLM."""
    if not question or not question.strip():
        return GuardrailResult(
            allowed=False,
            text="",
            violations=["entrada_vazia"],
            notes=["Nenhuma pergunta informada."],
        )

    violations: List[str] = []
    notes: List[str] = []

    if _OUT_OF_SCOPE.search(question):
        return GuardrailResult(
            allowed=False,
            text=OUT_OF_SCOPE_MSG,
            violations=["fora_de_escopo"],
            notes=["Pergunta fora do dominio clinico ou tentativa de "
                   "sobrescrever as instrucoes do sistema."],
        )

    # PII nunca deve chegar ao modelo nem ao log: substituida por
    # placeholders tipados, preservando a semantica clinica.
    sanitized, hits = anonymize(question)
    if hits:
        violations.append("pii_na_entrada")
        notes.append(f"PII removida da pergunta: {hits}")

    if _PRESCRIPTION_REQUEST.search(sanitized):
        # Nao bloqueia: o medico pode perguntar legitimamente. Marca para
        # que a saida seja redigida e o evento fique rastreado.
        violations.append("pedido_de_prescricao")
        notes.append("Pedido de dose/posologia detectado; a resposta sera "
                     "restrita a conduta, sem posologia.")

    return GuardrailResult(
        allowed=True,
        text=sanitized,
        violations=violations,
        notes=notes,
    )


# ----------------------------------------------------------------- saida
def check_output(answer: str, has_evidence: bool = True) -> GuardrailResult:
    """Valida e redige a resposta da LLM antes de exibi-la."""
    if not answer or not answer.strip():
        return GuardrailResult(
            allowed=False,
            text="Nao foi possivel gerar uma resposta fundamentada.",
            violations=["saida_vazia"],
            requires_human_validation=True,
        )

    text = answer
    violations: List[str] = []
    notes: List[str] = []

    text, n_dose = _DOSE.subn(REDACTION, text)
    if n_dose:
        violations.append("dose_na_saida")
        notes.append(f"{n_dose} ocorrencia(s) de dose redigida(s).")

    text, n_freq = _FREQUENCY.subn(REDACTION, text)
    if n_freq:
        violations.append("frequencia_na_saida")
        notes.append(f"{n_freq} ocorrencia(s) de posologia redigida(s).")

    if _PRESCRIPTIVE.search(text):
        violations.append("linguagem_prescritiva")
        notes.append("Linguagem prescritiva detectada; resposta marcada "
                     "como pendente de validacao medica.")

    # Explainability: resposta sem citacao nao e rastreavel ate a fonte.
    if has_evidence and not _CITATION.search(text):
        violations.append("sem_citacao")
        notes.append("Resposta nao cita PMID; confiabilidade reduzida.")

    if DISCLAIMER not in text:
        text = f"{text.strip()}\n\n{DISCLAIMER}"

    return GuardrailResult(
        allowed=True,
        text=text,
        violations=violations,
        requires_human_validation=True,
        notes=notes,
    )
