"""Trilha de auditoria do assistente medico.

Requisito: "Implementar logging detalhado para
rastreamento e auditoria".

Cada evento e gravado como uma linha JSON (JSONL) em `logs/audit.log`.
O formato append-only facilita a leitura incremental, o parsing por
ferramentas externas e a exibicao na interface.

Nenhum dado sensivel bruto deve entrar aqui: o texto da pergunta ja chega
sanitizado pelo guardrail de entrada, e o paciente e identificado apenas
pelo id sintetico (ex.: PAC-0001).
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config import AUDIT_LOG

EVENT_INPUT = "input_received"
EVENT_RETRIEVAL = "evidence_retrieved"
EVENT_PATIENT = "record_consulted"
EVENT_LLM = "answer_generated"
EVENT_GUARDRAIL = "guardrail_applied"
EVENT_ALERT = "team_alert"
EVENT_NODE = "node_executed"
EVENT_ERROR = "error"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_trace_id() -> str:
    """Identificador que correlaciona todos os eventos de uma execucao."""
    return uuid.uuid4().hex[:12]


def log_event(
    event: str,
    trace_id: str,
    detail: Optional[Dict[str, Any]] = None,
    patient_id: Optional[str] = None,
    log_path=None,
) -> Dict[str, Any]:
    """Grava um evento na trilha e devolve o registro escrito."""
    record = {
        "timestamp": _now(),
        "trace_id": trace_id,
        "event": event,
        "patient_id": patient_id,
        "detail": detail or {},
    }

    path = log_path or AUDIT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def read_events(
    limit: Optional[int] = None,
    trace_id: Optional[str] = None,
    log_path=None,
) -> List[Dict[str, Any]]:
    """Le a trilha, opcionalmente filtrando por execucao."""
    path = log_path or AUDIT_LOG
    if not path.exists():
        return []

    events: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if trace_id and record.get("trace_id") != trace_id:
                continue
            events.append(record)

    return events[-limit:] if limit else events
