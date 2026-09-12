"""Configuracao central do projeto: paths, modelos e hiperparametros."""

from pathlib import Path

# --------------------------------------------------------------------
# Diretorios
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"

MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"

for _d in (RAW_DIR, PROCESSED_DIR, TRAINING_DIR, MODELS_DIR, LOGS_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Dataset PubMedQA
# --------------------------------------------------------------------
PUBMEDQA_RAW_FILE = RAW_DIR / "ori_pqal.json"
PUBMEDQA_RAW_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/"
    "master/data/ori_pqal.json"
)

PROCESSED_FILE = PROCESSED_DIR / "pubmedqa_processed.json"
STATS_FILE = PROCESSED_DIR / "preprocessing_stats.json"

TRAIN_FILE = TRAINING_DIR / "train.jsonl"
VAL_FILE = TRAINING_DIR / "val.jsonl"
TEST_FILE = TRAINING_DIR / "test.jsonl"

# --------------------------------------------------------------------
# Curadoria e divisao dos dados
# --------------------------------------------------------------------
RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

MIN_ANSWER_CHARS = 20
MIN_CONTEXT_CHARS = 100
MAX_CONTEXT_CHARS = 6000
VALID_LABELS = {"yes", "no", "maybe"}

# Balanceamento do conjunto de treino (experimento 2).
# O dataset original tem 55% yes / 34% no / 11% maybe, o que leva o modelo
# a ignorar as classes minoritarias. Aplicado SOMENTE no treino: validacao
# e teste mantem a distribuicao real.
BALANCE_TRAIN = True

# --------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------
BASE_LLM = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = MODELS_DIR / "qwen2.5-1.5b-pubmedqa-lora"

# --------------------------------------------------------------------
# Fine-tuning (QLoRA)
# --------------------------------------------------------------------
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

TRAINING_ARGS = {
    # O experimento 1 mostrou overfitting ja a partir da 1a epoca:
    # eval_loss 1.580 -> 1.604 -> 1.632. Uma epoca e o suficiente.
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 5,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "optim": "paged_adamw_8bit",
    "max_grad_norm": 0.3,
    "seed": RANDOM_SEED,
    "report_to": "none",
}

MAX_SEQ_LENGTH = 2048

# --------------------------------------------------------------------
# Geracao / avaliacao
# --------------------------------------------------------------------
GENERATION_ARGS = {
    "max_new_tokens": 160,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "repetition_penalty": 1.05,
}

EVAL_RESULTS_FILE = DOCS_DIR / "evaluation_results.json"

# --------------------------------------------------------------------
# Prompt usado no treino e na inferencia
# --------------------------------------------------------------------
# Em ingles, alinhado ao idioma do PubMedQA. No experimento 1 a instrucao
# estava em portugues e as respostas de referencia em ingles: o modelo
# respondia em portugues e o ROUGE-L media diferenca de idioma, nao
# qualidade da justificativa.
SYSTEM_PROMPT = (
    "You are a clinical decision support assistant. "
    "Answer strictly based on the provided evidence. "
    "Never prescribe drugs or dosages directly: every suggestion "
    "requires validation by a licensed healthcare professional."
)

INSTRUCTION_TEMPLATE = (
    "Clinical question: {question}\n\n"
    "Scientific evidence:\n{context}\n\n"
    "Answer with a verdict (yes/no/maybe) and a rationale."
)

VERDICT_PREFIX = "Verdict:"
RATIONALE_PREFIX = "Rationale:"

# --------------------------------------------------------------------
# Banco vetorial (RAG)
# --------------------------------------------------------------------
# all-MiniLM-L6-v2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTORDB_DIR = BASE_DIR / "chromadb"
VECTORDB_COLLECTION = "pubmedqa"

# Quantos trechos de evidencia enviar ao modelo por pergunta.
RETRIEVER_TOP_K = 3

# --------------------------------------------------------------------
# Base estruturada de pacientes (dados sinteticos)
# --------------------------------------------------------------------
PATIENTS_DB = DATA_DIR / "patients.db"
PATIENTS_SEED_FILE = DATA_DIR / "patients_seed.json"

# --------------------------------------------------------------------
# Base estruturada de pacientes (dados sinteticos)
# --------------------------------------------------------------------
PATIENTS_DB = DATA_DIR / "patients.db"
PATIENTS_SEED_FILE = DATA_DIR / "patients_seed.json"

# --------------------------------------------------------------------
# Auditoria
# --------------------------------------------------------------------
AUDIT_LOG = LOGS_DIR / "audit.log"
