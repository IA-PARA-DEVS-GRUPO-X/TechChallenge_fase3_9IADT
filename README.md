# Assistente Médico com LLM Fine-tunada — Tech Challenge Fase 3

Assistente virtual de apoio à decisão clínica construído sobre uma LLM
fine-tunada com QLoRA no dataset **PubMedQA**, orquestrado com **LangChain**
e **LangGraph**, com camada de segurança, auditoria e explainability.

> **Aviso**: sistema acadêmico de apoio à decisão. Não substitui julgamento
> clínico e nunca emite prescrições sem validação humana.

---

## Arquitetura

```
data/raw/          PubMedQA bruto (não versionado, baixado por script)
data/processed/    dataset curado + estatísticas de preprocessing
data/training/     train/val/test em JSONL (formato de chat)
src/config.py      configuração central (paths, modelos, hiperparâmetros)
src/preprocessing/ download, anonimização, curadoria e split
src/finetuning/    treino QLoRA e avaliação base vs fine-tuned
src/rag/           banco vetorial Chroma e retriever
src/db/            base estruturada de pacientes (SQLite, dados sintéticos)
src/app/           pipeline LangChain e interface Streamlit
src/workflow/      fluxo LangGraph
src/security/      guardrails e trilha de auditoria
docs/              relatório técnico, diagrama do fluxo e resultados
```

O sistema combina três fontes numa única resposta rastreável:

| Fonte | Papel |
|---|---|
| LLM fine-tunada | Raciocínio clínico no formato veredito + justificativa |
| RAG (Chroma) | Evidência científica com PMID citável |
| SQLite | Prontuário do paciente (dados sintéticos) |

---

## Como executar

O fine-tuning e a inferência exigem GPU. Todo o projeto roda no **Google
Colab** (GPU T4 gratuita).

### 1. Preparação dos dados

```bash
python -m src.preprocessing.download_dataset
python -m src.preprocessing.build_dataset
python -m src.preprocessing.split_dataset
```

Aplica, nesta ordem: normalização Unicode NFKC, remoção de caracteres de
controle, **anonimização** (e-mail, URL, CPF, CNS, telefone, MRN, datas,
nomes com título clínico, idade >= 90 conforme HIPAA Safe Harbor),
curadoria por tamanho e validade do rótulo, deduplicação por SHA-256 e
split estratificado com seed fixa.

**Resultado:** 1000 exemplos (yes 552 / no 338 / maybe 110), divididos em
799 treino / 99 validação / 102 teste.

### 2. Fine-tuning (QLoRA)

```bash
python -m src.finetuning.train
python -m src.finetuning.train --no-balance --epochs 3   # experimento 1
```

| Parâmetro | Valor |
|---|---|
| Modelo base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Quantização | NF4 4-bit com double quantization |
| LoRA | r=16, alpha=32, dropout=0.05, atenção + MLP |
| Parâmetros treinados | 18.464.768 de 1.562.179.072 (1,18%) |
| Batch efetivo | 16 (batch 2 x grad. accumulation 8) |
| Épocas | 1 (o experimento 1 mostrou overfitting a partir da 2a) |
| Balanceamento | classes igualadas apenas no treino |

### 3. Avaliação

```bash
python -m src.finetuning.evaluate --batch-size 4
```

**F1 macro é a métrica principal**, não a acurácia: 55% do dataset é `yes`,
então um modelo que sempre responde "yes" atinge 55% de acurácia com F1
macro de apenas ~0,24.

### 4. Bases do assistente

```bash
python -m src.db.patients        # 20 pacientes sintéticos (seed fixa)
python -m src.rag.build_vectordb # indexa 1000 artigos no Chroma
```

### 5. Assistente e fluxo

```bash
streamlit run src/app/ui.py --server.enableCORS false --server.enableXsrfProtection false
```

```python
from src.workflow.graph import run_flow, export_diagram

estado = run_flow("Is early diagnosis associated with better outcomes in sepsis?",
                  patient_id="PAC-0009")
print(estado["path"])      # caminho percorrido no grafo
print(estado["alerts"])    # alertas para a equipe médica
export_diagram()           # docs/fluxo_langgraph.mmd
```

---

## Fluxo de decisão (LangGraph)

```
triagem ---(bloqueado)-------------------------------> FIM
   |
   +--(ok)--> prontuário --> verificar_exames --(pendentes)--> alerta_exames --+
                                    |                                          |
                                    +--(nenhum)--------------------------------+
                                                                               |
                                                                               v
              buscar_evidência --> sugerir_conduta --> guardrail --> alertar_equipe --> FIM
```

| Nó | Função |
|---|---|
| `triagem` | Sanitiza PII, barra prompt injection e temas fora de escopo |
| `carregar_prontuario` | Consulta a base estruturada do paciente |
| `verificar_exames` | Levanta exames pendentes; define o desvio condicional |
| `alerta_exames` | Sinaliza conduta sugerida com informação incompleta |
| `buscar_evidencia` | Recupera trechos científicos no Chroma |
| `sugerir_conduta` | Gera a sugestão com a LLM customizada |
| `guardrail` | Valida a saída fora do modelo |
| `alertar_equipe` | Consolida alertas por severidade |

---

## Segurança e conformidade

- Anonimização aplicada antes de qualquer treino ou indexação.
- Guardrails **programáticos** (`src/security/guardrails.py`), fora do
  modelo: o prompt sozinho é contornável. Na entrada, remove PII e barra
  tentativas de sobrescrever instruções; na saída, redige dose e posologia,
  detecta linguagem prescritiva e exige citação de PMID.
- Toda interação registrada em `logs/audit.log` (JSONL), com `trace_id`
  correlacionando os eventos de uma mesma execução.
- Respostas sempre acompanhadas da fonte (PMID) e do aviso de validação
  humana obrigatória.

---

## Documentação

- `docs/relatorio_tecnico.md` — relatório técnico completo
- `docs/evaluation_results.json` — métricas do experimento final
- `docs/evaluation_results_v1.json` — métricas do experimento 1
- `docs/loss_curve.png` — curva de aprendizado
- `docs/fluxo_langgraph.mmd` — diagrama do fluxo
