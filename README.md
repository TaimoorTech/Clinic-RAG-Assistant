# Clinic Assistant: a RAG question-answering API

A retrieval-augmented generation (RAG) service that answers patient and reception questions
**only from a clinic's own documents**, cites the passages it used, and refuses to guess when
the documents don't contain the answer.

> **Sample data only.** The clinic ("Greenfield Family Clinic") and every document in
> `data/docs/` are fictional. This project gives general information and is **not medical advice**.

Built as a learning project while moving from mobile development into AI engineering.

## What it does

- Answers questions such as *"How late can I cancel an appointment?"* or *"Do you accept cash?"*
  using a grounded prompt, with numbered citations back to the source passages.
- Says "I don't have that information" when the documents don't cover the question
  (for example, services the clinic doesn't list).
- Detects emergency wording (for example "chest pain") **before any AI call** and redirects
  the user to emergency services.
- Refuses to diagnose or give medication advice; every answer carries a disclaimer.

## How it works

```
question
   │
   ▼
1. Emergency check (fixed rules, no LLM)  ──► "call 999", stop
   │
   ▼
2. Retrieve: embed the question, search Chroma for the closest chunks
   │
   ▼
3. Filter: drop clearly unrelated chunks
   │        (nothing left ──► "contact reception", stop, no LLM call)
   ▼
4. Prompt: rules + numbered passages + question
   │
   ▼
5. LLM (OpenAI via LangChain ChatOpenAI) answers only from the passages
   │
   ▼
6. Response: answer + citations [1][2] + sources with scores + disclaimer
```

**Ingestion (run once, re-run when documents change):** markdown files in `data/docs/` are split
by heading, then into overlapping word windows. Each chunk keeps its heading and source file as
metadata, is embedded locally with Chroma's default model (all-MiniLM-L6-v2), and stored in a
persistent Chroma collection.

## Tech stack

Python · FastAPI · ChromaDB · LangChain (`langchain-openai`) · OpenAI chat model · pytest

## Setup

Requires Python 3.12+ and an OpenAI API key with available credits.

```powershell
git clone https://github.com/taimoortech/clinic-rag-assistant.git
cd clinic-rag-assistant
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
# edit .env: set OPENAI_API_KEY and OPENAI_MODEL
python -m app.ingest              # builds the vector index (first run downloads a small model)
uvicorn app.main:app --reload 
```

Open <http://127.0.0.1:8000/docs> for the interactive API page.

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Full pipeline: guardrails, retrieval, grounded LLM answer with citations |
| `POST /search` | Retrieval only (no LLM); useful for inspecting scores while tuning |
| `GET /health` | Liveness check |

Example request:

```json
POST /ask
{"question": "How late can I cancel an appointment?"}
```

## Example questions

Real responses from `POST /ask` (the disclaimer appended to every answer is omitted here for brevity):

| Question | Answer |
|---|---|
| How late can I cancel an appointment? | "You should cancel your appointment at least 24 hours in advance [1]." |
| Do you accept cash? | "No, the clinic does not accept cash. Payments can only be made with a card [1]." |
| Do you take cash, and what are your Saturday hours? | "The clinic does not accept cash; card payments are the only form of payment accepted [2]. On Saturdays, the clinic is open from 9:00 am to 1:00 pm [1]." (a question that needs two different documents) |
| Do you offer physiotherapy? | "I'm sorry, but the passages provided do not mention anything about offering physiotherapy services. I would recommend contacting the reception for further information." |
| I am facing Heart Attack | Emergency redirect to 999, no LLM call (`kind: "emergency"`) |

Full response shape for the first question:

```json
{
  "question": "How late can I cancel an appointment?",
  "answer": "You should cancel your appointment at least 24 hours in advance [1].\n\nThis is general information from the clinic's documents, not medical advice. Please speak to a clinician about your own health.",
  "kind": "answer",
  "sources": [
    {
      "n": 1,
      "source": "appointments_and_cancellation.md",
      "section": "Cancelling or changing an appointment",
      "score": 0.8226
    }
  ]
}
```

`sources` lists the passages that were retrieved and sent to the model, numbered to match the
`[n]` citations in the answer. The citations show which passages the answer actually used.

## Design decisions and what I learned

**A similarity-score threshold could not tell answerable from unanswerable questions.**
My first design discarded retrieved chunks below a fixed cosine-similarity score. Measuring real
questions showed why that fails with a small general-purpose embedding model:

| Question | Top score | Should be answered? |
|---|---|---|
| Do you accept cash? | 0.215 | yes |
| Do you offer physiotherapy? | 0.296 | no |
| Do you have a dental clinic? | 0.371 | no |
| Can I get a COVID vaccine here? | 0.398 | no (only childhood vaccinations are listed) |

Short questions score low against long passages, and unrelated questions can land close to generic
passages such as "appointments" or "vaccinations", so unanswerable questions scored *higher* than
a correct answer. No threshold separates them. The fix was to use the score only to remove obviously
unrelated chunks, and to make the grounded prompt ("answer only from these passages, otherwise say
you don't know") the judge of answerability.

**Deterministic checks run before the LLM.** Emergency detection and the "no relevant passages"
path never call the model, which makes them cheaper, predictable and impossible to talk around.

**Prompt-injection hygiene.** Retrieved passages and the user's question are passed as data, and
the system prompt tells the model to ignore instructions found inside them.

**Testable by design.** The retriever and the LLM are injected into the pipeline, so the unit tests
run with fakes: no API key, no network, no cost.

## Limitations

- **Sample data.** The documents are fictional and small (five files). Results on a real clinic's
  documents would differ.
- **Not medical advice.** The assistant only reports what the documents say. It is not a clinical tool
  and must not be used for diagnosis, triage or treatment decisions.
- **Emergency detection is keyword-based.** It catches common phrases but will miss unusual wording.
  It is a safety net, not a substitute for clinical triage.
- **Retrieval quality is limited by the embedding model.** As shown above, scores are not a reliable
  relevance signal, and compound questions ("cash and Saturday hours") rely on the LLM using all
  retrieved chunks. Hybrid keyword + vector search or a stronger embedding model should help.
- **Not yet evaluated systematically.** So far the pipeline has only been checked by hand on a
  handful of questions. An evaluation set with retrieval and answer-quality metrics is the next step.
- **`sources` shows retrieved passages, not only the ones cited.** For an unanswerable question
  (for example physiotherapy) it still lists the four closest passages even though the answer
  correctly says none of them cover it.
- **No authentication, rate limiting or conversation memory.** Each question is answered
  independently.
- **Answers can still be wrong.** Grounding reduces but does not eliminate model errors.

## Project structure

```
app/
  main.py        FastAPI app: /ask, /search, /health
  rag.py         Pipeline: guardrails -> retrieval -> prompt -> LLM
  guardrails.py  Emergency detection, disclaimer and fallback messages
  retrieval.py   Chroma query and score conversion
  filtering.py   Relevance filtering of retrieved chunks
  chunking.py    Heading-aware overlapping chunker
  ingest.py      Builds the vector index from data/docs/
  llm.py         LangChain ChatOpenAI wrapper and error handling
  config.py      Chunk size, top-k and filter settings
data/docs/       Fictional clinic documents (markdown)
tests/           pytest unit tests (no network or API key needed)
```

## Roadmap

- Evaluation set and script: retrieval hit rate, answer correctness, correct refusals, safety handling
- Experiments: chunk size, `top_k`, stronger embeddings, hybrid search
- Splitting compound questions into sub-questions
- Dockerfile and a deployed demo
- Simple chat front end
