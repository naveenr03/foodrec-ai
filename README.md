# Foodrec-AI – Restaurant Recommendation

A RAG-based restaurant recommendation app that uses semantic search (FAISS + embeddings) and an LLM (Groq) to suggest restaurants from a Zomato-style dataset. Built with LangChain and Streamlit.

## Features

- **Data pipeline**: Load, clean, and sample restaurant data from CSV
- **Document store**: Convert restaurants into LangChain documents for retrieval
- **Embeddings**: HuggingFace sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store**: FAISS index for fast similarity search
- **Retrieval**: Top-k restaurant retrieval by natural-language query, with **FAISS distance scores** surfaced in the UI and API
- **LLM recommendation**: Groq (Llama) picks and explains the best match
- **Streamlit UI**: Form with optional food/dish search, optional cuisine, budget, ambience, group size, and one-click recommendation

## Prerequisites

- Python 3.10+
- [Groq API key](https://console.groq.com) (free tier available)
- Restaurant dataset: default raw file is set in [`src/data_processing/constants.py`](src/data_processing/constants.py) (`DEFAULT_RAW_CSV_NAME`); place that file under `data/raw/`.

## Installation

1. **Clone and enter the project**
  ```bash
   git clone https://github.com/your-username/foodrec-ai.git
   cd foodrec-ai
  ```
2. **Create a virtual environment and install dependencies**
  ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\Activate.ps1
   pip install -r requirements.txt
  ```
3. **Environment variables** (project root `.env` and/or shell):

   | Variable | Purpose |
   |----------|---------|
   | `GROQ_API_KEY` | Required for LLM recommendations (Groq). |
   | `FOODREC_RETRIEVAL_DEBUG` | Set to `1` or `true` to log each retrieval query, FAISS score, and restaurant name to **stderr** (useful when running Streamlit or scripts from a terminal). |

## Data setup

1. Put the raw restaurant CSV at:
  ```text
   data/raw/<your file>.csv
  ```
   The default filename is defined in `src/data_processing/constants.py` (`DEFAULT_RAW_CSV_NAME`); `scripts/run_data_pipeline.py` uses the same value. You can use another file by editing that constant or the script’s path. Expected columns include: `name`, `location`, `cuisines`, `approx_cost(for two people)`, `rate`, `rest_type`. **Newer Zomato-style CSVs** (e.g. Chennai) often use `Name of Restaurant`, `Cuisine`, `Price for 2`, `Dining Rating`, and `Features` — these are mapped automatically in [`src/data_processing/column_aliases.py`](src/data_processing/column_aliases.py).
2. **Run the data pipeline** (clean, sample 2000 rows, save to `data/processed/`)
  ```bash
   python scripts/run_data_pipeline.py
  ```
3. **Build the FAISS vector store** (required before running the app)
  ```bash
   python scripts/build_vector_store.py
  ```
   Output is written to `data/vector_store/`.

## Running the app

From the **project root** (not inside `venv/`), with the virtual environment activated:

```bash
python -m streamlit run app/streamlit_app.py
```

On Windows PowerShell, activate the venv first: `.\venv\Scripts\Activate.ps1`.

Using `python -m streamlit` avoids “streamlit is not recognized” when the `streamlit` launcher is not on your `PATH`. You can use `streamlit run app/streamlit_app.py` instead if that command works.

Then open the URL shown in the terminal (e.g. `http://localhost:8501`). Optionally enter a **food or dish** (e.g. burger), **cuisine**, budget, ambience, and group size, then click **Find restaurant** to get a recommendation and the list of retrieved options.

## Architecture and code layout

End-to-end flow (data pipeline, FAISS build, retrieval, LLM) is documented in **[docs/TECHNICAL_OVERVIEW_AND_INTERVIEW_PREP.md](docs/TECHNICAL_OVERVIEW_AND_INTERVIEW_PREP.md)**. Use that doc for module-by-module behavior and interview-style Q&A; this README stays focused on setup and commands.

- **Paths**: [`src/paths.py`](src/paths.py) — resolved `Path`s for `data/raw`, processed CSV, and vector store (used by scripts and the app).
- **Column mapping**: [`src/data_processing/column_aliases.py`](src/data_processing/column_aliases.py) — raw header aliases; ETL lives in [`load_dataset.py`](src/data_processing/load_dataset.py).
- **CLI scripts**: each runnable under `scripts/` calls [`scripts/bootstrap.py`](scripts/bootstrap.py) once so `src.*` imports work without extra `PYTHONPATH` setup.

## Troubleshooting

**Wrong city or old restaurants after changing the raw CSV**

The app searches a **FAISS index** on disk (`data/vector_store/`), not the raw CSV. Changing `DEFAULT_RAW_CSV_NAME` or swapping files under `data/raw/` only takes effect after:

1. `python scripts/run_data_pipeline.py` — overwrites `data/processed/restaurants_clean.csv`
2. `python scripts/build_vector_store.py` — rebuilds the index and writes `index_build_manifest.json`

Then restart Streamlit (or use **C** “Clear cache” in the app menu). The UI caption under the header shows which raw file the current index was built from.

If `build_vector_store.py` fails with **“No columns to parse from file”**, your `restaurants_clean.csv` is invalid (often from a raw CSV whose headers did not match the expected schema, which previously produced a file with rows but **zero columns**). Delete `data/processed/restaurants_clean.csv`, run `run_data_pipeline.py` again (you should now get a clear error or a valid CSV), then `build_vector_store.py`.

## Project structure

```text
foodrec-ai/
├── app/
│   ├── streamlit_app.py          # Thin entry: loads index, wires UI
│   ├── query_builder.py          # Natural-language query from form fields
│   └── ui.py                     # Streamlit layout helpers (styles, hero, results)
├── data/
│   ├── raw/                      # Raw CSV (see constants.py; not in repo)
│   ├── processed/                # restaurants_clean.csv
│   └── vector_store/             # FAISS index (built by script)
├── scripts/
│   ├── bootstrap.py              # Puts repo root on sys.path for CLI scripts
│   ├── run_data_pipeline.py      # Clean + sample dataset
│   ├── build_vector_store.py     # Build FAISS from documents
│   ├── test_documents.py         # Test document builder
│   ├── test_embeddings.py        # Test embedding model
│   ├── test_retrieval.py         # Test semantic search
│   └── test_llm_recommendation.py # Test full RAG + LLM
├── src/
│   ├── paths.py                  # Project-root paths for data and vector store
│   ├── data_processing/
│   │   ├── load_dataset.py       # Load/clean/save CSV
│   │   ├── column_aliases.py     # Raw CSV header → canonical column names
│   │   ├── constants.py          # Default raw CSV filename
│   │   └── document_builder.py   # Rows → LangChain documents
│   ├── embeddings/
│   │   └── embedder.py           # HuggingFace embedding model
│   ├── retrieval/
│   │   ├── vector_store.py       # FAISS create/save/load
│   │   └── retriever.py          # Semantic search
│   ├── llm/
│   │   └── recommender.py        # Groq LLM recommendation
│   └── pipeline/
│       └── recommendation_pipeline.py  # Retrieve + recommend
├── .env                          # GROQ_API_KEY (not in repo)
├── .gitattributes
├── requirements.txt
└── README.md
```

## Scripts overview


| Script                       | Purpose                                                                          |
| ---------------------------- | -------------------------------------------------------------------------------- |
| `run_data_pipeline.py`       | Load raw CSV → clean → sample 2000 → save `data/processed/restaurants_clean.csv` |
| `build_vector_store.py`      | Load cleaned data → build documents → embed → save FAISS to `data/vector_store/` |
| `test_documents.py`          | Load cleaned data, build documents, print count and one sample                   |
| `test_embeddings.py`         | Load data, build docs, load embedder, print embedding dimension (e.g. 384)       |
| `test_retrieval.py`          | Load FAISS + embedder, run a sample query, print top 5 with **distance scores**        |
| `test_llm_recommendation.py` | Full flow: retrieve → Groq recommendation → print result                         |


## Deployment (Streamlit Cloud)

1. Push this repo to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io), connect the repo and set **Main file path** to `app/streamlit_app.py`.
3. Add a secret: `GROQ_API_KEY` = your Groq API key.
4. Ensure `data/vector_store/` is present in the repo (commit the built index), or run `build_vector_store.py` as part of a build step if your deployment supports it.



