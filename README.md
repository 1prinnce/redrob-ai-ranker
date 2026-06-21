# Redrob AI Ranker

Candidate ranking pipeline for the Redrob candidate discovery dataset.

The repository contains two command-line pipeline stages:

- `src.pipeline.prep`: reads candidate records, computes cached candidate features, generates embeddings, builds a FAISS index, and writes ranking artifacts.
- `src.pipeline.rank`: loads the artifacts, reads a job description, retrieves candidates with FAISS, scores the retrieved pool, and writes a top-100 submission CSV.

## Repository Layout

- `src/data`: JSONL candidate loading and top-level candidate validation.
- `src/jd`: deterministic regex-based job-description parsing and requirements-text extraction.
- `src/features`: career, behavioral, location, skills, honeypot, and disqualifier scoring helpers.
- `src/embeddings`: embedding backend loading, candidate/JD text fusion, embedding persistence, and FAISS index helpers.
- `src/scoring`: final score composition, score normalization, ranking validation, and recruiter-facing reasoning.
- `src/pipeline`: offline preprocessing and online ranking entry points.
- `dataset`: challenge data files and submission validator.
- `artifacts`: generated preprocessing artifacts used by the ranker.
- `outputs`: generated submission files.

## Inputs

The preprocessing pipeline accepts a `.jsonl` or `.jsonl.gz` candidate file. Each candidate must include these top-level keys:

- `candidate_id`
- `profile`
- `career_history`
- `education`
- `skills`
- `certifications`
- `languages`
- `redrob_signals`

Candidate IDs must match `CAND_0000000` with seven digits.

The ranking pipeline accepts a job description as `.txt` or `.docx`. Reading `.docx` files requires `python-docx`.

## Artifacts

`src.pipeline.prep` writes these files to the output artifact directory:

- `candidate_embeddings.npy`
- `candidate_features.npz`
- `candidate_meta.pkl`
- `faiss.index`
- `prep_manifest.json`

`src.pipeline.rank` expects all five files to exist in the artifact directory and validates candidate counts, embedding dimensions, metadata fields, feature arrays, and manifest values before ranking.

## Embeddings And Retrieval

The default embedding model name is `all-MiniLM-L6-v2`.

`src/embeddings/encoder.py` first attempts to load an ONNX Runtime backend through `onnxruntime`, `optimum`, and `transformers`. If that is unavailable or loading fails, it attempts a CPU SentenceTransformers backend.

Embeddings are stored as `float32` arrays. FAISS retrieval uses an `IndexFlatIP` index over L2-normalized 384-dimensional embeddings.

## Scoring

The online ranker retrieves up to 1000 candidates with FAISS, scores the retrieved candidates, sorts by raw score descending, keeps the top 100, and min-max normalizes the top-100 scores into `[0.40, 1.00]`.

The raw scoring formula implemented by `src/scoring/scorer.py` and reused by `src/pipeline/rank.py` is:

```text
base_score = clamp(
  0.35 * semantic_score
  + 0.25 * career_score
  + 0.15 * behavioral_score
  + 0.10 * location_score
  + 0.15 * skill_score
)

final_score = clamp(
  base_score
  * (1.0 - honeypot_penalty)
  * (1.0 - disqualifier_penalty)
)
```

Score values and penalties are clamped to the `[0.0, 1.0]` range.

The submitted CSV score is the normalized top-100 score, not the raw `final_score`.

## Reasoning

`src/scoring/reasoning.py` generates short recruiter-facing explanations from available structured fields. It uses only present candidate facts such as current title, current company, years of experience, recruiter response rate, notice period, rank, and normalized score.

## Commands

Run preprocessing:

```powershell
python -m src.pipeline.prep --candidates dataset\candidates.jsonl --out artifacts
```

Run ranking:

```powershell
python -m src.pipeline.rank --artifacts artifacts --jd dataset\job_description.docx --out outputs\submission.csv
```

Validate a submission with the dataset validator:

```powershell
python dataset\validate_submission.py outputs\submission.csv
```

## Known Limits

- There is no dependency lockfile in this workspace.
- There are no test files in the `tests` directory in this workspace.
- The code does not include benchmark results or measured ranking-quality metrics.
- Candidate validation in `src/data/validator.py` checks required top-level keys only; it does not enforce the full `dataset/candidate_schema.json` schema.
