# arxiv-astro

Minimal pipeline for fetching papers from one arXiv category, extracting usable text from HTML/PDF/abstract, sending it to an OpenAI-compatible LLM, and writing per-paper JSON blocks for an arXiv reader.

## Run

```bash
python -m arxiv_astro.cli --category astro-ph.CO --max-results 5
python -m arxiv_astro.cli fetch --category astro-ph.CO --max-results 5
python -m arxiv_astro.cli fetch --category astro-ph.CO --max-results 5 --debug
python -m arxiv_astro.cli content --input data/runs/2026-06-07_astro-ph.CO/manifest.json
python -m arxiv_astro.cli explain --input data/runs/2026-06-07_astro-ph.CO/manifest.json
python -m arxiv_astro.cli run --category astro-ph.CO --max-results 5
```

The root command defaults to metadata-only `fetch`. Use `run` explicitly for the full LLM pipeline.
Use `content` to load full text and images from a `metadata.json` or fetch `manifest.json`.
Use `explain` to generate LLM interpretation blocks from a `content.json` or content `manifest.json`.
The `run` command renders a live Rich table on stderr with each paper's ID, title, stage, content source, text length, and image count. Completed LLM interpretations are rendered below the table as per-paper details.

Outputs are organized by stable arXiv ID:

```text
data/
  papers/
    2401.12345v1/
      metadata.json
      content.json
      interpretation.json
      reader.json
  runs/
    2026-06-07_astro-ph.CO/
      manifest.json
```

`fetch` writes `metadata.json` plus a run manifest. `content` writes `content.json` plus a run manifest. `explain` writes `interpretation.json` and `reader.json` plus a run manifest. `run` writes all four per-paper JSON files and one manifest in a single pass.

Debug logging can also be enabled globally:

```bash
DEBUG=1 python -m arxiv_astro.cli fetch --category astro-ph.CO --max-results 5
```

Environment variables:

```bash
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
LLM_REQUEST_TIMEOUT=180
```

The LLM client uses the OpenAI SDK against the DeepSeek-compatible endpoint with:

```python
reasoning_effort="high"
extra_body={"thinking": {"type": "enabled"}}
```

## Test

```bash
pytest
```

Real network integration tests are opt-in:

```bash
RUN_REAL_NETWORK=1 pytest tests/integration --no-cov
RUN_REAL_NETWORK=1 ARXIV_ASTRO_TEST_CATEGORY=astro-ph.IM pytest tests/integration --no-cov
```
