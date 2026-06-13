# arxiv-reader

[![Run Tests](https://github.com/lintian233/arxiv-reader/actions/workflows/tests.yml/badge.svg)](https://github.com/lintian233/arxiv-reader/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/lintian233/arxiv-reader/badge.svg?branch=main)](https://codecov.io/gh/lintian233/arxiv-reader)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](LICENSE)

`arxiv-reader` is a local research reading pipeline for recent arXiv papers. It fetches papers from a category, optionally selects the most relevant candidates with an LLM, downloads full text and figures, and produces a structured academic reading report.

![arxiv-reader preview](docs/demo.gif)

## Overview

The project is designed for literature triage in astronomy and related research workflows: fast enough for daily arXiv scanning, structured enough for later reuse, and local-first so downloaded papers, figures, and interpretations remain available across runs.

```text
arXiv metadata -> LLM selection -> full text and figures -> interpretation -> HTML report
```

Typical inputs:

- an arXiv category, archive group, or category combination
- optional research interests for paper selection
- an OpenAI-compatible LLM endpoint

Typical outputs:

- a run manifest in the current project directory
- a reusable paper cache under the user data directory
- a local HTML report with selected figures and structured interpretation

## Quick Start

Install the project locally:

```bash
git clone https://github.com/lintian233/arxiv-reader.git
cd arxiv-reader
pip install -e .
```

Check the CLI:

```bash
arxiv-reader -h
```

Configure the LLM endpoint:

```bash
mkdir -p ~/.config/arxiv-reader
cat > ~/.config/arxiv-reader/.env <<'EOF'
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_SELECTION_MODEL=deepseek-v4-pro
DEEPSEEK_INTERPRETATION_MODEL=deepseek-v4-pro
EOF
```

Run a complete reading pipeline:

```bash
arxiv-reader run \
  --category astro-ph \
  --max-results 2 \
  --interests "papers related to the FAST radio telescope and radio interferometric methods"
```

When `--interests` is set and `--fetch-results` is omitted, `arxiv-reader` fetches all papers from the latest arXiv published day for the category, then asks the selection model to choose up to `--max-results` papers. Pass `--fetch-results N` only when you want to force a fixed-size recent candidate batch.

The command prints the generated `manifest.json` path when it finishes. Use that path to build the HTML report:

```bash
arxiv-reader report --input runs/2026-06-11_astro-ph/manifest.json
```

Serve reports and downloaded figures locally:

```bash
arxiv-reader serve --port 8765
```

Open:

```text
http://localhost:8765/
```

## Reading Workflow

For normal use, the workflow is:

```bash
arxiv-reader run --category astro-ph.IM --max-results 2
arxiv-reader report --input runs/2026-06-11_astro-ph.IM/manifest.json
arxiv-reader serve --port 8765
```

Use `--interests` when you want the model to select a smaller reading set from a larger metadata batch:

```bash
arxiv-reader run \
  --category astro-ph.IM,astro-ph.HE \
  --max-results 5 \
  --interests "FRB localization, radio interferometry, transient detection pipelines"
```

Use `--fetch-results N` for a fixed candidate window, for example when you want the selector to consider the latest 50 or 100 metadata records instead of the latest arXiv day.

## Local Data Layout

`arxiv-reader` separates long-lived paper data from per-run research outputs.

Running `arxiv-reader run ...` creates a `runs/` directory in the current working directory:

```text
runs/
└── 2026-06-11_astro-ph/
    ├── manifest.json
    ├── selection.json
    └── report.html
```

The paper cache is stored separately:

```text
~/.local/share/arxiv-reader/
└── papers/
    └── 2606.xxxxxv1/
        ├── metadata.json
        ├── content.json
        ├── figures.json
        ├── interpretation.json
        ├── reader.json
        ├── paper.pdf
        └── figures/
```

`manifest.json` records the run and points to the cached paper blocks. `selection.json` stores the LLM selection result when interests are provided. `report.html` is the local reading report for that run.

## Configuration

For most users, only the API key and model need to be configured:

```bash
mkdir -p ~/.config/arxiv-reader
cat > ~/.config/arxiv-reader/.env <<'EOF'
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_SELECTION_MODEL=deepseek-v4-pro
DEEPSEEK_INTERPRETATION_MODEL=deepseek-v4-flash
EOF
```

Configuration is loaded in this order:

```text
shell environment > .env in current directory > ~/.config/arxiv-reader/.env > defaults
```

Advanced options:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | empty | API key used by the OpenAI-compatible LLM client. Required for selection and interpretation. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible API endpoint. |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | Default model used when task-specific model variables are not set. |
| `DEEPSEEK_SELECTION_MODEL` | `DEEPSEEK_MODEL` | Model used for paper selection. |
| `DEEPSEEK_INTERPRETATION_MODEL` | `DEEPSEEK_MODEL` | Model used for paper interpretation. |
| `PAPER_DATA_DIR` | `~/.local/share/arxiv-reader` | Long-lived paper cache root for PDFs, figures, and normalized paper blocks. |
| `RUNS_DIR` | `.` | Root directory for run manifests, selection files, and generated reports. |
| `REQUEST_TIMEOUT` | `30` | Timeout in seconds for arXiv, content, and figure HTTP requests. |
| `LLM_REQUEST_TIMEOUT` | `180` | Timeout in seconds for LLM requests. |
| `MAX_INPUT_CHARS` | `400000` | Maximum paper text characters sent to the interpretation task. |
| `LLM_MAX_OUTPUT_TOKENS` | `12000` | Maximum output tokens requested from the LLM. |
| `PAPER_INTERESTS` | empty | Default research interests used by `run` when `--interests` is omitted. |
| `FETCH_RESULTS` | empty | Default number of metadata candidates fetched before selection. When empty, `run --interests ...` uses all papers from the latest arXiv published day in the returned category results. |
| `SELECTION_MAX_INPUT_CHARS` | `220000` | Maximum metadata prompt size for paper selection. |
| `SELECTION_SUMMARY_MAX_CHARS` | `4000` | Maximum abstract characters per paper used during selection. |
| `DEBUG` | false | Enables debug logging when set to `1`, `true`, `yes`, or `on`. |

Debug logging example:

```bash
DEBUG=1 arxiv-reader fetch --category astro-ph.IM --max-results 5
```

## Category Syntax

Single category:

```bash
--category astro-ph.IM
```

Astrophysics archive group:

```bash
--category astro-ph
```

`astro-ph` expands to:

```text
astro-ph.CO
astro-ph.EP
astro-ph.GA
astro-ph.HE
astro-ph.IM
astro-ph.SR
```

Multiple categories:

```bash
--category astro-ph.IM,astro-ph.HE
```

The category expression is sent as one arXiv API OR query where possible.

## License

This project is licensed under the GNU General Public License v2.0. See [LICENSE](LICENSE).
