# Version Control Notes

The root `.gitignore` is tuned for this repo's Python and `uv` workflow.

- Ignore local-only developer state such as `.venv/`, `__pycache__/`, `.pytest_cache/`, coverage output, and editor metadata.
- Ignore generated packaging metadata like `*.egg-info/`, `build/`, and `dist/`.
- Ignore `.env` variants while keeping `.env.example` committed as the configuration template.
- Ignore `data/runtime/` because the app writes uploaded files, batch metadata, and generated outputs there at runtime.

Keep `pyproject.toml`, `uv.lock`, docs, and source files under version control. Dependency changes are made through `pyproject.toml` and locked with `uv`; there is no second requirements file.
