# SimuQ / DiffSimuQ Development Guidelines

## Version Control
- Commit aggressively: every completed feature, bug fix, or meaningful milestone gets its own commit.
- Never commit broken code. All tests must pass before committing.
- Use descriptive commit messages that explain the "why", not just the "what".
- Stage only relevant files (never blindly `git add -A`).

## Testing
- Write unit tests whenever implementing a new function or module.
- Tests live in `differential_computing/tests/` (new modules) or alongside existing test structure.
- Run tests before every commit; if any fail, fix them first.
- For numerical/stochastic functions, validate against finite differences or an analytic reference.
- Use `conda activate qec_pg` to run tests (has all required packages).

## Environment
- Always use the `qec_pg` conda environment: `conda activate qec_pg`
- Local SimuQ fork in `src/` — do NOT pip install the public SimuQ package.
- PulseDSL_py path: `/Users/syue99/research/RISC-Q/PulseDSL/src/DSL/`
- Python path setup: `sys.path.insert(0, "src/")` and `sys.path.insert(0, "differential_computing/")`
