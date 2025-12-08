# Contributing to MEXsrfdcrPy

🎋 Thanks for your interest in contributing to **MEXsrfdcrPy: Spatial Random Forests for Daily Climate Records Reconstruction in Mexico**.  
This package is part of a broader open-source ecosystem for working with Mexican climate data and global gridded products.

We welcome contributions from students, researchers, and practitioners in climatology, agriculture, data science, and software engineering.

---

## How you can contribute

There are many ways to help:

- 🐛 **Report bugs** (unexpected behaviour, crashes, wrong results).
- 💡 **Propose improvements** to the API, performance, or documentation.
- 🧪 **Add tests** that increase coverage and guard against regressions.
- 🌱 **Add examples** or small workflows that show how to use the package in real studies.
- 📚 **Improve documentation** (docstrings, README, usage notes).

If you are unsure whether your idea fits, open an issue and start a discussion.

---

## Code of conduct

By participating in this project, you agree to interact respectfully and constructively.  
Be kind, assume good faith, and keep the focus on improving the software and its scientific value.

---

## Project structure (high level)

The core code lives under:

- `src/MEXsrfdcrPy/`
  - `loso.py` – leave-one-station-out evaluation, station-wise reconstruction, metrics.
  - `grid.py` – global Random Forest training and daily grid prediction (X, Y, Z, T).
  - `__init__.py` – public API exports.

Tests live in:

- `tests/`
  - `test_loso.py`
  - `test_grid.py`

Please follow the existing design patterns when adding functionality.

---

## Development setup

1. **Fork** the repository on GitHub.

2. **Clone** your fork:

   ```bash
   git clone https://github.com/<your-user>/MEXsrfdcrPy.git
   cd MEXsrfdcrPy
