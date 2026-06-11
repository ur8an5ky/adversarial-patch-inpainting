# Adversarial Patch Inpainting

Rekonstrukcja obrazu po usunięciu fragmentów (adversarial patches) —
klasyczne metody analizy obrazu z benchmarkiem względem modelu ML.
Projekt z Analizy i Przetwarzania Obrazów.

## Struktura
- `src/` — kod modułów (dane/detekcja, inpainting, ewaluacja)
- `notebooks/` — eksploracja i demo
- `tests/` — sanity-checki

## Setup
\`\`\`bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
\`\`\`