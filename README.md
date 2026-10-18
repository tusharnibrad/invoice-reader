# Invoice Reader (OCR + LLM)

Upload an invoice as a PDF or image. The app reads the text (directly from digital PDFs, or with OCR for scans and photos), asks an LLM to turn it into structured fields, and then checks that the numbers add up.

**Why the validation step matters:** LLMs sometimes return confident but wrong numbers. This project checks that line items sum to the subtotal, that subtotal + tax equals the total, and that quantity x price equals each row amount. Anything that fails is shown to the user as a warning.

## How it works

    upload -> read text (PyMuPDF, or Tesseract OCR for scans)
           -> LLM extracts JSON fields
           -> clean + validate (math checks)
           -> show fields and warnings in the web page

| File | Job |
|---|---|
| `extractor.py` | Reading, OCR, LLM call, JSON cleanup, validation |
| `app.py` | Flask server with `/` and `/extract` |
| `templates/index.html` | Upload page and results view |
| `evaluate.py` | Measures accuracy on your labelled samples |
| `tests/test_logic.py` | Unit tests for the parsing and validation logic |

## Run it locally

1. Install Tesseract OCR on your computer.
   - Windows: install from the UB Mannheim build and add it to PATH.
   - Mac: `brew install tesseract`
   - Ubuntu: `sudo apt install tesseract-ocr`
2. Create a virtual environment and install packages:

       python -m venv .venv
       .venv\Scripts\activate        # Windows
       source .venv/bin/activate     # Mac/Linux
       pip install -r requirements.txt

3. Copy `.env.example` to `.env` and fill in `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` for any OpenAI-compatible provider. Check the provider's docs for the current model name.
4. Start the app: `python app.py` and open http://localhost:5000

Run the tests (no API key needed): `python tests/test_logic.py`

## Measure accuracy (do this before putting numbers on your resume)

1. Add 20-30 invoices to `samples/` with matching `.json` answer files (see `samples/README.md`).
2. Run `python evaluate.py`.
3. Use the real percentages it prints.

## Deploy with Docker

    docker build -t invoice-reader .
    docker run -p 8000:8000 --env-file .env invoice-reader

The Dockerfile installs Tesseract for you. Hosts such as Render or Hugging Face Spaces (Docker) can run this image. Set the three LLM variables in the host's secret settings, never in the code.

## Limits

- Handwriting and very low-quality photos reduce OCR accuracy.
- Output is only as good as the model and the scan. Always verify important values.
- Uploaded files are deleted after processing, but the text is sent to the LLM provider, so do not use real sensitive documents.

## Ideas to extend

- Medical report mode: change the prompt and schema to lab values (this is a data-extraction tool, not medical advice).
- Confidence flags per field.
- Export results to CSV or Excel.
- Batch upload.
