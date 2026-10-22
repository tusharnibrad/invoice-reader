"""Core logic: read a document (PDF or image), run OCR when needed,
ask an LLM to turn the text into structured fields, then sanity-check the result.

Heavy libraries (PyMuPDF, pytesseract, openai) are imported inside the functions
that need them, so the pure-logic helpers can be unit-tested without them.
"""
import base64
import io
import json
import os
import re
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS

# A PDF page with fewer characters than this is treated as a scan and sent to OCR.
MIN_TEXT_CHARS_PER_PAGE = 30

FIELDS = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "currency",
    "subtotal",
    "tax",
    "total",
]

SYSTEM_PROMPT = """You extract structured data from invoice text.
Return ONLY a JSON object. No explanation, no markdown fences.

Use exactly this shape:
{
  "vendor_name": string or null,
  "invoice_number": string or null,
  "invoice_date": "YYYY-MM-DD" or null,
  "due_date": "YYYY-MM-DD" or null,
  "currency": 3-letter code such as "INR" or "USD", or null,
  "line_items": [
    {"description": string, "quantity": number or null,
     "unit_price": number or null, "amount": number or null}
  ],
  "subtotal": number or null,
  "tax": number or null,
  "total": number or null
}

Rules:
- Use null when a field is not present in the text. Never guess or invent values.
- Numbers must be plain numbers without currency symbols or thousands separators.
- The text may contain OCR mistakes. Fix obvious character errors only when the
  surrounding context makes the correct value certain."""


# ---------------------------------------------------------------- reading ---

def read_document(path):
    """Return (text, method) where method is 'pdf-text', 'ocr' or 'mixed'."""
    ext = Path(path).suffix.lower()
    if ext in PDF_EXTENSIONS:
        return _read_pdf(path)
    if ext in IMAGE_EXTENSIONS:
        return _ocr_image_file(path), "ocr"
    raise ValueError(f"Unsupported file type: {ext}")


def _find_tesseract():
    """On Windows the installer does not add Tesseract to PATH, so point pytesseract at it."""
    import pytesseract

    exe = os.getenv("TESSERACT_CMD") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(exe):
        pytesseract.pytesseract.tesseract_cmd = exe
    return pytesseract


def prepare_for_ocr(img):
    """Clean up a photo or screenshot so Tesseract can read it.

    Phone photos of screens are large and full of moire patterns. Shrinking them,
    blurring slightly and keeping only pixels darker than their surroundings turns
    them into clean black-on-white text. Small images are enlarged instead.
    """
    from PIL import Image, ImageChops, ImageFilter, ImageOps

    img = ImageOps.exif_transpose(img).convert("L")
    width, height = img.size
    longest = max(width, height)
    target = 1000 if longest > 1200 else 1400
    scale = target / longest
    img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    local_mean = img.filter(ImageFilter.GaussianBlur(15))
    ink = ImageChops.subtract(local_mean, img)  # how much darker than the neighbourhood
    return ink.point(lambda v: 0 if v > 8 else 255)


def _ocr_image_file(path):
    from PIL import Image

    pytesseract = _find_tesseract()
    with Image.open(path) as img:
        prepared = prepare_for_ocr(img)
    return pytesseract.image_to_string(prepared, config="--psm 6")


def _read_pdf(path):
    import fitz  # PyMuPDF
    from PIL import Image

    pytesseract = _find_tesseract()

    pages, used_text, used_ocr = [], False, False
    with fitz.open(path) as doc:
        for page in doc:
            text = page.get_text().strip()
            if len(text) >= MIN_TEXT_CHARS_PER_PAGE:
                pages.append(text)
                used_text = True
            else:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                pages.append(pytesseract.image_to_string(img))
                used_ocr = True

    if used_text and used_ocr:
        method = "mixed"
    elif used_ocr:
        method = "ocr"
    else:
        method = "pdf-text"
    return "\n\n".join(pages), method


# -------------------------------------------------------------------- LLM ---

def _client_and_model():
    from openai import OpenAI

    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not api_key or not model:
        raise RuntimeError(
            "Set LLM_API_KEY and LLM_MODEL (and LLM_BASE_URL if needed). "
            "See .env.example."
        )
    return OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL") or None), model


def call_llm(document_text):
    """Send the document text to an OpenAI-compatible chat endpoint."""
    client, model = _client_and_model()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Invoice text:\n\n{document_text[:12000]}"},
        ],
    )
    return response.choices[0].message.content


def _image_to_data_url(path, longest_side=1600):
    """Shrink the photo and turn it into a data URL the model can look at."""
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        scale = longest_side / max(img.size)
        if scale < 1:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_vision_messages(data_url):
    return [
        {"role": "system", "content": SYSTEM_PROMPT.replace("invoice text", "an invoice photo")},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the fields from this invoice image."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def call_llm_vision(path):
    """Let the model read the photo directly instead of relying on OCR text."""
    client, model = _client_and_model()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=build_vision_messages(_image_to_data_url(path)),
    )
    return response.choices[0].message.content


def clean_json(raw):
    """Parse the model's reply even if it added code fences or extra words."""
    if raw is None:
        raise ValueError("Empty response from model")
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start : end + 1])


# ------------------------------------------------------------- validation ---

def to_number(value):
    """Turn '1,234.50', '₹1,234.50' or 1234.5 into a float. Returns None if impossible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^\d.\-]", "", str(value).replace(",", ""))
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def validate(data, tolerance=0.02):
    """Return a list of human-readable problems. An empty list means the numbers add up."""
    issues = []

    for field in ("vendor_name", "invoice_number", "invoice_date", "total"):
        if not data.get(field):
            issues.append(f"Missing field: {field}")

    items = data.get("line_items") or []
    amounts = [to_number(item.get("amount")) for item in items if isinstance(item, dict)]
    known_amounts = [a for a in amounts if a is not None]

    subtotal = to_number(data.get("subtotal"))
    tax = to_number(data.get("tax"))
    total = to_number(data.get("total"))

    if known_amounts and subtotal is not None:
        if abs(sum(known_amounts) - subtotal) > tolerance:
            issues.append(
                f"Line items add up to {sum(known_amounts):.2f} but subtotal is {subtotal:.2f}"
            )

    if subtotal is not None and total is not None:
        expected = subtotal + (tax or 0.0)
        if abs(expected - total) > tolerance:
            issues.append(
                f"Subtotal + tax is {expected:.2f} but total is {total:.2f}"
            )

    for item in items:
        if not isinstance(item, dict):
            continue
        qty, price, amount = (
            to_number(item.get("quantity")),
            to_number(item.get("unit_price")),
            to_number(item.get("amount")),
        )
        if None not in (qty, price, amount) and abs(qty * price - amount) > max(tolerance, 0.01 * amount):
            issues.append(
                f"Row '{item.get('description', '?')}': {qty} x {price} is not {amount}"
            )

    return issues


# ------------------------------------------------------------------- main ---

def _is_quota_error(exc):
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def process(path):
    """Full pipeline for one file.

    Photos and screenshots are sent to the model as images (set USE_VISION=0 to
    turn this off). If that fails for a reason other than the daily quota, the
    app falls back to OCR + text.
    """
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS and os.getenv("USE_VISION", "1") != "0":
        try:
            data = clean_json(call_llm_vision(path))
            return {
                "fields": data,
                "issues": validate(data),
                "method": "vision",
                "text_preview": "(the photo was read directly by the model, no OCR text)",
            }
        except Exception as exc:
            if _is_quota_error(exc):
                raise

    text, method = read_document(path)
    if not text.strip():
        raise ValueError("No text could be read from this file")
    data = clean_json(call_llm(text))
    return {
        "fields": data,
        "issues": validate(data),
        "method": method,
        "text_preview": text[:1500],
    }
