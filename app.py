import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

load_dotenv()

from extractor import ALLOWED_EXTENSIONS, process  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/extract")
def extract():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify(error="Choose a PDF or image file first."), 400

    name = secure_filename(file.filename)
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify(error=f"Unsupported file type. Use one of: {allowed}"), 400

    # The upload is written to a temp file and deleted right after processing.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        file.save(path)
        try:
            result = process(path)
        except RuntimeError as exc:  # missing API key or model name
            return jsonify(error=str(exc)), 500
        except ValueError as exc:  # unreadable file or malformed model output
            return jsonify(error=str(exc)), 422
        except Exception as exc:  # network, quota or provider errors
            app.logger.exception("Extraction failed")
            return jsonify(error=f"Extraction failed: {exc}"), 502

    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File is larger than 10 MB."), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000)
