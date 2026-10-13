"""Vetorizador – servidor local.

Uso:  python app.py   (abre automaticamente http://127.0.0.1:5000)
"""

import json
import os
import threading
import webbrowser

from flask import Flask, jsonify, request, send_from_directory

from vectorizer import VectorizeError, vectorize_laser, vectorize_standard

BASE = os.path.dirname(os.path.abspath(__file__))
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", 5000))

app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/vectorize")
def api_vectorize():
    f = request.files.get("file")
    if not f:
        return jsonify(error="Nenhum arquivo enviado."), 400
    try:
        params = json.loads(request.form.get("params") or "{}")
    except json.JSONDecodeError:
        return jsonify(error="Parâmetros inválidos."), 400

    data = f.read()
    try:
        if request.form.get("mode") == "laser":
            result = vectorize_laser(data, params)
        else:
            result = vectorize_standard(data, params)
    except VectorizeError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:  # erro inesperado: mostra na interface
        app.logger.exception("falha na vetorização")
        return jsonify(error=f"Erro ao vetorizar: {exc}"), 500
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Arquivo muito grande (máx. 60 MB)."), 413


if __name__ == "__main__":
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    print(f"\n  Vetorizador rodando em http://{HOST}:{PORT}  (Ctrl+C para sair)\n")
    app.run(host=HOST, port=PORT, threaded=True, debug=False)
