import json
import time
from urllib import error, request
from urllib.parse import quote


def _build_prompt(question, contexts):
    context = "\n\n".join(
        f"[{number}] {item['text']}"
        for number, item in enumerate(contexts, start=1)
    )
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer:"
    )


def _read_json_response(response):
    return json.loads(response.read().decode("utf-8"))


class OllamaGenerator:
    """Generate answers through a local Ollama HTTP server."""

    def __init__(self, model, endpoint="http://localhost:11434/api/generate", timeout=120):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(self, question, contexts, system_prompt):
        prompt = f"{system_prompt}\n\n{_build_prompt(question, contexts)}"
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")
        started = time.perf_counter()
        try:
            response = request.urlopen(
                request.Request(
                    self.endpoint,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=self.timeout,
            )
            body = _read_json_response(response)
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Ollama generation failed at {self.endpoint} using model '{self.model}': {exc}"
            ) from exc
        answer = body.get("response")
        if not isinstance(answer, str):
            raise RuntimeError("Ollama returned no string response.")
        return answer, (time.perf_counter() - started) * 1000


class GeminiGenerator:
    """Generate answers through the Google Gemini generateContent API."""

    def __init__(self, model, api_key, endpoint=None, timeout=120):
        if not api_key:
            raise ValueError("Gemini requires a non-empty API key.")
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint or (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "{model}:generateContent"
        )
        self.timeout = timeout

    def generate(self, question, contexts, system_prompt):
        endpoint = self.endpoint.replace("{model}", quote(self.model, safe=""))
        separator = "&" if "?" in endpoint else "?"
        url = f"{endpoint}{separator}key={quote(self.api_key, safe='')}"
        payload = json.dumps({
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [{
                "role": "user",
                "parts": [{"text": _build_prompt(question, contexts)}],
            }],
        }).encode("utf-8")
        started = time.perf_counter()
        try:
            response = request.urlopen(
                request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=self.timeout,
            )
            body = _read_json_response(response)
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Gemini generation failed at {self.endpoint} using model "
                f"'{self.model}' (HTTP {exc.code}): {details}"
            ) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Gemini generation failed at {self.endpoint} using model '{self.model}': {exc}"
            ) from exc

        try:
            answer = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Gemini returned no text candidate.") from exc
        if not isinstance(answer, str):
            raise RuntimeError("Gemini returned a non-string text candidate.")
        return answer, (time.perf_counter() - started) * 1000
