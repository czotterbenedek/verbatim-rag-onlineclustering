import json
import time
from urllib import error, request


class OllamaGenerator:
    """Generate answers through a local Ollama HTTP server."""

    def __init__(self, model, endpoint="http://localhost:11434/api/generate", timeout=120):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(self, question, contexts, system_prompt):
        context = "\n\n".join(
            f"[{number}] {item['text']}"
            for number, item in enumerate(contexts, start=1)
        )
        prompt = (
            f"{system_prompt}\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer:"
        )
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
            body = json.loads(response.read().decode("utf-8"))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Ollama generation failed at {self.endpoint} using model '{self.model}': {exc}"
            ) from exc
        answer = body.get("response")
        if not isinstance(answer, str):
            raise RuntimeError("Ollama returned no string response.")
        return answer, (time.perf_counter() - started) * 1000
