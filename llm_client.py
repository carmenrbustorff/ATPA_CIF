"""
LLM Client for the Autonomous Research Agent.

Communicates with a locally-hosted LLM via the Ollama API
(OpenAI-compatible) at http://127.0.0.1:11434.

Supports any model available in the local Ollama instance,
e.g. gemma3, qwen2.5-coder, llama3, deepseek-coder, etc.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:32b"


class LLMClient:
    """
    Model-agnostic client for a locally-hosted Ollama LLM.

    Uses the OpenAI-compatible REST API exposed by Ollama.
    No external Python SDK is required – only the standard library.
    If the optional ``ollama`` package is installed it will be used
    automatically for a richer experience (streaming, error details).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._sdk_available = self._check_sdk()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_sdk(self) -> bool:
        try:
            import ollama  # noqa: F401
            return True
        except ImportError:
            return False

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        """Low-level HTTP POST using only the standard library."""
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`)."
            ) from exc

    def _get_json(self, endpoint: str) -> dict:
        """Low-level HTTP GET using only the standard library."""
        url = f"{self.base_url}{endpoint}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`)."
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a prompt to the LLM and return the response text.

        Parameters
        ----------
        prompt:
            The user-facing prompt.
        system:
            Optional system-level instruction prepended to the conversation.
        temperature:
            Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens:
            Maximum number of tokens to generate.

        Returns
        -------
        str
            The model's text response.
        """
        if self._sdk_available:
            return self._generate_sdk(prompt, system, temperature, max_tokens)
        return self._generate_http(prompt, system, temperature, max_tokens)

    def _generate_sdk(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        from ollama import Client  # type: ignore

        # Create a client that respects the 300-second timeout
        client = Client(host=self.base_url, timeout=self.timeout)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["message"]["content"]

    def _generate_http(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Fallback: use the Ollama /api/generate endpoint directly."""
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        result = self._post_json("/api/generate", payload)
        return result.get("response", "")

    def list_models(self) -> list[str]:
        """Return a list of model names available in the local Ollama instance."""
        result = self._get_json("/api/tags")
        return [m["name"] for m in result.get("models", [])]

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            self._get_json("/api/tags")
            return True
        except ConnectionError:
            return False

    def propose_architecture(self, context: str, previous_results: Optional[str] = None) -> str:
        """
        Ask the LLM to propose a new model architecture.

        Parameters
        ----------
        context:
            Information about the task, dataset, and constraints.
        previous_results:
            Optional summary of the previous iteration's results.

        Returns
        -------
        str
            Raw LLM response (usually contains Python code + explanation).
        """
        system = (
            "You are an expert machine-learning engineer specialising in audio "
            "classification and bioacoustic deep learning. "
            "Your goal is to propose improved model architectures for the "
            "BirdCLEF+ 2026 competition (Track B), which requires multi-label "
            "classification of 234 bird species from mel-spectrogram inputs. "
            "Always respond with working Python code (TensorFlow/Keras or PyTorch) "
            "followed by a brief explanation. Wrap code in ```python ... ``` blocks."
        )
        prev_section = (
            f"\n\nPrevious iteration results:\n{previous_results}"
            if previous_results
            else ""
        )
        prompt = (
            f"Task context:\n{context}{prev_section}\n\n"
            "Propose a new or improved model architecture. "
            "Prioritise small-scale, fast iterations for rapid experimentation. "
            "Include training loop, data loading assumptions, and evaluation code."
        )
        return self.generate(prompt, system=system)

    def analyse_results(self, results: str, context: str) -> str:
        """
        Ask the LLM to analyse training results and suggest improvements.

        Parameters
        ----------
        results:
            JSON or text summary of the latest experiment results.
        context:
            Task/experiment context.

        Returns
        -------
        str
            LLM analysis and recommendations.
        """
        system = (
            "You are an expert ML researcher. Analyse the provided experiment "
            "results and suggest concrete, actionable improvements for the next "
            "iteration. Be specific: mention hyperparameter values, architecture "
            "changes, and data-augmentation strategies."
        )
        prompt = (
            f"Context:\n{context}\n\n"
            f"Latest experiment results:\n{results}\n\n"
            "Provide a structured analysis:\n"
            "1. What worked well?\n"
            "2. What needs improvement?\n"
            "3. Specific changes for the next iteration (with code snippets if relevant)."
        )
        return self.generate(prompt, system=system)
