import logging

from app.core.config import settings


class GeminiConfigurationError(Exception):
    """Raised when Gemini client configuration is missing or invalid."""


class GeminiGenerationError(Exception):
    """Raised when Gemini text generation fails."""


class GeminiClient:
    """Google Gemini client adapter compatible with `ChatService` LLM interface."""

    ALLOWED_MODEL = "gemini-flash-latest"

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = settings.gemini_api_key
        self.model_name = settings.gemini_model

        if settings.llm_provider.lower() != "gemini":
            raise GeminiConfigurationError("Invalid LLM provider. Expected 'gemini'.")

        if self.model_name != self.ALLOWED_MODEL:
            raise GeminiConfigurationError(
                f"Invalid Gemini model configured. Expected '{self.ALLOWED_MODEL}'."
            )

        if not self.api_key:
            raise GeminiConfigurationError("GEMINI_API_KEY is not configured.")

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._genai = genai
            self.logger.info("Gemini client configured successfully")
        except Exception as exc:
            self.logger.exception("Failed to initialize Gemini SDK client")
            raise GeminiConfigurationError("Failed to initialize Gemini SDK client.") from exc

    def generate(self, prompt: str, model: str | None = None) -> str:
        """Generate text with Gemini using the only allowed model.

        The `model` argument is intentionally ignored to enforce
        `gemini-flash-latest` across the application.
        """
        _ = model
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty.")

        try:
            self.logger.info(
                "Calling Gemini generate_content",
                extra={"prompt_length": len(prompt), "model_name": self.ALLOWED_MODEL},
            )
            model_client = self._genai.GenerativeModel(self.ALLOWED_MODEL)
            response = model_client.generate_content(prompt)

            text = getattr(response, "text", "")
            if not text:
                raise GeminiGenerationError("Gemini response did not contain text output.")

            self.logger.info("Gemini generation completed", extra={"response_length": len(text)})
            return str(text)
        except GeminiGenerationError:
            self.logger.exception("Gemini returned an invalid generation response")
            raise
        except Exception as exc:
            self.logger.exception("Gemini generation request failed")
            raise GeminiGenerationError("Gemini generation request failed.") from exc
