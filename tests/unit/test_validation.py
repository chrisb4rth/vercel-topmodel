"""
Unit tests for the executor validation module.

Tests cover the three validation functions and the custom ValidationError exception:
- validate_query_length: rejects empty, whitespace-only, and oversized queries
- validate_language: rejects unsupported language codes
- validate_request_payload: rejects malformed payloads, missing fields, wrong types

Each test verifies both the error_code and message fields of the raised ValidationError
to ensure clients receive actionable, structured error information.
"""

import pytest

from executor.validation import (
    ValidationError,
    validate_language,
    validate_query_length,
    validate_request_payload,
)
from models import ChatRequest, Language


# ---------------------------------------------------------------------------
# ValidationError exception tests
# ---------------------------------------------------------------------------


class TestValidationError:
    """Tests for the custom ValidationError exception class."""

    def test_error_carries_code_and_message(self) -> None:
        """ValidationError stores both error_code and message as attributes."""

        error: ValidationError = ValidationError(
            error_code="test_error",
            message="Something went wrong",
        )

        assert error.error_code == "test_error"
        assert error.message == "Something went wrong"
        # The message is also the standard Exception string representation
        assert str(error) == "Something went wrong"

    def test_error_is_exception_subclass(self) -> None:
        """ValidationError can be caught as a generic Exception."""

        with pytest.raises(Exception):
            raise ValidationError(error_code="x", message="y")


# ---------------------------------------------------------------------------
# validate_query_length tests
# ---------------------------------------------------------------------------


class TestValidateQueryLength:
    """Tests for query length validation."""

    def test_rejects_empty_string(self) -> None:
        """Empty string has length 0, which is below the minimum of 1."""

        with pytest.raises(ValidationError) as exc_info:
            validate_query_length("")

        assert exc_info.value.error_code == "query_length_error"
        assert "1 and 2000" in exc_info.value.message

    def test_rejects_whitespace_only(self) -> None:
        """Whitespace-only strings are treated as empty after stripping."""

        with pytest.raises(ValidationError) as exc_info:
            validate_query_length("   \t\n  ")

        assert exc_info.value.error_code == "query_length_error"

    def test_rejects_oversized_query(self) -> None:
        """Queries exceeding 2000 characters are rejected."""

        oversized_query: str = "a" * 2001

        with pytest.raises(ValidationError) as exc_info:
            validate_query_length(oversized_query)

        assert exc_info.value.error_code == "query_length_error"
        assert "1 and 2000" in exc_info.value.message

    def test_accepts_minimum_length_query(self) -> None:
        """A single non-whitespace character is the minimum valid query."""

        # Should not raise
        validate_query_length("a")

    def test_accepts_maximum_length_query(self) -> None:
        """Exactly 2000 characters is the maximum valid query."""

        max_query: str = "b" * 2000

        # Should not raise
        validate_query_length(max_query)

    def test_accepts_typical_query(self) -> None:
        """A normal-length query passes validation without error."""

        validate_query_length("Was ist der Pfändungsfreibetrag?")


# ---------------------------------------------------------------------------
# validate_language tests
# ---------------------------------------------------------------------------


class TestValidateLanguage:
    """Tests for language validation."""

    def test_accepts_german(self) -> None:
        """German ('de') is a supported language."""

        result: Language = validate_language("de")

        assert result == Language.GERMAN

    def test_accepts_english(self) -> None:
        """English ('en') is a supported language."""

        result: Language = validate_language("en")

        assert result == Language.ENGLISH

    def test_rejects_unsupported_language(self) -> None:
        """French ('fr') is not supported and should be rejected."""

        with pytest.raises(ValidationError) as exc_info:
            validate_language("fr")

        assert exc_info.value.error_code == "unsupported_language"
        assert "German" in exc_info.value.message
        assert "English" in exc_info.value.message

    def test_rejects_empty_language_code(self) -> None:
        """An empty string is not a valid language code."""

        with pytest.raises(ValidationError) as exc_info:
            validate_language("")

        assert exc_info.value.error_code == "unsupported_language"

    def test_rejects_case_sensitive_mismatch(self) -> None:
        """Language codes are case-sensitive; 'DE' is not 'de'."""

        with pytest.raises(ValidationError) as exc_info:
            validate_language("DE")

        assert exc_info.value.error_code == "unsupported_language"


# ---------------------------------------------------------------------------
# validate_request_payload tests
# ---------------------------------------------------------------------------


class TestValidateRequestPayload:
    """Tests for full request payload validation."""

    def test_valid_payload_returns_chat_request(self) -> None:
        """A complete, valid payload produces a ChatRequest instance."""

        payload: dict = {
            "query": "Was ist der Pfändungsfreibetrag?",
            "session_id": "session-123",
            "language": "de",
        }

        result: ChatRequest = validate_request_payload(payload)

        assert result.query == "Was ist der Pfändungsfreibetrag?"
        assert result.session_id == "session-123"
        assert result.language == Language.GERMAN

    def test_valid_payload_defaults_language_to_german(self) -> None:
        """When language is omitted, it defaults to German."""

        payload: dict = {
            "query": "Test query",
            "session_id": "session-456",
        }

        result: ChatRequest = validate_request_payload(payload)

        assert result.language == Language.GERMAN

    def test_rejects_missing_query_field(self) -> None:
        """Payload without 'query' field is rejected."""

        payload: dict = {
            "session_id": "session-123",
            "language": "de",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_request_payload(payload)

        assert exc_info.value.error_code == "validation_error"
        assert "query" in exc_info.value.message

    def test_rejects_missing_session_id_field(self) -> None:
        """Payload without 'session_id' field is rejected."""

        payload: dict = {
            "query": "Test query",
            "language": "de",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_request_payload(payload)

        assert exc_info.value.error_code == "validation_error"
        assert "session_id" in exc_info.value.message

    def test_rejects_missing_both_required_fields(self) -> None:
        """Payload missing both required fields reports both in the error."""

        payload: dict = {"language": "de"}

        with pytest.raises(ValidationError) as exc_info:
            validate_request_payload(payload)

        assert exc_info.value.error_code == "validation_error"
        assert "query" in exc_info.value.message
        assert "session_id" in exc_info.value.message

    def test_rejects_non_string_query(self) -> None:
        """Query field must be a string, not an integer or other type."""

        payload: dict = {
            "query": 12345,
            "session_id": "session-123",
            "language": "de",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_request_payload(payload)

        assert exc_info.value.error_code == "validation_error"
        assert "query" in exc_info.value.message
        assert "string" in exc_info.value.message

    def test_rejects_non_string_session_id(self) -> None:
        """Session ID field must be a string."""

        payload: dict = {
            "query": "Test query",
            "session_id": 999,
            "language": "de",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_request_payload(payload)

        assert exc_info.value.error_code == "validation_error"
        assert "session_id" in exc_info.value.message

    def test_rejects_non_string_language(self) -> None:
        """Language field, if present, must be a string."""

        payload: dict = {
            "query": "Test query",
            "session_id": "session-123",
            "language": 42,
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_request_payload(payload)

        assert exc_info.value.error_code == "validation_error"
        assert "language" in exc_info.value.message

    def test_rejects_non_dict_payload(self) -> None:
        """A non-dictionary payload (e.g., a list) is rejected."""

        with pytest.raises(ValidationError) as exc_info:
            validate_request_payload(["not", "a", "dict"])  # type: ignore

        assert exc_info.value.error_code == "validation_error"
        assert "JSON object" in exc_info.value.message

    def test_rejects_empty_query_in_full_payload(self) -> None:
        """Full payload validation also enforces query length constraints."""

        payload: dict = {
            "query": "",
            "session_id": "session-123",
            "language": "de",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_request_payload(payload)

        assert exc_info.value.error_code == "query_length_error"

    def test_rejects_unsupported_language_in_full_payload(self) -> None:
        """Full payload validation also enforces language constraints."""

        payload: dict = {
            "query": "Test query",
            "session_id": "session-123",
            "language": "fr",
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_request_payload(payload)

        assert exc_info.value.error_code == "unsupported_language"

    def test_strips_whitespace_from_query(self) -> None:
        """The validated ChatRequest contains the stripped query."""

        payload: dict = {
            "query": "  Hello world  ",
            "session_id": "session-123",
            "language": "en",
        }

        result: ChatRequest = validate_request_payload(payload)

        assert result.query == "Hello world"
