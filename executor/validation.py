"""
Query and request validation for the Legal Advisor System executor layer.

This module implements input validation that runs before any query reaches
the supervisor or sub-agents. It enforces constraints on query length,
language support, and request payload structure. Validation failures are
raised as typed exceptions carrying structured error information (error_code
and message) that the executor layer translates into HTTP 400 responses.

Validation order:
1. Payload structure — ensures required fields are present and correctly typed
2. Query length — ensures the query is between 1 and 2000 characters
3. Language — ensures the declared language is one of the supported values (de, en)

This ordering means that structural issues are caught first (cheapest check),
followed by content constraints, so that error messages are as specific as
possible for the client.

Data Flow:
    raw dict payload → validate_request_payload() → validate_query_length()
    → validate_language() → ChatRequest (validated)
"""

from models import ChatRequest, Language


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum allowed query length (inclusive)
QUERY_MIN_LENGTH: int = 1

# Maximum allowed query length (inclusive)
QUERY_MAX_LENGTH: int = 2000

# Set of valid language codes accepted by the system
SUPPORTED_LANGUAGE_CODES: set[str] = {lang.value for lang in Language}


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """
    Raised when an incoming request fails validation.

    Carries structured error information that the executor layer maps
    directly to an HTTP 400 JSON response body. The error_code enables
    clients to programmatically distinguish between different validation
    failure categories without parsing the human-readable message.

    Attributes:
        error_code: Machine-readable error identifier used by clients for
            programmatic error handling (e.g., "query_length_error",
            "unsupported_language", "validation_error").
        message: Human-readable description of the validation failure,
            suitable for display to end users or inclusion in API responses.
    """

    def __init__(
        self,
        error_code: str,
        message: str,
    ) -> None:
        """
        Initialize a ValidationError with structured error information.

        Args:
            error_code: Machine-readable error identifier (e.g., "query_length_error").
            message: Human-readable description of the validation failure.
        """

        super().__init__(message)
        self.error_code: str = error_code
        self.message: str = message


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------


def validate_query_length(
    query: str,
) -> None:
    """
    Validate that the query string is within the acceptable length range.

    Rejects empty strings and strings exceeding 2000 characters. Whitespace-only
    strings are treated as empty because they carry no semantic content for
    classification or sub-agent processing.

    Args:
        query: The raw query string extracted from the request payload.

    Raises:
        ValidationError: If the query length is outside the 1–2000 character range,
            with error_code "query_length_error".

    Returns:
        None
    """

    # Strip whitespace to catch queries that are technically non-empty but
    # contain no meaningful content for legal classification
    stripped_query: str = query.strip()
    query_length: int = len(stripped_query)

    if query_length < QUERY_MIN_LENGTH or query_length > QUERY_MAX_LENGTH:
        raise ValidationError(
            error_code="query_length_error",
            message="Query must be between 1 and 2000 characters",
        )

    return None


def validate_language(
    language: str,
) -> Language:
    """
    Validate that the provided language code is supported by the system.

    The system only supports German ("de") and English ("en") because the
    legal knowledge base covers German banking law and the user base operates
    in these two languages. Queries in other languages would produce unreliable
    classification results and potentially hallucinated legal citations.

    Args:
        language: The language code string from the request payload
            (expected values: "de" or "en").

    Raises:
        ValidationError: If the language code is not "de" or "en",
            with error_code "unsupported_language".

    Returns:
        The corresponding Language enum member for downstream use.
    """

    if language not in SUPPORTED_LANGUAGE_CODES:
        raise ValidationError(
            error_code="unsupported_language",
            message="Supported languages: German, English",
        )

    # Convert the raw string to the typed enum for use in downstream components
    validated_language: Language = Language(language)

    return validated_language


def validate_request_payload(
    payload: dict,
) -> ChatRequest:
    """
    Validate the complete request payload structure and content.

    Performs all validation steps in sequence: structural checks first (required
    fields, correct types), then content checks (query length, language). This
    ordering ensures the most fundamental issues are reported first, avoiding
    confusing errors about content when the structure itself is broken.

    Args:
        payload: The raw request dictionary parsed from the incoming JSON body.
            Expected to contain keys "query" (str), "session_id" (str), and
            "language" (str).

    Raises:
        ValidationError: If any validation step fails. The error_code will be:
            - "validation_error" for missing/maltyped fields
            - "query_length_error" for query length violations
            - "unsupported_language" for unsupported language codes

    Returns:
        A fully validated ChatRequest instance that downstream components
        can trust satisfies all input constraints.
    """

    # --- Step 1: Verify payload is a dictionary ---
    if not isinstance(payload, dict):
        raise ValidationError(
            error_code="validation_error",
            message="Request payload must be a JSON object",
        )

    # --- Step 2: Check required fields are present ---
    missing_fields: list[str] = []

    if "query" not in payload:
        missing_fields.append("query")
    if "session_id" not in payload:
        missing_fields.append("session_id")

    if missing_fields:
        # Join field names for a descriptive error listing all missing fields at once
        fields_description: str = ", ".join(missing_fields)
        raise ValidationError(
            error_code="validation_error",
            message=f"Missing required fields: {fields_description}",
        )

    # --- Step 3: Check field types ---
    if not isinstance(payload["query"], str):
        raise ValidationError(
            error_code="validation_error",
            message="Field 'query' must be a string",
        )

    if not isinstance(payload["session_id"], str):
        raise ValidationError(
            error_code="validation_error",
            message="Field 'session_id' must be a string",
        )

    # Language field is optional in the payload — default to German if absent,
    # but if present it must be a string
    raw_language: str = payload.get("language", "de")

    if not isinstance(raw_language, str):
        raise ValidationError(
            error_code="validation_error",
            message="Field 'language' must be a string",
        )

    # --- Step 4: Validate query length ---
    validate_query_length(payload["query"])

    # --- Step 5: Validate language ---
    validated_language: Language = validate_language(raw_language)

    # --- Step 6: Construct validated request ---
    validated_request: ChatRequest = ChatRequest(
        query=payload["query"].strip(),
        session_id=payload["session_id"],
        language=validated_language,
    )

    return validated_request
