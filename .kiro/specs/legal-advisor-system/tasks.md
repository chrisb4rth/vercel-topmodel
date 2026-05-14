# Implementation Plan: Legal Advisor System

## Overview

Implement a Python-based agentic chatbot for legal questions in German banking operations (account seizures, insolvency proceedings). The system uses a supervisor/sub-agent architecture with registry-based extensibility, parallel dispatch, streaming responses, and conversation context management. Deployed via Vercel AI Gateway with existing API key infrastructure.

## Tasks

- [ ] 1. Set up project structure, data models, and base interfaces
  - [x] 1.1 Create directory structure and module scaffolding
    - Create directories: `executor/`, `supervisor/`, `agents/`, `registry/`, `context/`, `tests/property/`, `tests/unit/`, `tests/integration/`
    - Add `__init__.py` files for all packages
    - Set up `pyproject.toml` or `requirements.txt` with dependencies (hypothesis, pytest, pytest-asyncio, pydantic or dataclasses)
    - _Requirements: 8.4, 8.5_

  - [ ] 1.2 Implement data models (`models.py`)
    - Implement all dataclasses: `Language`, `ConfidenceLevel`, `ChatRequest`, `LegalReference`, `SubAgentMetadata`, `ClassificationResult`, `SubAgentResponse`, `SubAgentResult`, `SynthesizedResponse`, `Exchange`, `ConversationContext`, `StreamChunk`
    - Ensure full typing and docstrings for all fields
    - _Requirements: 7.1, 7.2, 7.5_

  - [-] 1.3 Implement the base sub-agent interface (`agents/base.py`)
    - Create `BaseSubAgent` abstract class with `handle_query` and `get_metadata` abstract methods
    - Define type signatures matching the design document
    - _Requirements: 8.4, 8.5_

- [ ] 2. Implement Sub-Agent Registry
  - [~] 2.1 Implement `SubAgentRegistry` class (`registry/registry.py`)
    - Implement `register()`, `get_agents_for_domains()`, `get_all_metadata()` methods
    - Store agents in a dict keyed by `domain_id`
    - Raise appropriate errors for duplicate registrations
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ]* 2.2 Write property test for registry discovery completeness
    - **Property 15: Registry discovery completeness**
    - **Validates: Requirements 8.1, 8.2**

- [ ] 3. Implement Query Validation and Classification
  - [~] 3.1 Implement query validation logic (`executor/validation.py`)
    - Validate query length (1–2000 characters), reject empty or oversized queries with descriptive error
    - Validate language detection (German/English only), reject unsupported languages
    - Validate request payload structure (required fields: query, session_id)
    - _Requirements: 1.4, 1.5, 1.6, 6.6_

  - [ ]* 3.2 Write property test for query length validation
    - **Property 1: Query length validation**
    - **Validates: Requirements 1.6**

  - [ ]* 3.3 Write property test for language validation
    - **Property 2: Language validation**
    - **Validates: Requirements 1.4, 1.5**

  - [ ]* 3.4 Write property test for malformed request rejection
    - **Property 14: Malformed request rejection**
    - **Validates: Requirements 6.6**

  - [~] 3.5 Implement query classifier (`supervisor/classifier.py`)
    - Classify queries into one or more legal sub-domains based on registry metadata
    - Return `ClassificationResult` with domain_ids, confidence, and detected language
    - If no domain matches, return empty domain_ids so supervisor can list available sub-domains
    - _Requirements: 1.1, 1.2, 1.3_

  - [ ]* 3.6 Write property test for classification producing valid registered domains
    - **Property 3: Classification produces valid registered domains**
    - **Validates: Requirements 1.1, 1.3**

- [~] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement Parallel Dispatcher
  - [~] 5.1 Implement `ParallelDispatcher` (`supervisor/dispatcher.py`)
    - Dispatch queries to multiple sub-agents concurrently using `asyncio.gather` with per-agent timeouts
    - Return `SubAgentResult` for each agent (success, timeout, or error)
    - Enforce 30-second per-agent timeout
    - _Requirements: 2.1, 2.3, 2.6_

  - [ ]* 5.2 Write property test for dispatch matching classification
    - **Property 4: Dispatch matches classification**
    - **Validates: Requirements 1.2, 2.1**

  - [ ]* 5.3 Write property test for graceful degradation on partial timeout
    - **Property 5: Graceful degradation on partial timeout**
    - **Validates: Requirements 2.4, 2.5**

- [ ] 6. Implement Response Synthesis
  - [~] 6.1 Implement `ResponseSynthesizer` (`supervisor/synthesizer.py`)
    - Merge multiple `SubAgentResponse` objects into a single `SynthesizedResponse`
    - Preserve all legal references from all sub-agent responses without alteration
    - Remove duplicate information, organize content by sub-domain
    - Set `recommend_professional=True` if any sub-agent has LOW confidence
    - Populate `unresolved_domains` for timed-out or errored agents
    - _Requirements: 2.2, 2.4, 7.3, 7.4_

  - [ ]* 6.2 Write property test for reference preservation through synthesis
    - **Property 6: Reference preservation through synthesis**
    - **Validates: Requirements 2.2, 7.4**

- [ ] 7. Implement Sub-Agents
  - [~] 7.1 Implement Account Seizure Agent (`agents/account_seizure.py`)
    - Extend `BaseSubAgent`, implement `handle_query` and `get_metadata`
    - Cover topics: seizure order processing, protected amounts (Pfändungsfreigrenzen), third-party debt orders, priority of claims
    - Cite ZPO and PfÜB provisions in responses
    - Set `is_out_of_scope=True` for queries outside covered topics
    - Assign confidence levels based on provision match quality
    - Include `limitation_note` when no provision matches
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [~] 7.2 Implement Insolvency Agent (`agents/insolvency.py`)
    - Extend `BaseSubAgent`, implement `handle_query` and `get_metadata`
    - Cover topics: account blocking (§ 89 InsO), administrator rights (§ 80 InsO), payment prohibitions (§ 82 InsO), estate segregation (§ 35 InsO)
    - Cite InsO provisions in responses
    - Set `is_out_of_scope=True` for queries outside covered topics
    - Assign confidence levels based on provision match quality
    - Include `limitation_note` when no provision matches
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ]* 7.3 Write property test for substantive answers citing legal sources
    - **Property 7: Substantive answers always cite legal sources**
    - **Validates: Requirements 3.1, 3.2, 4.1, 4.2, 7.1**

  - [ ]* 7.4 Write property test for confidence qualifier always present
    - **Property 8: Confidence qualifier is always present**
    - **Validates: Requirements 7.2**

  - [ ]* 7.5 Write property test for low confidence triggering professional consultation
    - **Property 9: Low confidence triggers professional consultation recommendation**
    - **Validates: Requirements 7.3**

  - [ ]* 7.6 Write property test for unresolvable queries stating limitation
    - **Property 10: Unresolvable queries state limitation with LOW confidence**
    - **Validates: Requirements 3.3, 4.3, 7.5**

  - [ ]* 7.7 Write property test for out-of-scope queries flagged
    - **Property 11: Out-of-scope queries are flagged**
    - **Validates: Requirements 3.5, 4.5**

- [~] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement Conversation Context Management
  - [~] 9.1 Implement `ContextStore` (`context/store.py`)
    - Implement `get_context()`, `append_exchange()`, `summarize_if_needed()` methods
    - Store per-session conversation history (minimum 20 exchanges before truncation)
    - Implement summarization that preserves key entities (legal references, case identifiers, monetary amounts, party names)
    - Set `is_truncated=True` and populate `preserved_entities` after summarization
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 9.2 Write property test for context round-trip preservation
    - **Property 12: Context round-trip preservation**
    - **Validates: Requirements 5.1, 5.2, 5.3**

  - [ ]* 9.3 Write property test for summarization preserving key entities
    - **Property 13: Summarization preserves key entities**
    - **Validates: Requirements 5.4, 5.5**

- [ ] 10. Implement Supervisor Orchestration
  - [~] 10.1 Implement `Supervisor` class (`supervisor/supervisor.py`)
    - Wire together: registry, classifier, dispatcher, synthesizer, context store
    - Implement `process_query()` orchestrating the full pipeline: validate → classify → dispatch → synthesize → stream
    - Inject conversation context when delegating to sub-agents
    - Handle edge cases: empty registry (reject with 503), all timeouts (return 504-style error), partial timeouts (graceful degradation)
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.4, 2.5, 8.3_

- [ ] 11. Implement Executor Layer and Streaming
  - [~] 11.1 Implement `ExecutorLayer` (`executor/executor.py`)
    - Expose REST API compatible with Vercel AI Gateway routing conventions
    - Implement `handle_request()`, `authenticate()`, `validate_request()` methods
    - Return streaming SSE responses, deliver first token within 3 seconds
    - Implement authentication via API key validation
    - Return appropriate HTTP status codes: 400, 401, 403, 429, 503, 504
    - Implement rate limiting for excess requests
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 11.2 Write unit tests for authentication and error responses
    - Test HTTP 401 for missing API key
    - Test HTTP 403 for invalid/expired API key
    - Test HTTP 429 for rate limit exceeded
    - _Requirements: 6.3, 6.4, 6.7_

  - [ ]* 11.3 Write unit tests for streaming format compliance
    - Verify SSE chunk format
    - Verify first-token latency contract
    - _Requirements: 6.5_

- [ ] 12. Integration and Wiring
  - [~] 12.1 Wire all components together at application entry point
    - Create application startup: instantiate registry, register sub-agents, create context store, supervisor, executor
    - Ensure registry discovery happens at startup
    - Configure Vercel AI Gateway compatibility
    - _Requirements: 6.1, 6.2, 8.1, 8.2_

  - [ ]* 12.2 Write integration tests for full pipeline
    - Test end-to-end: submit query → classify → dispatch → synthesize → stream response
    - Test multi-domain query spanning seizure + insolvency
    - Test conversation context carried across exchanges
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.6, 5.1, 5.2_

- [~] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (15 properties defined in design)
- Unit tests validate specific examples and edge cases
- The system uses Python with asyncio for concurrent sub-agent dispatch
- Hypothesis is used for property-based testing with minimum 100 examples per property
- All sub-agents implement the `BaseSubAgent` interface for registry compatibility

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "3.1"] },
    { "id": 3, "tasks": ["2.2", "3.2", "3.3", "3.4", "3.5"] },
    { "id": 4, "tasks": ["3.6", "5.1"] },
    { "id": 5, "tasks": ["5.2", "5.3", "6.1"] },
    { "id": 6, "tasks": ["6.2", "7.1", "7.2"] },
    { "id": 7, "tasks": ["7.3", "7.4", "7.5", "7.6", "7.7", "9.1"] },
    { "id": 8, "tasks": ["9.2", "9.3", "10.1"] },
    { "id": 9, "tasks": ["11.1"] },
    { "id": 10, "tasks": ["11.2", "11.3", "12.1"] },
    { "id": 11, "tasks": ["12.2"] }
  ]
}
```
