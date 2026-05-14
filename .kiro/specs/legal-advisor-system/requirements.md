# Requirements Document

## Introduction

The Legal Advisor System is an agentic chatbot designed to answer legal questions in the domain of banking operations, specifically around account seizures (Kontopfändungen) and insolvency proceedings (Insolvenzverfahren). The system follows a supervisor/sub-agent architecture where a supervisor layer orchestrates specialized sub-agents that handle specific question types. The system acts as an "executor" layer and is deployed via the Vercel AI Gateway.

## Glossary

- **Supervisor**: The top-level orchestration agent that receives user queries, classifies them, and delegates to appropriate Sub_Agents
- **Sub_Agent**: A specialized agent responsible for answering questions within a specific legal sub-domain (e.g., account seizures, insolvency proceedings)
- **Executor_Layer**: The runtime layer that manages agent execution, tool invocation, and response assembly
- **Query_Classifier**: The component within the Supervisor that determines which Sub_Agent should handle a given user query
- **Vercel_AI_Gateway**: The deployment platform providing API access, routing, and key management for the system
- **User**: A banking operations professional who submits legal questions to the system
- **Knowledge_Base**: The structured legal reference data that Sub_Agents consult to formulate answers
- **Session**: A single conversation thread between a User and the system, maintaining context across multiple exchanges

## Requirements

### Requirement 1: Query Reception and Classification

**User Story:** As a banking operations professional, I want to submit legal questions in natural language, so that I receive accurate answers without needing to know which sub-domain my question belongs to.

#### Acceptance Criteria

1. WHEN a User submits a query of between 1 and 2000 characters, THE Supervisor SHALL classify the query into one or more legal sub-domains within 2 seconds
2. WHEN a query spans multiple sub-domains, THE Supervisor SHALL identify all relevant sub-domains and delegate to the corresponding Sub_Agents
3. IF the Supervisor cannot classify a query into any known sub-domain, THEN THE Supervisor SHALL respond with a clarification request listing the supported sub-domain names
4. THE Supervisor SHALL accept queries in German and English language
5. IF a User submits a query in a language other than German or English, THEN THE Supervisor SHALL reject the query with an error message indicating the supported languages
6. IF a User submits an empty query or a query exceeding 2000 characters, THEN THE Supervisor SHALL reject the query with an error message indicating the acceptable query length range

### Requirement 2: Supervisor Orchestration

**User Story:** As a system operator, I want the supervisor to coordinate sub-agents effectively, so that complex multi-domain questions are answered coherently.

#### Acceptance Criteria

1. WHEN a query is classified, THE Supervisor SHALL delegate the query to the Sub_Agent or Sub_Agents matching the sub-domains identified by the Query_Classifier
2. WHEN multiple Sub_Agents return partial answers, THE Supervisor SHALL synthesize them into a single response that preserves all source references, removes duplicate information, and organizes content by sub-domain
3. WHILE a Sub_Agent is processing a query, THE Supervisor SHALL enforce a timeout of 30 seconds per Sub_Agent invocation
4. IF a Sub_Agent fails to respond within the timeout, THEN THE Supervisor SHALL return a partial response from the remaining Sub_Agents and include an indication of which sub-domain could not be resolved
5. IF all delegated Sub_Agents fail to respond within the timeout, THEN THE Supervisor SHALL return an error response indicating that no sub-domain could be resolved and listing the sub-domains that were attempted
6. WHEN a query requires delegation to multiple Sub_Agents, THE Supervisor SHALL invoke the Sub_Agents in parallel so that total response time does not exceed 30 seconds plus synthesis overhead

### Requirement 3: Account Seizure Sub-Agent

**User Story:** As a banking operations professional, I want to ask questions about account seizures (Kontopfändungen), so that I can handle seizure orders correctly and in compliance with legal requirements.

#### Acceptance Criteria

1. WHEN a query about account seizures is delegated, THE Account_Seizure_Agent SHALL provide an answer derived from the Knowledge_Base and cite the applicable German legal provisions (ZPO, PfÜB regulations) used to formulate the answer
2. IF the Account_Seizure_Agent provides a substantive legal answer, THEN THE Account_Seizure_Agent SHALL reference at least one specific legal paragraph (law abbreviation, section number) supporting the answer
3. IF the Account_Seizure_Agent cannot match the query to any provision in the Knowledge_Base or the query is too ambiguous to resolve, THEN THE Account_Seizure_Agent SHALL state which information is missing or unclear and decline to provide a speculative answer
4. THE Account_Seizure_Agent SHALL cover topics including seizure order processing, protected amounts (Pfändungsfreigrenzen), third-party debt orders, and priority of claims
5. IF a delegated query does not fall within the Account_Seizure_Agent's covered topics, THEN THE Account_Seizure_Agent SHALL return a response indicating the query is outside its scope so the Supervisor can re-route or inform the User

### Requirement 4: Insolvency Sub-Agent

**User Story:** As a banking operations professional, I want to ask questions about insolvency proceedings, so that I can handle insolvency-related account operations correctly.

#### Acceptance Criteria

1. WHEN a query about insolvency is delegated, THE Insolvency_Agent SHALL provide an answer derived from and citing provisions of the German Insolvency Statute (InsO) within 15 seconds
2. WHEN the answer relates to a specific InsO provision, THE Insolvency_Agent SHALL reference the applicable paragraph number (e.g., § 80 InsO) in its response
3. IF the Insolvency_Agent cannot identify a matching InsO provision or established legal interpretation for a query, THEN THE Insolvency_Agent SHALL state the limitation explicitly and indicate which aspect of the query could not be resolved, rather than speculating
4. THE Insolvency_Agent SHALL cover at minimum the following topics: account blocking upon insolvency filing (§ 89 InsO), insolvency administrator rights (§ 80 InsO), payment prohibitions (§ 82 InsO), and estate segregation (§ 35 InsO)
5. IF a delegated query does not fall within the Insolvency_Agent's covered topics, THEN THE Insolvency_Agent SHALL return a response indicating the query is outside its scope so the Supervisor can re-route or inform the User

### Requirement 5: Conversation Context Management

**User Story:** As a banking operations professional, I want the system to remember context within a conversation, so that I can ask follow-up questions without repeating background information.

#### Acceptance Criteria

1. WHILE a Session is active, THE Executor_Layer SHALL maintain the full conversation history (each exchange consisting of one user query and one system response) accessible to the Supervisor and Sub_Agents for the duration of that Session
2. WHEN a follow-up query references prior context, THE Supervisor SHALL include the conversation history from the current Session when delegating to Sub_Agents
3. THE Executor_Layer SHALL support a minimum of 20 exchanges (where one exchange equals one user query and one system response) per Session before context truncation occurs
4. IF a Session exceeds the context window limit, THEN THE Executor_Layer SHALL summarize earlier exchanges while preserving key entities (legal references, case identifiers, monetary amounts, and party names) and retain the summary for the remainder of the Session
5. IF context truncation or summarization occurs during a Session, THEN THE Executor_Layer SHALL indicate to the User that earlier context has been condensed and that repeating critical details may improve answer accuracy

### Requirement 6: Vercel AI Gateway Deployment

**User Story:** As a system operator, I want the system deployed via Vercel AI Gateway, so that I can leverage existing API key infrastructure and scalable hosting.

#### Acceptance Criteria

1. THE Executor_Layer SHALL expose a REST API compatible with the Vercel AI Gateway routing conventions
2. THE Executor_Layer SHALL authenticate incoming requests using API keys managed through the Vercel_AI_Gateway
3. IF a request is received without an API key, THEN THE Executor_Layer SHALL reject the request with HTTP 401 status and an error message indicating missing authentication credentials
4. IF a request is received with an invalid or expired API key, THEN THE Executor_Layer SHALL reject the request with HTTP 403 status and an error message indicating invalid credentials
5. THE Executor_Layer SHALL return responses in a streaming-compatible format and deliver the first token to the client within 3 seconds of request receipt
6. IF a request payload is malformed or missing required fields, THEN THE Executor_Layer SHALL reject the request with HTTP 400 status and an error message indicating the validation failure
7. IF the Executor_Layer receives requests exceeding its capacity, THEN THE Executor_Layer SHALL reject excess requests with HTTP 429 status and an error message indicating rate limit exceeded

### Requirement 7: Response Quality and Traceability

**User Story:** As a banking operations professional, I want answers that cite their legal sources, so that I can verify the information and use it in compliance documentation.

#### Acceptance Criteria

1. WHEN a Sub_Agent provides an answer that interprets, applies, or explains a legal provision, THE Sub_Agent SHALL include at least one legal source reference specifying the law name, paragraph number, and section number where available
2. WHEN a Sub_Agent provides an answer, THE Sub_Agent SHALL indicate a confidence qualifier: "high" if the answer maps to a directly applicable legal provision with explicit wording, "medium" if the answer is derived by analogy or from general provisions, or "low" if no directly matching provision is identified and the answer relies on interpretation
3. IF a Sub_Agent's confidence is low, THEN THE Sub_Agent SHALL recommend consulting a qualified legal professional
4. THE Supervisor SHALL preserve source references from Sub_Agents in the final synthesized response without alteration
5. IF a Sub_Agent cannot identify any legal source for a query that requires legal interpretation, THEN THE Sub_Agent SHALL explicitly state that no applicable legal provision was found and SHALL assign a confidence qualifier of "low"

### Requirement 8: Extensibility for Additional Sub-Agents

**User Story:** As a system operator, I want to add new sub-agents for additional legal domains without modifying the core supervisor logic, so that the system can grow over time.

#### Acceptance Criteria

1. THE Supervisor SHALL discover available Sub_Agents through a registry mechanism at startup, where each Sub_Agent entry provides a sub-domain identifier, a natural-language description of covered topics, and a list of supported query categories
2. WHEN a new Sub_Agent is registered in the registry, THE Supervisor SHALL include the new sub-domain in its classification logic upon the next system startup without code changes to the Supervisor
3. IF the registry is empty or unavailable at startup, THEN THE Supervisor SHALL log an error indicating no Sub_Agents are available and reject incoming queries with a message indicating the system is not ready
4. THE Executor_Layer SHALL define a standard interface that all Sub_Agents implement, including operations for receiving a classified query with conversation context, returning a response containing an answer body, legal source references, and a confidence qualifier
5. THE Executor_Layer SHALL provide a Sub_Agent template or base class that new Sub_Agents extend, pre-implementing the standard interface contract and response formatting requirements
