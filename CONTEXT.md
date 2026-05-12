# Ubiquitous Language — conjecture

> Maintained by the Director. Suggestions via `/suggest-terminology` skill or the webUI Context browser.
> All prompt templates, evaluator names, and benchmark outputs should use these canonical terms.

---

## claim

**Definition:** The atomic unit of knowledge in Conjecture — a single, specific statement with a confidence score, epistemological type, and bidirectional relationships; never a fact, always provisional.

**Aliases:** knowledge claim

**Scope:** conjecture

**Status:** canonical

---

## confidence score

**Definition:** A float in [0.0, 1.0] attached to every claim representing the strength of supporting evidence, not the probability the claim is true; even a score of 0.99 does not make a claim a fact.

**Aliases:** confidence, confidence level

**Scope:** conjecture

**Status:** canonical

---

## claim type

**Definition:** The epistemological category assigned to a claim, drawn from the nine canonical values: IMPRESSION, ASSUMPTION, OBSERVATION, CONJECTURE, CONCEPT, EXAMPLE, GOAL, REFERENCE, ASSERTION; a claim may carry multiple types.

**Scope:** conjecture

**Status:** canonical

---

## claim graph

**Definition:** The directed acyclic graph (DAG) formed by all claims and their `supers`/`subs` relationships, which the framework traverses to build reasoning context and propagate dirty-flag cascades.

**Aliases:** claim DAG, claim graph

**Scope:** conjecture

**Status:** canonical

---

## supers / subs

**Definition:** Bidirectional relationship fields on every claim: `subs` are claims that provide evidence FOR this claim (children, toward leaves); `supers` are claims this claim provides evidence FOR (parents, toward root); decomposition produces subs, synthesis builds toward supers.

**Aliases:** super-claims, sub-claims, parent claims, supporting claims

**Scope:** conjecture

**Status:** canonical

---

## root context

**Definition:** The entire conversation (user and framework messages) stored as a single claim; all other claims in a session are subs of the root context, and root similarity measures how relevant any claim is to the current conversation.

**Scope:** conjecture

**Status:** canonical

---

## decomposition

**Definition:** The process by which the LLM breaks compound input (a user prompt, a complex question, or an existing claim) into constituent sub-claims, making implicit assumptions and sub-problems explicit and traceable; decomposition is how the framework reasons, not a separate preprocessing step.

**Aliases:** claim decomposition, input decomposition

**Scope:** conjecture

**Status:** canonical

---

## dirty flag

**Definition:** A boolean marker on a claim indicating it needs re-evaluation, set automatically when a claim's content, confidence, or relationships change; dirty status cascades unidirectionally toward `supers` (never to `subs`), ensuring parent conclusions are re-evaluated when evidence changes.

**Aliases:** dirty, dirty bit

**Scope:** conjecture

**Status:** canonical

---

## evaluation priority tuple

**Definition:** The ordered triple [dirty, confidence, root_similarity] used to select which dirty claims are evaluated first; dirty claims with low confidence and high relevance to the root context are processed earliest.

**Scope:** conjecture

**Status:** canonical

---

## hallucination

**Definition:** A model response that presents a low- or zero-evidence belief as a high-confidence assertion; in Conjecture, hallucinations are detectable as confidence gaps — claims asserted at high confidence that lack supporting sub-claims or contradict existing evidence in the graph.

**Scope:** conjecture

**Status:** canonical

---

## verified claim

**Definition:** A claim that has passed at least one tier of the fact-checking pipeline (self-consistency, vector search, or live web) and been marked with an updated confidence score reflecting that verification; verified claims are still provisional, not facts.

**Aliases:** validated claim

**Scope:** conjecture

**Status:** canonical

---

## fact-checking pipeline

**Definition:** The tiered verification architecture (Tier 1: self-consistency against existing claims; Tier 2: vector-search similarity; Tier 3: live web search) that attempts to confirm or contradict a claim using progressively more expensive evidence sources.

**Aliases:** verification pipeline

**Scope:** conjecture

**Status:** canonical

---

## provenance

**Definition:** The traceable chain or tree of supporting claims that leads from any conclusion back to its primary evidence sources; users can audit any claim's provenance via the support-tree view.

**Aliases:** claim provenance, evidence chain

**Scope:** conjecture

**Status:** canonical

---

## grounding

**Definition:** The act of anchoring a claim to external, retrievable evidence (web search results, database records, or prior session claims) rather than relying solely on the model's parametric memory; grounding is the primary defense against hallucination.

**Scope:** conjecture

**Status:** canonical

---

## three-prompt architecture

**Definition:** The production reasoning mode in which the framework issues three coordinated LLM calls — decompose, reason with claim context, respond — used for hard reasoning tasks (BBH, complex math); requires 70B+ models and must not be applied to recall or commonsense tasks.

**Aliases:** three-prompt, 3-prompt

**Scope:** conjecture

**Status:** canonical

---

## cot_lite

**Definition:** A lightweight chain-of-thought prompt strategy that adds minimal step-by-step scaffolding without full decomposition; used for recall and commonsense tasks (MMLU, TruthfulQA, HellaSwag) where full decomposition causes regressions.

**Aliases:** lightweight CoT, direct-with-steps

**Scope:** conjecture

**Status:** canonical

---

## task-type routing

**Definition:** The production requirement to classify every incoming query by task type (hard reasoning vs. recall/commonsense) before selecting a prompting strategy; three-prompt for hard reasoning, cot_lite for recall; omitting routing causes measurable accuracy regressions on recall tasks.

**Aliases:** query routing, prompt routing

**Scope:** conjecture

**Status:** canonical

---

## benchmark task

**Definition:** A single scored problem drawn from a standardized dataset (GSM8K, MMLU, BBH, TruthfulQA, HellaSwag, MathQA, LogiQA, BoolQ, Winogrande, etc.) used to measure Conjecture's accuracy relative to a direct-prompting baseline on the same model.

**Scope:** conjecture

**Status:** canonical

---

## evaluator

**Definition:** A component (human rubric, automated script, or LLM judge) that assigns a score to a claim or a model response; in the benchmark suite the evaluator compares extracted answers against ground truth and records accuracy in STATS.yaml.

**Scope:** conjecture

**Status:** canonical

---

## rubric

**Definition:** A weighted scoring instrument with explicit criteria and point values used to evaluate the quality of a Conjecture implementation, prompt template, or LLM integration (e.g., the 50-point LLM evaluation rubric or the Phase 3 architecture rubric).

**Scope:** conjecture

**Status:** canonical

---

## answer extraction

**Definition:** The pattern-matching process that parses a raw LLM response to identify the final answer in the format expected by a benchmark (e.g., `####` for GSM8K, `\boxed{}` for MATH, A–D for MMLU); extraction failures produce false accuracy readings and must use the unified extraction module.

**Scope:** conjecture

**Status:** canonical

---

## direct prompting baseline

**Definition:** The accuracy score obtained by sending a benchmark task to a model with no Conjecture enhancement (no decomposition, no claim context, no structured scaffolding); every Conjecture strategy is measured against this baseline, and regressions below it are not acceptable in production.

**Aliases:** direct baseline, passthrough baseline

**Scope:** conjecture

**Status:** canonical

---

## domain-adaptive prompting

**Definition:** The practice of detecting the problem domain (mathematical, logical, commonsense, verification) and selecting a specialized system prompt matched to that domain before generating a response; demonstrated a 100% improvement in mathematical reasoning in Cycle 1.

**Aliases:** domain-adaptive reasoning, problem-type detection

**Scope:** conjecture

**Status:** canonical

---

## halt-or-explore decision

**Definition:** The LLM-driven choice at each iteration of the reasoning loop to either emit a `respond_to_user` tool call (halt) or create new claims to investigate further (explore); the decision is not system-imposed — the LLM halts when it judges evidence sufficient.

**Aliases:** halt or explore, reasoning halt

**Scope:** conjecture

**Status:** canonical

---

## persistent session

**Definition:** A single Conjecture session kept open across all test cases in a benchmark run so that claims created from early problems accumulate in the graph and can enhance later queries via semantic retrieval; required by O-0006 for benchmark suite runs.

**Aliases:** benchmark session, accumulated session

**Scope:** conjecture

**Status:** canonical

---

## counter-claim

**Definition:** A claim created by the LLM to represent an alternative position, correction, or disproof of an existing claim; counter-claims are implicit LLM behavior when the model questions the validity of a claim, and the framework must treat them as first-class claims subject to the same confidence and verification rules.

**Scope:** conjecture

**Status:** canonical

---
