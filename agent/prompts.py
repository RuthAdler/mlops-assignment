"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = (
    "You are a SQL expert translating natural-language questions into a single "
    "SQLite query against the provided schema. Output ONLY the SQL inside a "
    "```sql ... ``` fenced block - no commentary, no multiple queries, no "
    "trailing prose. Use only tables and columns that appear in the schema, "
    "and double-quote identifiers when they contain unusual characters or "
    "reserved words. Prefer explicit JOIN ... ON over implicit joins."
)

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = (
    "Schema:\n{schema}\n\n"
    "Question:\n{question}\n\n"
    "Return one SQLite query that answers the question."
)


VERIFY_SYSTEM = (
    "You are a careful SQL reviewer. Given a question, the SQL that was "
    "generated, and the execution result, judge whether the result plausibly "
    "answers the question. Respond with a single JSON object on one line: "
    '{{"ok": <true|false>, "issue": "<short reason>"}}. No prose, no fences.\n'
    "Mark ok=false when the execution errored, when zero rows were returned "
    "but the question implies rows should exist, when the returned columns "
    "clearly do not match what was asked, or when the rows obviously "
    "contradict the question. Otherwise mark ok=true with an empty issue."
)

# Available placeholders: {question}, {sql}, {execution}
VERIFY_USER = (
    "Question:\n{question}\n\n"
    "SQL:\n{sql}\n\n"
    "Execution result:\n{execution}\n\n"
    "Reply with only the JSON object."
)


REVISE_SYSTEM = (
    "You are a SQL expert revising a prior attempt that did not correctly "
    "answer the question. Output ONLY the revised SQL inside a ```sql ... ``` "
    "fenced block. Address the reviewer's issue directly and do not repeat "
    "the previous mistake. Use only tables and columns from the schema."
)

# Available placeholders: {schema}, {question}, {previous_sql}, {execution}, {issue}
REVISE_USER = (
    "Schema:\n{schema}\n\n"
    "Question:\n{question}\n\n"
    "Previous SQL:\n{previous_sql}\n\n"
    "Previous execution result:\n{execution}\n\n"
    "Reviewer's issue:\n{issue}\n\n"
    "Return one corrected SQLite query that fixes the issue above."
)
