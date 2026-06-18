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
    "reserved words. Prefer explicit JOIN ... ON over implicit joins. Select "
    "only the columns requested by the question; do not include helper columns "
    "used only for sorting or filtering. Preserve the direction of arithmetic "
    "comparisons exactly as written, for example 'A and B difference' means "
    "A minus B unless the question says otherwise. For dates, match the stored "
    "SQLite text format carefully and consider LIKE when the question omits "
    "fractional seconds."
)

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = (
    "Schema:\n{schema}\n\n"
    "Question:\n{question}\n\n"
    "Return one SQLite query that answers the question."
)


VERIFY_SYSTEM = (
    "You are a strict SQL reviewer. Given a question, the SQL that was "
    "generated, and the execution result, judge whether the result answers the "
    "question exactly, not merely plausibly. Respond with a single JSON object "
    "on one line: "
    '{{"ok": <true|false>, "issue": "<short reason>"}}. No prose, no fences.\n'
    "Mark ok=false when the execution errored; when zero rows were returned "
    "but the question names a specific entity, date, category, or expects a "
    "list/count; when the selected columns do not match the requested output "
    "exactly; when the SQL returns helper columns used only for ORDER BY; when "
    "a calculation reverses the requested direction; when a requested label "
    "such as yes/no, status, full name, address, or ID is not what the SQL "
    "returns; or when a literal value looks guessed instead of matching the "
    "question. If uncertain, mark ok=false and give a concrete fix. Mark "
    "ok=true only when columns, filters, aggregation, ordering, limit, and "
    "row shape all line up with the question. For highest, lowest, top, first, "
    "largest, or smallest questions, accept ORDER BY ... LIMIT 1 as exact even "
    "when the requested noun is plural, unless the question explicitly asks "
    "for all tied rows or all rows with the same maximum/minimum value."
)

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
    "the previous mistake. Use only tables and columns from the schema. Keep "
    "the output shape exact: return only requested columns, not sorting or "
    "debug columns. If the prior SQL returned zero rows for a named entity, "
    "date, or category, revise the literal/date predicate rather than adding "
    "irrelevant joins. If a computation was backwards, reverse only that "
    "operation and keep the rest of the query stable."
)

REVISE_USER = (
    "Schema:\n{schema}\n\n"
    "Question:\n{question}\n\n"
    "Previous SQL:\n{previous_sql}\n\n"
    "Previous execution result:\n{execution}\n\n"
    "Reviewer's issue:\n{issue}\n\n"
    "Return one corrected SQLite query that fixes the issue above."
)
