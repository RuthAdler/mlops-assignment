# Project Guidelines

See `.cursor/rules/project.mdc` for full coding standards and safety rules.

## Critical safety rules
- Ask for my permission before accessing or using in any form any
  authentication data (SSH keys, passwords, tokens, etc.).
- Ask for my permission before running `git commit`, `git push`, or any
  command that creates a commit or rewrites history. Show me what will be
  committed first, then wait for my confirmation.

## Key conventions
- PEP 8 with 120-char lines; PEP 287 docstrings; type all function args.
- Prefer `NamedTuple`/`dataclass` over composite types or fixed-key dicts.
- Functions < 40 lines, files < 300 lines, single responsibility.
- pytest for tests; don't test private (`_`-prefixed) members.
- Don't invent APIs — check docs. Don't write code you don't understand.
