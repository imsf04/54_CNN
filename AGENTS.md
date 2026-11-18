## Agent skills

### Issue tracker

Issues and specifications live in GitHub Issues for `imsf04/54_CNN`. See `docs/agents/issue-tracker.md`.

### Triage labels

This repository uses the default five triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. See `docs/agents/domain.md`.

## Engineering rules

- Follow the let-it-crash principle: fail loudly at the first invalid state.
- Do not add `try`/`except` blocks that catch, suppress, translate, or silently handle exceptions. Exceptions are evidence of a violated contract and must remain visible.
- Do not use nested `if`/`else` chains when a guard clause, mapping, comprehension, or polymorphic module expresses the rule more clearly.
- Prefer modern Python, Pythonic expressions, comprehensions, pure functions, and small deep modules with narrow interfaces.
- Follow the Zen of Python and keep domain rules explicit rather than hiding them in defensive plumbing.
- The layered modules under `src/price_trends/models`, `services`, `domain`, and `adapters` are the only formal interfaces. Do not add compatibility re-exports from deleted flat modules.
