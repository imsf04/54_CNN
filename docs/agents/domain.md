# Domain Docs

How the engineering skills should consume this repository's domain documentation.

## Before exploring, read these

- `GLOSSARY.md` at the repository root, if it exists.
- `GLOSSARY-MAP.md` instead, if it exists.
- `docs/adr/` for decisions touching the area being changed.

If these files do not exist, proceed silently. Create them only when domain terms or architectural decisions are actually resolved.

## File structure

This is a single-context repository:

```
/
├── GLOSSARY.md
├── docs/adr/
└── src/
```

## Use the glossary's vocabulary

When naming a domain concept in an issue, proposal, refactor, or test, use the term defined in `GLOSSARY.md`. If a required concept is missing, record it as a domain-modeling gap.

## Flag ADR conflicts

If a proposal contradicts an existing ADR, surface the conflict explicitly instead of silently overriding it.
