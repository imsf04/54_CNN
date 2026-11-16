# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues in `imsf04/54_CNN`. Use the `gh` CLI for all operations.

## Conventions

- Create an issue with `gh issue create --title "..." --body "..."`.
- Read an issue with `gh issue view <number> --json number,title,body,labels,comments`.
- List and filter issues with `gh issue list`.
- Comment with `gh issue comment <number> --body "..."`.
- Apply or remove labels with `gh issue edit <number> --add-label "..."` / `--remove-label "..."`.
- Close an issue after posting the resolution comment.

## Pull requests as a triage surface

PRs as a request surface: no.

## When a skill says "publish to the issue tracker"

Create a GitHub issue in `imsf04/54_CNN`.

## Wayfinding

If a future skill uses a wayfinding map, use one GitHub issue labelled `wayfinder:map` and link child tickets as GitHub sub-issues when supported.
