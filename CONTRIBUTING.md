# Contributing Workflow

This repository is set up so CI runs on both `push` and `pull_request`.

## Recommended Solo Workflow

1. Create a feature branch from `main`.
2. Make changes and run:

```bash
pytest -q -m "not verification"
pytest -q -m verification
```

3. Push the branch and wait for GitHub Actions to pass.
4. If the change is substantial or you want a merge checkpoint, open a PR from your branch into `main`.
5. Merge only after CI is green.

## When A PR Is Still Useful For Solo Work

- it gives you a clean review boundary before merge
- it preserves a discussion thread for why a change was made
- it makes baseline changes and CI results easier to inspect later
- it reduces the chance of merging an unreviewed experiment into `main`

## When You Can Skip A PR

- the change is local-only and not meant for `main`
- you are iterating quickly and using branch pushes only as temporary checkpoints

Even in solo development, keep the branch + CI workflow. It gives you most of the safety with very little extra cost.
