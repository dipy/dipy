# AGENTS Instructions

This file contains guidance for AI agents and other AI-assisted editors working
on DIPY. It complements the
[Automated Contributions Policy](.github/CONTRIBUTING.md#automated-contributions-policy),
which applies to every contribution.

## REQUIRED: AI usage disclosure

Every pull request description MUST fill the "AI Usage Disclosure" section of
the [pull request template](.github/PULL_REQUEST_TEMPLATE.md): list what the
AI tool was used for and name the tool.

Until a human has reviewed the changes, also add this note at the end of every
pull request description or work summary you generate:

> This pull request includes code written with the assistance of AI.
> The code has **not yet been reviewed** by a human.

This is a mandatory requirement, not optional. Pull requests without the
disclosure are closed without review.

## Working on an issue

Before working on any issue, run `gh issue view <number> --repo dipy/dipy` to
check its labels. Do not open a pull request against an issue labeled
`type:needs triage`. Such pull requests are closed without review until a
maintainer removes the label.

Never open issues or pull requests in a fully-automated way. A human must
review, understand and test every change, and be able to explain it on request.

## Project conventions

- Follow the [DIPY coding style](doc/devel/coding_style_guideline.rst): PEP8,
  numpy docstring style, `pathlib` for path manipulation.
- Prefix commit messages with one of the
  [DIPY commit codes](doc/devel/commit_codes.rst) (`BF:`, `NF:`, `RF:`,
  `OPT:`, `DOC:`, `TEST:`, `STYLE:`, ...).
- Add tests next to the module you change (e.g. `dipy/core/tests/`) and run
  them with `pytest` before submitting.
- Fill every section of the pull request template.

## Generated summaries

When generating a summary of your work:

- Describe the "why" of the changes, and why the proposed solution is the
  right one.
- Highlight the parts of the change that require careful review.
- Keep it short. More text is not better. Avoid flattery, avoid stating the
  obvious, avoid filler phrases, and prefer technical clarity over marketing
  tone.
