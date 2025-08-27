name: "Bug Report \U0001F41B"
description: Report an error or unexpected behavior
labels: ["type:bug"]
body:
  - type: markdown
    attributes:
      value: |
        **Please review [our guidelines to post an issue](https://github.com/dipy/dipy/discussions/3585) before opening a new issue.**

  - type: textarea
    attributes:
      label: Summary
      description: |
        A clear and concise description of the bug, including a minimal reproducible example.
        If we cannot reproduce the bug, it is unlikely that we will be able to help you.

        Please, include the full output of DIPY with the complete error message.
    validations:
      required: true

  - type: input
    attributes:
      label: Platform
      description: What operating system and architecture are you using? (e.g. `uname -orsm`)
      placeholder: e.g., macOS 14 arm64, Windows 11 x86_64, Ubuntu 20.04 amd64
    validations:
      required: true

  - type: input
    attributes:
      label: DIPY version
      description: What version of DIPY are you using? (e.g. `pip show dipy`)
      placeholder: e.g., dipy 1.11.0 (or 2cec3c8 if a commit is appropriate)
    validations:
      required: true

  - type: input
    attributes:
      label: Environment
      description: What are the package versions, if relevant, that are causing the issue? (e.g. `pip freeze | grep numpy`)
      placeholder: e.g., numpy 2.0.0
    validations:
      required: false

  - type: input
    attributes:
      label: Python version
      description: What version of Python are you using? (see `python --version`)
      placeholder: e.g., Python 3.12.6
    validations:
      required: true

  - type: textarea
    attributes:
      label: Additional information
      description: Any additional information, configuration or data that might be necessary to reproduce the issue.
    validations:
      required: false
