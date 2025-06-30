name: "Infrastructure Impact Report \U0001F4D0"
description: "File an infrastructure support report to improve DIPY's compatibility or build system"
labels: ["type:infrastructure"]
body:
  - type: markdown
    attributes:
      value: |
        **Please review [our guidelines to post an issue](https://github.com/dipy/dipy/discussions/3585) before opening a new issue.**

  - type: textarea
    attributes:
      label: Summary
      description: |
        A clear and concise description of the compatibility issue with an OS, Python version or another package or the
        changes proposed to improve the build system in DIPY, including a minimal reproducible example.

        Please, explain whate you expect the infrastructure to be, and what actually happens without the proposed
        change. Include the relevant error trace (if applicable).
    validations:
      required: true

  - type: textarea
    attributes:
      label: Impact analysis
      description: |
        Analysis of the benefits/downsides of the current design, and the benefits/downsides of the proposed
        infrastructure.
    validations:
      required: false

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
