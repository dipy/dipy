name: "Feature Request \U0001F4A1"
description: "Suggest a feature request or idea to improve DIPY"
labels: ["type:new feature"]
body:
  - type: markdown
    attributes:
      value: |
        **Please review [our guidelines to post an issue](https://github.com/dipy/dipy/discussions/3585) before opening a new issue.**

  - type: textarea
    attributes:
      label: Summary
      description: |
        Description of the feature request, and how adding that feature would benefit a particular case or why that
        feature is necessary or not covered by the current up-to-date version.
    validations:
      required: true

  - type: input
    attributes:
      label: DIPY version
      description: What version of DIPY are you using? (e.g. `pip show dipy`)
      placeholder: e.g., dipy 1.11.0 (or 2cec3c8 if a commit is appropriate)
    validations:
      required: true

  - type: textarea
    attributes:
      label: Additional information
      description: Any additional information, configuration or data that might be necessary to reproduce the issue.
    validations:
      required: false
