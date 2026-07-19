# Security policy

## Supported versions

Security fixes are applied to the latest version of the `main` branch. Older
commits, forks, experimental branches, and modified model packages are not
supported.

TurboFieldfare is a research project. It is not intended for production,
multi-user, or security-critical deployments.

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/drumih/turbo-fieldfare/security/advisories/new)
instead.

Include:

- a description of the vulnerability;
- the affected commit or version;
- reproduction steps or a minimal proof of concept;
- the expected and observed behavior;
- the potential security impact; and
- a suggested mitigation, if known.

Do not include personal data, credentials, access tokens, or copyrighted model
weights in the report.

Reports are especially useful when they involve unsafe model-package handling,
path traversal, buffer or offset safety, malformed remote data, verification
bypasses, command-line injection, credential exposure, or unexpected file
access. Model quality problems, incorrect generated text, expected high
resource use, and performance regressions are not normally security
vulnerabilities.

Please allow the issue to be investigated and a fix to be prepared before
publishing vulnerability details. Credit can be included in the eventual
advisory unless the reporter prefers to remain anonymous.
