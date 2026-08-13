# Security policy

## Supported versions

Until the first production release, security fixes are prepared on the default branch.
After release, the latest published `0.x` version receives security fixes. Older
pre-release and `0.x` versions are not guaranteed patches unless a security advisory
states otherwise.

## Report a vulnerability privately

Do **not** open a public issue for a suspected vulnerability or include private source
material, credentials, absolute paths, or evidence bodies in a report.

Use GitHub's private vulnerability reporting form:

<https://github.com/matuteiglesias/kb-artifacts/security/advisories/new>

Include the affected version or commit, impact, minimal reproduction, and suggested
mitigation when known. Use sanitized inputs only. The maintainer will acknowledge the
report, assess impact, coordinate a fix and disclosure, and credit reporters who want
credit. Response timelines depend on severity and maintainer availability; this file
does not promise an unsupported service-level agreement.

For ordinary defects without security impact, use the public issue tracker.

## Scope

Reports about unsafe path handling, source mutation, provenance disclosure, artifact
promotion boundaries, dependency compromise, and distribution or publishing integrity
are in scope. Vulnerabilities in an upstream dependency should also be reported to that
dependency's maintainers when appropriate.
