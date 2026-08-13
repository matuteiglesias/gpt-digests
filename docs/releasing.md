# Releasing

Python-package publication is separate from ordinary CI and from evidence-artifact
promotion. A successful build is not release approval.

## One-time PyPI setup

On PyPI, configure a Trusted Publisher for the GitHub repository
`matuteiglesias/kb-artifacts`, workflow `.github/workflows/release.yml`, and GitHub
environment `pypi`. Do not create or store a long-lived `PYPI_TOKEN`.

In GitHub, create the protected `pypi` environment and restrict deployment to the
release authority. The workflow requests `id-token: write` only in its publish job so
PyPI can exchange GitHub's short-lived OIDC identity for upload authorization.

## Release procedure

1. Complete the [TestPyPI rehearsal](#testpypi-rehearsal).
2. Obtain production release acceptance from Matías.
3. Set the approved stable version in `pyproject.toml` and ensure release notes are ready.
4. Require green CI and documentation checks for the exact source commit.
5. Create and push an annotated tag matching the version exactly, such as `v0.1.0`.
6. Approve the protected `pypi` environment deployment when prompted.
7. Confirm that the workflow verifies the exact version from PyPI and creates the
   matching GitHub release.
8. Confirm that the documentation workflow for the released commit is live at the
   canonical site.

The release workflow rejects a tag that does not equal `v` plus the package version.
It tests the tagged source, builds one wheel and one sdist, runs strict metadata and
installed-package checks, records SHA-256 checksums, and stores those exact files as a
workflow artifact. Only the downstream publish job downloads and uploads the validated
distributions.

After publishing, a job with no repository checkout creates a clean environment,
installs the exact tagged version from PyPI, imports the public API, runs
`kb-artifact --help`, and completes the tutorial. Only after that succeeds does a
least-privilege job create the matching GitHub release with the validated wheel,
sdist, checksum file, and generated release notes. A publication that fails external
verification therefore does not receive a GitHub release claiming success.

The official PyPA publish action uses Trusted Publishing and emits PyPI attestations.
No publishing step exists in the ordinary CI workflow.

## TestPyPI rehearsal

Before production, set a release-candidate version such as `0.1.0rc1` in
`pyproject.toml`, obtain approval for the rehearsal, and push the matching annotated
tag `v0.1.0rc1`. Configure a TestPyPI Trusted Publisher for
`.github/workflows/testpypi.yml` with the protected GitHub environment `testpypi`; no
TestPyPI token is stored.

The rehearsal workflow builds and validates the exact candidate distributions,
publishes them to TestPyPI, and then creates a new environment with no repository
checkout. It installs the exact candidate with TestPyPI as the package index and PyPI
as the dependency fallback, imports the public API, runs `kb-artifact --help`, and
executes the 60-second tutorial. TestPyPI index propagation is retried for a bounded
period.

A failed external-consumer verification blocks production release acceptance. After a
successful rehearsal, change the version to the approved stable version and repeat the
normal CI checks before creating the production tag. TestPyPI artifacts are rehearsals;
they are never promoted or copied into PyPI.
