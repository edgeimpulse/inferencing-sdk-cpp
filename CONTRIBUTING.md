# Contributing

Contributions to this ESP32 packaging fork of the Edge Impulse inferencing SDK are welcome.

Use pull requests for fixes and improvements, and open an issue first if you want to discuss a larger change in behavior, packaging, or release policy.

## Scope

This fork exists to package the Edge Impulse inferencing SDK cleanly for ESP-IDF and the Espressif Component Registry.

Changes that affect the ESP32 integration should keep the following files aligned:

- `README.md`
- `idf_component.yml`
- `.github/workflows/upload_component.yml`
- `.github/workflows/upload_component_staging.yml`

If you need to change the underlying shared SDK sources, keep the fork-specific packaging adjustments minimal and avoid unnecessary divergence from upstream.

## Pull Requests

Before opening a pull request:

- Make sure the change is scoped to a clear problem.
- Update user-facing documentation when install steps, registry metadata, or behavior changes.
- Do not bump component versions, create release tags, or change release intent unless the maintainer explicitly asks for it.
- Prefer small, reviewable changes over large repo-wide rewrites.

## Development Notes

This repository vendors third-party code and multiple platform ports. Be deliberate about what gets compiled for ESP32 targets.

In particular:

- ESP-IDF component metadata lives in `idf_component.yml`.
- Staging uploads are triggered manually with `.github/workflows/upload_component_staging.yml`.
- Production uploads are handled by `.github/workflows/upload_component.yml` on stable version tags.
- Espressif-specific SDK ports are under `src/edge-impulse-sdk/porting/espressif/`.

## Release Process

Releases are handled by the repository maintainer.

Contributors should not prepare or publish releases unless explicitly asked to do so.

When a release is needed, the maintainer should use the following process.

Use plain component versions such as `X.Y.Z` and prerelease versions such as `X.Y.Z-rc1`.

For example:

- component version: `X.Y.Z-rc1`
- matching Git tag: `vX.Y.Z-rc1`

1. Update the version in `idf_component.yml` to a prerelease such as `X.Y.Z-rc1`.
2. Review `README.md`, `idf_component.yml`, `.github/workflows/upload_component_staging.yml`, and `.github/workflows/upload_component.yml` to make sure the registry metadata still matches the intended published component.
3. Push the release-prep branch and run the `upload_component_staging.yml` workflow manually.
4. Verify the uploaded prerelease on the staging registry page and test consuming it from the staging registry.
5. When the prerelease looks correct, update `idf_component.yml` to the stable version `X.Y.Z`.
6. Commit the stable release changes on the default branch.
7. Create an annotated tag whose version matches the component version, such as `vX.Y.Z`.
8. The `upload_component.yml` workflow will upload the tagged version to the production registry using GitHub OIDC.
9. Verify the published version on the production registry page and create a GitHub release if you want release notes attached to the tag.

## Reporting Problems

If you are not sure whether a bug belongs in this fork or upstream, open the issue here first and include enough context to reproduce the problem on ESP-IDF.
