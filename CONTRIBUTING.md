# Contributing

Contributions to this ESP32 packaging fork of the Edge Impulse inferencing SDK are welcome.

Use pull requests for fixes and improvements, and open an issue first if you want to discuss a larger change in behavior, packaging, or release policy.

## Scope

This fork exists to package the Edge Impulse inferencing SDK cleanly for ESP-IDF and the Espressif Component Registry.

Changes that affect the ESP32 integration should keep the following files aligned:

- `README.md`
- `idf_component.yml`
- `component.mk`
- `.github/workflows/upload_component.yml`

If you need to change the underlying shared SDK sources, keep the fork-specific packaging adjustments minimal and avoid unnecessary divergence from upstream.

## Pull Requests

Before opening a pull request:

- Make sure the change is scoped to a clear problem.
- Update user-facing documentation when install steps, registry metadata, or behavior changes.
- Keep legacy GNU Make support in `component.mk` in sync with any source layout changes, since the component still ships that file.
- Prefer small, reviewable changes over large repo-wide rewrites.

## Development Notes

This repository vendors third-party code and multiple platform ports. Be deliberate about what gets compiled for ESP32 targets.

In particular:

- ESP-IDF component metadata lives in `idf_component.yml`.
- Registry publishing is handled by GitHub Actions on version tags.
- The legacy ESP-IDF GNU Make integration is defined in `component.mk`.
- Espressif-specific SDK ports are under `src/porting/espressif/`.

## Release Process

Maintainers should use the following process to publish a new component version. Assuming the new version is `vX.Y.Z`:

1. Update the version in `idf_component.yml` to `X.Y.Z`.
2. Review `README.md`, `idf_component.yml`, and `.github/workflows/upload_component.yml` to make sure the registry metadata still matches the intended published component.
3. Commit the release changes.
4. Create an annotated tag named `vX.Y.Z`.
5. Push the branch and the tag to GitHub.
6. The `upload_component.yml` workflow will upload the tagged version to the Espressif Component Registry using the `IDF_COMPONENT_API_TOKEN` repository secret.
7. Verify the published version on the registry page and create a GitHub release if you want release notes attached to the tag.

## Reporting Problems

If you are not sure whether a bug belongs in this fork or upstream, open the issue here first and include enough context to reproduce the problem on ESP-IDF.
