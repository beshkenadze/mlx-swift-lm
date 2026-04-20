# Development Notes

## Running DFlash integration tests

The current unit-test placeholder for the DFlash parity gate can be invoked
with:

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -only-testing:MLXLMTests/DFlashIteratorParityTests
```

Current behavior:

- This target is expected to skip in normal local and CI runs.
- `MLXLMTests` does not own the Hugging Face downloader/tokenizer wiring needed
  for the real parity harness.
- The real end-to-end parity test should live in the IntegrationTesting
  surface backed by `Libraries/IntegrationTestHelpers/`.

Until that IntegrationTesting target lands, use:

- `scripts/smoke-dflash.sh --dflash` for manual end-to-end DFlash validation
- targeted `swift test --filter ...` or `xcodebuild test -only-testing:...`
  commands for unit-level coverage around HTTP wiring and iterator compile
  compatibility
