// Copyright © 2026 Apple Inc.

import Foundation
import Testing

/// Lossless parity regression gate for `DFlashIterator`.
///
/// The real regression test requires:
///   1. A `Downloader` and `TokenizerLoader` to resolve an HF target checkpoint
///      via `LLMModelFactory.shared.loadContainer(from:using:configuration:)`.
///   2. A metallib-capable environment to actually execute MLX forward passes.
///   3. `DFLASH_TEST_HF=1` to opt into a ~10GB download of
///      `mlx-community/Qwen3-4B-bf16` plus `z-lab/Qwen3-4B-DFlash-b16`.
///
/// `MLXLMTests` intentionally depends only on `MLXLLM` + `MLXLMCommon` + `MLX*`
/// core products — NOT on the `MLXHuggingFace` downloader/tokenizer bridge used
/// by `MLXLMServer.BaselineEngine` — so the real parity harness belongs in the
/// integration-testing surface (mirror `Libraries/IntegrationTestHelpers` which
/// already injects a downloader + tokenizer loader).
///
/// This stub keeps the gated-test surface alive on the unit-test branch so that
/// `@Test(.enabled(if: dflashHFTestsEnabled && dflashMetallibAvailable))` wiring
/// compiles end-to-end. It intentionally records an `Issue.record` rather than
/// running anything, because neither gate is meant to be satisfied in a unit
/// test context.
struct DFlashIteratorParityTests {

    @Test(.enabled(if: dflashHFTestsEnabled && dflashMetallibAvailable))
    func testDFlashLosslessAgainstBaselineQwen34B_pending() {
        // Intentional placeholder. Real parity test must live in the
        // integration-testing target where Downloader + TokenizerLoader are
        // injected, and MLX metallib resolves at runtime.
        //
        // Expected shape of the real test (future):
        //   1. Load target container via LLMModelFactory.shared.loadContainer
        //      with injected HF downloader + tokenizer loader.
        //   2. Load drafter via DFlashWeightLoader.load().
        //   3. Encode prompt "Solve 2x + 5 = 17 step by step."; maxTokens=48.
        //   4. Run baseline greedy TokenIterator → baselineTokens.
        //   5. Run DFlashIterator on the same prompt → dflashTokens.
        //   6. #expect(baselineTokens == dflashTokens)
        //   7. #expect(iterator.acceptanceRate >= 0.5)
        Issue.record(
            """
            DFlash lossless parity gate pending integration-test wiring. \
            Move this test into IntegrationTesting target where downloader + \
            tokenizer loader are already injected. Drop this placeholder once \
            the real test lands.
            """)
    }
}
