// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

/// End-to-end check that TranslateGemma (a Gemma 3 fine-tune) loads and generates a
/// translation through the existing `gemma3` text path.
@Suite(.serialized)
struct TranslateGemmaIntegrationTests {

    @Test func translateGemma4bEndToEnd() async throws {
        let container = try await models.translateGemmaContainer()
        try await ChatSessionTests.translation(container: container)
    }

    /// Fuller check: translate a WMT14 sample set (longer sentences, en->fr/de/ru) and
    /// score each output against the human reference with chrF.
    @Test func translateGemma4bDataset() async throws {
        let container = try await models.translateGemmaContainer()
        try await ChatSessionTests.translationDataset(container: container)
    }

    /// Latency: prefill the constant translation-instruction prefix once and reuse the KV
    /// cache across requests; verify identical output and measure TTFT before/after.
    @Test func translateGemma4bPrefixCacheBenchmark() async throws {
        let container = try await models.translateGemmaContainer()
        try await ChatSessionTests.translationPrefixCacheBenchmark(container: container)
    }
}
