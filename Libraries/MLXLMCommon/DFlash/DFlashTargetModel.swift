// Copyright © 2026 Apple Inc.

import MLX

/// Target model operations required by the DFlash speculative decoding loop.
/// Conformance exposes the three forward-pass variants DFlash needs:
/// - Token embedding lookup for building the noise block
/// - LM-head projection for converting drafter hidden states to vocab logits
/// - Forward-with-captured-hidden-states for verify/prefill passes
public protocol DFlashTargetModel: LanguageModel {
    /// Maps token IDs to input embeddings.
    /// Shape: `[batch, seqLen] -> [batch, seqLen, hiddenSize]`.
    func embedTokens(_ inputs: MLXArray) -> MLXArray

    /// Projects hidden states to vocab logits via the LM head or tied embeddings.
    /// Shape: `[batch, seqLen, hiddenSize] -> [batch, seqLen, vocabSize]`.
    func applyLMHead(_ hidden: MLXArray) -> MLXArray

    /// Runs a forward pass while capturing selected decoder-layer hidden states
    /// plus the final logits in one call.
    func captureHiddenStatesAndLogits(
        inputs: MLXArray,
        layerIndices: [Int],
        cache: [KVCache]?
    ) -> (hiddenStates: [MLXArray], logits: MLXArray)
}
