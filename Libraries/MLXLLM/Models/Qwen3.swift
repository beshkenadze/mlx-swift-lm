//
//  Qwen3.swift
//  LLM
//
//  Created by John Mai on 2025/4/28.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py

class Qwen3Attention: Module {
    let args: Qwen3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Qwen3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta,
            scale: ropeScale)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Apply RoPE positioning
        queries = applyRotaryPosition(rope, to: queries, cache: cache, kind: .query)
        keys = applyRotaryPosition(rope, to: keys, cache: cache, kind: .key)

        // Use the automatic attention router that handles both quantized and regular caches
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class Qwen3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class Qwen3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3Attention
    let mlp: Qwen3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        _attention.wrappedValue = Qwen3Attention(args)
        self.mlp = Qwen3MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class Qwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                Qwen3TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }

    /// Number of decoder layers in this inner model.
    public var layerCount: Int { layers.count }

    /// Runs a forward pass and captures hidden states at the specified layer indices.
    ///
    /// Mirrors ``callAsFunction(_:cache:)`` but captures the post-layer hidden state at each
    /// requested index and skips the final ``norm`` projection and any LM head.
    ///
    /// - Parameters:
    ///   - inputs: input token IDs, shape `[batch, seqLen]`
    ///   - layerIndices: zero-based decoder-layer indices whose post-layer outputs to capture
    ///   - cache: optional KV cache (advances as usual); pass `nil` for a cache-free forward
    /// - Returns: captured hidden states in the same order as `layerIndices`, each with
    ///            shape `[batch, seqLen, hiddenSize]`.
    public func captureHiddenStates(
        inputs: MLXArray,
        layerIndices: [Int],
        cache: [KVCache]? = nil
    ) -> [MLXArray] {
        for index in layerIndices {
            precondition(
                index >= 0 && index < layers.count,
                "captureHiddenStates: layer index \(index) out of range 0..<\(layers.count)")
        }

        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        let wanted = Set(layerIndices)
        var captured: [Int: MLXArray] = [:]
        captured.reserveCapacity(wanted.count)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if wanted.contains(i) {
                captured[i] = h
            }
        }

        return layerIndices.map { captured[$0]! }
    }

    /// Runs a single forward pass and captures BOTH the requested layer hidden states
    /// and the final post-norm hidden (ready for LM-head projection).
    ///
    /// Mirrors ``captureHiddenStates(inputs:layerIndices:cache:)`` but also applies
    /// the terminal ``norm`` — the caller can project through the LM head to obtain
    /// logits. Intended for speculative-decoding hot loops (e.g. dFlash) where the
    /// target needs both intermediate taps and final logits per cycle; doing both
    /// from a single forward avoids re-running the target prefix.
    ///
    /// - Parameters:
    ///   - inputs: input token IDs, shape `[batch, seqLen]`
    ///   - layerIndices: zero-based decoder-layer indices whose post-layer outputs to capture
    ///   - cache: optional KV cache (advances as usual); pass `nil` for a cache-free forward
    /// - Returns: `(hiddenStates, finalHidden)` — `hiddenStates` follows `layerIndices`
    ///   order, each shape `[batch, seqLen, hiddenSize]`; `finalHidden` is
    ///   `norm(last_layer)` shape `[batch, seqLen, hiddenSize]`.
    public func captureHiddenStatesAndFinalHidden(
        inputs: MLXArray,
        layerIndices: [Int],
        cache: [KVCache]? = nil
    ) -> (hiddenStates: [MLXArray], finalHidden: MLXArray) {
        for index in layerIndices {
            precondition(
                index >= 0 && index < layers.count,
                "captureHiddenStatesAndFinalHidden: layer index \(index) out of range 0..<\(layers.count)")
        }

        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        let wanted = Set(layerIndices)
        var captured: [Int: MLXArray] = [:]
        captured.reserveCapacity(wanted.count)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if wanted.contains(i) {
                captured[i] = h
            }
        }

        let finalHidden = norm(h)
        return (layerIndices.map { captured[$0]! }, finalHidden)
    }
}

public class Qwen3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3ModelInner
    let configuration: Qwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }

    /// Runs a forward pass and captures post-decoder-layer hidden states at the specified
    /// layer indices. Does not apply the final ``Qwen3ModelInner/norm`` nor the LM head.
    ///
    /// Useful for speculative-decoding drafts (e.g. dFlash) that consume the target model's
    /// intermediate hidden states.
    ///
    /// - Parameters:
    ///   - inputs: input token IDs, shape `[batch, seqLen]`
    ///   - layerIndices: zero-based decoder-layer indices whose outputs to capture
    ///   - cache: optional KV cache (advances as usual); pass `nil` for a cache-free forward
    /// - Returns: captured hidden states in the order of `layerIndices`, each with shape
    ///            `[batch, seqLen, hiddenSize]`.
    public func captureHiddenStates(
        inputs: MLXArray,
        layerIndices: [Int],
        cache: [KVCache]? = nil
    ) -> [MLXArray] {
        model.captureHiddenStates(
            inputs: inputs, layerIndices: layerIndices, cache: cache)
    }

    /// Runs a single forward pass and returns BOTH the selected-layer hidden states
    /// AND the per-position logits. Doing both in one forward pass avoids the
    /// O(N·prefill) cost of re-running the target from scratch to capture hidden
    /// states in speculative-decoding hot loops (e.g. dFlash).
    ///
    /// - Parameters:
    ///   - inputs: input token IDs, shape `[batch, seqLen]`
    ///   - layerIndices: zero-based decoder-layer indices whose outputs to capture
    ///   - cache: optional KV cache (advances as usual); pass `nil` for a cache-free forward
    /// - Returns: `(hiddenStates, logits)` — `hiddenStates` in the order of
    ///   `layerIndices`, each shape `[batch, seqLen, hiddenSize]`; `logits` shape
    ///   `[batch, seqLen, vocabSize]`.
    public func captureHiddenStatesAndLogits(
        inputs: MLXArray,
        layerIndices: [Int],
        cache: [KVCache]? = nil
    ) -> (hiddenStates: [MLXArray], logits: MLXArray) {
        let (hiddenStates, finalHidden) = model.captureHiddenStatesAndFinalHidden(
            inputs: inputs, layerIndices: layerIndices, cache: cache)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(finalHidden)
        } else {
            logits = model.embedTokens.asLinear(finalHidden)
        }
        return (hiddenStates, logits)
    }

    /// Returns the LM-head projection weight. For models with ``Qwen3Configuration/tieWordEmbeddings``
    /// set, there is no separate ``lmHead`` and callers should fall back to the token embedding
    /// weight (exposed as ``tiedEmbeddingWeight``); this property returns `nil` in that case.
    public var lmHeadWeight: MLXArray? {
        lmHead?.weight
    }

    /// Returns the token embedding weight (used as the tied LM head when
    /// ``Qwen3Configuration/tieWordEmbeddings`` is `true`).
    public var tiedEmbeddingWeight: MLXArray {
        model.embedTokens.weight
    }
}

public struct Qwen3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var headDim: Int
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false
    var maxPositionEmbeddings: Int = 32768

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Qwen3Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: Qwen3Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: Qwen3Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: Qwen3Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Qwen3Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.headDim = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: Qwen3Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    }
}

// MARK: - LoRA

extension Qwen3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
