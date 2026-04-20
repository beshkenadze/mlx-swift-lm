// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import MLXNN

final class DFlashMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlashAttention
    @ModuleInfo(key: "mlp") var mlp: DFlashMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: DFlashDraftConfig) {
        _selfAttn.wrappedValue = DFlashAttention(config)
        _mlp.wrappedValue = DFlashMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.intermediateSize
        )
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        hidden: MLXArray,
        targetHidden: MLXArray,
        rope: RoPE,
        cache: KVCache?
    ) -> MLXArray {
        let attentionOutput = selfAttn(
            noise: inputLayerNorm(hidden),
            targetHidden: targetHidden,
            rope: rope,
            cache: cache
        )
        let hiddenAfterAttention = hidden + attentionOutput
        let mlpOutput = mlp(postAttentionLayerNorm(hiddenAfterAttention))
        return hiddenAfterAttention + mlpOutput
    }
}

public final class DFlashDraftModel: Module {
    @ModuleInfo(key: "layers") var layers: [DFlashDecoderLayer]
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let rope: RoPE
    public let config: DFlashDraftConfig

    public init(_ config: DFlashDraftConfig) {
        self.config = config
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            DFlashDecoderLayer(config)
        }

        let targetProjectionSize = config.dflashConfig.targetLayerIds.count * config.hiddenSize
        self._fc.wrappedValue = Linear(targetProjectionSize, config.hiddenSize, bias: false)
        self._hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self.rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta
        )
    }

    public func callAsFunction(
        noiseEmbedding: MLXArray,
        targetHidden: MLXArray,
        caches: [KVCache?]
    ) -> MLXArray {
        precondition(
            caches.count == layers.count,
            "DFlashDraftModel expects one cache entry per decoder layer")

        var hidden = noiseEmbedding
        let projectedTargetHidden = hiddenNorm(fc(targetHidden))

        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden: hidden,
                targetHidden: projectedTargetHidden,
                rope: rope,
                cache: caches[index]
            )
        }

        return norm(hidden)
    }
}

extension DFlashDraftModel: DFlashDraftingModel {
    public var numDraftLayers: Int { config.numHiddenLayers }
}
