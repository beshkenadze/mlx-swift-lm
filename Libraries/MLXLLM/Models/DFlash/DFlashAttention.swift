// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class DFlashAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    public init(_ config: DFlashDraftConfig) {
        let hiddenSize = config.hiddenSize
        let queryDim = config.numAttentionHeads * config.headDim
        let kvDim = config.numKeyValueHeads * config.headDim

        _qProj.wrappedValue = Linear(hiddenSize, queryDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(hiddenSize, kvDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(hiddenSize, kvDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(queryDim, hiddenSize, bias: config.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)

        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Foundation.sqrt(Float(config.headDim))
    }

    public func callAsFunction(
        noise: MLXArray,
        targetHidden: MLXArray,
        rope: RoPE,
        cache: KVCache? = nil
    ) -> MLXArray {
        let batchSize = noise.dim(0)
        let queryLength = noise.dim(1)
        let contextLength = targetHidden.dim(1)
        let totalLength = contextLength + queryLength

        precondition(
            targetHidden.dim(0) == batchSize,
            "DFlashAttention expects matching batch sizes for noise and targetHidden")

        var queries = qProj(noise)
        var keys = concatenated([kProj(targetHidden), kProj(noise)], axis: 1)
        var values = concatenated([vProj(targetHidden), vProj(noise)], axis: 1)

        queries = qNorm(queries.reshaped(batchSize, queryLength, numHeads, headDim))
            .transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(batchSize, totalLength, numKVHeads, headDim))
            .transposed(0, 2, 1, 3)
        values = values.reshaped(batchSize, totalLength, numKVHeads, headDim)
            .transposed(0, 2, 1, 3)

        let (rotatedQueries, rotatedKeys) = rope.applyDFlash(
            q: queries,
            k: keys,
            qLen: queryLength,
            offset: cache?.offset ?? 0
        )

        let output = attentionWithCacheUpdate(
            queries: rotatedQueries,
            keys: rotatedKeys,
            values: values,
            cache: cache,
            scale: scale,
            mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, queryLength, numHeads * headDim)

        return oProj(output)
    }
}

public extension RoPE {
    /// Applies RoPE over the full K span while aligning Q to the tail positions.
    func applyDFlash(
        q: MLXArray,
        k: MLXArray,
        qLen: Int,
        offset: Int = 0
    ) -> (MLXArray, MLXArray) {
        let keyLength = k.dim(2)
        let rotatedKeys = self(k, offset: offset)
        let rotatedQueries = self(q, offset: offset + keyLength - qLen)
        return (rotatedQueries, rotatedKeys)
    }
}
