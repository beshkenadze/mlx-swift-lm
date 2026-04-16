import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    let qHeads = queries.dim(1)
    let kvHeads = keys.dim(1)

    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let turboQuantCache = cache as? TurboQuantKVCacheProtocol {
        if shouldUseTurboQuantMaterializedFallback(
            qHeads: qHeads,
            kvHeads: kvHeads,
            totalSequenceLength: cache.offset + keys.dim(2),
            mask: mask,
            turboQuantBits: turboQuantCache.turboQuantBits
        ) {
            let (cachedKeys, cachedValues) = turboQuantCache.updateTurboQuant(keys: keys, values: values)
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: mask
            )
        }
        let (packedKeys, packedValues) = turboQuantCache.updateTurboQuantPacked(
            keys: keys, values: values)
        return turboQuantScaledDotProductAttention(
            queries: queries,
            packedKeys: packedKeys,
            packedValues: packedValues,
            scale: scale,
            mask: mask,
            bits: turboQuantCache.turboQuantBits,
            seed: turboQuantCache.turboQuantSeed
        )
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }
}

private func shouldUseTurboQuantMaterializedFallback(
    qHeads: Int,
    kvHeads: Int,
    totalSequenceLength: Int,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    turboQuantBits: Int
) -> Bool {
    let isGroupedQuery = qHeads > kvHeads
    if !isGroupedQuery { return false }
    if turboQuantBits != 3 { return false }

    switch mask {
    case .none:
        return totalSequenceLength <= 256
    case .causal:
        return true
    case .array, .arrays:
        return false
    }
}

public func turboQuantScaledDotProductAttention(
    queries: MLXArray,
    packedKeys: TurboQuantPackedTensorState,
    packedValues: TurboQuantPackedTensorState,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    bits: Int,
    seed: Int,
    sequenceChunkSize: Int = 128
) -> MLXArray {
    switch mask {
    case .none:
        return turboQuantChunkedScaledDotProductAttention(
            queries: queries,
            packedKeys: packedKeys,
            packedValues: packedValues,
            scale: scale,
            bits: bits,
            seed: seed,
            sequenceChunkSize: sequenceChunkSize
        )
    case .causal, .array, .arrays:
        return turboQuantMaterializedScaledDotProductAttention(
            queries: queries,
            packedKeys: packedKeys,
            packedValues: packedValues,
            scale: scale,
            mask: mask,
            bits: bits,
            seed: seed
        )
    }
}

private func turboQuantMaterializedScaledDotProductAttention(
    queries: MLXArray,
    packedKeys: TurboQuantPackedTensorState,
    packedValues: TurboQuantPackedTensorState,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    bits: Int,
    seed: Int
) -> MLXArray {
    let qHeads = queries.dim(1)
    let kvHeads = packedKeys.indices.dim(1)
    precondition(qHeads % kvHeads == 0, "TurboQuant attention expects qHeads to be divisible by kvHeads")
    let repeats = qHeads / kvHeads

    var outputs = [MLXArray]()
    outputs.reserveCapacity(kvHeads)

    for kvHead in 0..<kvHeads {
        let queryStart = kvHead * repeats
        let queryEnd = queryStart + repeats
        let querySlice = queries[0..., queryStart ..< queryEnd, 0..., 0...]
        let keySlice = materializeTurboQuantHead(packedKeys, headIndex: kvHead, bits: bits, seed: seed)
        let valueSlice = materializeTurboQuantHead(packedValues, headIndex: kvHead, bits: bits, seed: seed)
        let output = MLXFast.scaledDotProductAttention(
            queries: querySlice,
            keys: keySlice,
            values: valueSlice,
            scale: scale,
            mask: mask
        )
        outputs.append(output)
    }

    return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 1)
}

private func turboQuantChunkedScaledDotProductAttention(
    queries: MLXArray,
    packedKeys: TurboQuantPackedTensorState,
    packedValues: TurboQuantPackedTensorState,
    scale: Float,
    bits: Int,
    seed: Int,
    sequenceChunkSize: Int
) -> MLXArray {
    precondition(sequenceChunkSize > 0, "sequenceChunkSize must be positive")
    let qHeads = queries.dim(1)
    let kvHeads = packedKeys.indices.dim(1)
    precondition(qHeads % kvHeads == 0, "TurboQuant attention expects qHeads to be divisible by kvHeads")
    let repeats = qHeads / kvHeads

    var outputs = [MLXArray]()
    outputs.reserveCapacity(kvHeads)

    for kvHead in 0..<kvHeads {
        let queryStart = kvHead * repeats
        let queryEnd = queryStart + repeats
        let querySlice = queries[0..., queryStart ..< queryEnd, 0..., 0...].asType(.float32)
        let output = turboQuantChunkedHeadAttention(
            queries: querySlice,
            packedKeys: packedKeys,
            packedValues: packedValues,
            headIndex: kvHead,
            scale: scale,
            bits: bits,
            seed: seed,
            sequenceChunkSize: sequenceChunkSize
        )
        outputs.append(output.asType(queries.dtype))
    }

    return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 1)
}

private func turboQuantChunkedHeadAttention(
    queries: MLXArray,
    packedKeys: TurboQuantPackedTensorState,
    packedValues: TurboQuantPackedTensorState,
    headIndex: Int,
    scale: Float,
    bits: Int,
    seed: Int,
    sequenceChunkSize: Int
) -> MLXArray {
    let totalSequence = packedKeys.norms.dim(2)
    let repeats = queries.dim(1)

    var runningMax: MLXArray?
    var runningDenom: MLXArray?
    var runningNumer: MLXArray?

    for start in stride(from: 0, to: totalSequence, by: sequenceChunkSize) {
        let end = min(start + sequenceChunkSize, totalSequence)
        let keyChunk = materializeTurboQuantHeadRange(
            packedKeys, headIndex: headIndex, sequenceRange: start..<end, bits: bits, seed: seed
        ).asType(.float32)
        let valueChunk = materializeTurboQuantHeadRange(
            packedValues, headIndex: headIndex, sequenceRange: start..<end, bits: bits, seed: seed
        ).asType(.float32)

        let repeatedKeys = repeated(keyChunk, count: repeats, axis: 1)
        let repeatedValues = repeated(valueChunk, count: repeats, axis: 1)
        let scores = matmul(queries * scale, repeatedKeys.transposed(0, 1, 3, 2))
        let chunkMax = scores.max(axis: -1, keepDims: true)
        let expScores = exp(scores - chunkMax)
        let chunkDenom = sum(expScores, axis: -1, keepDims: true)
        let chunkNumer = matmul(expScores, repeatedValues)

        if let currentMax = runningMax, let currentDenom = runningDenom, let currentNumer = runningNumer {
            let nextMax = maximum(currentMax, chunkMax)
            let currentScale = exp(currentMax - nextMax)
            let chunkScale = exp(chunkMax - nextMax)
            runningMax = nextMax
            runningDenom = currentDenom * currentScale + chunkDenom * chunkScale
            runningNumer = currentNumer * currentScale + chunkNumer * chunkScale
        } else {
            runningMax = chunkMax
            runningDenom = chunkDenom
            runningNumer = chunkNumer
        }
    }

    guard let denom = runningDenom, let numer = runningNumer else {
        fatalError("TurboQuant chunked attention requires at least one KV chunk")
    }
    return numer / denom
}
