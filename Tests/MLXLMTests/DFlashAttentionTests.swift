// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import MLXNN
import Testing

struct DFlashAttentionTests {

    @Test(
        "DFlash attention returns [batch, qLen, hiddenSize]",
        .disabled(if: !dflashMetallibAvailable, "MLX metallib unavailable")
    )
    func testDFlashAttentionForwardShape() throws {
        let config = makeTestDFlashConfig()
        let attention = DFlashAttention(config)
        let batchSize = 1
        let queryLength = config.blockSize
        let contextLength = 32
        let rope = RoPE(
            dimensions: config.headDim,
            traditional: false,
            base: config.ropeTheta
        )

        let noise = MLXRandom.normal([batchSize, queryLength, config.hiddenSize], dtype: .float32)
        let targetHidden = MLXRandom.normal(
            [batchSize, contextLength, config.hiddenSize],
            dtype: .float32
        )

        let output = attention(
            noise: noise,
            targetHidden: targetHidden,
            rope: rope,
            cache: nil
        )
        MLX.eval(output)

        #expect(output.shape == [batchSize, queryLength, config.hiddenSize])
    }
}
