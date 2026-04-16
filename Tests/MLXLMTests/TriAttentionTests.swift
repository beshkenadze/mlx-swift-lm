import Foundation
import MLX
@testable import MLXLMCommon
import MLXNN
import Testing

@Test
func testExtractTriAttentionRoPEConfigFromStandardRoPE() async throws {
    let rope = RoPE(dimensions: 8, traditional: true, base: 1000, scale: 0.5)

    let config = try TriAttentionRoPEConfig.extract(from: rope, headDim: 16)

    #expect(config.headDim == 16)
    #expect(config.rotatedDims == 8)
    #expect(config.traditional)
    #expect(!config.proportional)

    let expected = MLXArray([Float32(2.0), 0.35565588, 0.06324556, 0.011246826])
    #expect(allClose(config.omega, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
}

@Test
func testCreateTriAttentionConfigurationFromCalibrationFileAndRoPE() async throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try save(
        arrays: [
            "layer.0.q_center_real": MLXArray.zeros([2, 4], dtype: .float32),
            "layer.0.q_center_imag": MLXArray.zeros([2, 4], dtype: .float32),
            "layer.0.q_mean_norm": MLXArray.ones([2, 4], dtype: .float32),
        ],
        metadata: [
            "n_layers": "1",
            "n_q_heads": "2",
            "n_kv_heads": "2",
        ],
        url: url
    )

    let rope = RoPE(dimensions: 8)
    let config = try TriAttentionConfiguration.load(
        calibrationURL: url,
        rope: rope,
        headDim: 16,
        budget: 512,
        divideLength: 32,
        protectRecent: 64,
        protectInitial: 8
    )

    #expect(config.budget == 512)
    #expect(config.divideLength == 32)
    #expect(config.protectRecent == 64)
    #expect(config.protectInitial == 8)
    #expect(config.calibration.qHeads == 2)
    #expect(config.calibration.kvHeads == 2)
    #expect(config.rope.rotatedDims == 8)
}

@Test
func testComputeTriAttentionCalibrationStatistics() async throws {
    let q = MLXArray(
        [Float32(1), 2, 3, 4, 5, 6, 7, 8,
         10, 20, 30, 40, 50, 60, 70, 80]
    ).reshaped(1, 2, 1, 8)

    let calibration = TriAttentionCalibrationData.computeStatistics(
        captures: [0: [q]],
        rope: TriAttentionRoPEConfig(
            headDim: 8,
            rotatedDims: 8,
            traditional: false,
            omega: MLXArray.ones([4], dtype: .float32)
        ),
        qHeads: 2,
        kvHeads: 2,
        nLayers: 2
    )

    let expectedReal = MLXArray(
        [Float32(1), 3, 5, 7,
         10, 30, 50, 70]
    ).reshaped(2, 4)
    let expectedImag = MLXArray(
        [Float32(2), 4, 6, 8,
         20, 40, 60, 80]
    ).reshaped(2, 4)
    let expectedNorm = sqrt(expectedReal * expectedReal + expectedImag * expectedImag + MLXArray(1e-12))

    let layer0 = try #require(calibration.layerCalibration(0))
    #expect(allClose(layer0.qCenterReal, expectedReal, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    #expect(allClose(layer0.qCenterImag, expectedImag, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    #expect(allClose(layer0.qMeanNorm, expectedNorm, rtol: 1e-5, atol: 1e-5).item(Bool.self))

    let layer1 = try #require(calibration.layerCalibration(1))
    #expect(allClose(layer1.qCenterReal, MLXArray.zeros([2, 4], dtype: .float32)).item(Bool.self))
    #expect(allClose(layer1.qCenterImag, MLXArray.zeros([2, 4], dtype: .float32)).item(Bool.self))
    #expect(allClose(layer1.qMeanNorm, MLXArray.zeros([2, 4], dtype: .float32)).item(Bool.self))
}

@Test
func testComputeTriAttentionCalibrationStatisticsUsesOnlyPartialRotarySlice() async throws {
    let q = MLXArray(
        [Float32(1), 2, 3, 4, 5, 6, 7, 8,
         9, 10, 11, 12, 13, 14, 15, 16]
    ).reshaped(1, 2, 1, 8)

    let calibration = TriAttentionCalibrationData.computeStatistics(
        captures: [0: [q]],
        rope: TriAttentionRoPEConfig(
            headDim: 8,
            rotatedDims: 4,
            traditional: false,
            omega: MLXArray.ones([2], dtype: .float32)
        ),
        qHeads: 2,
        kvHeads: 2,
        nLayers: 1
    )

    let layer0 = try #require(calibration.layerCalibration(0))
    let expectedReal = MLXArray([Float32(1), 3, 9, 11]).reshaped(2, 2)
    let expectedImag = MLXArray([Float32(2), 4, 10, 12]).reshaped(2, 2)
    let expectedNorm = sqrt(expectedReal * expectedReal + expectedImag * expectedImag + MLXArray(1e-12))

    #expect(layer0.qCenterReal.shape == [2, 2])
    #expect(layer0.qCenterImag.shape == [2, 2])
    #expect(layer0.qMeanNorm.shape == [2, 2])
    if layer0.qCenterReal.shape == [2, 2] {
        #expect(allClose(layer0.qCenterReal, expectedReal, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    }
    if layer0.qCenterImag.shape == [2, 2] {
        #expect(allClose(layer0.qCenterImag, expectedImag, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    }
    if layer0.qMeanNorm.shape == [2, 2] {
        #expect(allClose(layer0.qMeanNorm, expectedNorm, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    }
}

@Test
func testSaveTriAttentionCalibrationRoundTrip() async throws {
    let calibration = TriAttentionCalibrationData(
        layers: [
            TriAttentionLayerCalibration(
                qCenterReal: MLXArray.ones([2, 4], dtype: .float32),
                qCenterImag: MLXArray.zeros([2, 4], dtype: .float32),
                qMeanNorm: MLXArray.ones([2, 4], dtype: .float32) * 3
            )
        ],
        qHeads: 2,
        kvHeads: 2
    )

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try calibration.save(to: url)
    let loaded = try TriAttentionCalibrationData.load(from: url)

    #expect(loaded.qHeads == 2)
    #expect(loaded.kvHeads == 2)
    let layer = try #require(loaded.layerCalibration(0))
    #expect(allClose(layer.qCenterReal, MLXArray.ones([2, 4], dtype: .float32)).item(Bool.self))
    #expect(allClose(layer.qCenterImag, MLXArray.zeros([2, 4], dtype: .float32)).item(Bool.self))
    #expect(allClose(layer.qMeanNorm, MLXArray.ones([2, 4], dtype: .float32) * 3).item(Bool.self))
}

private func makeGroupedQueryTriAttentionConfiguration(
    qHeads: Int,
    kvHeads: Int,
    nFreqs: Int,
    headDim: Int,
    budget: Int = 4
) -> TriAttentionConfiguration {
    let layer = TriAttentionLayerCalibration(
        qCenterReal: MLXArray.zeros([qHeads, nFreqs], dtype: .float32),
        qCenterImag: MLXArray.zeros([qHeads, nFreqs], dtype: .float32),
        qMeanNorm: MLXArray.ones([qHeads, nFreqs], dtype: .float32)
    )
    let calibration = TriAttentionCalibrationData(layers: [layer], qHeads: qHeads, kvHeads: kvHeads)
    let rope = TriAttentionRoPEConfig(
        headDim: headDim,
        rotatedDims: nFreqs * 2,
        traditional: false,
        omega: MLXArray.ones([nFreqs], dtype: .float32)
    )

    return TriAttentionConfiguration(
        calibration: calibration,
        rope: rope,
        budget: budget,
        divideLength: 1,
        protectRecent: 0,
        protectInitial: 0
    )
}

private func compressTriAttentionKeys(
    calibrationQHeads: Int,
    calibrationKVHeads: Int,
    runtimeKVHeads: Int,
    tokenCount: Int = 1025,
    headDim: Int = 128,
    nFreqs: Int = 32,
    budget: Int = 4
) -> (MLXArray, MLXArray) {
    let cache = TriAttentionCache(
        base: KVCacheSimple(),
        configuration: makeGroupedQueryTriAttentionConfiguration(
            qHeads: calibrationQHeads,
            kvHeads: calibrationKVHeads,
            nFreqs: nFreqs,
            headDim: headDim,
            budget: budget
        ),
        layerIndex: 0
    )

    let keys = MLXArray.zeros([1, runtimeKVHeads, tokenCount, headDim], dtype: .float32)
    let values = MLXArray.zeros([1, runtimeKVHeads, tokenCount, headDim], dtype: .float32)
    let (compressedKeys, compressedValues) = cache.update(keys: keys, values: values)
    eval(compressedKeys, compressedValues)
    return (compressedKeys, compressedValues)
}

@Test
func testTriAttentionGroupedQueryCalibrationShapeIsAccepted() async {
    let (compressedKeys, compressedValues) = compressTriAttentionKeys(
        calibrationQHeads: 16,
        calibrationKVHeads: 8,
        runtimeKVHeads: 8,
        headDim: 64
    )

    #expect(compressedKeys.shape == [1, 8, 4, 64])
    #expect(compressedValues.shape == [1, 8, 4, 64])
}

@Test
func testTriAttentionRejectsGroupedQueryCalibrationMismatch() async {
    let configuration = makeGroupedQueryTriAttentionConfiguration(
        qHeads: 8,
        kvHeads: 8,
        nFreqs: 32,
        headDim: 128
    )
    let layerCalibration = try! #require(configuration.calibration.layerCalibration(0))
    let keys = MLXArray.zeros([1, 4, 1025, 128], dtype: .float32)

    do {
        _ = try scoreKeys(
            cachedKeys: keys,
            currentPosition: 1025,
            layerCalibration: layerCalibration,
            calibration: configuration.calibration,
            rope: configuration.rope,
            offsets: MLXArray([Float32(1), Float32(2), Float32(4)])
        )
        Issue.record("Expected grouped-query calibration mismatch to throw")
    } catch let error as TriAttentionError {
        switch error {
        case .incompatibleCalibration(let message):
            #expect(message.contains("runtime KV heads"))
            #expect(message.contains("calibration KV heads"))
        default:
            Issue.record("Unexpected TriAttentionError: \(error)")
        }
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test
func testTriAttentionRejectsMalformedGroupedQueryCalibrationTensorShapes() async {
    let layer = TriAttentionLayerCalibration(
        qCenterReal: MLXArray.zeros([8, 32], dtype: .float32),
        qCenterImag: MLXArray.zeros([7, 32], dtype: .float32),
        qMeanNorm: MLXArray.ones([8, 32], dtype: .float32)
    )
    let calibration = TriAttentionCalibrationData(layers: [layer], qHeads: 8, kvHeads: 8)
    let rope = TriAttentionRoPEConfig(
        headDim: 128,
        rotatedDims: 64,
        traditional: false,
        omega: MLXArray.ones([32], dtype: .float32)
    )
    let keys = MLXArray.zeros([1, 8, 1025, 128], dtype: .float32)

    do {
        _ = try scoreKeys(
            cachedKeys: keys,
            currentPosition: 1025,
            layerCalibration: layer,
            calibration: calibration,
            rope: rope,
            offsets: MLXArray([Float32(1), Float32(2), Float32(4)])
        )
        Issue.record("Expected malformed calibration tensor shapes to throw")
    } catch let error as TriAttentionError {
        switch error {
        case .incompatibleCalibration(let message):
            #expect(message.contains("qCenterImag head count"))
        default:
            Issue.record("Unexpected TriAttentionError: \(error)")
        }
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test
func testTriAttentionRejectsMismatchedCalibrationTensorShapes() async {
    let rope = TriAttentionRoPEConfig(
        headDim: 128,
        rotatedDims: 64,
        traditional: false,
        omega: MLXArray.ones([32], dtype: .float32)
    )
    let keys = MLXArray.zeros([1, 8, 1025, 128], dtype: .float32)
    let offsets = MLXArray([Float32(1), Float32(2), Float32(4)])

    do {
        _ = try scoreKeys(
            cachedKeys: keys,
            currentPosition: 1025,
            layerCalibration: TriAttentionLayerCalibration(
                qCenterReal: MLXArray.zeros([16, 32], dtype: .float32),
                qCenterImag: MLXArray.zeros([15, 32], dtype: .float32),
                qMeanNorm: MLXArray.ones([16, 32], dtype: .float32)
            ),
            calibration: TriAttentionCalibrationData(layers: [], qHeads: 16, kvHeads: 8),
            rope: rope,
            offsets: offsets
        )
        Issue.record("Expected qCenterImag shape mismatch to throw")
    } catch let error as TriAttentionError {
        switch error {
        case .incompatibleCalibration(let message):
            #expect(message.contains("qCenterImag"))
        default:
            Issue.record("Unexpected TriAttentionError: \(error)")
        }
    } catch {
        Issue.record("Unexpected error: \(error)")
    }

    do {
        _ = try scoreKeys(
            cachedKeys: keys,
            currentPosition: 1025,
            layerCalibration: TriAttentionLayerCalibration(
                qCenterReal: MLXArray.zeros([16, 32], dtype: .float32),
                qCenterImag: MLXArray.zeros([16, 32], dtype: .float32),
                qMeanNorm: MLXArray.ones([16, 31], dtype: .float32)
            ),
            calibration: TriAttentionCalibrationData(layers: [], qHeads: 16, kvHeads: 8),
            rope: rope,
            offsets: offsets
        )
        Issue.record("Expected qMeanNorm shape mismatch to throw")
    } catch let error as TriAttentionError {
        switch error {
        case .incompatibleCalibration(let message):
            #expect(message.contains("qMeanNorm"))
        default:
            Issue.record("Unexpected TriAttentionError: \(error)")
        }
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test
func testTriAttentionLongContextCompressionMismatchReportsTypedError() async {
    let result = await #expect(processExitsWith: .failure, observing: [\.standardErrorContent]) {
        _ = compressTriAttentionKeys(
            calibrationQHeads: 8,
            calibrationKVHeads: 8,
            runtimeKVHeads: 4
        )
    }

    let standardError = result.map { String(decoding: $0.standardErrorContent, as: UTF8.self) } ?? ""
    #expect(standardError.contains("runtime KV heads"))
    #expect(!standardError.contains("[reshape]"))
}

private final class DummyCalibrationModel: Module, LanguageModel, KVCacheDimensionProvider {
    let kvHeads: [Int] = [2]
    let rope = RoPE(dimensions: 8)

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func sanitize(weights: [String : MLXArray]) -> [String : MLXArray] { weights }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = inputs.dim(0)
        let length = inputs.dim(1)
        let queries = MLXArray(
            (1...(batch * length * 2 * 8)).map(Float32.init)
        ).reshaped(batch, length, 2, 8).transposed(0, 2, 1, 3)
        let keys = queries * 2

        _ = applyRotaryPosition(rope, to: queries, cache: cache?[0], kind: .query)
        _ = applyRotaryPosition(rope, to: keys, cache: cache?[0], kind: .key)

        return MLXArray.zeros([batch, length, 16], dtype: .float32)
    }
}

@Test
func testTriAttentionCalibrationRunnerCapturesModelQueries() async throws {
    let model = DummyCalibrationModel()
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3, 4]))

    let calibration = try TriAttentionCalibrationRunner.calibrate(
        model: model,
        input: input,
        prefillStepSize: 4
    )

    #expect(calibration.qHeads == 2)
    #expect(calibration.kvHeads == 2)
    let layer = try #require(calibration.layerCalibration(0))
    #expect(layer.qCenterReal.shape == [2, 4])
    #expect(layer.qCenterImag.shape == [2, 4])
    #expect(layer.qMeanNorm.shape == [2, 4])
}

@Test
func testTriAttentionCalibrationRunnerSavesOutput() async throws {
    let model = DummyCalibrationModel()
    let input = LMInput(tokens: MLXArray([Int32(4), 3, 2, 1]))
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    let calibration = try TriAttentionCalibrationRunner.calibrate(
        model: model,
        input: input,
        outputURL: url,
        prefillStepSize: 4
    )
    let loaded = try TriAttentionCalibrationData.load(from: url)

    #expect(loaded.qHeads == calibration.qHeads)
    #expect(loaded.kvHeads == calibration.kvHeads)
    #expect(allClose(try #require(loaded.layerCalibration(0)).qCenterReal, try #require(calibration.layerCalibration(0)).qCenterReal).item(Bool.self))
}

private final class DummyAttention: Module {
    let headDim: Int
    let rope: Module

    init(headDim: Int, rope: Module) {
        self.headDim = headDim
        self.rope = rope
        super.init()
    }
}

private final class DummyLayer: Module {
    let selfAttn: DummyAttention

    init(selfAttn: DummyAttention) {
        self.selfAttn = selfAttn
        super.init()
    }
}

private final class DummyBackbone: Module {
    let layers: [DummyLayer]

    init(layers: [DummyLayer]) {
        self.layers = layers
        super.init()
    }
}

private final class DummyModel: Module {
    let model: DummyBackbone

    init(model: DummyBackbone) {
        self.model = model
        super.init()
    }
}

@Test
func testExtractTriAttentionRoPEConfigFromModel() async throws {
    let model = DummyModel(
        model: DummyBackbone(
            layers: [
                DummyLayer(
                    selfAttn: DummyAttention(
                        headDim: 32,
                        rope: RoPE(dimensions: 16, traditional: false, base: 10000, scale: 2)
                    ))
            ]))

    let config = try TriAttentionRoPEConfig.extract(from: model)

    #expect(config.headDim == 32)
    #expect(config.rotatedDims == 16)
    #expect(!config.traditional)
}

@Test
func testCreateTriAttentionConfigurationFromCalibrationFileAndModel() async throws {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try save(
        arrays: [
            "layer.0.q_center_real": MLXArray.zeros([2, 4], dtype: .float32),
            "layer.0.q_center_imag": MLXArray.zeros([2, 4], dtype: .float32),
            "layer.0.q_mean_norm": MLXArray.ones([2, 4], dtype: .float32),
        ],
        metadata: [
            "n_layers": "1",
            "n_q_heads": "2",
            "n_kv_heads": "2",
        ],
        url: url
    )

    let model = DummyModel(
        model: DummyBackbone(
            layers: [DummyLayer(selfAttn: DummyAttention(headDim: 24, rope: RoPE(dimensions: 8)))]))

    let config = try TriAttentionConfiguration.load(
        calibrationURL: url,
        model: model,
        budget: 256
    )

    #expect(config.budget == 256)
    #expect(config.rope.headDim == 24)
    #expect(config.rope.rotatedDims == 8)
}

private final class NoRoPEModel: Module {
    let attention: DummyAttention

    override init() {
        self.attention = DummyAttention(headDim: 16, rope: DummyRoPE())
        super.init()
    }
}

@Test
func testExtractTriAttentionRoPEConfigFromModelRejectsUnsupportedModelShape() async throws {
    do {
        _ = try TriAttentionRoPEConfig.extract(from: NoRoPEModel())
        Issue.record("Expected model-level extraction to throw")
    } catch let error as TriAttentionError {
        switch error {
        case .unsupportedRoPE:
            break
        default:
            Issue.record("Unexpected TriAttentionError: \(error)")
        }
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

private final class DummyRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray { x }
    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray { x }
}

@Test
func testExtractTriAttentionRoPEConfigRejectsUnsupportedRoPE() async throws {
    do {
        _ = try TriAttentionRoPEConfig.extract(from: DummyRoPE(), headDim: 8)
        Issue.record("Expected unsupported RoPE extraction to throw")
    } catch let error as TriAttentionError {
        switch error {
        case .unsupportedRoPE:
            break
        default:
            Issue.record("Unexpected TriAttentionError: \(error)")
        }
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}
