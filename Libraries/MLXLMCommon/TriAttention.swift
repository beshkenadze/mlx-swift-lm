import Foundation
import MLX
import MLXNN

private protocol TriAttentionCalibrationCacheLayerProvider {
    var triAttentionCalibrationLayerIndex: Int { get }
}

private final class TriAttentionCalibrationCaptureCache: BaseKVCache,
    TriAttentionCalibrationCacheLayerProvider
{
    var base: KVCache
    let triAttentionCalibrationLayerIndex: Int

    init(base: KVCache, layerIndex: Int) {
        self.base = base
        self.triAttentionCalibrationLayerIndex = layerIndex
        super.init()
        self.offset = base.offset
    }

    override var maxSize: Int? { base.maxSize }

    override var state: [MLXArray] {
        get { base.state }
        set { base.state = newValue }
    }

    override var metaState: [String] {
        get { base.metaState }
        set { base.metaState = newValue }
    }

    override var isTrimmable: Bool { base.isTrimmable }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = base.update(keys: keys, values: values)
        offset = base.offset
        return result
    }

    @discardableResult
    override func trim(_ n: Int) -> Int {
        let trimmed = base.trim(n)
        offset = base.offset
        return trimmed
    }

    override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        base.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    override func copy() -> any KVCache {
        TriAttentionCalibrationCaptureCache(base: base.copy(), layerIndex: triAttentionCalibrationLayerIndex)
    }
}

private final class TriAttentionCalibrationCaptureSession: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var captures = [Int: [MLXArray]]()

    func record(layerIndex: Int, tensor: MLXArray) {
        lock.lock()
        captures[layerIndex, default: []].append(tensor)
        lock.unlock()
    }
}

private enum TriAttentionCalibrationContext {
    @TaskLocal static var session: TriAttentionCalibrationCaptureSession?
}

func recordTriAttentionCalibrationCapture(
    kind: RotaryPositionApplicationKind,
    cache: KVCache?,
    tensor: MLXArray
) {
    guard kind == .query,
        let layerIndex = (cache as? TriAttentionCalibrationCacheLayerProvider)?
            .triAttentionCalibrationLayerIndex
    else {
        return
    }
    TriAttentionCalibrationContext.session?.record(layerIndex: layerIndex, tensor: tensor)
}

public enum TriAttentionCalibrationRunner {
    public static func calibrate<M: LanguageModel & KVCacheDimensionProvider>(
        model: M,
        input: LMInput,
        outputURL: URL? = nil,
        prefillStepSize: Int = 512
    ) throws -> TriAttentionCalibrationData {
        let kvHeads = try uniformKVHeads(model.kvHeads)

        var parameters = GenerateParameters(maxTokens: 0, prefillStepSize: prefillStepSize)
        parameters.kvBits = nil
        parameters.triAttention = nil

        let captureCaches = wrapTriAttentionCalibrationCaches(model.newCache(parameters: parameters))
        let session = TriAttentionCalibrationCaptureSession()

        try TriAttentionCalibrationContext.$session.withValue(session) {
            _ = try TokenIterator(input: input, model: model, cache: captureCaches, parameters: parameters)
        }

        let (qHeads, headDim) = try inferredCaptureShape(from: session.captures)
        let ropeModule = try discoverRoPEModule(in: model)
        let rope = try TriAttentionRoPEConfig.extract(from: ropeModule, headDim: headDim)
        let calibration = TriAttentionCalibrationData.computeStatistics(
            captures: session.captures,
            rope: rope,
            qHeads: qHeads,
            kvHeads: kvHeads,
            nLayers: model.kvHeads.count
        )
        if let outputURL {
            try calibration.save(to: outputURL)
        }
        return calibration
    }
}

public struct TriAttentionRoPEConfig: @unchecked Sendable {
    public let headDim: Int
    public let rotatedDims: Int
    public let traditional: Bool
    public let omega: MLXArray
    public let proportional: Bool

    public init(
        headDim: Int,
        rotatedDims: Int,
        traditional: Bool,
        omega: MLXArray,
        proportional: Bool = false
    ) {
        self.headDim = headDim
        self.rotatedDims = rotatedDims
        self.traditional = traditional
        self.omega = omega
        self.proportional = proportional
    }

    public static func extract(from rope: Module, headDim: Int) throws -> TriAttentionRoPEConfig {
        if let rope = rope as? ProportionalRoPE, let freqs = rope._freqs {
            return .init(
                headDim: headDim,
                rotatedDims: rope.rotatedDims,
                traditional: rope.traditional,
                omega: MLXArray(1.0) / freqs,
                proportional: true
            )
        }

        if let rope = rope as? Llama3RoPE {
            return .init(
                headDim: headDim,
                rotatedDims: rope.dims,
                traditional: rope.traditional,
                omega: MLXArray(1.0) / rope._freqs
            )
        }

        if let rope = rope as? YarnRoPE {
            guard let freqs: MLXArray = reflectedValue(named: "_freqs", in: rope) else {
                throw TriAttentionError.unsupportedRoPE("Unable to reflect YarnRoPE frequencies")
            }
            return .init(
                headDim: headDim,
                rotatedDims: rope.dimensions,
                traditional: rope.traditional,
                omega: MLXArray(1.0) / freqs
            )
        }

        if let rope = rope as? SuScaledRoPE {
            guard let freqs: MLXArray = reflectedValue(named: "_freqs", in: rope) else {
                throw TriAttentionError.unsupportedRoPE(
                    "Unable to reflect SuScaledRoPE frequencies")
            }
            return .init(
                headDim: headDim,
                rotatedDims: rope.dimensions,
                traditional: false,
                omega: MLXArray(1.0) / freqs
            )
        }

        if rope is RoPE {
            guard let dimensions: Int = reflectedValue(named: "dimensions", in: rope),
                let traditional: Bool = reflectedValue(named: "traditional", in: rope),
                let base: Float = reflectedValue(named: "base", in: rope),
                let scale: Float = reflectedValue(named: "scale", in: rope)
            else {
                throw TriAttentionError.unsupportedRoPE(
                    "Unable to reflect standard RoPE internals")
            }

            let exponents = arange(0, dimensions, step: 2, dtype: .float32) / dimensions
            let omega = (MLXArray(1.0) / pow(base, exponents)) / scale
            return .init(
                headDim: headDim,
                rotatedDims: dimensions,
                traditional: traditional,
                omega: omega
            )
        }

        throw TriAttentionError.unsupportedRoPE("Unsupported RoPE type: \(type(of: rope))")
    }

    public static func extract(from model: Module) throws -> TriAttentionRoPEConfig {
        for (_, module) in model.namedModules() {
            if let attention = reflectedModule(named: attentionPropertyNames, in: module),
                let rope = reflectedRoPEModule(in: attention),
                let headDim = reflectedHeadDim(in: attention) ?? reflectedHeadDim(in: module)
            {
                return try extract(from: rope, headDim: headDim)
            }

            if let rope = reflectedRoPEModule(in: module),
                let headDim = reflectedHeadDim(in: module)
            {
                return try extract(from: rope, headDim: headDim)
            }
        }

        var visited = Set<ObjectIdentifier>()
        if let (rope, headDim) = discoverRoPEAndHeadDim(in: model, visited: &visited) {
            return try extract(from: rope, headDim: headDim)
        }

        throw TriAttentionError.unsupportedRoPE(
            "Could not discover supported RoPE + headDim from model \(type(of: model))")
    }
}

public struct TriAttentionLayerCalibration: @unchecked Sendable {
    public let qCenterReal: MLXArray
    public let qCenterImag: MLXArray
    public let qMeanNorm: MLXArray

    public init(qCenterReal: MLXArray, qCenterImag: MLXArray, qMeanNorm: MLXArray) {
        self.qCenterReal = qCenterReal
        self.qCenterImag = qCenterImag
        self.qMeanNorm = qMeanNorm
    }
}

public struct TriAttentionCalibrationData: @unchecked Sendable {
    public let layers: [TriAttentionLayerCalibration]
    public let qHeads: Int
    public let kvHeads: Int

    public init(layers: [TriAttentionLayerCalibration], qHeads: Int, kvHeads: Int) {
        self.layers = layers
        self.qHeads = qHeads
        self.kvHeads = kvHeads
    }

    public func layerCalibration(_ layerIndex: Int) -> TriAttentionLayerCalibration? {
        guard layers.indices.contains(layerIndex) else {
            return nil
        }
        return layers[layerIndex]
    }

    public static func computeStatistics(
        captures: [Int: [MLXArray]],
        rope: TriAttentionRoPEConfig,
        qHeads: Int,
        kvHeads: Int,
        nLayers: Int
    ) -> TriAttentionCalibrationData {
        let nFreqs = rope.rotatedDims / 2
        let zero = TriAttentionLayerCalibration(
            qCenterReal: MLXArray.zeros([qHeads, nFreqs], dtype: .float32),
            qCenterImag: MLXArray.zeros([qHeads, nFreqs], dtype: .float32),
            qMeanNorm: MLXArray.zeros([qHeads, nFreqs], dtype: .float32)
        )

        var layers = Array(repeating: zero, count: nLayers)

        for layerIndex in 0..<nLayers {
            guard let layerCaptures = captures[layerIndex], !layerCaptures.isEmpty else {
                continue
            }

            let allQ = concatenated(layerCaptures, axis: 2)
            var centerReal = [MLXArray]()
            var centerImag = [MLXArray]()
            var meanNorm = [MLXArray]()
            centerReal.reserveCapacity(qHeads)
            centerImag.reserveCapacity(qHeads)
            meanNorm.reserveCapacity(qHeads)

            for headIndex in 0..<qHeads {
                let qHead = allQ[0, headIndex, 0..., 0...]
                let (real, imag) = decomposeComplex(qHead, rope: rope)
                let magnitude = sqrt(real * real + imag * imag + MLXArray(1e-12))

                centerReal.append(mean(real, axis: 0))
                centerImag.append(mean(imag, axis: 0))
                meanNorm.append(mean(magnitude, axis: 0))
            }

            layers[layerIndex] = TriAttentionLayerCalibration(
                qCenterReal: stacked(centerReal),
                qCenterImag: stacked(centerImag),
                qMeanNorm: stacked(meanNorm)
            )
        }

        return .init(layers: layers, qHeads: qHeads, kvHeads: kvHeads)
    }

    public static func load(from url: URL) throws -> TriAttentionCalibrationData {
        let (tensors, metadata) = try loadArraysAndMetadata(url: url)

        guard let nLayersString = metadata["n_layers"], let nLayers = Int(nLayersString),
            let qHeadsString = metadata["n_q_heads"], let qHeads = Int(qHeadsString),
            let kvHeadsString = metadata["n_kv_heads"], let kvHeads = Int(kvHeadsString)
        else {
            throw TriAttentionError.invalidCalibrationFile(
                "Missing TriAttention metadata in \(url.lastPathComponent)")
        }

        var layers = [TriAttentionLayerCalibration]()
        layers.reserveCapacity(nLayers)

        for layerIndex in 0 ..< nLayers {
            guard let qCenterReal = tensors["layer.\(layerIndex).q_center_real"],
                let qCenterImag = tensors["layer.\(layerIndex).q_center_imag"],
                let qMeanNorm = tensors["layer.\(layerIndex).q_mean_norm"]
            else {
                throw TriAttentionError.invalidCalibrationFile(
                    "Missing calibration tensors for layer \(layerIndex)")
            }
            layers.append(
                .init(qCenterReal: qCenterReal, qCenterImag: qCenterImag, qMeanNorm: qMeanNorm))
        }

        return .init(layers: layers, qHeads: qHeads, kvHeads: kvHeads)
    }

    public func save(to url: URL) throws {
        var arrays = [String: MLXArray]()
        for (layerIndex, layer) in layers.enumerated() {
            arrays["layer.\(layerIndex).q_center_real"] = layer.qCenterReal
            arrays["layer.\(layerIndex).q_center_imag"] = layer.qCenterImag
            arrays["layer.\(layerIndex).q_mean_norm"] = layer.qMeanNorm
        }

        try writeTriAttentionCalibrationArrays(
            arrays: arrays,
            metadata: [
                "n_layers": String(layers.count),
                "n_q_heads": String(qHeads),
                "n_kv_heads": String(kvHeads),
            ],
            url: url
        )
    }
}

public struct TriAttentionConfiguration: @unchecked Sendable {
    public let calibration: TriAttentionCalibrationData
    public let rope: TriAttentionRoPEConfig
    public let budget: Int
    public let divideLength: Int
    public let protectRecent: Int
    public let protectInitial: Int

    public init(
        calibration: TriAttentionCalibrationData,
        rope: TriAttentionRoPEConfig,
        budget: Int = 2048,
        divideLength: Int = 128,
        protectRecent: Int = 128,
        protectInitial: Int = 4
    ) {
        self.calibration = calibration
        self.rope = rope
        self.budget = budget
        self.divideLength = divideLength
        self.protectRecent = protectRecent
        self.protectInitial = protectInitial
    }

    public static func load(
        calibrationURL: URL,
        rope: Module,
        headDim: Int,
        budget: Int = 2048,
        divideLength: Int = 128,
        protectRecent: Int = 128,
        protectInitial: Int = 4
    ) throws -> TriAttentionConfiguration {
        try .init(
            calibration: TriAttentionCalibrationData.load(from: calibrationURL),
            rope: TriAttentionRoPEConfig.extract(from: rope, headDim: headDim),
            budget: budget,
            divideLength: divideLength,
            protectRecent: protectRecent,
            protectInitial: protectInitial
        )
    }

    public static func load(
        calibrationURL: URL,
        model: Module,
        budget: Int = 2048,
        divideLength: Int = 128,
        protectRecent: Int = 128,
        protectInitial: Int = 4
    ) throws -> TriAttentionConfiguration {
        let calibration = try TriAttentionCalibrationData.load(from: calibrationURL)
        let ropeModule = try discoverRoPEModule(in: model)
        let inferredHeadDim = try discoverHeadDim(in: model) ?? inferHeadDimFromCalibration(
            calibration: calibration,
            rope: ropeModule
        )
        return try TriAttentionConfiguration(
            calibration: calibration,
            rope: TriAttentionRoPEConfig.extract(from: ropeModule, headDim: inferredHeadDim),
            budget: budget,
            divideLength: divideLength,
            protectRecent: protectRecent,
            protectInitial: protectInitial
        )
    }
}

public enum TriAttentionError: LocalizedError, Equatable {
    case invalidCalibrationFile(String)
    case incompatibleWithQuantizedKV
    case cacheSerializationUnsupported
    case unsupportedRoPE(String)
    case calibrationCaptureFailed(String)
    case incompatibleCalibration(String)

    public var errorDescription: String? {
        switch self {
        case .invalidCalibrationFile(let message):
            return message
        case .incompatibleWithQuantizedKV:
            return "TriAttention and KV cache quantization cannot be enabled together in v1."
        case .cacheSerializationUnsupported:
            return "TriAttention prompt-cache serialization is not supported yet."
        case .unsupportedRoPE(let message):
            return message
        case .calibrationCaptureFailed(let message):
            return message
        case .incompatibleCalibration(let message):
            return message
        }
    }
}

struct TriAttentionScoringState {
    let kvHeads: Int
    let repeats: Int
    let nFreqs: Int
    let qCenterMagGrouped: MLXArray
    let qCenterPhaseGrouped: MLXArray
    let normWeight: MLXArray
    let offsetCos: MLXArray
    let offsetSin: MLXArray
    let omega: MLXArray
}

func makeTriAttentionScoringState(
    layerCalibration: TriAttentionLayerCalibration,
    calibration: TriAttentionCalibrationData,
    rope: TriAttentionRoPEConfig,
    offsets: MLXArray
) throws -> TriAttentionScoringState {
    try validateTriAttentionShapeCompatibility(
        calibration: calibration,
        layerCalibration: layerCalibration,
        runtimeKVHeads: calibration.kvHeads,
        runtimeHeadDim: rope.headDim,
        rope: rope
    )

    let repeats = calibration.qHeads / calibration.kvHeads
    let nFreqs = rope.rotatedDims / 2

    let qCenterMag = sqrt(
        layerCalibration.qCenterReal * layerCalibration.qCenterReal
            + layerCalibration.qCenterImag * layerCalibration.qCenterImag
            + MLXArray(1e-12)
    )
    let qCenterPhase = atan2(layerCalibration.qCenterImag, layerCalibration.qCenterReal)

    let qCenterMagGrouped = qCenterMag.reshaped(calibration.kvHeads, repeats, nFreqs)
    let qCenterPhaseGrouped = qCenterPhase.reshaped(calibration.kvHeads, repeats, nFreqs)
    let qMeanNormGrouped = layerCalibration.qMeanNorm.reshaped(calibration.kvHeads, repeats, nFreqs)
    let normWeight = qMeanNormGrouped - qCenterMagGrouped

    let offsetOmega = expandedDimensions(offsets, axis: -1) * rope.omega[.newAxis, 0...]
    let offsetCos = cos(offsetOmega)
    let offsetSin = sin(offsetOmega)

    return TriAttentionScoringState(
        kvHeads: calibration.kvHeads,
        repeats: repeats,
        nFreqs: nFreqs,
        qCenterMagGrouped: qCenterMagGrouped,
        qCenterPhaseGrouped: qCenterPhaseGrouped,
        normWeight: normWeight,
        offsetCos: offsetCos,
        offsetSin: offsetSin,
        omega: rope.omega
    )
}

public final class TriAttentionCache: BaseKVCache {
    private static let compressionHysteresis = 128
    nonisolated(unsafe) private static let defaultOffsets = MLXArray((0 ..< 17).map { Float(pow(2.0, Double($0))) })

    public var base: KVCache
    public let configuration: TriAttentionConfiguration
    public let layerIndex: Int
    private var tokensSinceCompress: Int
    private var scoringState: TriAttentionScoringState?

    public init(base: KVCache, configuration: TriAttentionConfiguration, layerIndex: Int) {
        self.base = base
        self.configuration = configuration
        self.layerIndex = layerIndex
        self.tokensSinceCompress = base.offset
        super.init()
        self.offset = base.offset
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let nNew = keys.dim(2)
        var (cachedKeys, cachedValues) = base.update(keys: keys, values: values)

        offset += nNew
        tokensSinceCompress += nNew

        if cachedKeys.dim(2) > configuration.budget
            && tokensSinceCompress >= configuration.divideLength
            && base is KVCacheSimple
        {
            let (prunedKeys, prunedValues) = compress(keys: cachedKeys, values: cachedValues)
            base.state = [prunedKeys, prunedValues]
            cachedKeys = prunedKeys
            cachedValues = prunedValues
            tokensSinceCompress = 0
        }

        return (cachedKeys, cachedValues)
    }

    public override var maxSize: Int? { base.maxSize }

    public override var state: [MLXArray] {
        get { base.state }
        set { base.state = newValue }
    }

    public override var metaState: [String] {
        get { base.metaState }
        set { base.metaState = newValue }
    }

    public override var isTrimmable: Bool { base.isTrimmable }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = base.trim(n)
        offset -= trimmed
        return trimmed
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        base.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public override func copy() -> any KVCache {
        let copy = TriAttentionCache(
            base: base.copy(), configuration: configuration, layerIndex: layerIndex)
        copy.offset = offset
        copy.tokensSinceCompress = tokensSinceCompress
        copy.scoringState = scoringState
        return copy
    }

    private func compress(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let tokenCount = keys.dim(2)
        let hysteresis = configuration.budget > Self.compressionHysteresis
            ? Self.compressionHysteresis : 0
        let targetKeepCount = configuration.budget - hysteresis
        let protectedKeepCount = min(
            tokenCount, max(configuration.protectInitial, 0) + max(configuration.protectRecent, 0))
        let keepCount = min(tokenCount, max(targetKeepCount, protectedKeepCount))
        guard keepCount < tokenCount,
            let layerCalibration = configuration.calibration.layerCalibration(layerIndex)
        else {
            return (keys, values)
        }

        let rawScores: MLXArray
        do {
            let scoringState: TriAttentionScoringState
            if let cached = self.scoringState {
                scoringState = cached
            } else {
                let prepared = try makeTriAttentionScoringState(
                    layerCalibration: layerCalibration,
                    calibration: configuration.calibration,
                    rope: configuration.rope,
                    offsets: Self.defaultOffsets
                )
                self.scoringState = prepared
                scoringState = prepared
            }
            rawScores = try scoreKeys(
                cachedKeys: keys,
                currentPosition: offset,
                prepared: scoringState,
                rope: configuration.rope
            )
        } catch let error as TriAttentionError {
            fatalError(error.localizedDescription)
        } catch {
            fatalError("TriAttention compression failed: \(error)")
        }

        let scores = mean(rawScores, axis: 1)
        if configuration.protectInitial > 0 {
            let protected = min(configuration.protectInitial, tokenCount)
            scores[0..., ..<protected] = MLXArray(1e9, dtype: scores.dtype)
        }
        if configuration.protectRecent > 0 && tokenCount > configuration.protectRecent {
            let start = tokenCount - configuration.protectRecent
            scores[0..., start ..< tokenCount] = MLXArray(1e9, dtype: scores.dtype)
        }

        let indices = MLX.argPartition(-scores[0], kth: keepCount - 1, axis: -1)[..<keepCount]
        let keepIndices = sorted(indices)
        let prunedKeys = keys[0..., 0..., keepIndices, 0...]
        let prunedValues = values[0..., 0..., keepIndices, 0...]
        return (prunedKeys, prunedValues)
    }
}

public func validateGenerationCacheConfiguration(parameters: GenerateParameters) throws {
    if parameters.triAttention != nil && parameters.kvBits != nil {
        throw TriAttentionError.incompatibleWithQuantizedKV
    }
    if parameters.turboQuant != nil && parameters.kvBits != nil {
        throw TurboQuantError.incompatibleWithQuantizedKV
    }
    if parameters.turboQuant != nil && parameters.triAttention != nil {
        throw TurboQuantError.incompatibleWithTriAttention
    }
}

public func validateTriAttentionConfiguration(parameters: GenerateParameters) throws {
    try validateGenerationCacheConfiguration(parameters: parameters)
}

public func wrapGenerationCaches(
    _ caches: [KVCache],
    parameters: GenerateParameters?
) -> [KVCache] {
    let turboWrapped: [KVCache]
    if let turboQuant = parameters?.turboQuant {
        turboWrapped = caches.enumerated().map { layerIndex, cache in
            wrapTurboQuantCache(cache, layerIndex: layerIndex, configuration: turboQuant)
        }
    } else {
        turboWrapped = caches
    }

    guard let triAttention = parameters?.triAttention else {
        return turboWrapped
    }

    return turboWrapped.enumerated().map { layerIndex, cache in
        wrapTriAttentionCache(cache, layerIndex: layerIndex, configuration: triAttention)
    }
}

public func wrapTriAttentionCaches(
    _ caches: [KVCache],
    parameters: GenerateParameters?
) -> [KVCache] {
    wrapGenerationCaches(caches, parameters: parameters)
}

private func wrapTurboQuantCache(
    _ cache: KVCache,
    layerIndex: Int,
    configuration: TurboQuantConfiguration
) -> KVCache {
    _ = layerIndex
    switch cache {
    case is TurboQuantKVCacheProtocol, is RotatingKVCache, is MambaCache, is ArraysCache:
        return cache
    case let cacheList as CacheList:
        let wrapped = cacheList.elements.map {
            wrapTurboQuantCache($0, layerIndex: layerIndex, configuration: configuration)
        }
        return CacheList(wrapped)
    case let simple as KVCacheSimple:
        return simple.toTurboQuant(bits: configuration.bits, seed: configuration.seed)
    default:
        return cache
    }
}

private func wrapTriAttentionCache(
    _ cache: KVCache,
    layerIndex: Int,
    configuration: TriAttentionConfiguration
) -> KVCache {
    switch cache {
    case is TriAttentionCache, is RotatingKVCache, is MambaCache, is ArraysCache:
        return cache
    case let cacheList as CacheList:
        let wrapped = cacheList.elements.map {
            wrapTriAttentionCache($0, layerIndex: layerIndex, configuration: configuration)
        }
        return CacheList(wrapped)
    case let simple as KVCacheSimple:
        return TriAttentionCache(base: simple, configuration: configuration, layerIndex: layerIndex)
    default:
        return cache
    }
}

private func validateTriAttentionShapeCompatibility(
    calibration: TriAttentionCalibrationData,
    layerCalibration: TriAttentionLayerCalibration,
    runtimeKVHeads: Int,
    runtimeHeadDim: Int,
    rope: TriAttentionRoPEConfig
) throws {
    guard calibration.kvHeads > 0, calibration.qHeads % calibration.kvHeads == 0 else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: qHeads \(calibration.qHeads) must be divisible by kvHeads \(calibration.kvHeads)."
        )
    }

    guard runtimeKVHeads == calibration.kvHeads else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: runtime KV heads \(runtimeKVHeads) do not match calibration KV heads \(calibration.kvHeads)."
        )
    }

    guard layerCalibration.qCenterReal.dim(0) == calibration.qHeads else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: qCenterReal head count \(layerCalibration.qCenterReal.dim(0)) does not match calibration qHeads \(calibration.qHeads)."
        )
    }

    guard layerCalibration.qCenterImag.dim(0) == calibration.qHeads else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: qCenterImag head count \(layerCalibration.qCenterImag.dim(0)) does not match calibration qHeads \(calibration.qHeads)."
        )
    }

    guard layerCalibration.qMeanNorm.dim(0) == calibration.qHeads else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: qMeanNorm head count \(layerCalibration.qMeanNorm.dim(0)) does not match calibration qHeads \(calibration.qHeads)."
        )
    }

    let expectedFreqs = rope.rotatedDims / 2
    guard layerCalibration.qCenterReal.dim(1) == expectedFreqs else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: qCenterReal frequency count \(layerCalibration.qCenterReal.dim(1)) does not match expected rotated frequency count \(expectedFreqs)."
        )
    }

    guard layerCalibration.qCenterImag.dim(1) == expectedFreqs else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: qCenterImag frequency count \(layerCalibration.qCenterImag.dim(1)) does not match expected rotated frequency count \(expectedFreqs)."
        )
    }

    guard layerCalibration.qMeanNorm.dim(1) == expectedFreqs else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: qMeanNorm frequency count \(layerCalibration.qMeanNorm.dim(1)) does not match expected rotated frequency count \(expectedFreqs)."
        )
    }

    guard runtimeHeadDim >= rope.rotatedDims else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: runtime head dim \(runtimeHeadDim) must be at least rotated dims \(rope.rotatedDims)."
        )
    }
}

func scoreKeys(
    cachedKeys: MLXArray,
    currentPosition: Int,
    layerCalibration: TriAttentionLayerCalibration,
    calibration: TriAttentionCalibrationData,
    rope: TriAttentionRoPEConfig,
    offsets: MLXArray
) throws -> MLXArray {
    let prepared = try makeTriAttentionScoringState(
        layerCalibration: layerCalibration,
        calibration: calibration,
        rope: rope,
        offsets: offsets
    )
    return try scoreKeys(cachedKeys: cachedKeys, currentPosition: currentPosition, prepared: prepared, rope: rope)
}

func scoreKeys(
    cachedKeys: MLXArray,
    currentPosition: Int,
    prepared: TriAttentionScoringState,
    rope: TriAttentionRoPEConfig
) throws -> MLXArray {
    let kvHeads = cachedKeys.dim(1)
    let runtimeHeadDim = cachedKeys.dim(3)
    guard kvHeads == prepared.kvHeads else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: runtime KV heads \(kvHeads) do not match calibration KV heads \(prepared.kvHeads)."
        )
    }
    guard runtimeHeadDim >= rope.rotatedDims else {
        throw TriAttentionError.incompatibleCalibration(
            "TriAttention incompatible calibration: runtime head dim \(runtimeHeadDim) must be at least rotated dims \(rope.rotatedDims)."
        )
    }

    let (kReal, kImag) = decomposeComplex(cachedKeys, rope: rope)
    let kMag = sqrt(kReal * kReal + kImag * kImag + MLXArray(1e-12))
    let kPhase = atan2(kImag, kReal)

    let phi = prepared.qCenterPhaseGrouped[.newAxis, 0..., .newAxis, 0..., 0...]
        - kPhase[0..., 0..., 0..., .newAxis, 0...]
    let amp = prepared.qCenterMagGrouped[.newAxis, 0..., .newAxis, 0..., 0...]
        * kMag[0..., 0..., 0..., .newAxis, 0...]

    let a = amp * cos(phi)
    let b = amp * sin(phi)

    let currentOmega = MLXArray(Float(currentPosition)) * prepared.omega
    let currentCos = cos(currentOmega)[.newAxis, 0...]
    let currentSin = sin(currentOmega)[.newAxis, 0...]
    let cosTw = prepared.offsetCos * currentCos - prepared.offsetSin * currentSin
    let sinTw = prepared.offsetSin * currentCos + prepared.offsetCos * currentSin

    let flatShape = [cachedKeys.dim(0) * kvHeads * cachedKeys.dim(2) * prepared.repeats, prepared.nFreqs]
    let sTrigFlat = matmul(a.reshaped(flatShape), cosTw.T) - matmul(b.reshaped(flatShape), sinTw.T)
    let sTrig = mean(sTrigFlat, axis: -1).reshaped(cachedKeys.dim(0), kvHeads, cachedKeys.dim(2), prepared.repeats)

    let sNorm = sum(
        prepared.normWeight[.newAxis, 0..., .newAxis, 0..., 0...]
            * kMag[0..., 0..., 0..., .newAxis, 0...],
        axis: -1)

    let combined = sTrig + sNorm
    if prepared.repeats > 1 {
        let meanScores = mean(combined, axis: 2, keepDims: true)
        let variance = mean(square(combined - meanScores), axis: 2, keepDims: true)
        let z = (combined - meanScores) / sqrt(variance + MLXArray(1e-8))
        return max(z, axis: -1)
    } else {
        return combined[.ellipsis, 0]
    }
}

func decomposeComplex(
    _ vectors: MLXArray,
    rope: TriAttentionRoPEConfig
) -> (MLXArray, MLXArray) {
    let nFreqs = rope.rotatedDims / 2
    if rope.proportional {
        let half = rope.headDim / 2
        let rotatedHalf = rope.rotatedDims / 2
        let left = vectors[.ellipsis, ..<rotatedHalf]
        let right = vectors[.ellipsis, half ..< (half + rotatedHalf)]
        if rope.traditional {
            return (left, right)
        } else {
            let real = concatenated(
                [left[.ellipsis, .stride(by: 2)], right[.ellipsis, .stride(by: 2)]], axis: -1)
            let imag = concatenated(
                [left[.ellipsis, .stride(from: 1, by: 2)], right[.ellipsis, .stride(from: 1, by: 2)]],
                axis: -1)
            return (real, imag)
        }
    } else if rope.traditional {
        return (vectors[.ellipsis, ..<nFreqs], vectors[.ellipsis, nFreqs ..< (2 * nFreqs)])
    } else {
        let rotated = vectors[.ellipsis, ..<rope.rotatedDims]
        return (rotated[.ellipsis, .stride(by: 2)], rotated[.ellipsis, .stride(from: 1, by: 2)])
    }
}

private func reflectedValue<T>(named name: String, in value: Any) -> T? {
    Mirror(reflecting: value).children.first { $0.label == name }?.value as? T
}

private func reflectedModule(named name: String, in value: Any) -> Module? {
    Mirror(reflecting: value).children.first { $0.label == name }?.value as? Module
}

private func reflectedModule(named names: [String], in value: Any) -> Module? {
    for name in names {
        if let module = reflectedModule(named: name, in: value) {
            return module
        }
    }
    return nil
}

private func reflectedRoPEModule(in value: Any) -> Module? {
    for name in ropePropertyNames {
        if let named = reflectedChild(named: name, in: value),
            let module = discoverImmediateRoPEModule(in: named)
        {
            return module
        }
    }

    for child in Mirror(reflecting: value).children {
        if let module = discoverImmediateRoPEModule(in: child.value) {
            return module
        }
    }

    return nil
}

private func discoverImmediateRoPEModule(in value: Any) -> Module? {
    if let module = value as? Module, isRoPEModule(module) {
        return module
    }
    for child in Mirror(reflecting: value).children {
        if let module = child.value as? Module, isRoPEModule(module) {
            return module
        }
    }
    return nil
}

private func reflectedHeadDim(in value: Any) -> Int? {
    for key in ["headDim", "effectiveHeadDim", "headDimensions", "resolvedHeadDimensions", "globalHeadDim"] {
        if let headDim: Int = reflectedValue(named: key, in: value) {
            return headDim
        }
        if let headDim: Int? = reflectedValue(named: key, in: value) {
            return headDim
        }
    }

    let hiddenSize = reflectedInt(named: ["hiddenSize", "dim"], in: value)
    let attentionHeads = reflectedInt(
        named: ["attentionHeads", "numAttentionHeads", "nHeads", "numHeads"], in: value)
    if let hiddenSize, let attentionHeads, attentionHeads > 0 {
        return hiddenSize / attentionHeads
    }

    for nestedKey in ["args", "configuration", "config", "llmConfig", "llm_config"] {
        if let nested = reflectedChild(named: nestedKey, in: value),
            let headDim = reflectedHeadDim(in: nested)
        {
            return headDim
        }
    }

    return nil
}

private func isRoPEModule(_ module: Module) -> Bool {
    module is RoPE || module is Llama3RoPE || module is ProportionalRoPE || module is YarnRoPE
        || module is SuScaledRoPE
}

private let attentionPropertyNames = ["attention", "selfAttn", "selfAttention", "attn"]
private let ropePropertyNames = ["rope", "rotaryEmbedding", "rotaryEmb", "rotaryPosEmb"]

private func writeTriAttentionCalibrationArrays(
    arrays: [String: MLXArray],
    metadata: [String: String],
    url: URL
) throws {
    try save(arrays: arrays, metadata: metadata, url: url)
}

private func reflectedInt(named names: [String], in value: Any) -> Int? {
    for name in names {
        if let intValue: Int = reflectedValue(named: name, in: value) {
            return intValue
        }
        if let intValue: Int? = reflectedValue(named: name, in: value) {
            return intValue
        }
    }
    return nil
}

private func reflectedChild(named name: String, in value: Any) -> Any? {
    Mirror(reflecting: value).children.first { $0.label == name }?.value
}

private func wrapTriAttentionCalibrationCaches(_ caches: [KVCache]) -> [KVCache] {
    caches.enumerated().map { layerIndex, cache in
        wrapTriAttentionCalibrationCache(cache, layerIndex: layerIndex)
    }
}

private func wrapTriAttentionCalibrationCache(_ cache: KVCache, layerIndex: Int) -> KVCache {
    switch cache {
    case is TriAttentionCalibrationCaptureCache, is MambaCache, is ArraysCache:
        return cache
    case let cacheList as CacheList:
        cacheList.elements = cacheList.elements.map {
            wrapTriAttentionCalibrationCache($0, layerIndex: layerIndex)
        }
        return cacheList
    default:
        return TriAttentionCalibrationCaptureCache(base: cache, layerIndex: layerIndex)
    }
}

private func uniformKVHeads(_ kvHeads: [Int]) throws -> Int {
    guard let first = kvHeads.first else {
        throw TriAttentionError.calibrationCaptureFailed("Model exposes no KV heads for calibration")
    }
    guard kvHeads.allSatisfy({ $0 == first }) else {
        throw TriAttentionError.calibrationCaptureFailed(
            "Calibration runner currently requires uniform kvHeads across layers")
    }
    return first
}

private func inferredCaptureShape(from captures: [Int: [MLXArray]]) throws -> (qHeads: Int, headDim: Int) {
    for layerCaptures in captures.values {
        if let capture = layerCaptures.first {
            return (capture.dim(1), capture.dim(3))
        }
    }
    throw TriAttentionError.calibrationCaptureFailed(
        "No pre-RoPE query captures were recorded during calibration")
}

private func discoverRoPEModule(in model: Module) throws -> Module {
    for (_, module) in model.namedModules() {
        if let attention = reflectedModule(named: attentionPropertyNames, in: module),
            let rope = reflectedRoPEModule(in: attention)
        {
            return rope
        }

        if let rope = reflectedRoPEModule(in: module) {
            return rope
        }
    }

    var visited = Set<ObjectIdentifier>()
    if let rope = discoverRoPEModule(in: model, visited: &visited) {
        return rope
    }

    throw TriAttentionError.unsupportedRoPE(
        "Could not discover supported RoPE from model \(type(of: model))")
}

private func discoverHeadDim(in model: Module) throws -> Int? {
    for (_, module) in model.namedModules() {
        if let attention = reflectedModule(named: attentionPropertyNames, in: module),
            let headDim = reflectedHeadDim(in: attention) ?? reflectedHeadDim(in: module)
        {
            return headDim
        }

        if let headDim = reflectedHeadDim(in: module) {
            return headDim
        }
    }

    var visited = Set<ObjectIdentifier>()
    if let (_, headDim) = discoverRoPEAndHeadDim(in: model, visited: &visited) {
        return headDim
    }
    return nil
}

private func discoverRoPEAndHeadDim(
    in value: Any,
    visited: inout Set<ObjectIdentifier>
) -> (Module, Int)? {
    if let module = value as? Module {
        let identity = ObjectIdentifier(module)
        guard visited.insert(identity).inserted else {
            return nil
        }

        if let attention = reflectedModule(named: attentionPropertyNames, in: module),
            let rope = reflectedRoPEModule(in: attention),
            let headDim = reflectedHeadDim(in: attention) ?? reflectedHeadDim(in: module)
        {
            return (rope, headDim)
        }

        if let rope = reflectedRoPEModule(in: module),
            let headDim = reflectedHeadDim(in: module)
        {
            return (rope, headDim)
        }
    }

    for child in Mirror(reflecting: value).children {
        if let result = discoverRoPEAndHeadDim(in: child.value, visited: &visited) {
            return result
        }
    }

    return nil
}

private func discoverRoPEModule(
    in value: Any,
    visited: inout Set<ObjectIdentifier>
) -> Module? {
    if let module = value as? Module {
        let identity = ObjectIdentifier(module)
        guard visited.insert(identity).inserted else {
            return nil
        }

        if let attention = reflectedModule(named: attentionPropertyNames, in: module),
            let rope = reflectedRoPEModule(in: attention)
        {
            return rope
        }

        if let rope = reflectedRoPEModule(in: module) {
            return rope
        }
    }

    for child in Mirror(reflecting: value).children {
        if let rope = discoverRoPEModule(in: child.value, visited: &visited) {
            return rope
        }
    }

    return nil
}

private func inferHeadDimFromCalibration(
    calibration: TriAttentionCalibrationData,
    rope: Module
) throws -> Int {
    guard let firstLayer = calibration.layers.first else {
        throw TriAttentionError.invalidCalibrationFile("TriAttention calibration file is empty")
    }
    let rotatedDims = firstLayer.qCenterReal.dim(1) * 2

    return rotatedDims
}
