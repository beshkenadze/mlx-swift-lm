import Foundation
import MLX
import MLXLMCommon
import Testing

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { KVCacheSimple() },
    { RotatingKVCache(maxSize: 32) },
    { QuantizedKVCache() },
    { ChunkedKVCache(chunkSize: 16) },
    { ArraysCache(size: 2) },
    { MambaCache() },
]

private final class DummyTurboQuantCache: TurboQuantKVCacheProtocol {
    var offset: Int = 0
    var maxSize: Int? { nil }
    let turboQuantBits: Int = 3
    let turboQuantSeed: Int = 11
    private(set) var updateCalls = 0
    private(set) var packedUpdateCalls = 0
    let materializedKeys: MLXArray
    let materializedValues: MLXArray
    let packedKeys: TurboQuantPackedTensorState
    let packedValues: TurboQuantPackedTensorState

    init(materializedKeys: MLXArray, materializedValues: MLXArray) {
        let simple = KVCacheSimple()
        _ = simple.update(keys: materializedKeys, values: materializedValues)
        let packedCache = simple.toTurboQuant(bits: turboQuantBits, seed: turboQuantSeed)
        let packedState = packedCache.getTurboQuantPackedState()!
        let reconstructed = packedCache.getTurboQuantState()!
        self.materializedKeys = reconstructed.0
        self.materializedValues = reconstructed.1
        self.packedKeys = packedState.0
        self.packedValues = packedState.1
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("Regular update path should not be used")
    }

    func innerState() -> [MLXArray] { [] }

    var state: [MLXArray] {
        get { [] }
        set {}
    }
    var metaState: [String] {
        get { [""] }
        set {}
    }
    var isTrimmable: Bool { false }
    @discardableResult func trim(_ n: Int) -> Int { 0 }
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode { .none }

    func updateTurboQuant(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        updateCalls += 1
        offset += keys.dim(2)
        return (materializedKeys, materializedValues)
    }

    func updateTurboQuantPacked(keys: MLXArray, values: MLXArray) -> (
        TurboQuantPackedTensorState, TurboQuantPackedTensorState
    ) {
        packedUpdateCalls += 1
        offset += keys.dim(2)
        return (packedKeys, packedValues)
    }

    func getTurboQuantState() -> (MLXArray, MLXArray)? {
        (materializedKeys, materializedValues)
    }

    func getTurboQuantPackedState() -> (TurboQuantPackedTensorState, TurboQuantPackedTensorState)? {
        (packedKeys, packedValues)
    }

    func copy() -> any KVCache {
        DummyTurboQuantCache(materializedKeys: materializedKeys, materializedValues: materializedValues)
    }
}

private func makeTriAttentionConfiguration() -> TriAttentionConfiguration {
    let layer = TriAttentionLayerCalibration(
        qCenterReal: MLXArray.zeros([2, 4], dtype: .float32),
        qCenterImag: MLXArray.zeros([2, 4], dtype: .float32),
        qMeanNorm: MLXArray.ones([2, 4], dtype: .float32)
    )
    let calibration = TriAttentionCalibrationData(layers: [layer], qHeads: 2, kvHeads: 2)
    let rope = TriAttentionRoPEConfig(
        headDim: 8,
        rotatedDims: 8,
        traditional: false,
        omega: MLXArray.ones([4], dtype: .float32)
    )
    return TriAttentionConfiguration(
        calibration: calibration,
        rope: rope,
        budget: 4,
        divideLength: 1,
        protectRecent: 1,
        protectInitial: 1
    )
}

private func makeTurboQuantConfiguration() -> TurboQuantConfiguration {
    TurboQuantConfiguration(bits: 3, seed: 7)
}

@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheSerialization(creator: (() -> any KVCache)) async throws {
    let cache = (0 ..< 10).map { _ in creator() }
    let keys = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
    for item in cache {
        switch item {
        case let arrays as ArraysCache:
            arrays[0] = keys
            arrays[1] = values
        case let quantized as QuantizedKVCache:
            _ = quantized.updateQuantized(keys: keys, values: values)
        default:
            _ = item.update(keys: keys, values: values)
        }
    }

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(cache.count == loadedCache.count)
    for (lhs, rhs) in zip(cache, loadedCache) {
        #expect(type(of: lhs) == type(of: rhs))
        #expect(lhs.metaState == rhs.metaState)
        #expect(lhs.state.count == rhs.state.count)
    }
}

/// Verify that copy() produces an independent cache: same type, same state,
/// but mutating the copy does not affect the original.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyIsIndependent(creator: (() -> any KVCache)) async throws {
    let original = creator()

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)

    // populate the original
    switch original {
    case let arrays as ArraysCache:
        arrays[0] = keys
        arrays[1] = values
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: keys, values: values)
    default:
        _ = original.update(keys: keys, values: values)
    }

    let originalOffset = original.offset
    let originalState = original.state
    eval(originalState)
    let originalMeta = original.metaState

    // copy
    let copied = original.copy()

    // same type
    #expect(type(of: original) == type(of: copied))

    // same offset and metadata
    #expect(copied.offset == originalOffset)
    #expect(copied.metaState == originalMeta)

    // same state values
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (origArr, copyArr) in zip(originalState, copiedState) {
        #expect(origArr.shape == copyArr.shape)
        #expect(allClose(origArr, copyArr).item(Bool.self))
    }

    // mutate the copy — push more tokens through it
    let moreKeys = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    let moreValues = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)

    switch copied {
    case let arrays as ArraysCache:
        // overwrite slot 0 with a different array
        arrays[0] = moreKeys
    case let quantized as QuantizedKVCache:
        _ = quantized.updateQuantized(keys: moreKeys, values: moreValues)
    default:
        _ = copied.update(keys: moreKeys, values: moreValues)
    }

    // original must be unchanged
    #expect(original.offset == originalOffset)
    #expect(original.metaState == originalMeta)
    let currentState = original.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (origArr, savedArr) in zip(currentState, originalState) {
        #expect(origArr.shape == savedArr.shape)
        #expect(allClose(origArr, savedArr).item(Bool.self))
    }
}

/// copy() on an empty (unpopulated) cache must not crash.
@Test(
    .serialized,
    arguments: cacheCreators)
func testCacheCopyOnEmptyCache(creator: (() -> any KVCache)) async throws {
    let empty = creator()
    let copied = empty.copy()

    #expect(type(of: empty) == type(of: copied))
    #expect(copied.offset == 0)
    #expect(copied.state.count == empty.state.count)
}

/// CacheList.copy() produces independent sub-caches.
@Test
func testCacheListCopyIsIndependent() async throws {
    let sub1 = KVCacheSimple()
    let sub2 = RotatingKVCache(maxSize: 32)
    let composite = CacheList(sub1, sub2)

    let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
    _ = sub1.update(keys: keys, values: values)
    _ = sub2.update(keys: keys, values: values)

    // snapshot original state — eval to materialize before copy
    let originalState = composite.state
    eval(originalState)
    let originalOffset0 = sub1.offset
    let originalOffset1 = sub2.offset

    let copied = composite.copy()

    #expect(copied is CacheList)
    let copiedState = copied.state
    eval(copiedState)
    #expect(copiedState.count == originalState.count)
    for (orig, copy) in zip(originalState, copiedState) {
        #expect(orig.shape == copy.shape)
        #expect(allClose(orig, copy).item(Bool.self))
    }

    // mutate inside the copy
    let copiedList = copied as! CacheList
    _ = copiedList[0].update(
        keys: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16),
        values: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
    )

    // originals unchanged
    #expect(sub1.offset == originalOffset0)
    #expect(sub2.offset == originalOffset1)
    let currentState = composite.state
    eval(currentState)
    #expect(currentState.count == originalState.count)
    for (orig, saved) in zip(currentState, originalState) {
        #expect(orig.shape == saved.shape)
        #expect(allClose(orig, saved).item(Bool.self))
    }
}

@Test
func testWrapTriAttentionCachesWrapsKVLeavesOnly() async throws {
    let parameters = GenerateParameters(triAttention: makeTriAttentionConfiguration())
    let caches: [KVCache] = [
        KVCacheSimple(),
        RotatingKVCache(maxSize: 32),
        CacheList(MambaCache(), KVCacheSimple()),
    ]

    let wrapped = wrapTriAttentionCaches(caches, parameters: parameters)

    #expect(wrapped[0] is TriAttentionCache)
    #expect(!(wrapped[1] is TriAttentionCache))

    let composite = try #require(wrapped[2] as? CacheList)
    #expect(composite[0] is MambaCache)
    #expect(composite[1] is TriAttentionCache)
}

@Test
func testTriAttentionConfigurationRejectsKVQuantization() async throws {
    let parameters = GenerateParameters(
        kvBits: 4,
        triAttention: makeTriAttentionConfiguration()
    )

    do {
        try validateTriAttentionConfiguration(parameters: parameters)
        Issue.record("Expected TriAttention + kvBits validation to throw")
    } catch let error as TriAttentionError {
        #expect(error == .incompatibleWithQuantizedKV)
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test
func testSavePromptCacheRejectsTriAttentionCache() async throws {
    let cache = TriAttentionCache(
        base: KVCacheSimple(),
        configuration: makeTriAttentionConfiguration(),
        layerIndex: 0
    )
    _ = cache.update(
        keys: MLXArray.ones([1, 2, 2, 8], dtype: .float32),
        values: MLXArray.ones([1, 2, 2, 8], dtype: .float32)
    )

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    do {
        try savePromptCache(url: url, cache: [cache])
        Issue.record("Expected TriAttention cache serialization to fail")
    } catch let error as TriAttentionError {
        #expect(error == .cacheSerializationUnsupported)
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test
func testAttentionWithCacheUpdateUsesTurboQuantPackedPath() async throws {
    let queries = MLXArray.ones([1, 2, 1, 4], dtype: .float32)
    let incomingKeys = MLXArray.zeros([1, 2, 1, 4], dtype: .float32)
    let incomingValues = MLXArray.zeros([1, 2, 1, 4], dtype: .float32)
    let cachedKeys = MLXArray.ones([1, 2, 3, 4], dtype: .float32) * 2
    let cachedValues = MLXArray.ones([1, 2, 3, 4], dtype: .float32) * 3
    let cache = DummyTurboQuantCache(materializedKeys: cachedKeys, materializedValues: cachedValues)

    let output = attentionWithCacheUpdate(
        queries: queries,
        keys: incomingKeys,
        values: incomingValues,
        cache: cache,
        scale: 0.5,
        mask: .none
    )

    let expected = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: cache.materializedKeys,
        values: cache.materializedValues,
        scale: 0.5,
        mask: .none
    )

    #expect(cache.updateCalls == 0)
    #expect(cache.packedUpdateCalls == 1)
    #expect(allClose(output, expected).item(Bool.self))
}

@Test
func testTurboQuantPackedAttentionMatchesMaterializedAttention() async throws {
    let queries = MLXArray(
        [Float32(1), 0, 1, 0,
         0, 1, 0, 1]
    ).reshaped(1, 2, 1, 4)
    let keys = MLXArray(
        [Float32(1), 2, 3, 4,
         2, 3, 4, 5,
         3, 4, 5, 6,
         6, 5, 4, 3,
         5, 4, 3, 2,
         4, 3, 2, 1]
    ).reshaped(1, 2, 3, 4)
    let values = MLXArray(
        [Float32(2), 1, 0, -1,
         3, 2, 1, 0,
         4, 3, 2, 1,
         1, 0, -1, -2,
         2, 1, 0, -1,
         3, 2, 1, 0]
    ).reshaped(1, 2, 3, 4)

    let turbo = KVCacheSimple().toTurboQuant(bits: 3, seed: 5)
    let packed = turbo.updateTurboQuantPacked(keys: keys, values: values)
    let materialized = try #require(turbo.getTurboQuantState())

    let packedOutput = turboQuantScaledDotProductAttention(
        queries: queries,
        packedKeys: packed.0,
        packedValues: packed.1,
        scale: 0.5,
        mask: .none,
        bits: turbo.turboQuantBits,
        seed: turbo.turboQuantSeed,
        sequenceChunkSize: 1
    )
    let materializedOutput = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: materialized.0,
        values: materialized.1,
        scale: 0.5,
        mask: .none
    )

    #expect(allClose(packedOutput, materializedOutput, rtol: 1e-4, atol: 1e-4).item(Bool.self))
}

@Test
func testKVCacheSimpleCanConvertToTurboQuantCache() async throws {
    let simple = KVCacheSimple()
    let keys = MLXArray(
        [Float32(1), 2, 3, 4,
         2, 3, 4, 5,
         3, 4, 5, 6,
         6, 5, 4, 3,
         5, 4, 3, 2,
         4, 3, 2, 1]
    ).reshaped(1, 2, 3, 4)
    let values = MLXArray(
        [Float32(2), 1, 0, -1,
         3, 2, 1, 0,
         4, 3, 2, 1,
         1, 0, -1, -2,
         2, 1, 0, -1,
         3, 2, 1, 0]
    ).reshaped(1, 2, 3, 4)
    _ = simple.update(keys: keys, values: values)

    let turbo = simple.toTurboQuant(seed: 123)
    let state = try #require(turbo.getTurboQuantState())

    #expect(turbo.offset == simple.offset)
    #expect(turbo.state.count == 4)
    #expect(turbo.state[0].dtype == .uint8)
    #expect(turbo.state[1].dtype == .float32)
    #expect(turbo.state[2].dtype == .uint8)
    #expect(turbo.state[3].dtype == .float32)
    #expect(turbo.state[0].shape == [1, 2, 3, 2])
    #expect(turbo.state[2].shape == [1, 2, 3, 2])
    #expect(state.0.shape == keys.shape)
    #expect(state.1.shape == values.shape)

    let keyNorms = sqrt(sum(keys * keys, axis: -1))
    let valueNorms = sqrt(sum(values * values, axis: -1))
    let recoveredKeyNorms = sqrt(sum(state.0 * state.0, axis: -1))
    let recoveredValueNorms = sqrt(sum(state.1 * state.1, axis: -1))

    #expect(allClose(recoveredKeyNorms, keyNorms, rtol: 0.25, atol: 0.25).item(Bool.self))
    #expect(allClose(recoveredValueNorms, valueNorms, rtol: 0.25, atol: 0.25).item(Bool.self))
}

@Test
func testWrapGenerationCachesWrapsTurboQuantLeavesOnly() async throws {
    let parameters = GenerateParameters(turboQuant: makeTurboQuantConfiguration())
    let caches: [KVCache] = [
        KVCacheSimple(),
        RotatingKVCache(maxSize: 32),
        CacheList(MambaCache(), KVCacheSimple()),
    ]

    let wrapped = wrapGenerationCaches(caches, parameters: parameters)

    #expect(wrapped[0] is TurboQuantKVCache)
    #expect(!(wrapped[1] is TurboQuantKVCache))

    let composite = try #require(wrapped[2] as? CacheList)
    #expect(composite[0] is MambaCache)
    #expect(composite[1] is TurboQuantKVCache)
}

@Test
func testTurboQuantConfigurationRejectsKVQuantization() async throws {
    let parameters = GenerateParameters(
        kvBits: 4,
        turboQuant: makeTurboQuantConfiguration()
    )

    do {
        try validateGenerationCacheConfiguration(parameters: parameters)
        Issue.record("Expected TurboQuant + kvBits validation to throw")
    } catch let error as TurboQuantError {
        #expect(error == .incompatibleWithQuantizedKV)
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test
func testTurboQuantPromptCacheSerializationRoundTrip() async throws {
    let simple = KVCacheSimple()
    let keys = MLXArray(
        [Float32(1), 2, 3, 4,
         2, 3, 4, 5,
         3, 4, 5, 6,
         6, 5, 4, 3,
         5, 4, 3, 2,
         4, 3, 2, 1]
    ).reshaped(1, 2, 3, 4)
    let values = MLXArray(
        [Float32(2), 1, 0, -1,
         3, 2, 1, 0,
         4, 3, 2, 1,
         1, 0, -1, -2,
         2, 1, 0, -1,
         3, 2, 1, 0]
    ).reshaped(1, 2, 3, 4)
    _ = simple.update(keys: keys, values: values)
    let turbo = simple.toTurboQuant(bits: 3, seed: 17)

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: [turbo], metadata: ["kind": "turbo"])
    let (loadedCache, metadata) = try loadPromptCache(url: url)

    #expect(metadata["kind"] == "turbo")
    #expect(loadedCache.count == 1)
    let loadedTurbo = try #require(loadedCache[0] as? TurboQuantKVCache)
    #expect(loadedTurbo.metaState == turbo.metaState)
    #expect(loadedTurbo.state.count == turbo.state.count)

    let loadedState = try #require(loadedTurbo.getTurboQuantState())
    let originalState = try #require(turbo.getTurboQuantState())
    #expect(allClose(loadedState.0, originalState.0, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    #expect(allClose(loadedState.1, originalState.1, rtol: 1e-5, atol: 1e-5).item(Bool.self))
}

@Test
func testCacheListWithTurboQuantPromptCacheSerializationRoundTrip() async throws {
    let mamba = MambaCache()
    let mambaA = MLXArray.ones([1, 4], dtype: .float32)
    let mambaB = MLXArray.ones([1, 4], dtype: .float32) * 2
    mamba[0] = mambaA
    mamba[1] = mambaB

    let simple = KVCacheSimple()
    let keys = MLXArray(
        [Float32(1), 2, 3, 4,
         2, 3, 4, 5,
         3, 4, 5, 6,
         6, 5, 4, 3,
         5, 4, 3, 2,
         4, 3, 2, 1]
    ).reshaped(1, 2, 3, 4)
    let values = MLXArray(
        [Float32(2), 1, 0, -1,
         3, 2, 1, 0,
         4, 3, 2, 1,
         1, 0, -1, -2,
         2, 1, 0, -1,
         3, 2, 1, 0]
    ).reshaped(1, 2, 3, 4)
    _ = simple.update(keys: keys, values: values)
    let turbo = simple.toTurboQuant(bits: 3, seed: 19)

    let composite = CacheList(mamba, turbo)
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: [composite], metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    let loadedComposite = try #require(loadedCache.first as? CacheList)
    #expect(loadedComposite.elements.count == 2)
    #expect(loadedComposite[0] is MambaCache)
    #expect(loadedComposite[1] is TurboQuantKVCache)

    let loadedMamba = try #require(loadedComposite[0] as? MambaCache)
    #expect(allClose(try #require(loadedMamba[0]), mambaA).item(Bool.self))
    #expect(allClose(try #require(loadedMamba[1]), mambaB).item(Bool.self))

    let loadedTurbo = try #require(loadedComposite[1] as? TurboQuantKVCache)
    let loadedState = try #require(loadedTurbo.getTurboQuantState())
    let originalState = try #require(turbo.getTurboQuantState())
    #expect(allClose(loadedState.0, originalState.0, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    #expect(allClose(loadedState.1, originalState.1, rtol: 1e-5, atol: 1e-5).item(Bool.self))
}
