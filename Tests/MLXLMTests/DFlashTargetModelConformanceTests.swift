// Copyright © 2026 Apple Inc.

import MLXLLM
import MLXLMCommon
import Testing

struct DFlashTargetModelConformanceTests {

    @Test("Qwen3Model conforms to DFlashTargetModel")
    func testQwen3Conforms() {
        func accepts<T: DFlashTargetModel>(_ model: T) { _ = model }
        _ = accepts as (Qwen3Model) -> Void
    }
}
