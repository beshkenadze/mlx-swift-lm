// Copyright © 2026 Apple Inc.

@testable import MLXLMCommon
import Testing

struct DFlashIteratorEOSTests {

    private let specHelper = DFlashIteratorSpecRoundTests()

    @Test(
        "DFlashIterator truncates speculative output at EOS and terminates",
        .disabled(if: !dflashMetallibAvailable, "MLX metallib unavailable")
    )
    func testSpeculationRoundStopsAtEOSInsideAcceptedBlock() throws {
        let eosToken = Int32(7)
        let proposals: [Int32] = [1, 2, 3, 4, 5, eosToken, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        let posterior = proposals + [Int32(42)]

        var iterator = try specHelper.makeIterator(
            proposals: proposals,
            posterior: posterior,
            stopTokenIds: [Int(eosToken)]
        )

        #expect(iterator.next() == 3)
        #expect(iterator.next() == Int(proposals[0]))

        #expect(iterator.pendingTokens == proposals.prefix(6).map(Int.init))
        #expect(iterator.pendingTokens.count == 6)
        #expect(iterator.pendingIndex == 1)
        #expect(iterator.isTerminated)
        #expect(iterator.committedLength == 3 + 6)
        #expect(iterator.targetCache.first?.offset == iterator.committedLength)
        #expect(iterator.draftCache.first?.offset == 3)

        let lastTargetHidden = try #require(iterator.lastTargetHidden)
        #expect(lastTargetHidden.shape[1] == 6)

        let drained = Array((0..<5).compactMap { _ in iterator.next() })
        #expect(drained == proposals[1...5].map(Int.init))
        #expect(iterator.next() == nil)
    }

    @Test(
        "DFlashIterator truncates speculative output to remaining maxTokens budget",
        .disabled(if: !dflashMetallibAvailable, "MLX metallib unavailable")
    )
    func testSpeculationRoundStopsAtMaxTokensBudget() throws {
        let proposals = Array(21...35).map(Int32.init)
        let posterior = proposals + [Int32(36)]

        var iterator = try specHelper.makeIterator(
            proposals: proposals,
            posterior: posterior,
            maxTokens: 3
        )

        var emitted: [Int] = []
        while let token = iterator.next() {
            emitted.append(token)
        }

        #expect(emitted == [3, Int(proposals[0]), Int(proposals[1])])
        #expect(iterator.tokenCount == 3)
        #expect(iterator.pendingTokens == proposals.prefix(2).map(Int.init))
        #expect(iterator.pendingIndex == 2)
        #expect(iterator.committedLength == 3 + 2)
        #expect(iterator.committedLength - 3 <= 3)
        #expect(iterator.targetCache.first?.offset == iterator.committedLength)

        let lastTargetHidden = try #require(iterator.lastTargetHidden)
        #expect(lastTargetHidden.shape[1] == 2)
        #expect(iterator.next() == nil)
    }
}
