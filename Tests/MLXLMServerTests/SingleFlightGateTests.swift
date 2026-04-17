import XCTest
@testable import MLXLMServer

final class SingleFlightGateTests: XCTestCase {
    func testAcquireReleaseAllowsSequential() async throws {
        let gate = SingleFlightGate()
        try await gate.acquire()
        await gate.release()
        try await gate.acquire()
        await gate.release()
    }

    func testAcquireWhileHeldThrows() async throws {
        let gate = SingleFlightGate()
        try await gate.acquire()
        do {
            try await gate.acquire()
            XCTFail("second acquire should have thrown")
        } catch SingleFlightError.busy {
            // expected
        }
        await gate.release()
    }

    func testReleaseTwiceIsIdempotent() async throws {
        let gate = SingleFlightGate()
        try await gate.acquire()
        await gate.release()
        await gate.release()   // no crash
    }
}
