import XCTest
@testable import MLXLMServer

final class SSEFrameTests: XCTestCase {
    func testFrameWrapsJSONInDataPrefix() {
        let frame = SSEFrame.data(#"{"a":1}"#)
        XCTAssertEqual(frame, #"data: {"a":1}\#n\#n"#)
    }

    func testDoneMarker() {
        XCTAssertEqual(SSEFrame.done, "data: [DONE]\n\n")
    }

    func testChatCompletionChunkShape() {
        let chunk = SSEFrame.chatCompletionChunk(
            id: "chatcmpl-x",
            model: "m",
            created: 42,
            contentDelta: "hello",
            finishReason: nil
        )
        XCTAssertTrue(chunk.hasPrefix("data: "))
        XCTAssertTrue(chunk.hasSuffix("\n\n"))
        let jsonStart = chunk.index(chunk.startIndex, offsetBy: "data: ".count)
        let jsonEnd = chunk.index(chunk.endIndex, offsetBy: -2)
        let jsonString = String(chunk[jsonStart..<jsonEnd])
        let obj = try! JSONSerialization.jsonObject(with: Data(jsonString.utf8)) as! [String: Any]
        XCTAssertEqual(obj["object"] as? String, "chat.completion.chunk")
        XCTAssertEqual(obj["model"] as? String, "m")
        let choices = obj["choices"] as! [[String: Any]]
        let delta = choices[0]["delta"] as! [String: Any]
        XCTAssertEqual(delta["content"] as? String, "hello")
    }

    func testFinalChunkCarriesFinishReasonAndEmptyDelta() {
        let chunk = SSEFrame.chatCompletionChunk(
            id: "chatcmpl-x",
            model: "m",
            created: 42,
            contentDelta: nil,
            finishReason: "stop",
            usage: Usage(promptTokens: 2, completionTokens: 3, acceptanceRate: 0.8)
        )
        let jsonStart = chunk.index(chunk.startIndex, offsetBy: "data: ".count)
        let jsonEnd = chunk.index(chunk.endIndex, offsetBy: -2)
        let obj = try! JSONSerialization.jsonObject(
            with: Data(String(chunk[jsonStart..<jsonEnd]).utf8)
        ) as! [String: Any]
        let choices = obj["choices"] as! [[String: Any]]
        XCTAssertEqual(choices[0]["finish_reason"] as? String, "stop")
        let delta = choices[0]["delta"] as! [String: Any]
        XCTAssertTrue(delta.isEmpty)
        let usage = obj["usage"] as! [String: Any]
        XCTAssertEqual(usage["acceptance_rate"] as? Double, 0.8)
        XCTAssertEqual(usage["total_tokens"] as? Int, 5)
    }
}
