import Foundation
import XCTest
@testable import MLXLMServer

final class ChatRequestDecoderTests: XCTestCase {
    func testDecodeMinimalRequest() throws {
        let json = #"""
        {
          "model": "stub-model",
          "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "say hi"}
          ]
        }
        """#
        let request = try ChatRequestDecoder.decode(Data(json.utf8))
        XCTAssertEqual(request.modelID, "stub-model")
        XCTAssertEqual(request.messages.count, 2)
        XCTAssertEqual(request.messages[0].role, "system")
        XCTAssertEqual(request.messages[1].content, "say hi")
        XCTAssertEqual(request.maxTokens, 256)     // default
        XCTAssertEqual(request.stream, false)      // default
        XCTAssertEqual(request.stopSequences, [])
    }

    func testDecodeWithOverrides() throws {
        let json = #"""
        {
          "model": "m",
          "messages": [{"role":"user","content":"hi"}],
          "max_tokens": 10,
          "stream": true,
          "stop": ["\n\n", "<|end|>"]
        }
        """#
        let request = try ChatRequestDecoder.decode(Data(json.utf8))
        XCTAssertEqual(request.maxTokens, 10)
        XCTAssertEqual(request.stream, true)
        XCTAssertEqual(request.stopSequences, ["\n\n", "<|end|>"])
    }

    func testDecodeWithStopAsSingleString() throws {
        let json = #"""
        { "model": "m",
          "messages": [{"role":"user","content":"x"}],
          "stop": "DONE" }
        """#
        let request = try ChatRequestDecoder.decode(Data(json.utf8))
        XCTAssertEqual(request.stopSequences, ["DONE"])
    }

    func testDecodeRejectsEmptyMessages() {
        let json = #"""
        { "model": "m", "messages": [] }
        """#
        XCTAssertThrowsError(try ChatRequestDecoder.decode(Data(json.utf8))) { error in
            guard let e = error as? ChatRequestDecoderError else {
                return XCTFail("wrong error type")
            }
            XCTAssertEqual(e, .emptyMessages)
        }
    }

    func testDecodeRejectsMissingModel() {
        let json = #"""
        { "messages": [{"role":"user","content":"x"}] }
        """#
        XCTAssertThrowsError(try ChatRequestDecoder.decode(Data(json.utf8)))
    }
}
