import Foundation

public struct BaselineEngineConfiguration: Sendable {
    public let modelID: String
    public let defaultMaxTokens: Int
    public let contextWindow: Int

    public init(
        modelID: String,
        defaultMaxTokens: Int = 256,
        contextWindow: Int = 4096
    ) {
        self.modelID = modelID
        self.defaultMaxTokens = defaultMaxTokens
        self.contextWindow = contextWindow
    }
}
