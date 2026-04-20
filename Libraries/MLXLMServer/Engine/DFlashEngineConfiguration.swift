import Foundation

public struct DFlashEngineConfiguration: Sendable {
    public let targetModelID: String
    public let draftRepositoryID: String
    public let modelAlias: String
    public let defaultMaxTokens: Int

    public init(
        targetModelID: String,
        draftRepositoryID: String,
        modelAlias: String,
        defaultMaxTokens: Int = 256
    ) {
        self.targetModelID = targetModelID
        self.draftRepositoryID = draftRepositoryID
        self.modelAlias = modelAlias
        self.defaultMaxTokens = defaultMaxTokens
    }
}
