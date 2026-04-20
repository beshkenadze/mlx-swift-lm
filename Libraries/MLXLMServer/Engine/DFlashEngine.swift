import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

public final class DFlashEngine: InferenceEngine, @unchecked Sendable {
    public let configuration: DFlashEngineConfiguration

    private let start: Date
    private let stateLock = NSLock()

    private var targetContainer: ModelContainer?
    private var drafter: DFlashDraftModel?
    private var draftConfig: DFlashDraftConfig?
    private var loaded: Bool = false
    private var loadTask: Task<Void, Error>?

    public init(configuration: DFlashEngineConfiguration) {
        self.configuration = configuration
        self.start = Date()
    }

    public func availableModels() async -> [ModelInfo] {
        [ModelInfo(
            id: configuration.modelAlias,
            created: Int(start.timeIntervalSince1970),
            ownedBy: "dflash"
        )]
    }

    public func health() async -> EngineHealth {
        let ready = stateLock.withLock { loaded }
        return EngineHealth(
            ready: ready,
            modelIDs: [configuration.modelAlias],
            uptimeSeconds: Date().timeIntervalSince(start)
        )
    }

    public func load() async throws {
        let task = stateLock.withLock { () -> Task<Void, Error>? in
            if loaded {
                return nil
            }
            if let loadTask {
                return loadTask
            }
            let task = Task { [self] in
                try await performLoad()
            }
            self.loadTask = task
            return task
        }

        guard let task else { return }
        try await task.value
    }

    public func generate(_ request: ChatRequest) -> AsyncThrowingStream<ChatDelta, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard request.modelID == configuration.modelAlias else {
                        throw DFlashEngineError.unknownModel(request.modelID)
                    }

                    try await self.load()

                    let state = self.stateLock.withLock {
                        (
                            loaded: self.loaded,
                            targetContainer: self.targetContainer,
                            drafter: self.drafter,
                            draftConfig: self.draftConfig
                        )
                    }
                    guard state.loaded, let container = state.targetContainer else {
                        throw DFlashEngineError.modelLoadFailed("call load() first")
                    }
                    guard let drafter = state.drafter, let draftConfig = state.draftConfig else {
                        throw DFlashEngineError.draftLoadFailed("call load() first")
                    }

                    try await container.perform(nonSendable: drafter) { ctx, drafter in
                        let rawMessages: [[String: any Sendable]] = request.messages.map { msg in
                            ["role": msg.role, "content": msg.content]
                        }

                        let promptTokenIDs: [Int]
                        do {
                            promptTokenIDs = try ctx.tokenizer.applyChatTemplate(messages: rawMessages)
                        } catch {
                            throw DFlashEngineError.tokenizationFailed(String(describing: error))
                        }

                        let promptTokens = MLXArray(promptTokenIDs).reshaped([1, promptTokenIDs.count])
                        let stopTokenIds = Self.buildStopTokenIds(
                            modelConfiguration: ctx.configuration,
                            tokenizer: ctx.tokenizer
                        )

                        guard let target = ctx.model as? any DFlashTargetModel else {
                            throw DFlashEngineError.invalidTarget(String(describing: type(of: ctx.model)))
                        }

                        let generationCap = max(0, request.maxTokens)
                        var iterator = try DFlashIterator(
                            promptTokens: promptTokens,
                            target: target,
                            drafter: drafter,
                            draftConfig: draftConfig,
                            stopTokenIds: stopTokenIds,
                            maxTokens: generationCap
                        )

                        var generatedTokenIDs: [Int] = []
                        generatedTokenIDs.reserveCapacity(generationCap)
                        var detokenizer = NaiveStreamingDetokenizer(tokenizer: ctx.tokenizer)
                        var finishReason: FinishReason = .stop
                        var sawStopToken = false

                        while let token = iterator.next() {
                            if Task.isCancelled {
                                throw CancellationError()
                            }

                            if token == ctx.tokenizer.unknownTokenId || stopTokenIds.contains(token) {
                                sawStopToken = true
                                finishReason = .stop
                                break
                            }

                            generatedTokenIDs.append(token)
                            detokenizer.append(token: token)

                            guard request.stream else {
                                continue
                            }

                            if let fragment = detokenizer.next(), !fragment.isEmpty {
                                continuation.yield(ChatDelta(textFragments: [fragment]))
                            }
                        }

                        if !sawStopToken && iterator.tokenCount >= generationCap {
                            finishReason = .length
                        }

                        let acceptanceRate =
                            iterator.totalProposed > 0 ? iterator.acceptanceRate : nil
                        log(
                            "dflash usage: prompt=\(promptTokens.dim(1)) completion=\(iterator.tokenCount) acceptanceRate=\(iterator.acceptanceRate) totalProposed=\(iterator.totalProposed) totalAccepted=\(iterator.totalAccepted)"
                        )
                        let usage = Usage(
                            promptTokens: promptTokens.dim(1),
                            completionTokens: iterator.tokenCount,
                            acceptanceRate: acceptanceRate
                        )

                        if request.stream {
                            continuation.yield(
                                ChatDelta(
                                    textFragments: [],
                                    finishReason: finishReason,
                                    usage: usage
                                ))
                        } else {
                            let text = ctx.tokenizer.decode(tokenIds: generatedTokenIDs)
                            continuation.yield(
                                ChatDelta(
                                    textFragments: text.isEmpty ? [] : [text],
                                    finishReason: finishReason,
                                    usage: usage
                                ))
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private func performLoad() async throws {
        do {
            log("loading dflash target model: \(configuration.targetModelID)")
            let targetContainer = try await LLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: .init(id: configuration.targetModelID)
            )

            log("loading dflash draft weights: \(configuration.draftRepositoryID)")
            let drafterAndConfig: (DFlashDraftModel, DFlashDraftConfig)
            do {
                drafterAndConfig = try await DFlashWeightLoader.load(
                    from: configuration.draftRepositoryID
                )
            } catch {
                throw DFlashEngineError.draftLoadFailed(String(describing: error))
            }

            stateLock.withLock {
                self.targetContainer = targetContainer
                self.drafter = drafterAndConfig.0
                self.draftConfig = drafterAndConfig.1
                self.loaded = true
                self.loadTask = nil
            }
            log("dflash model ready: \(configuration.modelAlias)")
        } catch let error as DFlashEngineError {
            stateLock.withLock {
                self.loadTask = nil
            }
            throw error
        } catch {
            stateLock.withLock {
                self.loadTask = nil
            }
            throw DFlashEngineError.modelLoadFailed(String(describing: error))
        }
    }

    private static func buildStopTokenIds(
        modelConfiguration: ModelConfiguration,
        tokenizer: any MLXLMCommon.Tokenizer
    ) -> Set<Int> {
        var stopTokenIds = modelConfiguration.eosTokenIds
        if let tokenizerEOS = tokenizer.eosTokenId {
            stopTokenIds.insert(tokenizerEOS)
        }
        for token in modelConfiguration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) {
                stopTokenIds.insert(id)
            }
        }
        return stopTokenIds
    }

    private func log(_ message: String) {
        FileHandle.standardError.write(Data("\(message)\n".utf8))
    }
}

public enum DFlashEngineError: Error {
    case notYetImplemented
    case unknownModel(String)
    case modelLoadFailed(String)
    case draftLoadFailed(String)
    case tokenizationFailed(String)
    case invalidTarget(String)
}
