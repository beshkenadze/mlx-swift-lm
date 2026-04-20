// Copyright © 2026 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN

public enum DFlashWeightLoaderError: LocalizedError {
    case unexpectedWeightKey(String)

    public var errorDescription: String? {
        switch self {
        case .unexpectedWeightKey(let key):
            return "Unexpected DFlash weight key: \(key)"
        }
    }
}

public enum DFlashWeightLoader {
    public static let repositoryID = "z-lab/Qwen3-4B-DFlash-b16"

    private static let configFilename = "config.json"
    private static let weightsFilename = "model.safetensors"
    private static let allowedLeafKeys: Set<String> = [
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
    ]
    private static let allowedLayerRange = 0 ..< 5

    public static func load(from repositoryID: String = Self.repositoryID) async throws
        -> (DFlashDraftModel, DFlashDraftConfig)
    {
        let hub = HubApi()
        let modelDirectory = try await hub.snapshot(from: repositoryID)
        return try load(modelDirectory: modelDirectory)
    }

    static func load(modelDirectory: URL) throws -> (DFlashDraftModel, DFlashDraftConfig) {
        let configurationURL = modelDirectory.appending(component: configFilename)
        let weightsURL = modelDirectory.appending(component: weightsFilename)

        let configData = try Data(contentsOf: configurationURL)
        let config = try JSONDecoder().decode(DFlashDraftConfig.self, from: configData)

        let (weights, _) = try loadArraysAndMetadata(url: weightsURL)
        try validateWeightKeys(weights.keys)

        let model = DFlashDraftModel(config)
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])

        eval(model)
        return (model, config)
    }

    private static func validateWeightKeys<S: Sequence>(_ keys: S) throws where S.Element == String {
        for key in keys {
            if allowedLeafKeys.contains(key) {
                continue
            }

            let components = key.split(separator: ".", omittingEmptySubsequences: false)
            if components.count >= 3,
                components[0] == "layers",
                let layerIndex = Int(components[1]),
                allowedLayerRange.contains(layerIndex)
            {
                continue
            }

            throw DFlashWeightLoaderError.unexpectedWeightKey(key)
        }
    }
}
