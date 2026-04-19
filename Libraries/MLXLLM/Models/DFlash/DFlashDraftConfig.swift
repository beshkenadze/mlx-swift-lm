// Copyright © 2026 Apple Inc.

import Foundation

public struct DFlashDraftConfig: Sendable, Codable {
    public let modelType: String
    public let blockSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let numTargetLayers: Int
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let attentionBias: Bool
    public let tieWordEmbeddings: Bool
    public let vocabSize: Int
    public let dflashConfig: DFlashSubConfig

    public init(
        modelType: String,
        blockSize: Int,
        hiddenSize: Int,
        intermediateSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        numTargetLayers: Int,
        maxPositionEmbeddings: Int,
        ropeTheta: Float,
        rmsNormEps: Float,
        attentionBias: Bool = false,
        tieWordEmbeddings: Bool,
        vocabSize: Int,
        dflashConfig: DFlashSubConfig
    ) {
        self.modelType = modelType
        self.blockSize = blockSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.numTargetLayers = numTargetLayers
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.attentionBias = attentionBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.vocabSize = vocabSize
        self.dflashConfig = dflashConfig
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case blockSize = "block_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numTargetLayers = "num_target_layers"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case vocabSize = "vocab_size"
        case dflashConfig = "dflash_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decode(String.self, forKey: .modelType)
        self.blockSize = try container.decode(Int.self, forKey: .blockSize)
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.numTargetLayers = try container.decode(Int.self, forKey: .numTargetLayers)
        self.maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        self.ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.tieWordEmbeddings = try container.decode(Bool.self, forKey: .tieWordEmbeddings)
        self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        self.dflashConfig = try container.decode(DFlashSubConfig.self, forKey: .dflashConfig)
    }

    public struct DFlashSubConfig: Sendable, Codable {
        public let maskTokenId: Int
        public let targetLayerIds: [Int]

        public init(maskTokenId: Int, targetLayerIds: [Int]) {
            self.maskTokenId = maskTokenId
            self.targetLayerIds = targetLayerIds
        }

        enum CodingKeys: String, CodingKey {
            case maskTokenId = "mask_token_id"
            case targetLayerIds = "target_layer_ids"
        }
    }
}
