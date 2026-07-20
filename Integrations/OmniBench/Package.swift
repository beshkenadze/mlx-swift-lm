// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "mlx-swift-lm-omni-bench",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MLXLMOmniBench", targets: ["MLXLMOmniBench"]),
        .executable(name: "omni-bench-mlx-lm", targets: ["OmniBenchMLXLMCLI"]),
    ],
    dependencies: [
        .package(name: "mlx-swift-lm", path: "../.."),
        .package(
            url: "https://github.com/beshkenadze/omni-bench.git",
            revision: "cd302017303270853cff99b7d0c8f7c2acbd04a4"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            .upToNextMinor(from: "0.31.4")
        ),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MLXLMOmniBench",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "OmniBench", package: "omni-bench"),
            ]
        ),
        .executableTarget(
            name: "OmniBenchMLXLMCLI",
            dependencies: [
                "MLXLMOmniBench",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "OmniBench", package: "omni-bench"),
            ]
        ),
        .testTarget(
            name: "MLXLMOmniBenchTests",
            dependencies: [
                "MLXLMOmniBench",
                .product(name: "OmniBench", package: "omni-bench"),
            ]
        ),
    ]
)
