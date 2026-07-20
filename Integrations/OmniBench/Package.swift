// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "mlx-swift-lm-omni-bench",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MLXLMOmniBench", targets: ["MLXLMOmniBench"])
    ],
    dependencies: [
        .package(name: "mlx-swift-lm", path: "../.."),
        .package(
            url: "https://github.com/beshkenadze/omni-bench.git",
            revision: "17e39c9be2dbc7eaca4a968379984bf97c00134d"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            .upToNextMinor(from: "0.31.4")
        ),
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
        .testTarget(
            name: "MLXLMOmniBenchTests",
            dependencies: [
                "MLXLMOmniBench",
                .product(name: "OmniBench", package: "omni-bench"),
            ]
        ),
    ]
)
