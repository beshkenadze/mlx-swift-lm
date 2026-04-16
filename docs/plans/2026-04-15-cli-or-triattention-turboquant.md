# Minimal CLI for TriAttention or TurboQuant Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal user-facing CLI to `mlx-swift-lm` that supports local-directory inference with **either** TurboQuant **or** TriAttention (not both), plus a dedicated TriAttention calibration command.

**Architecture:** Add a new SwiftPM executable target that depends on `MLXLMCommon`, `MLXLLM`, and a tokenizer adapter for local-directory model loading. Keep v1 intentionally narrow: local model directories only, no remote downloader flow, and two explicit user paths: `generate --turboquant ...` or `generate --triattention-calibration ...`, plus `triattention calibrate ...`.

**Tech Stack:** SwiftPM executable target, Apple `swift-argument-parser`, `MLXLLM`, `MLXLMCommon`, `MLXLMTokenizers` adapter, `xcodebuild` for verification.

---

## Product Scope

### Supported v1 commands

```bash
# Plain generation
swift run mlx-lm-cli generate \
  --model-directory /path/to/model \
  --prompt "Hello"

# TurboQuant generation (OR mode)
swift run mlx-lm-cli generate \
  --model-directory /path/to/model \
  --prompt "Hello" \
  --turboquant \
  --turboquant-bits 3 \
  --turboquant-seed 7

# TriAttention calibration
swift run mlx-lm-cli triattention calibrate \
  --model-directory /path/to/model \
  --prompt-file calibration.txt \
  --output triattention.safetensors

# TriAttention generation (OR mode)
swift run mlx-lm-cli generate \
  --model-directory /path/to/model \
  --prompt "Hello" \
  --triattention-calibration triattention.safetensors \
  --triattention-budget 2048
```

### Explicit non-goals for v1

- No simultaneous `--turboquant` and `--triattention-calibration`
- No remote Hugging Face download flow
- No VLM-specific image/video CLI in the first slice
- No prompt-cache import/export flags in the first slice

---

### Task 1: Add CLI dependencies and executable target

**Files:**
- Modify: `Package.swift`
- Test: `Tests/MLXLMTests/CLIOptionsTests.swift`

**Step 1: Write the failing parser smoke test**

Create `Tests/MLXLMTests/CLIOptionsTests.swift` with a first parser test that references the root command type:

```swift
import Testing
@testable import MLXLMCLI

@Test
func testGenerateCommandParsesTurboQuantMode() throws {
    let command = try RootCommand.parse([
        "generate",
        "--model-directory", "/tmp/model",
        "--prompt", "Hello",
        "--turboquant"
    ])

    let generate = try #require(command.subcommand as? GenerateCommand)
    #expect(generate.turboQuant.enabled)
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
xcodebuild test -scheme "mlx-swift-lm-Package" -destination "platform=macOS" -configuration Debug -only-testing "MLXLMTests/testGenerateCommandParsesTurboQuantMode()"
```

Expected: fail because `MLXLMCLI` target/types do not exist.

**Step 3: Add package dependencies and executable target**

Modify `Package.swift`:

- add dependency:

```swift
.package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", from: "0.1.0"),
```

- add executable product and target:

```swift
.executable(name: "mlx-lm-cli", targets: ["MLXLMCLI"]),

.executableTarget(
    name: "MLXLMCLI",
    dependencies: [
        "MLXLMCommon",
        "MLXLLM",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "MLXLMTokenizers", package: "swift-tokenizers-mlx"),
    ],
    path: "Sources/MLXLMCLI"
)
```

**Step 4: Run build to verify package graph resolves**

Run:

```bash
xcodebuild build -scheme "mlx-swift-lm-Package" -destination "platform=macOS" -configuration Debug
```

Expected: build still fails because command source files do not exist yet, but dependency resolution succeeds.

**Step 5: Commit**

```bash
git add Package.swift Tests/MLXLMTests/CLIOptionsTests.swift
git commit -m "feat: add CLI package scaffolding"
```

---

### Task 2: Add root command and subcommand skeletons

**Files:**
- Create: `Sources/MLXLMCLI/main.swift`
- Create: `Sources/MLXLMCLI/RootCommand.swift`
- Create: `Sources/MLXLMCLI/Commands/GenerateCommand.swift`
- Create: `Sources/MLXLMCLI/Commands/TriAttentionCommand.swift`
- Create: `Sources/MLXLMCLI/Commands/TriAttentionCalibrateCommand.swift`
- Test: `Tests/MLXLMTests/CLIOptionsTests.swift`

**Step 1: Write the next failing test for subcommand structure**

```swift
@Test
func testTriAttentionCalibrateParsesOutputFile() throws {
    let command = try RootCommand.parse([
        "triattention", "calibrate",
        "--model-directory", "/tmp/model",
        "--prompt-file", "/tmp/calibration.txt",
        "--output", "/tmp/out.safetensors"
    ])

    let tri = try #require(command.subcommand as? TriAttentionCommand)
    let calibrate = try #require(tri.subcommand as? TriAttentionCalibrateCommand)
    #expect(calibrate.output.path == "/tmp/out.safetensors")
}
```

**Step 2: Run test to verify it fails**

Use `xcodebuild test` with the exact selector.

**Step 3: Add minimal command declarations**

Implement:

- `main.swift`

```swift
import ArgumentParser

RootCommand.main()
```

- `RootCommand.swift`

```swift
import ArgumentParser

struct RootCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx-lm-cli",
        subcommands: [GenerateCommand.self, TriAttentionCommand.self]
    )
}
```

- `TriAttentionCommand.swift`

```swift
import ArgumentParser

struct TriAttentionCommand: ParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [TriAttentionCalibrateCommand.self])
}
```

- `GenerateCommand.swift` and `TriAttentionCalibrateCommand.swift` can initially contain only parsed options and `run()` stubs that `fatalError("not implemented")`.

**Step 4: Run parser tests to verify they pass**

Run both CLI parsing tests with `xcodebuild test`.

**Step 5: Commit**

```bash
git add Sources/MLXLMCLI Tests/MLXLMTests/CLIOptionsTests.swift
git commit -m "feat: add CLI command skeletons"
```

---

### Task 3: Define shared option groups and OR-mode validation

**Files:**
- Create: `Sources/MLXLMCLI/Options/LocalModelOptions.swift`
- Create: `Sources/MLXLMCLI/Options/PromptOptions.swift`
- Create: `Sources/MLXLMCLI/Options/GenerationOptions.swift`
- Create: `Sources/MLXLMCLI/Options/TurboQuantOptions.swift`
- Create: `Sources/MLXLMCLI/Options/TriAttentionOptions.swift`
- Test: `Tests/MLXLMTests/CLIOptionsTests.swift`

**Step 1: Write the failing validation tests**

```swift
@Test
func testGenerateRejectsTurboQuantAndTriAttentionTogether() throws {
    #expect(throws: Error.self) {
        _ = try RootCommand.parse([
            "generate",
            "--model-directory", "/tmp/model",
            "--prompt", "Hello",
            "--turboquant",
            "--triattention-calibration", "/tmp/calibration.safetensors"
        ])
    }
}

@Test
func testGenerateRequiresPromptOrPromptFile() throws {
    #expect(throws: Error.self) {
        _ = try RootCommand.parse([
            "generate",
            "--model-directory", "/tmp/model"
        ])
    }
}
```

**Step 2: Run tests to verify they fail**

Run targeted `xcodebuild test`.

**Step 3: Implement shared option types**

Suggested shapes:

```swift
struct LocalModelOptions: ParsableArguments {
    @Option(name: .long) var modelDirectory: String
}

struct PromptOptions: ParsableArguments {
    @Option(name: .long) var prompt: String?
    @Option(name: .long) var promptFile: String?
}

struct TurboQuantOptions: ParsableArguments {
    @Flag(name: .long) var enabled = false
    @Option(name: .long) var bits: Int = 3
    @Option(name: .long) var seed: Int = 0
}

struct TriAttentionOptions: ParsableArguments {
    @Option(name: .long) var calibration: String?
    @Option(name: .long) var budget: Int = 2048
    @Option(name: .long) var divideLength: Int = 128
    @Option(name: .long) var protectRecent: Int = 128
    @Option(name: .long) var protectInitial: Int = 4
}
```

Put validation in `GenerateCommand.validate()`:
- require exactly one of `prompt` / `promptFile`
- forbid simultaneous TurboQuant and TriAttention
- enforce `turboquantBits == 3` in v1

**Step 4: Run parser tests to verify they pass**

**Step 5: Commit**

```bash
git add Sources/MLXLMCLI/Options Sources/MLXLMCLI/Commands/GenerateCommand.swift Tests/MLXLMTests/CLIOptionsTests.swift
git commit -m "feat: add CLI option validation"
```

---

### Task 4: Add local-directory model loading helper

**Files:**
- Create: `Sources/MLXLMCLI/Support/CLIModelLoader.swift`
- Test: `Tests/MLXLMTests/CLILoaderTests.swift`

**Step 1: Write the failing loader test**

Create a test that verifies a local directory path is normalized to a `URL` and the helper asks `loadModelContainer(from:using:)` for local loading.

Because direct model loading is expensive, make this a narrow unit-style test around helper decomposition, e.g. prompt file reading and URL validation. Example:

```swift
@Test
func testPromptFileIsLoadedAsString() throws {
    let url = URL(filePath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString)
    try "hello".write(to: url, atomically: true, encoding: .utf8)
    #expect(try readPrompt(prompt: nil, promptFile: url.path) == "hello")
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement helper**

`CLIModelLoader.swift` should provide:

```swift
import Foundation
import MLXLLM
import MLXLMCommon
import MLXLMTokenizers

func loadLocalModelContainer(modelDirectory: String) async throws -> ModelContainer {
    try await loadModelContainer(
        from: URL(filePath: modelDirectory),
        using: TokenizersLoader()
    )
}

func readPrompt(prompt: String?, promptFile: String?) throws -> String { ... }
```

**Step 4: Run tests to verify they pass**

**Step 5: Commit**

```bash
git add Sources/MLXLMCLI/Support/CLIModelLoader.swift Tests/MLXLMTests/CLILoaderTests.swift
git commit -m "feat: add local CLI model loading helper"
```

---

### Task 5: Map CLI flags to `GenerateParameters` for OR-mode generation

**Files:**
- Modify: `Sources/MLXLMCLI/Commands/GenerateCommand.swift`
- Create: `Sources/MLXLMCLI/Support/CLIParameterBuilder.swift`
- Test: `Tests/MLXLMTests/CLIParameterBuilderTests.swift`

**Step 1: Write the failing mapping tests**

```swift
@Test
func testTurboQuantFlagsMapToGenerateParameters() throws {
    let params = try buildGenerateParameters(
        turbo: .init(enabled: true, bits: 3, seed: 7),
        tri: .init(calibration: nil),
        generation: .init(maxTokens: 128, temperature: 0.7)
    )

    #expect(params.turboQuant?.bits == 3)
    #expect(params.turboQuant?.seed == 7)
    #expect(params.triAttention == nil)
}

@Test
func testTriAttentionFlagsMapToGenerateParameters() async throws {
    let model = DummyCalibrationModel()
    let params = try buildGenerateParameters(
        turbo: .init(enabled: false),
        tri: .init(
            calibration: "/tmp/calibration.safetensors",
            budget: 1024,
            divideLength: 64,
            protectRecent: 64,
            protectInitial: 8
        ),
        generation: .init(),
        model: model
    )
    #expect(params.turboQuant == nil)
    #expect(params.triAttention?.budget == 1024)
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `buildGenerateParameters(...)`**

Rules:
- common sampling/top-k/top-p/maxTokens/prefillStepSize map directly
- if TurboQuant enabled, set `GenerateParameters.turboQuant = TurboQuantConfiguration(bits: seed:)`
- if TriAttention calibration path is present, require loaded model and build:

```swift
let tri = try TriAttentionConfiguration.load(
    calibrationURL: URL(filePath: calibrationPath),
    model: model,
    budget: budget,
    divideLength: divideLength,
    protectRecent: protectRecent,
    protectInitial: protectInitial
)
```

**Step 4: Run mapping tests to verify they pass**

**Step 5: Commit**

```bash
git add Sources/MLXLMCLI/Support/CLIParameterBuilder.swift Sources/MLXLMCLI/Commands/GenerateCommand.swift Tests/MLXLMTests/CLIParameterBuilderTests.swift
git commit -m "feat: map CLI flags to generation parameters"
```

---

### Task 6: Implement `generate` command execution

**Files:**
- Modify: `Sources/MLXLMCLI/Commands/GenerateCommand.swift`
- Test: `Tests/MLXLMTests/CLIGenerateCommandTests.swift`

**Step 1: Write the failing command test around dry command flow**

Keep it unit-sized. Do not hit a real model in the first test. Test the execution helper that returns parsed prompt + selected mode + built parameter summary.

Example:

```swift
@Test
func testGenerateModeSummaryPrefersTurboQuant() throws {
    let summary = try describeGenerationMode(
        turbo: .init(enabled: true, bits: 3, seed: 7),
        tri: .init(calibration: nil)
    )
    #expect(summary == "turboquant")
}
```

Then a second integration-flavored test can use a tiny local fixture model later if available.

**Step 2: Run test to verify it fails**

**Step 3: Implement command run path**

`GenerateCommand.run()` should:
1. read prompt text from `--prompt` or `--prompt-file`
2. load local model container
3. prepare `UserInput(prompt:)`
4. build `GenerateParameters`
5. call `modelContainer.generate(input:parameters:)`
6. print chunks to stdout

Keep output simple in v1:
- stream `.chunk` text to stdout
- ignore `.info` unless `--verbose`
- if `.toolCall`, print to stderr and exit non-zero (tool loop is out of scope)

**Step 4: Run tests/build to verify they pass**

Use `xcodebuild build` plus unit tests.

**Step 5: Commit**

```bash
git add Sources/MLXLMCLI/Commands/GenerateCommand.swift Tests/MLXLMTests/CLIGenerateCommandTests.swift
git commit -m "feat: implement CLI generate command"
```

---

### Task 7: Implement `triattention calibrate` command

**Files:**
- Modify: `Sources/MLXLMCLI/Commands/TriAttentionCalibrateCommand.swift`
- Test: `Tests/MLXLMTests/CLITriAttentionCalibrateTests.swift`

**Step 1: Write the failing calibration command mapping test**

```swift
@Test
func testCalibrateCommandBuildsLMInputFromPromptFile() throws {
    let command = TriAttentionCalibrateCommand(
        modelDirectory: "/tmp/model",
        prompt: nil,
        promptFile: "/tmp/prompt.txt",
        output: URL(filePath: "/tmp/out.safetensors"),
        prefillStepSize: 256
    )

    #expect(command.prefillStepSize == 256)
}
```

Then add a helper-level test for the runner invocation if needed.

**Step 2: Run test to verify it fails**

**Step 3: Implement command run path**

`TriAttentionCalibrateCommand.run()` should:
1. load local model container
2. read prompt text from `--prompt` or `--prompt-file`
3. prepare `LMInput` from `UserInput(prompt:)`
4. inside `modelContainer.perform`, call:

```swift
let calibration = try TriAttentionCalibrationRunner.calibrate(
    model: context.model as! (any LanguageModel & KVCacheDimensionProvider),
    input: lmInput,
    outputURL: output,
    prefillStepSize: prefillStepSize
)
```

If the cast is too broad/unsafe, add a narrow helper in `MLXLMCommon` instead of duplicating logic in CLI.

**Step 4: Run build/tests to verify they pass**

**Step 5: Commit**

```bash
git add Sources/MLXLMCLI/Commands/TriAttentionCalibrateCommand.swift Tests/MLXLMTests/CLITriAttentionCalibrateTests.swift
git commit -m "feat: implement TriAttention calibration command"
```

---

### Task 8: Add CLI docs and user-facing examples

**Files:**
- Modify: `README.md`
- Modify: `skills/mlx-swift-lm/SKILL.md`
- Create: `docs/cli.md`

**Step 1: Write the failing doc test / checklist**

Create a simple manual checklist in the plan execution notes:
- README mentions `mlx-lm-cli`
- README shows OR-mode examples for TurboQuant and TriAttention
- docs mention local-directory-only scope for v1

**Step 2: Add docs**

README additions should include:
- installation note for CLI target
- `swift run mlx-lm-cli generate ... --turboquant`
- `swift run mlx-lm-cli triattention calibrate ...`
- `swift run mlx-lm-cli generate ... --triattention-calibration ...`

`docs/cli.md` should explain:
- local model requirement
- OR semantics: choose TurboQuant or TriAttention, not both
- current non-goals (remote download, VLM CLI, tool loop)

Update `skills/mlx-swift-lm/SKILL.md` GenerateParameters section to include the new fields:

```swift
let params = GenerateParameters(
    turboQuant: TurboQuantConfiguration(bits: 3, seed: 7)
)

let tri = try TriAttentionConfiguration.load(
    calibrationURL: calibrationURL,
    model: model
)
let params = GenerateParameters(triAttention: tri)
```

**Step 3: Verify docs**

Run:

```bash
scripts/verify-docs.sh
```

If that script does not cover README/markdown examples, manually inspect rendered commands.

**Step 4: Commit**

```bash
git add README.md docs/cli.md skills/mlx-swift-lm/SKILL.md
git commit -m "docs: add CLI usage for turboquant and triattention"
```

---

### Task 9: Final verification

**Files:**
- Verify all touched files above

**Step 1: Run CLI parser/unit tests**

Run targeted xcodebuild tests for all new `CLI*Tests`.

**Step 2: Run existing regression for cache features**

Re-run the TurboQuant + TriAttention regression suite already used in this branch to ensure CLI-related changes did not break internals.

**Step 3: Build the package**

```bash
xcodebuild build -scheme "mlx-swift-lm-Package" -destination "platform=macOS" -configuration Debug
```

**Step 4: Smoke-test the executable help output**

```bash
swift run mlx-lm-cli --help
swift run mlx-lm-cli generate --help
swift run mlx-lm-cli triattention calibrate --help
```

Expected:
- help exits 0
- TurboQuant and TriAttention flags appear

**Step 5: Final commit**

```bash
git add Package.swift Sources/MLXLMCLI Tests/MLXLMTests README.md docs/cli.md skills/mlx-swift-lm/SKILL.md
git commit -m "feat: add minimal CLI for turboquant or triattention"
```
