# llm-tool TriAttention/TurboQuant Demo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a runnable demo path in `mlx-swift-examples/Tools/llm-tool` that proves **either** TurboQuant **or** TriAttention works end-to-end against the new `mlx-swift-lm` APIs.

**Architecture:** Extend `llm-tool`'s existing generation argument surface instead of inventing a new executable. Keep the user model simple: one generation path with TurboQuant flags, one generation path with TriAttention calibration flags, and one dedicated `triattention calibrate` command. Enforce OR semantics in CLI validation so users choose one optimization mode at a time.

**Tech Stack:** `swift-argument-parser`, `MLXLLM`, `MLXLMCommon`, existing `llm-tool` command structure in `mlx-swift-examples`, `xcodebuild` for verification.

---

## Scope and assumptions

- Target repo: `mlx-swift-examples`
- Target tool: `Tools/llm-tool`
- Dependency update required so `llm-tool` sees the new `mlx-swift-lm` APIs:
  - `GenerateParameters.turboQuant`
  - `GenerateParameters.triAttention`
  - `TurboQuantConfiguration`
  - `TriAttentionConfiguration.load(...)`
  - `TriAttentionCalibrationRunner.calibrate(...)`
- v1 keeps the product model intentionally narrow:
  - **TurboQuant OR TriAttention**, not both together
  - no interactive mid-session toggling
  - no new app UI, only `llm-tool`

---

## User-facing commands to support

```bash
# TurboQuant generation
swift run llm-tool eval \
  --model /path/to/model \
  --prompt "Hello" \
  --turboquant \
  --turboquant-bits 3 \
  --turboquant-seed 7

# TriAttention calibration
swift run llm-tool triattention calibrate \
  --model /path/to/model \
  --prompt-file calibration.txt \
  --output triattention.safetensors

# TriAttention generation
swift run llm-tool eval \
  --model /path/to/model \
  --prompt "Hello" \
  --triattention-calibration triattention.safetensors \
  --triattention-budget 2048 \
  --triattention-divide-length 128 \
  --triattention-protect-recent 128 \
  --triattention-protect-initial 4
```

---

### Task 1: Pin the `mlx-swift-lm` dependency to a revision/branch that includes TriAttention + TurboQuant

**Files:**
- Modify: `Package.swift` in `mlx-swift-examples`
- Test: package resolve/build

**Step 1: Write the failing build expectation**

No code test yet. The failure here is dependency-level: until `mlx-swift-examples` depends on the new `mlx-swift-lm`, references like `TurboQuantConfiguration` will not compile.

**Step 2: Update the dependency**

In `mlx-swift-examples/Package.swift`, point `mlx-swift-lm` to the branch or revision that contains your new work.

Example temporary form while developing:

```swift
.package(url: "https://github.com/<your-fork>/mlx-swift-lm", branch: "your-feature-branch")
```

Or pin by revision if the repo conventions prefer it.

**Step 3: Run package resolution/build**

Run:

```bash
xcodebuild build -scheme llm-tool -destination 'platform=macOS'
```

Expected: dependency resolves; build still fails later because CLI flags and command code are not added yet.

**Step 4: Commit**

```bash
git add Package.swift Package.resolved
git commit -m "build: point llm-tool to TriAttention/TurboQuant-enabled mlx-swift-lm"
```

---

### Task 2: Extend `GenerateArguments` with TurboQuant flags

**Files:**
- Modify: `Tools/llm-tool/LLMTool.swift`
- Test: `Tests/LLMToolTests/LLMToolArgumentTests.swift` (or the existing llm-tool test target/file if named differently)

**Step 1: Write the failing parser test**

```swift
func testEvalParsesTurboQuantFlags() throws {
    let command = try LLMTool.parse([
        "eval",
        "--model", "/tmp/model",
        "--prompt", "Hello",
        "--turboquant",
        "--turboquant-bits", "3",
        "--turboquant-seed", "7"
    ])

    let eval = try XCTUnwrap(command.subcommand as? EvaluateCommand)
    XCTAssertTrue(eval.generate.turboQuant)
    XCTAssertEqual(eval.generate.turboQuantBits, 3)
    XCTAssertEqual(eval.generate.turboQuantSeed, 7)
}
```

**Step 2: Run test to verify it fails**

```bash
xcodebuild test -scheme llm-tool -destination 'platform=macOS' -only-testing:LLMToolTests/LLMToolArgumentTests/testEvalParsesTurboQuantFlags
```

Expected: parser/field not found failure.

**Step 3: Add minimal flags to `GenerateArguments`**

In `Tools/llm-tool/LLMTool.swift`, inside `GenerateArguments`, add:

```swift
@Flag(name: .long, help: "Enable TurboQuant KV cache mode")
var turboQuant = false

@Option(name: .long, help: "TurboQuant bit-width (v1 supports 3 only)")
var turboQuantBits: Int = 3

@Option(name: .long, help: "TurboQuant deterministic sign seed")
var turboQuantSeed: Int = 0
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add Tools/llm-tool/LLMTool.swift Tests/LLMToolTests/LLMToolArgumentTests.swift
git commit -m "feat: add turboquant CLI flags to llm-tool"
```

---

### Task 3: Extend `GenerateArguments` with TriAttention generation flags

**Files:**
- Modify: `Tools/llm-tool/LLMTool.swift`
- Test: `Tests/LLMToolTests/LLMToolArgumentTests.swift`

**Step 1: Write the failing parser test**

```swift
func testEvalParsesTriAttentionFlags() throws {
    let command = try LLMTool.parse([
        "eval",
        "--model", "/tmp/model",
        "--prompt", "Hello",
        "--triattention-calibration", "/tmp/tri.safetensors",
        "--triattention-budget", "1024"
    ])

    let eval = try XCTUnwrap(command.subcommand as? EvaluateCommand)
    XCTAssertEqual(eval.generate.triAttentionCalibration, "/tmp/tri.safetensors")
    XCTAssertEqual(eval.generate.triAttentionBudget, 1024)
}
```

**Step 2: Run test to verify it fails**

**Step 3: Add minimal flags**

In `GenerateArguments` add:

```swift
@Option(name: .long, help: "TriAttention calibration file")
var triAttentionCalibration: String?

@Option(name: .long, help: "TriAttention budget")
var triAttentionBudget: Int = 2048

@Option(name: .long, help: "TriAttention divide length")
var triAttentionDivideLength: Int = 128

@Option(name: .long, help: "TriAttention protect recent tokens")
var triAttentionProtectRecent: Int = 128

@Option(name: .long, help: "TriAttention protect initial tokens")
var triAttentionProtectInitial: Int = 4
```

**Step 4: Run parser test to verify it passes**

**Step 5: Commit**

```bash
git add Tools/llm-tool/LLMTool.swift Tests/LLMToolTests/LLMToolArgumentTests.swift
git commit -m "feat: add triattention CLI flags to llm-tool"
```

---

### Task 4: Add OR-mode CLI validation

**Files:**
- Modify: `Tools/llm-tool/LLMTool.swift`
- Test: `Tests/LLMToolTests/LLMToolArgumentTests.swift`

**Step 1: Write the failing validation tests**

```swift
func testEvalRejectsTurboQuantAndTriAttentionTogether() {
    XCTAssertThrowsError(
        try LLMTool.parse([
            "eval",
            "--model", "/tmp/model",
            "--prompt", "Hello",
            "--turboquant",
            "--triattention-calibration", "/tmp/tri.safetensors"
        ])
    )
}

func testEvalRejectsUnsupportedTurboQuantBits() {
    XCTAssertThrowsError(
        try LLMTool.parse([
            "eval",
            "--model", "/tmp/model",
            "--prompt", "Hello",
            "--turboquant",
            "--turboquant-bits", "4"
        ])
    )
}
```

**Step 2: Run tests to verify they fail**

**Step 3: Add `validate()` to `GenerateArguments` or owning command**

Rules:
- forbid TurboQuant + TriAttention together
- require `--triattention-calibration` if any TriAttention tuning flag differs from default
- require `turboquantBits == 3` in v1

Minimal logic:

```swift
if turboQuant && triAttentionCalibration != nil {
    throw ValidationError("Choose either --turboquant or --triattention-calibration, not both.")
}
if turboQuant && turboQuantBits != 3 {
    throw ValidationError("TurboQuant currently supports only --turboquant-bits 3.")
}
```

**Step 4: Run tests to verify they pass**

**Step 5: Commit**

```bash
git add Tools/llm-tool/LLMTool.swift Tests/LLMToolTests/LLMToolArgumentTests.swift
git commit -m "feat: validate triattention or turboquant CLI modes"
```

---

### Task 5: Map TurboQuant flags into `GenerateParameters`

**Files:**
- Modify: `Tools/llm-tool/LLMTool.swift`
- Test: `Tests/LLMToolTests/GenerateParameterMappingTests.swift`

**Step 1: Write the failing mapping test**

```swift
func testTurboQuantFlagsMapToGenerateParameters() {
    let args = GenerateArguments(
        turboQuant: true,
        turboQuantBits: 3,
        turboQuantSeed: 7
    )

    let params = args.generateParameters
    XCTAssertEqual(params.turboQuant?.bits, 3)
    XCTAssertEqual(params.turboQuant?.seed, 7)
    XCTAssertNil(params.triAttention)
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement the mapping**

Inside the existing `GenerateArguments.generateParameters` builder, extend the initializer call:

```swift
turboQuant: turboQuant
    ? TurboQuantConfiguration(bits: turboQuantBits, seed: turboQuantSeed)
    : nil,
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add Tools/llm-tool/LLMTool.swift Tests/LLMToolTests/GenerateParameterMappingTests.swift
git commit -m "feat: map turboquant flags to generate parameters"
```

---

### Task 6: Add a helper that builds `TriAttentionConfiguration` after model load

**Files:**
- Modify: `Tools/llm-tool/LLMTool.swift`
- Create: `Tools/llm-tool/TriAttentionSupport.swift`
- Test: `Tests/LLMToolTests/TriAttentionSupportTests.swift`

**Step 1: Write the failing builder test**

```swift
func testTriAttentionOptionsBuildConfiguration() throws {
    let options = GenerateArguments(
        triAttentionCalibration: "/tmp/tri.safetensors",
        triAttentionBudget: 1024,
        triAttentionDivideLength: 64,
        triAttentionProtectRecent: 32,
        triAttentionProtectInitial: 8
    )

    // Use a tiny dummy model if the llm-tool test target already has one;
    // otherwise test a helper that just packages the file path + numeric options first.
    XCTAssertEqual(options.triAttentionBudget, 1024)
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement a helper**

`TriAttentionSupport.swift` should provide something like:

```swift
import Foundation
import MLXLMCommon

func buildTriAttentionConfiguration(
    calibrationPath: String,
    budget: Int,
    divideLength: Int,
    protectRecent: Int,
    protectInitial: Int,
    model: some LanguageModel
) throws -> TriAttentionConfiguration {
    try TriAttentionConfiguration.load(
        calibrationURL: URL(filePath: calibrationPath),
        model: model,
        budget: budget,
        divideLength: divideLength,
        protectRecent: protectRecent,
        protectInitial: protectInitial
    )
}
```

**Step 4: Run test/build to verify it passes**

**Step 5: Commit**

```bash
git add Tools/llm-tool/TriAttentionSupport.swift Tests/LLMToolTests/TriAttentionSupportTests.swift Tools/llm-tool/LLMTool.swift
git commit -m "feat: add triattention configuration builder for llm-tool"
```

---

### Task 7: Wire TriAttention into `eval` / `generate` execution path

**Files:**
- Modify: `Tools/llm-tool/LLMTool.swift`
- Test: `Tests/LLMToolTests/LLMToolEvaluateTests.swift`

**Step 1: Write the failing execution-path test**

Because full model loading is expensive, first write a narrow test against a helper that decides whether the command will request TurboQuant, TriAttention, or neither after parsing and after model load.

```swift
func testRequestedOptimizationModeIsTriAttention() {
    let args = GenerateArguments(triAttentionCalibration: "/tmp/tri.safetensors")
    XCTAssertEqual(args.requestedOptimizationMode, .triAttention)
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement wiring inside `EvaluateCommand.run()`**

Pattern:
1. parse prompt as usual
2. load model container as usual
3. if `triAttentionCalibration != nil`, build `TriAttentionConfiguration` **after** loading the model
4. construct `GenerateParameters` with that `triAttention` inserted
5. run existing generation flow unchanged

Pseudo-shape:

```swift
let base = generate.generateParameters
let stream = try await modelContainer.perform { context in
    var params = base
    if let calibration = generate.triAttentionCalibration {
        params.triAttention = try buildTriAttentionConfiguration(..., model: context.model)
    }
    return try MLXLMCommon.generate(input: input, parameters: params, context: context)
}
```

Do not try to force TriAttention through a pre-load config file path.

**Step 4: Run tests/build to verify it passes**

**Step 5: Commit**

```bash
git add Tools/llm-tool/LLMTool.swift Tests/LLMToolTests/LLMToolEvaluateTests.swift
git commit -m "feat: wire triattention into llm-tool eval"
```

---

### Task 8: Add `triattention calibrate` command

**Files:**
- Modify: `Tools/llm-tool/LLMTool.swift` (or command registration file)
- Create: `Tools/llm-tool/TriAttentionCommands.swift`
- Test: `Tests/LLMToolTests/TriAttentionCommandTests.swift`

**Step 1: Write the failing parser test**

```swift
func testTriAttentionCalibrateCommandParses() throws {
    let command = try LLMTool.parse([
        "triattention", "calibrate",
        "--model", "/tmp/model",
        "--prompt-file", "/tmp/prompt.txt",
        "--output", "/tmp/out.safetensors"
    ])

    XCTAssertNotNil(command)
}
```

**Step 2: Run test to verify it fails**

**Step 3: Implement the command**

Minimal command shape:

```swift
@main
struct LLMTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        subcommands: [EvaluateCommand.self, ChatCommand.self, TriAttentionCommand.self]
    )
}

struct TriAttentionCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [TriAttentionCalibrateCommand.self])
}
```

`TriAttentionCalibrateCommand.run()` should:
1. load model container
2. load prompt text from `--prompt` or `--prompt-file`
3. prepare `LMInput`
4. invoke `TriAttentionCalibrationRunner.calibrate(...)`
5. print path to output file on success

**Step 4: Run parser/build tests to verify they pass**

**Step 5: Commit**

```bash
git add Tools/llm-tool/TriAttentionCommands.swift Tools/llm-tool/LLMTool.swift Tests/LLMToolTests/TriAttentionCommandTests.swift
git commit -m "feat: add triattention calibration command to llm-tool"
```

---

### Task 9: Add help text / demo examples to `llm-tool`

**Files:**
- Modify: `Tools/llm-tool/README.md` or existing tool docs file if present
- Modify: `Tools/llm-tool/LLMTool.swift` help strings

**Step 1: Add explicit examples**

Document exactly these:

```bash
swift run llm-tool eval --model /path/to/model --prompt "Hello" --turboquant --turboquant-bits 3 --turboquant-seed 7

swift run llm-tool triattention calibrate --model /path/to/model --prompt-file calibration.txt --output triattention.safetensors

swift run llm-tool eval --model /path/to/model --prompt "Hello" --triattention-calibration triattention.safetensors
```

Also document:
- choose **either** TurboQuant **or** TriAttention
- TriAttention requires a precomputed calibration file

**Step 2: Verify help output**

Run:

```bash
swift run llm-tool --help
swift run llm-tool eval --help
swift run llm-tool triattention calibrate --help
```

Expected: new flags and command descriptions appear.

**Step 3: Commit**

```bash
git add Tools/llm-tool/README.md Tools/llm-tool/LLMTool.swift Tools/llm-tool/TriAttentionCommands.swift
git commit -m "docs: add llm-tool examples for turboquant and triattention"
```

---

### Task 10: Final verification

**Files:**
- Verify all touched files above

**Step 1: Run targeted llm-tool tests**

Run all new parser/mapping/calibration tests with `xcodebuild test -scheme llm-tool -destination 'platform=macOS'`.

**Step 2: Build the tool**

```bash
xcodebuild build -scheme llm-tool -destination 'platform=macOS'
```

**Step 3: Smoke test help output**

```bash
swift run llm-tool eval --help
swift run llm-tool triattention calibrate --help
```

**Step 4: If a local model is available, run one smoke command for each mode**

TurboQuant:

```bash
swift run llm-tool eval \
  --model /path/to/model \
  --prompt "Hello" \
  --turboquant
```

TriAttention generation:

```bash
swift run llm-tool eval \
  --model /path/to/model \
  --prompt "Hello" \
  --triattention-calibration triattention.safetensors
```

TriAttention calibration:

```bash
swift run llm-tool triattention calibrate \
  --model /path/to/model \
  --prompt-file calibration.txt \
  --output triattention.safetensors
```

**Step 5: Final commit**

```bash
git add Tools/llm-tool Tests/LLMToolTests
git commit -m "feat: demo triattention or turboquant in llm-tool"
```
