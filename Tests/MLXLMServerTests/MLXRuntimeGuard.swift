import Foundation

#if canImport(Metal)
    import Metal
#endif

/// Returns true when MLX's Metal backend is usable in this process:
/// - A default Metal device exists
/// - A metallib (either `mlx.metallib` colocated with the test binary, or
///   `default.metallib` in a SwiftPM resource bundle) is reachable on disk
///
/// Used to SKIP tests that load real models or run MLX forward passes
/// on hosts that don't have Metal available (CI without GPU, `swift test`
/// builds where the metallib isn't compiled, etc.).
///
/// Checking for the metallib file on disk before loading MLX is essential
/// because `mlx-c` aborts the process with a C++ fatal error when it cannot
/// find the library — that abort cannot be caught from Swift.
///
/// The probe caches its result so callers can invoke it freely.
func runtimeMetallibAvailable() -> Bool {
    _metallibProbeResult
}

private let _metallibProbeResult: Bool = {
    // Allow an explicit opt-out for hosts where we know the metallib is broken.
    if ProcessInfo.processInfo.environment["MLX_METALLIB_OK"] == "0" {
        return false
    }

    #if canImport(Metal)
        guard MTLCreateSystemDefaultDevice() != nil else { return false }
    #else
        return false
    #endif

    return metallibFileReachable()
}()

private func metallibFileReachable() -> Bool {
    let fm = FileManager.default
    let candidateNames = ["mlx.metallib", "default.metallib"]

    // 1) Colocated with the running binary (Bundle.main or executable path).
    let bundleCandidates: [URL] = {
        var urls: [URL] = []
        urls.append(Bundle.main.bundleURL)
        if let resURL = Bundle.main.resourceURL { urls.append(resURL) }
        for bundle in Bundle.allBundles {
            urls.append(bundle.bundleURL)
            if let resURL = bundle.resourceURL { urls.append(resURL) }
        }
        return urls
    }()

    for root in bundleCandidates {
        for name in candidateNames {
            if fm.fileExists(atPath: root.appendingPathComponent(name).path) {
                return true
            }
            // SwiftPM nests resources under `<target>_<module>.bundle/`.
            if let contents = try? fm.contentsOfDirectory(atPath: root.path) {
                for entry in contents where entry.hasSuffix(".bundle") {
                    let nested = root.appendingPathComponent(entry)
                        .appendingPathComponent(name)
                    if fm.fileExists(atPath: nested.path) { return true }
                }
            }
        }
    }
    return false
}
