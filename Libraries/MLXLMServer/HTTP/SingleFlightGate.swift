import Foundation

public enum SingleFlightError: Error, Equatable {
    case busy
}

public actor SingleFlightGate {
    private var held: Bool = false

    public init() {}

    public func acquire() async throws {
        guard !held else { throw SingleFlightError.busy }
        held = true
    }

    public func release() {
        held = false
    }
}
