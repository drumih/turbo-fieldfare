import Foundation

/// The plugin's own load arithmetic: the window and the bytes it plans from
/// what the user asked (`--max-context`, `--ram-budget`). A real plugin sizes
/// weights, expert slots and K/V here; the host prints the plan and uses
/// `min(asked, context)` as the session's window.
public struct TemplateLoadPlan: Sendable, Equatable {
    /// The longest window the template serves.
    public static let modelContext = 4_096
    /// State bytes per position: one token id.
    public static let bytesPerPosition = MemoryLayout<Int32>.size

    public let context: Int
    public let plannedBytes: Int

    public static func make(requested: Int, ramBudget: Int?) -> TemplateLoadPlan {
        var context = min(requested, modelContext)
        if let ramBudget { context = min(context, max(ramBudget / bytesPerPosition, 1)) }
        return TemplateLoadPlan(context: context, plannedBytes: context * bytesPerPosition)
    }
}
