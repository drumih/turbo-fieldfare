public enum AppTextSize: Int, CaseIterable, Codable, Identifiable, Sendable {
    case standard = 100
    case large = 125
    case larger = 150
    case extraLarge = 175
    case largest = 200

    public var id: Int { rawValue }
    public var label: String { "\(rawValue)%" }
    public var scale: Double { Double(rawValue) / 100 }
}
