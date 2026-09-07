import SwiftUI
import TurboFieldfareMacPresentation

struct HUDMetricView: View {
    let value: String
    let label: String
    var animated = true
    let identifier: AccessibilityID

    var body: some View {
        VStack(spacing: 1) {
            Text(value)
                .font(.system(.callout, design: .rounded).weight(.semibold))
                .monospacedDigit()
                .contentTransition(animated ? .numericText() : .identity)
                .animation(animated ? .snappy(duration: 0.25) : nil, value: value)
            Text(label)
                .font(.caption2)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 56)
        // One element, so the driver reads the figure by its identifier.
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier(identifier)
    }
}
