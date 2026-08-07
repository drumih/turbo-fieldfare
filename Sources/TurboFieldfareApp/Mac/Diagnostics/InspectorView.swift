import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel

    var body: some View {
        Form {
            modelSection
            composerSection
            memorySection
            generationSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        // Keep system grouped form backgrounds so rows/fields separate from chrome.
        .scrollContentBackground(.automatic)
        .background(TurboFieldfareMacTheme.elevatedSurface.opacity(0.35))
    }

    private var modelSection: some View {
        Section("Model") {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("Path")
                Spacer(minLength: 8)
                Text(model.modelPathText)
                    .font(.caption)
                    .truncationMode(.middle)
                    .lineLimit(1)
                    .foregroundStyle(.secondary)
                    .help(model.modelPathText)
                    .textSelection(.enabled)
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(model.modelPathText, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Copy model path")
                .accessibilityHint("Copies the model directory path to the clipboard")
                .help("Copy model path")
            }
            if model.canUnloadModel {
                Button("Unload Model", action: model.unloadModel)
                    .accessibilityHint("Frees model memory until you load it again")
            }
            LabeledContent("State") {
                Text(model.presentation.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.requiresModelInstallation {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(model.installDescriptor.approximateDownloadBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Installed size") {
                    Text(MetricFormat.storage(model.installDescriptor.installedBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let requirement = model.installRequirement {
                    LabeledContent("Available") {
                        Text(MetricFormat.storage(requirement.availableBytes))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }

    private var composerSection: some View {
        Section("Composer") {
            Picker("Send with", selection: newlineShortcutBinding) {
                ForEach(AppNewlineShortcut.sendMessageOptions) { shortcut in
                    Text(shortcut.sendMessageLabel).tag(shortcut)
                }
            }
            Picker("After sending", selection: sentPromptBehaviorBinding) {
                ForEach(AppSentPromptBehavior.allCases) { behavior in
                    Text(behavior.settingsLabel).tag(behavior)
                }
            }
            Toggle("Show prompt examples", isOn: showPromptExamplesBinding)
            Text("Send with chooses whether Return or Command-Return generates. Prompt examples appear when the prompt is empty.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isRunning)
    }

    private var memorySection: some View {
        Section("Memory") {
            LabeledContent("Context") {
                Picker("Context", selection: $model.maxContextTokens) {
                    ForEach(AppContextLengthOption.allCases) { option in
                        Text(option.menuLabel).tag(option.tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            LabeledContent("Expert cache") {
                Picker("Expert cache", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            Text("More expert-cache slots can improve decode speed by keeping more experts in memory, but they also use more RAM. Changes are compared with 4K context and 16 slots and apply after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var generationSection: some View {
        Section("Generation") {
            Text("Response length uses the context space left after your prompt. There is no separate max-length control.")
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                    numericChip(
                        model.temperature,
                        format: .number.precision(.fractionLength(2)))
                }
            }
            Text("0 uses deterministic greedy decoding. Higher values make sampling more varied.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
            if model.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)")
                            .monospacedDigit()
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background {
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(TurboFieldfareMacTheme.fieldSurface)
                                    .overlay {
                                        RoundedRectangle(cornerRadius: 6)
                                            .stroke(TurboFieldfareMacTheme.fieldBorder, lineWidth: 1)
                                    }
                            }
                    }
                    .fixedSize()
                }
            }
            Toggle("Top-P", isOn: $model.topPEnabled)
                .toggleStyle(.switch)
                .disabled(!model.topKEnabled)
            if !model.topKEnabled {
                Text("Top-P applies only while Top-K is on.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.topKEnabled && model.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                        numericChip(
                            model.topP,
                            format: .number.precision(.fractionLength(2)))
                    }
                }
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private func numericChip<F: FormatStyle>(_ value: Double, format: F) -> some View
    where F.FormatInput == Double, F.FormatOutput == String {
        Text(value, format: format)
            .monospacedDigit()
            .frame(minWidth: 40, alignment: .trailing)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background {
                RoundedRectangle(cornerRadius: 6)
                    .fill(TurboFieldfareMacTheme.fieldSurface)
                    .overlay {
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(TurboFieldfareMacTheme.fieldBorder, lineWidth: 1)
                    }
            }
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            Toggle("Prefill", isOn: $model.runtimeOptions.prefillEnabled)
            Text("Prefill processes the prompt in larger chunks before token-by-token generation. Leave it on for normal use.")
                .font(.caption)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 8) {
                Text("RDADVISE")
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            Text(rdadviseHelp)
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.hasStaleLoadedRuntime {
                Label("Reload required before generating", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(TurboFieldfareMacTheme.warningEmphasis)
                    .accessibilityLabel("Reload required before generating")
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var rdadviseHelp: String {
        switch model.runtimeOptions.rdadvisePolicy {
        case .off:
            return "RDADVISE is off (recommended). Experimental modes may speed short decodes but slow long ones."
        case .default:
            return "Default RDADVISE applies the stock read-advice policy. Experimental — reload required."
        case .bounded:
            return "Bounded RDADVISE limits advice volume. Experimental — reload required."
        case .adaptive:
            return "Adaptive RDADVISE adjusts advice during decode. Experimental — reload required."
        }
    }

    private var newlineShortcutBinding: Binding<AppNewlineShortcut> {
        Binding {
            model.newlineShortcut
        } set: { shortcut in
            model.setNewlineShortcut(shortcut)
        }
    }

    private var showPromptExamplesBinding: Binding<Bool> {
        Binding {
            model.showPromptExamples
        } set: { show in
            model.setShowPromptExamples(show)
        }
    }

    private var sentPromptBehaviorBinding: Binding<AppSentPromptBehavior> {
        Binding {
            model.sentPromptBehavior
        } set: { behavior in
            model.setSentPromptBehavior(behavior)
        }
    }
}
