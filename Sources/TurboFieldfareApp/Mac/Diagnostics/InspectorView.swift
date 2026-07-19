import AppKit
import TurboFieldfareAppCore
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel

    var body: some View {
        Form {
            modelSection
            samplingSection
            expertCacheSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var modelSection: some View {
        Section("Model") {
            LabeledContent("Path") {
                HStack(spacing: 6) {
                    Text((model.modelPathText as NSString).abbreviatingWithTildeInPath)
                        .font(.caption)
                        .truncationMode(.middle)
                        .lineLimit(1)
                        .foregroundStyle(.secondary)
                        .help(model.modelPathText)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.modelPathText, forType: .string)
                    } label: {
                        Label("Copy model path", systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Copy model path")
                }
            }
            if model.canUnloadModel {
                Button("Unload Model", action: model.unloadModel)
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

    private var samplingSection: some View {
        Section("Sampling") {
            LabeledContent("Max response") {
                Stepper(value: $model.maxNewTokens, in: 1...4096, step: 16) {
                    Text("\(model.maxNewTokens)").monospacedDigit()
                }
                .fixedSize()
            }
            LabeledContent("Max context") {
                Stepper(value: $model.maxContextTokens, in: 256...8192, step: 256) {
                    Text("\(model.maxContextTokens)").monospacedDigit()
                }
                .fixedSize()
            }
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
            if model.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)").monospacedDigit()
                    }
                    .fixedSize()
                }
            }
            Toggle("Top-P", isOn: $model.topPEnabled)
                .toggleStyle(.switch)
                .disabled(!model.topKEnabled)
            if model.topKEnabled && model.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                        Text(model.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }
            Toggle("Repetition penalty", isOn: $model.repetitionPenaltyEnabled)
                .toggleStyle(.switch)
            if model.repetitionPenaltyEnabled {
                LabeledContent("Penalty value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.repetitionPenalty, in: 1...1.8, step: 0.05)
                        Text(model.repetitionPenalty, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }
            Text("Defaults: 1,024 tokens, temperature 1.00, Top-K 64, Top-P 0.95. Repetition penalty is off.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var expertCacheSection: some View {
        Section("Expert cache") {
            LabeledContent("Slots") {
                Picker("Slots", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("Policy")
                Picker("Policy", selection: $model.runtimeOptions.expertCachePolicy) {
                    ForEach(AppExpertCachePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            if model.runtimeOptions.expertCacheSlots > 16 {
                Text("Slot counts above 16 trade extra RAM for fewer expert reads.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            Toggle("Prefill", isOn: $model.runtimeOptions.prefillEnabled)
            LabeledContent("Prefill chunk") {
                Picker("Prefill chunk", selection: $model.runtimeOptions.prefillChunkTokens) {
                    ForEach(AppRuntimeOptions.allowedPrefillChunkTokens, id: \.self) { tokens in
                        Text("\(tokens) tokens").tag(tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            .disabled(!model.runtimeOptions.prefillEnabled)
            Toggle("TurboQuant K4/V4 (Experimental)",
                   isOn: $model.runtimeOptions.turboQuantKVEnabled)
            Text("This is an experimental feature and may reduce output quality.")
                .font(.caption)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 8) {
                Text("RDADVISE")
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            Text("RDADVISE is experimental. It may speed up short decodes but slow down long decodes.")
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent("Model verification") {
                Picker("Model verification", selection: $model.runtimeOptions.modelVerification) {
                    ForEach(AppModelVerification.allCases) { choice in
                        Text(choice.label).tag(choice)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            if model.runtimeOptions.modelVerification == .trustedInstall {
                Text("Trust verified install checks the signed-off receipt and file sizes instead of hashing all model files again.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

}
