import AppKit
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel

    var body: some View {
        Form {
            modelSection
            // Second, beside Model: image support is an install concern, not a
            // diagnostic. Last put it under the runner diagnostics and below the
            // fold, where the one screen that must mention it - the empty state
            // before any model exists - could not.
            if showsVisionSection {
                visionSection
            }
            memorySection
            generationSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    /// The section stays visible after installation because Remove is part of
    /// the supported lifecycle. It is also visible before text installation so
    /// image support is discoverable before the larger download starts.
     private var showsVisionSection: Bool {
        VisionSectionVisibility.shows(
            visionRuntimeEnabled: model.visionRuntimeEnabled,
            visionRuntimeSupported: model.isVisionRuntimeSupported,
            isModelInstalled: model.isModelInstalled,
            isVisionPackInstalled: model.isVisionPackInstalled,
            isCompanionOperationInProgress: model.isVisionCompanionOperationInProgress,
            installState: model.visionInstallState)
    }

    private var visionSection: some View {
        Section("Image Support") {
            LabeledContent("State") {
                Text(visionStatusLabel)
                    .font(.caption)
                    .foregroundStyle(visionStatusColor)
            }
            if model.isVisionRuntimeSupported && !model.isVisionPackInstalled {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(
                        model.visionInstallDescriptor.approximateDownloadBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if let fraction = model.visionInstallProgressFraction {
                ProgressView(value: fraction)
                    .accessibilityValue(visionAccessibleProgress(fraction: fraction))
                HStack(alignment: .firstTextBaseline) {
                    Text(MetricFormat.percent(fraction * 100))
                    Spacer(minLength: 8)
                    if let eta = model.visionInstallETAText {
                        Text(eta)
                    }
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            } else if model.isInstallingVisionPack {
                ProgressView()
                    .controlSize(.small)
                if let eta = model.visionInstallETAText {
                    Text(eta)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            if !model.isVisionRuntimeSupported {
                Text("Image support requires an M2 or newer Mac. "
                    + "Text generation remains available on this Mac.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if case .failed(let message) = model.visionInstallState {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
            } else if case .recoverable(let message) = model.visionInstallState {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if case .partial(let message) = model.visionInstallationStatus,
                      !model.isInstallingVisionPack {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if case .unsupportedLayout = model.visionInstallationStatus {
                Text("Image support needs a model directory named "
                    + "“<name>.gturbo”, which is where the companion pack lives.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if case .failed(let message) = model.visionInstallReadiness,
                      !model.isInstallingVisionPack {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if case .insufficientSpace(let requirement) = model.visionInstallReadiness,
                      !model.isInstallingVisionPack {
                Text("Free \(MetricFormat.storage(requirement.shortfallBytes)) more storage.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if model.isVisionFilesystemMutationInProgress {
                Text("Model actions stay unavailable until this filesystem change finishes. "
                    + "Your prompt, images, and transcript are kept.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if !model.isVisionPackInstalled && model.loadState.isReady {
                Text("Unload the model before preparing image support.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if model.isVisionPackInstalled && model.loadState.isReady {
                Text("Unload the model before removing image support.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            HStack {
                if model.isInstallingVisionPack {
                    Button("Cancel", action: model.cancelVisionInstall)
                        .disabled(!model.canCancelVisionInstall)
                        .accessibilityIdentifier(.visionCancel)
                } else if case .readyToActivate = model.visionInstallState {
                    Button("Discard", role: .destructive) {
                        model.discardVisionPackDownload()
                    }
                    .disabled(!model.canDiscardVisionPackDownload)
                    .accessibilityIdentifier(.visionDiscard)
                    if model.isVisionRuntimeSupported {
                        Button("Activate", action: model.activateVisionPack)
                            .buttonStyle(.borderedProminent)
                            .disabled(!model.canActivateVisionPack)
                            .accessibilityIdentifier(.visionActivate)
                    }
                } else if model.isVisionPackInstalled {
                    Button("Remove", role: .destructive) {
                        model.requestVisionPackRemoval()
                    }
                    .disabled(!model.canRemoveVisionPack)
                    .accessibilityIdentifier(.visionRemove)
                } else {
                    if model.hasVisionPackDirectory {
                        Button("Remove", role: .destructive) {
                            model.requestVisionPackRemoval()
                        }
                    .disabled(!model.canRemoveVisionPack)
                    .accessibilityIdentifier(.visionRemove)
                    }
                    if model.hasPartialVisionPackDownload {
                        Button("Discard", role: .destructive) {
                            model.discardVisionPackDownload()
                        }
                        .disabled(!model.canDiscardVisionPackDownload)
                        .accessibilityIdentifier(.visionDiscard)
                    }
                    if model.isVisionRuntimeSupported {
                        Button(visionInstallButtonLabel) {
                            model.installVisionPack()
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.canInstallVisionPack)
                        .accessibilityIdentifier(.visionInstall)
                    }
                }
            }
        }
    }

    private var visionInstallButtonLabel: String {
        if model.hasPartialVisionPackDownload { return "Resume" }
        if model.hasVisionPackDirectory { return "Repair" }
        return "Download"
    }

    private func visionAccessibleProgress(fraction: Double) -> String {
        let percent = MetricFormat.percent(fraction * 100)
        guard let eta = model.visionInstallETAText else { return percent }
        return "\(percent), \(eta)"
    }

    private var visionStatusLabel: String {
        guard model.isVisionRuntimeSupported else { return "Requires M2 or newer" }
        if model.visionInstallState != .idle {
            return model.visionInstallPhaseLabel
        }
        switch model.visionInstallationStatus {
        case .missing: return "Not installed"
        case .partial: return "Needs repair"
        case .complete: return "Installed"
        case .unsupportedLayout: return "Not available for this model"
        }
    }

    private var visionStatusColor: Color {
        guard model.isVisionRuntimeSupported else { return .secondary }
        switch model.visionInstallationStatus {
        case .partial: return .orange
        case .missing, .complete, .unsupportedLayout: return .secondary
        }
    }

    private var modelSection: some View {
        Section("Model") {
            LabeledContent("Path") {
                HStack(spacing: 6) {
                    Text(model.modelPathText)
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
                    .accessibilityIdentifier(.inspectorCopyPath)
                }
            }
            if model.canUnloadModel {
                Button("Unload Model", action: model.unloadModel)
                    .accessibilityIdentifier(.inspectorUnload)
            }
            LabeledContent("State") {
                Text(model.presentation.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            LabeledContent("Download") {
                Text(MetricFormat.storage(model.installDescriptor.approximateDownloadBytes))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            if let metrics = model.modelStorageMetrics {
                LabeledContent("Installed logical") {
                    Text(MetricFormat.storage(metrics.logicalBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Installed allocated") {
                    Text(MetricFormat.storage(metrics.allocatedBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let available = metrics.availableBytes {
                    LabeledContent("Available") {
                        Text(MetricFormat.storage(available))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            } else if let metricsError = model.modelStorageMetricsError {
                Text(metricsError)
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if model.requiresModelInstallation {
                LabeledContent("Expected installed") {
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
        .disabled(model.isRunning || model.isInstallingModel
            || model.isVisionFilesystemMutationInProgress)
    }

    /// The context is settable only through the model, because changing it has
    /// to redraw a stored conversation against the new answer to "can this be
    /// continued". A plain binding would set the value and leave the window
    /// describing the old one.
    private var contextTokensBinding: Binding<Int> {
        Binding {
            model.maxContextTokens
        } set: { tokens in
            model.setMaxContextTokens(tokens)
        }
    }

    private var memorySection: some View {
        Section("Memory") {
            LabeledContent("Context") {
                Picker("Context", selection: contextTokensBinding) {
                    ForEach(AppContextLengthOption.allCases) { option in
                        Text(option.menuLabel).tag(option.tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .accessibilityIdentifier(.inspectorContext)
            }
            LabeledContent("Slots") {
                Picker("Slots", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .accessibilityIdentifier(.inspectorSlots)
            }
            Text("More slots can improve decode speed by keeping more experts in memory, but they also use more RAM. Changes are compared with 8K context and 16 slots and apply after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionFilesystemMutationInProgress)
    }

    private var generationSection: some View {
        Section("Generation") {
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                        .accessibilityIdentifier(.inspectorTemperature)
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Text("0 uses deterministic greedy decoding. Higher values make sampling more varied.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
                .accessibilityIdentifier(.inspectorTopK)
            if model.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)").monospacedDigit()
                    }
                    .fixedSize()
                    .accessibilityIdentifier(.inspectorTopKValue)
                }
            }
            Toggle("Top-P", isOn: $model.topPEnabled)
                .toggleStyle(.switch)
                .disabled(!model.topKEnabled)
                .accessibilityIdentifier(.inspectorTopP)
            if model.topKEnabled && model.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                            .accessibilityIdentifier(.inspectorTopPValue)
                        Text(model.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionFilesystemMutationInProgress)
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            Toggle("Prefill", isOn: $model.runtimeOptions.prefillEnabled)
                .accessibilityIdentifier(.inspectorPrefill)
            VStack(alignment: .leading, spacing: 8) {
                Text("RDADVISE")
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .accessibilityIdentifier(.inspectorRDAdvise)
            }
            Text("RDADVISE is experimental. It may speed up short decodes but slow down long decodes.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionFilesystemMutationInProgress)
    }

}
