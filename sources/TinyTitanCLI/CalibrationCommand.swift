import Foundation
import Metal
import TinyTitanCore
import TinyTitanLegacyEngine
import TinyTitanLib
import TinyTitanVariants

/// `tinytitan calibrate`: write or validate a `CalibrationProfile`.
///
/// T15 is the interface and the schema; the measurement arms are T16. Until
/// they land the writer produces an unmeasured draft (`measured: false`,
/// every knob nil) from the install's identity, so the file, the validator and
/// the help text exist and a later calibration run only fills the knobs.
public struct CalibrationOptions: Equatable, Sendable {
    public var model: String?
    public var maxContext: Int = 4096
    public var out: String?
    public var check: String?
    public var plan: Bool = false
    public var measure: Bool = false
    public var tokens: Int = 64
    public var repeats: Int = 3
    public var help: Bool = false

    public init(
        model: String? = nil, maxContext: Int = 4096,
        out: String? = nil, check: String? = nil, plan: Bool = false,
        measure: Bool = false, tokens: Int = 64, repeats: Int = 3,
        help: Bool = false
    ) {
        self.model = model
        self.maxContext = maxContext
        self.out = out
        self.check = check
        self.plan = plan
        self.measure = measure
        self.tokens = tokens
        self.repeats = repeats
        self.help = help
    }

    public static let usage = """
        usage: tinytitan calibrate (--model <dir> [--max-context <tokens>] [--out <path>]
                                   | --check <path>)

        options:
          --model <dir>          Path to a .ssdai model directory. Writes an
                                 unmeasured profile draft for this machine,
                                 install and context.
          --max-context <tokens> Context the profile is for; one of 4096, 8192,
                                 16384, 32768, 65536, 131072, 262144.
                                 Default 4096.
          --out <path>           Write the profile here instead of stdout.
          --check <path>         Decode and validate an existing profile, then
                                 print it normalized.
          --plan                 List the arms this machine can measure and the
                                 plan's arms that have no runtime switch yet.
          --measure              Run the measurable arms and write a measured
                                 profile (needs --model). One engine is loaded
                                 per sample; load time is outside the number.
          --tokens <n>           Decoded tokens per measurement sample
                                 (default 64, 8...512).
          --repeats <n>          Samples per arm (default 3, 2...9).
          --help                 Show this help.

        The profile is JSON. A draft has "measured": false and no knobs; a
        calibration run (T16) fills the knobs and sets "measured": true.
        """

    public static func parse(_ arguments: [String]) throws -> CalibrationOptions {
        var options = CalibrationOptions()
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            func value(_ name: String) throws -> String {
                index += 1
                guard index < arguments.count else {
                    throw CalibrationCommandError.missingValue(name)
                }
                return arguments[index]
            }
            switch argument {
            case "--help":
                options.help = true
            case "--model":
                options.model = try value("--model")
            case "--out":
                options.out = try value("--out")
            case "--check":
                options.check = try value("--check")
            case "--plan":
                options.plan = true
            case "--measure":
                options.measure = true
            case "--tokens":
                let raw = try value("--tokens")
                guard let parsed = Int(raw), (8...512).contains(parsed) else {
                    throw CalibrationCommandError.invalidValue("--tokens", raw)
                }
                options.tokens = parsed
            case "--repeats":
                let raw = try value("--repeats")
                guard let parsed = Int(raw), (2...9).contains(parsed) else {
                    throw CalibrationCommandError.invalidValue("--repeats", raw)
                }
                options.repeats = parsed
            case "--max-context":
                let raw = try value("--max-context")
                guard let parsed = Int(raw), parsed > 0 else {
                    throw CalibrationCommandError.invalidValue(
                        "--max-context", raw)
                }
                options.maxContext = parsed
            default:
                throw CalibrationCommandError.unknownArgument(argument)
            }
            index += 1
        }
        return options
    }
}

public enum CalibrationCommandError: Error, CustomStringConvertible {
    case unknownArgument(String)
    case missingValue(String)
    case invalidValue(String, String)
    case missingModel
    case conflictingInput(String, String)
    case profileTooLarge(Int)
    case noMetalDevice

    public var description: String {
        switch self {
        case .unknownArgument(let value):
            return "unknown argument \(value)"
        case .missingValue(let name):
            return "\(name) needs a value"
        case .invalidValue(let name, let value):
            return "\(name) got an invalid value \(value)"
        case .missingModel:
            return "calibrate needs --model <dir> or --check <path>"
        case .conflictingInput(let first, let second):
            return "calibrate \(first) and \(second) cannot be combined"
        case .profileTooLarge(let cap):
            return "calibration profile exceeds \(cap) bytes"
        case .noMetalDevice:
            return "calibrate --measure needs a Metal device"
        }
    }
}

public enum CalibrationCommand {
    private static func planText() -> String {
        var lines = ["baseline: family defaults", "arms:"]
        for arm in CalibrationPlan.defaultArms() {
            lines.append("  \(arm.name): \(describe(arm.knobs))")
            if let note = arm.note { lines.append("    \(note)") }
        }
        lines.append("deferred (needs a fresh process):")
        for item in CalibrationPlan.deferred {
            lines.append("  \(item.name): \(item.reason)")
        }
        lines.append("no runtime switch yet:")
        for item in CalibrationPlan.unavailable {
            lines.append("  \(item.name): \(item.reason)")
        }
        return lines.joined(separator: "\n")
    }

    private static func measure(
        modelPath: String, options: CalibrationOptions
    ) async throws -> Int32 {
        let directory = URL(fileURLWithPath: modelPath)
        let identity = try EngineFactories.gpuFactory(directoryURL: directory).metadata(directory)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CalibrationCommandError.noMetalDevice
        }
        let target = CalibrationProfile.Target(
            modelID: identity.modelID,
            family: identity.family,
            weightBits: identity.weightBits,
            contextTokens: options.maxContext)
        let measurer = CalibrationMeasurer(
            directory: directory, contextTokens: options.maxContext,
            tokens: options.tokens, device: device)
        let report = try await measurementReport(
            target: target, repeats: options.repeats
        ) { arm in
            try await measurer.measure(arm)
        }
        for decision in report.decisions {
            let text =
                "arm \(decision.arm.name): \(decision.reason) "
                + "(baseline \(format(decision.baselineMedian)), "
                + "arm \(format(decision.armMedian)) tok/s)\n"
            FileHandle.standardError.write(Data(text.utf8))
        }
        let profile = report.measuredProfile(
            machine: CalibrationProfile.currentMachine(), target: target)
        try profile.validate()
        let data = try profile.encodedJSON()
        if let out = options.out {
            try data.write(to: URL(fileURLWithPath: out))
            FileHandle.standardError.write(Data("wrote \(out)\n".utf8))
        } else {
            writeStdout(data)
        }
        return 0
    }

    /// The decision half of `--measure`, with the sampler injected so the
    /// profile/keep wiring is testable without loading a model.
    static func measurementReport(
        target: CalibrationProfile.Target, repeats: Int,
        sampler: @Sendable (CalibrationArm) async throws -> Double
    ) async throws -> CalibrationReport {
        try await CalibrationRunner().run(
            baseline: CalibrationPlan.baselineArm(),
            arms: CalibrationPlan.defaultArms(),
            repeats: repeats
        ) { arm in
            try await sampler(arm)
        }
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.2f", value)
    }

    private static func describe(_ knobs: CalibrationProfile.Knobs) -> String {
        var parts: [String] = []
        if let value = knobs.prefillChunkTokens {
            parts.append("prefillChunkTokens=\(value)")
        }
        if let value = knobs.expertCacheSlots {
            parts.append("expertCacheSlots=\(value)")
        }
        if let value = knobs.q8Attention { parts.append("q8Attention=\(value)") }
        if let value = knobs.readerThreads { parts.append("readerThreads=\(value)") }
        if let value = knobs.boundedIO { parts.append("boundedIO=\(value)") }
        if let value = knobs.prefillTileDepth {
            parts.append("prefillTileDepth=\(value)")
        }
        return parts.isEmpty ? "defaults" : parts.joined(separator: " ")
    }

    /// Profiles are small metadata documents; bound the read so a stray path
    /// cannot pull a model payload into memory.
    private static func readProfile(at path: String) throws -> Data {
        let cap = 1 << 20
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        let data = try handle.read(upToCount: cap + 1) ?? Data()
        guard data.count <= cap else {
            throw CalibrationCommandError.profileTooLarge(cap)
        }
        return data
    }

    private static func writeStdout(_ data: Data) {
        var line = data
        line.append(0x0A)
        FileHandle.standardOutput.write(line)
    }

    /// Runs the subcommand and returns the process exit code.
    public static func run(_ options: CalibrationOptions) async throws -> Int32 {
        if options.help {
            print(CalibrationOptions.usage)
            return 0
        }
        if options.plan {
            // `--plan` prints the plan and ignores everything else; refusing
            // the combination says so instead of silently dropping input.
            if options.check != nil {
                throw CalibrationCommandError.conflictingInput("--plan", "--check")
            }
            if options.model != nil {
                throw CalibrationCommandError.conflictingInput("--plan", "--model")
            }
            print(planText())
            return 0
        }
        if options.check != nil, options.model != nil {
            throw CalibrationCommandError.conflictingInput("--model", "--check")
        }
        if options.measure, options.check != nil {
            throw CalibrationCommandError.conflictingInput("--measure", "--check")
        }
        if options.measure, let modelPath = options.model {
            return try await measure(modelPath: modelPath, options: options)
        }
        if let path = options.check {
            let data = try readProfile(at: path)
            let profile = try CalibrationProfile.decode(from: data)
            writeStdout(try profile.encodedJSON())
            return 0
        }
        guard let modelPath = options.model else {
            throw CalibrationCommandError.missingModel
        }
        let directory = URL(fileURLWithPath: modelPath)
        let identity = try EngineFactories.gpuFactory(directoryURL: directory).metadata(directory)
        let profile = CalibrationProfile(
            machine: CalibrationProfile.currentMachine(),
            target: CalibrationProfile.Target(
                modelID: identity.modelID,
                family: identity.family,
                weightBits: identity.weightBits,
                contextTokens: options.maxContext),
            source: "unmeasured")
        try profile.validate()
        let data = try profile.encodedJSON()
        if let out = options.out {
            try data.write(to: URL(fileURLWithPath: out))
            FileHandle.standardError.write(
                Data("wrote \(out)\n".utf8))
        } else {
            writeStdout(data)
        }
        return 0
    }
}
