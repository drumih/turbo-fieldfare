import Foundation
import TinyTitan
import TinyTitanLib
import TinyTitanCore

package struct DoctorOptions: Equatable, Sendable {
    package var modelsDirectory: String = "models"
    package var json: Bool = false
    package var maxContext: Int = 4_096
    package var calibrationProfile: String?
    package var help: Bool = false

    package init(
        modelsDirectory: String = "models",
        json: Bool = false,
        maxContext: Int = 4_096,
        calibrationProfile: String? = nil,
        help: Bool = false
    ) {
        self.modelsDirectory = modelsDirectory
        self.json = json
        self.maxContext = maxContext
        self.calibrationProfile = calibrationProfile
        self.help = help
    }

    package static let usage = """
        usage: TinyTitanCLI doctor [--json] [--models-dir <dir>]
                                   [--max-context <tokens>]
                                   [--calibration-profile <path>]

        Reports what this machine can do without loading a model or its weights:
        chip, memory, disk, GPU/ANE capabilities, the chosen tensor-ops prefill
        path, installed models with a memory fit, and whether another TinyTitan
        model process is running.

        options:
          --json                    Print the report as JSON instead of text.
          --models-dir <dir>        Model store to inspect (default: models/).
          --max-context <tokens>    Context used for the KV share of each fit
                                    (default: 4096).
          --calibration-profile <path>
                                    Validate and report a calibration profile.
          --help                    Show this message.
        """

    package static func parse(_ arguments: [String]) throws -> DoctorOptions {
        var options = DoctorOptions()
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            func value(_ name: String) throws -> String {
                index += 1
                guard index < arguments.count else {
                    throw DoctorCommandError.missingValue(name)
                }
                return arguments[index]
            }
            switch argument {
            case "--help":
                options.help = true
            case "--json":
                options.json = true
            case "--models-dir":
                let raw = try value("--models-dir")
                guard !raw.isEmpty else {
                    throw DoctorCommandError.invalidValue("--models-dir", raw)
                }
                options.modelsDirectory = raw
            case "--max-context":
                let raw = try value("--max-context")
                guard let parsed = Int(raw),
                    RuntimeConfiguration.supportedContextTokens.contains(parsed)
                else {
                    throw DoctorCommandError.invalidValue("--max-context", raw)
                }
                options.maxContext = parsed
            case "--calibration-profile":
                let raw = try value("--calibration-profile")
                guard !raw.isEmpty else {
                    throw DoctorCommandError.invalidValue("--calibration-profile", raw)
                }
                options.calibrationProfile = raw
            default:
                throw DoctorCommandError.unknownArgument(argument)
            }
            index += 1
        }
        return options
    }
}

package enum DoctorCommandError: Error, CustomStringConvertible {
    case unknownArgument(String)
    case missingValue(String)
    case invalidValue(String, String)

    package var description: String {
        switch self {
        case .unknownArgument(let value):
            return "unknown argument \(value)"
        case .missingValue(let name):
            return "\(name) needs a value"
        case .invalidValue(let name, let value):
            return "\(name) got an invalid value \(value)"
        }
    }
}

package enum DoctorCommand {
    /// Runs the subcommand and returns the process exit code. Absent models are
    /// a report, not an error: the machine facts are still the answer.
    package static func run(_ options: DoctorOptions) throws -> Int32 {
        if options.help {
            print(DoctorOptions.usage)
            return 0
        }
        let environment = liveEnvironment(for: options)
        let report = DoctorReport.build(environment)
        if options.json {
            var data = try report.jsonData()
            data.append(0x0A)
            FileHandle.standardOutput.write(data)
        } else {
            print(report.renderedHuman())
        }
        return 0
    }

    /// The live facts. Tests call `DoctorReport.build` with their own
    /// environment instead, so no assertion here depends on the host.
    static func liveEnvironment(for options: DoctorOptions) -> DoctorEnvironment {
        let directory = URL(
            fileURLWithPath: options.modelsDirectory,
            relativeTo: URL(
                fileURLWithPath: FileManager.default.currentDirectoryPath)
        ).standardizedFileURL
        return DoctorEnvironment(
            capabilities: HardwareCapabilities.probe(),
            modelsDirectory: directory,
            environment: ProcessInfo.processInfo.environment,
            maxContextTokens: options.maxContext,
            disk: DoctorSystemProbe.disk(for: directory),
            otherProcesses: DoctorSystemProbe.otherProcesses(),
            calibration: calibration(at: options.calibrationProfile))
    }

    static func calibration(at path: String?) -> DoctorReport.Calibration {
        guard let path else { return .notConfigured }
        do {
            let profile = try CalibrationProfile.load(
                fileAt: URL(fileURLWithPath: path))
            let machine = CalibrationProfile.currentMachine()
            let matches =
                profile.machine.identifier == machine.identifier
                && profile.machine.memoryBytes == machine.memoryBytes
                && profile.machine.osVersion == machine.osVersion
            return DoctorReport.Calibration(
                configured: true,
                path: path,
                status: "valid",
                detail: nil,
                measured: profile.measured,
                source: profile.source,
                modelID: profile.target.modelID,
                contextTokens: profile.target.contextTokens,
                machineMatches: matches)
        } catch {
            return DoctorReport.Calibration(
                configured: true,
                path: path,
                status: "invalid",
                detail: "\(error)",
                measured: nil,
                source: nil,
                modelID: nil,
                contextTokens: nil,
                machineMatches: nil)
        }
    }
}
