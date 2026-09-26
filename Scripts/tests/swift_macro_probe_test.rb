#!/usr/bin/env ruby
# Compile bounded fixtures with the selected tools, then remove only platform
# plugin discovery from the emitted frontend command to reproduce issue 185.
require 'tmpdir'
require 'open3'
require 'shellwords'
require 'json'

require_sdk_27 = ARGV.delete('--require-sdk-27')
configuration = nil
unless ARGV.empty?
  unless ARGV.length == 2 && ARGV[0] == '--build' && %w[debug release].include?(ARGV[1])
    abort 'Usage: ruby Scripts/tests/swift_macro_probe_test.rb [--build debug|release] [--require-sdk-27]'
  end
  configuration = ARGV[1]
end

fixtures = {
  'Import' => "import SwiftUI\n",
  'Observation' => "import Observation\n@Observable final class Model { var count = 0 }\n",
  'State' => <<~SWIFT,
    import SwiftUI
    struct Probe: View {
        @State private var search: String
        init() { _search = State(initialValue: "") }
        var body: some View {
            TextField("Search", text: $search)
            Button("Clear") { search = "" }
        }
    }
  SWIFT
  'Preview' => "import SwiftUI\n#Preview { Text(\"Build check\") }\n"
}

alias_source = File.expand_path('../../Sources/TurboFieldfareApp/MacPresentation/StoredState.swift', __dir__)
fixtures['StoredState'] = fixtures['State'].gsub('@State', '@StoredState').sub('_search = State(', '_search = StoredState(')
fixtures['StoredState'] = fixtures['StoredState'].sub('    var body:', "    @StoredState private var optional: UUID?\n    @StoredState private var flag = false\n    var body:")
fixtures['PreviewProvider'] = 'import SwiftUI; struct Preview: PreviewProvider { static var previews: some View { Text("Build check") } }'

sdk, diagnostic, status = Open3.capture3('/usr/bin/xcrun', '--show-sdk-version')
abort diagnostic unless status.success?
abort "Cannot parse SDK version: #{sdk}" unless sdk.match?(/\A\d+\.\d+/)
platform_macros = sdk.to_i >= 27
abort 'The State macro regression gate requires macOS SDK 27 or newer' if require_sdk_27 && !platform_macros
Dir.mktmpdir('tff-real-macro-tests.') do |directory|
  fixtures.each do |name, source|
    file = File.join(directory, "#{name}.swift")
    File.write(file, File.read(alias_source) + "\n" + source)
    driver = ['/usr/bin/xcrun', 'swiftc', '-typecheck', '-swift-version', '6',
              '-target', 'arm64-apple-macosx26.0', '-module-name', 'MacroProbe', file]
    output, status = Open3.capture2e(*driver)
    missing = { 'State' => 'SwiftUIMacros', 'Preview' => 'PreviewsMacros' }[name]
    unless status.success? || (missing && output.include?("plugin for module '#{missing}' not found"))
      abort "#{name}: selected-tool baseline failed\n#{output}"
    end
    puts "PASS #{name}: selected tools (#{status.success? ? 'compiles' : 'expected missing platform plugin'})"
    next unless platform_macros

    output, status = Open3.capture2e(*driver, '-###')
    abort output unless status.success?
    commands = output.lines.map { |line| Shellwords.split(line) }
    frontend = commands.find { |argv| argv.include?('-frontend') }
    abort 'Cannot find compiler frontend command' unless frontend
    stripped = []
    removed = 0
    until frontend.empty?
      argument = frontend.shift
      if argument == '-external-plugin-path'
        abort 'Missing plugin path argument' if frontend.empty?
        frontend.shift
        removed += 1
      else
        stripped << argument
      end
    end
    abort 'No platform plugin paths found; negative fixture cannot be validated' if removed.zero?
    output, status = Open3.capture2e(*stripped)
    if missing
      unless !status.success? && output.include?("plugin for module '#{missing}' not found")
        abort "#{name}: expected missing #{missing}, got exit #{status.exitstatus}\n#{output}"
      end
    else
      abort "#{name}: unexpectedly failed without platform plugins\n#{output}" unless status.success?
    end
    puts "PASS #{name}: platform discovery removed (#{missing || 'still compiles'})"
  end
  if configuration
    # Exercise every Swift source, including future framework dependencies, with
    # fresh package objects and platform search paths absent from each frontend.
    frontend, status = Open3.capture2('/usr/bin/xcrun', '--find', 'swift-frontend')
    abort 'Cannot find swift-frontend' unless status.success?
    info, status = Open3.capture2('/usr/bin/xcrun', 'swiftc', '-print-target-info')
    abort 'Cannot read compiler resource directory' unless status.success?
    resources = JSON.parse(info).fetch('paths').fetch('runtimeResourcePath')
    proxy = File.join(directory, 'frontend')
    audit = File.join(directory, 'removed-paths')
    File.write(audit, '')
    File.write(proxy, <<~CODE)
      #!/usr/bin/ruby
      # Refuse opaque response files rather than accidentally retaining plugin
      # paths in one invocation while other invocations make the audit green.
      abort 'Response-file arguments cannot be validated by the plugin-removal proxy' if ARGV.any? { |arg| arg.start_with?('@') }
      arguments = []
      removed = 0
      until ARGV.empty?
        argument = ARGV.shift
        if argument == '-external-plugin-path'
          abort 'Missing plugin path argument' if ARGV.empty?
          ARGV.shift
          removed += 1
        else
          arguments << argument
        end
      end
      File.open(#{audit.dump}, 'a') { |file| file.puts(removed) }
      exec(#{frontend.strip.dump}, *arguments)
    CODE
    File.chmod(0755, proxy)
    response = File.join(directory, 'response')
    File.write(response, '-external-plugin-path /unexpected/plugin')
    output, status = Open3.capture2e(proxy, "@#{response}")
    unless !status.success? && output.include?('Response-file arguments cannot be validated')
      abort 'Plugin-removal proxy accepted an opaque response file'
    end
    puts 'PASS plugin-removal proxy rejects response files'
    root = File.expand_path('../..', __dir__)
    # SwiftBuild's integrated driver does not support replacing its frontend
    # consistently (missing ABI sidecars); use SwiftPM's external-driver path.
    success = system('/usr/bin/xcrun', 'swift', 'build', '--build-system', 'native', '--package-path', root,
                     '--scratch-path', File.join(directory, 'build'), '-c', configuration,
                     '-Xswiftc', '-driver-use-frontend-path', '-Xswiftc', proxy,
                     '-Xswiftc', '-resource-dir', '-Xswiftc', resources)
    abort 'Full package build without platform plugins failed' unless success
    removed = File.readlines(audit).sum(&:to_i)
    abort 'Negative build did not exercise plugin-path removal' if platform_macros && removed.zero?
    puts "PASS full #{configuration} package build: #{removed} platform plugin paths removed"
  end
end
puts 'SKIP platform-plugin removal: this fixture requires macOS SDK 27+' unless platform_macros
