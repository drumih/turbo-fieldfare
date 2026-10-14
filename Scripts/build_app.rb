#!/usr/bin/env ruby
require 'fileutils'
require 'open3'
require 'tmpdir'
require 'cgi'
require 'digest'

module LocalMacApp
  ROOT = File.expand_path('..', __dir__)
  RESOURCES = %w[TurboFieldfare_TurboFieldfare.bundle TurboFieldfare_TurboFieldfareAppCore.bundle TurboFieldfare_TurboFieldfareMac.bundle SwiftMath_SwiftMath.bundle swift-transformers_Hub.bundle swift-crypto_Crypto.bundle].freeze
  PRODUCTS = %w[TurboFieldfareMac TurboFieldfareDecodeService].freeze

  def self.run(*arguments)
    output, error, status = Open3.capture3(*arguments, chdir: ROOT)
    raise "#{arguments.first} failed (#{status.exitstatus}): #{error}\n#{output}" unless status.success?
    output.strip
  end

  def self.version
    source = File.read(File.join(ROOT, 'Sources/TurboFieldfareApp/MacPresentation/AboutPanelPresentation.swift'))
    source[/fallbackShortVersion\s*=\s*"([0-9]+(?:\.[0-9]+)+)"/, 1] || raise('About fallback version is missing')
  end

  def self.active?(destination)
    destination = File.realpath(destination) if File.exist?(destination)
    processes = run('/bin/ps', '-axww', '-o', 'comm=')
    processes.lines.any? do |line|
      executable = line.strip
      begin
        executable = File.realpath(executable)
      rescue Errno::ENOENT, Errno::EACCES
        # A process may exit during inspection, and other users' executables
        # may be inaccessible. Their reported path still supports comparison.
      end
      executable.start_with?(File.join(destination, 'Contents/MacOS') + '/')
    end
  end

  def self.publish(staged, destination)
    raise "Quit the app and its decode helper before replacing #{destination}. No process was stopped." if active?(destination)
    if File.exist?(destination)
      raise "Refusing to replace an unrelated directory at #{destination}" unless File.file?(File.join(destination, 'Contents/Info.plist')) &&
        File.read(File.join(destination, 'Contents/Info.plist')).include?('<string>com.turbofieldfare.mac</string>')
    end
    backup = destination.sub(/\.app\z/, '') + '.previous-' + Time.now.utc.strftime('%Y%m%dT%H%M%S') + '-'+ Process.pid.to_s + '.app'
    raise "Previous-build archive already exists at #{backup}; retry after moving it aside." if File.exist?(backup)
    File.rename(destination, backup) if File.exist?(destination)
    begin
      File.rename(staged, destination)
    rescue
      File.rename(backup, destination) if File.exist?(backup)
      raise
    end
  end

  def self.resource_root(bundle)
    structured = File.join(bundle, 'Contents/Resources')
    File.directory?(structured) ? structured : bundle
  end

  def self.package(bin, stage, mode:, sign: true)
    raise 'Unknown resource layout' unless %w[standard legacy].include?(mode)
    PRODUCTS.each { |name| raise "Missing executable: #{name}" unless File.file?(File.join(bin, name)) && File.executable?(File.join(bin, name)) }
    RESOURCES.each { |name| raise "Missing required resource bundle: #{name}" unless File.directory?(File.join(bin, name)) }
    required_files = {
      'TurboFieldfare_TurboFieldfare.bundle' => 'Metal',
      'TurboFieldfare_TurboFieldfareAppCore.bundle' => 'app-prompts.json',
      'TurboFieldfare_TurboFieldfareMac.bundle' => 'turbofieldfare-app-icon.png',
      'SwiftMath_SwiftMath.bundle' => 'mathFonts.bundle',
      'swift-transformers_Hub.bundle' => 'gpt2_tokenizer_config.json'
    }
    required_files.each { |bundle, file| raise "Missing resource: #{bundle}/#{file}" unless File.exist?(File.join(resource_root(File.join(bin, bundle)), file)) }
    RESOURCES.each do |name|
      Dir.glob(File.join(bin, name, '**', '*'), File::FNM_DOTMATCH).each do |entry|
        raise "Resource symlink is not portable: #{entry}" if File.symlink?(entry)
        raise "Model weights cannot be packaged: #{entry}" if entry.match?(/\.(?:gturbo|safetensors|bin|gguf)(?:\/|$)/)
      end
    end
    executables = File.join(stage, 'Contents/MacOS')
    resources = File.join(stage, 'Contents/Resources')
    FileUtils.mkdir_p([executables, resources])
    PRODUCTS.each { |name| FileUtils.cp(File.join(bin, name), executables) }
    RESOURCES.each { |name| FileUtils.cp_r(File.join(bin, name), resources) }
    if mode == 'legacy'
      # Native SwiftPM's accessor uses Bundle.main.bundleURL, which resolves to
      # the outer app for both packaged executables on the supported local layout.
      RESOURCES.each do |name|
        FileUtils.cp_r(File.join(bin, name), stage)
      end
    end
    iconset = File.join(File.dirname(stage), 'TurboFieldfare.iconset')
    FileUtils.mkdir_p(iconset)
    source_icon = File.join(resource_root(File.join(resources, 'TurboFieldfare_TurboFieldfareMac.bundle')), 'turbofieldfare-app-icon.png')
    [16, 32, 128, 256, 512].each do |size|
      [1, 2].each do |scale|
        target = File.join(iconset, "icon_#{size}x#{size}#{scale == 2 ? '@2x' : ''}.png")
        run('/usr/bin/sips', '-z', (size * scale).to_s, (size * scale).to_s, source_icon, '--out', target)
      end
    end
    run('/usr/bin/iconutil', '-c', 'icns', iconset, '-o', File.join(resources, 'TurboFieldfare.icns'))
    FileUtils.rm_r(iconset)
    values = {
      'CFBundleExecutable' => 'TurboFieldfareMac', 'CFBundleIdentifier' => 'com.turbofieldfare.mac',
      'CFBundleName' => 'TurboFieldfare', 'CFBundleDisplayName' => 'TurboFieldfare',
      'CFBundlePackageType' => 'APPL', 'CFBundleShortVersionString' => version,
      'CFBundleVersion' => version, 'LSMinimumSystemVersion' => '26.0',
      'CFBundleIconFile' => 'TurboFieldfare.icns', 'NSHighResolutionCapable' => true
    }
    entries = values.map { |key, value| "<key>#{key}</key>" + (value == true ? '<true/>' : "<string>#{CGI.escapeHTML(value)}</string>") }.join("\n")
    File.write(File.join(stage, 'Contents/Info.plist'), <<~PLIST)
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
      <plist version="1.0"><dict>#{entries}</dict></plist>
    PLIST
    run('/usr/bin/plutil', '-lint', File.join(stage, 'Contents/Info.plist'))
    if sign
      if mode == 'standard'
        PRODUCTS.each { |name| run('/usr/bin/codesign', '--force', '--sign', '-', File.join(executables, name)) }
        run('/usr/bin/codesign', '--force', '--sign', '-', stage)
        run('/usr/bin/codesign', '--verify', '--strict', stage)
      else
        # codesign interprets a main executable inside .app as an outer bundle
        # check. Native SwiftPM's root resources cannot be sealed; verify each
        # preserved Mach-O signature outside that bundle and pin identical bytes.
        Dir.mktmpdir('.signature-check-', File.dirname(stage)) do |verification|
          PRODUCTS.each do |name|
            executable = File.join(executables, name)
            bare = File.join(verification, name)
            FileUtils.cp(executable, bare)
            run('/usr/bin/codesign', '--verify', bare)
            raise "Signature-probe bytes changed for #{name}" unless Digest::SHA256.file(bare).hexdigest == Digest::SHA256.file(executable).hexdigest
          end
        end
      end
    end
  end

  def self.main(arguments)
    install = arguments == ['--install']
    raise 'Usage: Scripts/build_app.sh [--install]' unless arguments.empty? || install
    raise 'Apple Silicon is required' unless run('/usr/bin/uname', '-m') == 'arm64'
    raise 'macOS 26 or newer is required' unless run('/usr/bin/sw_vers', '-productVersion').split('.').first.to_i >= 26
    compiler = run('/usr/bin/xcrun', 'swift', '--version')[/Swift version (\d+)\.(\d+)/, 0]
    raise 'Swift 6.2 or newer is required' unless compiler && (compiler[/\d+\.\d+/].split('.').map(&:to_i) <=> [6, 2]) >= 0
    raise 'macOS SDK 26 or newer is required' unless run('/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-version').split('.').first.to_i >= 26
    PRODUCTS.each { |name| puts "Building #{name}…"; run('/usr/bin/xcrun', 'swift', 'build', '-c', 'release', '--product', name) }
    bin = run('/usr/bin/xcrun', 'swift', 'build', '-c', 'release', '--show-bin-path')
    # Inspect only the accessor for this exact build output, never a stale build.
    native = File.join(bin, 'TurboFieldfareMac.build/DerivedSources/resource_bundle_accessor.swift')
    swift_build = File.expand_path('../../Intermediates.noindex/TurboFieldfare.build/Release/TurboFieldfareMac-p.build/DerivedSources/resource_bundle_accessor.swift', bin)
    accessor = [native, swift_build].find { |path| File.file?(path) } || raise('Cannot locate the generated Mac resource accessor for this build')
    accessor_paths = if accessor == native
      Dir.glob(File.join(bin, '*.build/DerivedSources/resource_bundle_accessor.swift'))
    else
      Dir.glob(File.join(File.expand_path('../../Intermediates.noindex', bin), '**/Release/**/DerivedSources/resource_bundle_accessor.swift'))
    end
    required_accessors = RESOURCES.map do |bundle|
      name = bundle.delete_suffix('.bundle')
      accessor_paths.find { |path| File.read(path).include?(name) } || raise("Missing resource accessor for #{bundle}")
    end
    mode = required_accessors.all? { |path| File.read(path).include?('Bundle.main.resourceURL') } ? 'standard' : 'legacy'
    output_root = File.join(ROOT, 'dist')
    FileUtils.mkdir_p(output_root)
    Dir.mktmpdir('.build-app-', output_root) do |temporary|
      staged = File.join(temporary, 'TurboFieldfare.app')
      package(bin, staged, mode: mode)
      # Deny the generated accessors' build fallback instead of hiding a live build tree.
      escaped_bin = bin.gsub('\\', '\\\\').gsub('"', '\\"')
      policy = "(version 1)(allow default)(deny file-read* (subpath \"#{escaped_bin}\"))"
      PRODUCTS.each do |name|
        puts run('/usr/bin/sandbox-exec', '-p', policy, File.join(staged, 'Contents/MacOS', name), '--check-resources')
      end
      publish(staged, File.join(output_root, 'TurboFieldfare.app'))
    end
    puts "Built dist/TurboFieldfare.app (#{mode == 'standard' ? 'standard resources; ad-hoc app signature' : 'local SwiftPM resource layout; verified linker ad-hoc executable signatures'})."
    return unless install
    applications = File.join(Dir.home, 'Applications')
    FileUtils.mkdir_p(applications)
    Dir.mktmpdir('.turbofieldfare-install-', applications) do |temporary|
      staged = File.join(temporary, 'TurboFieldfare.app')
      FileUtils.cp_r(File.join(output_root, 'TurboFieldfare.app'), staged)
      publish(staged, File.join(applications, 'TurboFieldfare.app'))
    end
    puts 'Installed ~/Applications/TurboFieldfare.app; previous builds retained as TurboFieldfare.previous-*.app when replaced.'
  end
end

if $PROGRAM_NAME == __FILE__
  begin
    LocalMacApp.main(ARGV)
  rescue => error
    warn "App build failed: #{error}"
    exit 1
  end
end
