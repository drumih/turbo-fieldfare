#!/usr/bin/env ruby
require 'tmpdir'
require 'fileutils'
require_relative '../build_app'

def check(value, message)
  raise message unless value
end

def rejects(message)
  begin
    yield
  rescue => error
    check(error.message.include?(message), "wrong failure: #{error}")
    return
  end
  raise "expected failure: #{message}"
end

rejects('Usage:') { LocalMacApp.main(['--unknown']) }
Dir.mktmpdir('tf app packaging тест ') do |root|
  bin = File.join(root, 'bin')
  FileUtils.mkdir_p(bin)
  stage = File.join(root, 'TurboFieldfare.app')
  rejects('Missing executable') { LocalMacApp.package(bin, stage, mode: 'standard', sign: false) }
  LocalMacApp::PRODUCTS.each { |name| File.write(File.join(bin, name), '#!/bin/sh\nexit 0\n'); File.chmod(0755, File.join(bin, name)) }
  rejects('Missing required resource') { LocalMacApp.package(bin, stage, mode: 'standard', sign: false) }
  LocalMacApp::RESOURCES.each { |name| FileUtils.mkdir_p(File.join(bin, name)) }
  FileUtils.mkdir_p(File.join(bin, 'TurboFieldfare_TurboFieldfare.bundle/Metal'))
  File.write(File.join(bin, 'TurboFieldfare_TurboFieldfareAppCore.bundle/app-prompts.json'), '{}')
  File.write(File.join(bin, 'swift-transformers_Hub.bundle/gpt2_tokenizer_config.json'), '{}')
  FileUtils.mkdir_p(File.join(bin, 'SwiftMath_SwiftMath.bundle/mathFonts.bundle'))
  icon = File.join(LocalMacApp::ROOT, 'Sources/TurboFieldfareApp/Mac/Resources/turbofieldfare-app-icon.png')
  FileUtils.cp(icon, File.join(bin, 'TurboFieldfare_TurboFieldfareMac.bundle/turbofieldfare-app-icon.png'))
  File.write(File.join(bin, 'unrelated-tests.bundle'), 'exclude')
  LocalMacApp.package(bin, stage, mode: 'standard', sign: false)
  check(File.file?(File.join(stage, 'Contents/MacOS/TurboFieldfareDecodeService')), 'helper missing')
  check(File.file?(File.join(stage, 'Contents/Resources/TurboFieldfare.icns')), 'icon missing')
  check(File.read(File.join(stage, 'Contents/Info.plist')).include?(LocalMacApp.version), 'version mismatch')
  check(!File.exist?(File.join(stage, 'Contents/Resources/unrelated-tests.bundle')), 'test bundle included')
  destination = File.join(root, 'published.app')
  LocalMacApp.publish(stage, destination)
  before = File.read(File.join(destination, 'Contents/Info.plist'))
  FileUtils.rm(File.join(bin, 'TurboFieldfareDecodeService'))
  rejects('Missing executable') { LocalMacApp.package(bin, stage, mode: 'standard', sign: false) }
  check(File.read(File.join(destination, 'Contents/Info.plist')) == before, 'failed build changed output')
  helper = File.join(destination, 'Contents/MacOS/TurboFieldfareDecodeService')
  FileUtils.cp('/bin/sleep', helper)
  LocalMacApp.run('/usr/bin/codesign', '--force', '--sign', '-', helper)
  process = Process.spawn(helper, '30')
  begin
    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + 3
    until LocalMacApp.active?(destination)
      if Process.clock_gettime(Process::CLOCK_MONOTONIC) >= deadline
        raise "dummy helper was not detected: #{LocalMacApp.run('/bin/ps', '-p', process.to_s, '-o', 'comm=')}"
      end
      sleep 0.01
    end
    rejects('Quit the app and its decode helper') { LocalMacApp.publish(stage, destination) }
    rejects('Quit the app and its decode helper') { LocalMacApp.publish(stage, File.realpath(destination)) }
    check(File.read(File.join(destination, 'Contents/Info.plist')) == before, 'running helper changed output')
  ensure
    Process.kill('TERM', process)
    Process.wait(process)
  end
  FileUtils.mkdir_p(stage)
  rejects('unrelated directory') { LocalMacApp.publish(stage, bin) }
  FileUtils.rm_r(stage)
  FileUtils.cp_r(destination, stage)
  LocalMacApp.publish(stage, destination)
  check(Dir.glob(File.join(root, 'published.previous-*.app/Contents/Info.plist')).length == 1, 'previous app not retained')
end
puts 'PASS packaging: arguments, missing helper/resources, metadata/icon, exclusions, staged failure, running helper refusal and preserved replacement'
