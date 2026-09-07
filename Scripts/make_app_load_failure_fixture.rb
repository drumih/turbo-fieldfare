#!/usr/bin/env ruby
# frozen_string_literal: true

require "digest"
require "fileutils"
require "json"
require "time"

abort "usage: #{$PROGRAM_NAME} SOURCE.gturbo OUTPUT.gturbo" unless ARGV.length == 2

source = File.expand_path(ARGV[0])
output = File.expand_path(ARGV[1])
abort "source is not a .gturbo directory: #{source}" unless File.directory?(source)
abort "output already exists: #{output}" if File.exist?(output)

manifest_source = File.join(source, "manifest.json")
layout_source = File.join(source, "packed_experts", "layout.json")
abort "source manifest is missing: #{manifest_source}" unless File.file?(manifest_source)
abort "source expert layout is missing: #{layout_source}" unless File.file?(layout_source)

manifest = File.binread(manifest_source)
layout = File.binread(layout_source)
FileUtils.mkdir_p(File.join(output, "packed_experts"))
File.binwrite(File.join(output, "manifest.json"), manifest)
File.binwrite(File.join(output, "packed_experts", "layout.json"), layout)

receipt = {
  "schemaVersion" => 1,
  "manifestSha256" => Digest::SHA256.hexdigest(manifest),
  "modelDirectoryPath" => output,
  "verificationTimestamp" => Time.now.utc.iso8601,
  "toolVersion" => "make_app_load_failure_fixture.rb",
  "files" => {}
}
File.binwrite(
  File.join(output, "verified-install.json"),
  JSON.generate(receipt)
)

puts output
warn "fixture is metadata-complete but intentionally omits model_weights.bin"
