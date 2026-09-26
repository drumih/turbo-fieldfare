#!/usr/bin/env ruby
# Keep the public app buildable without Xcode's SwiftUI platform plugins.
require 'pathname'

forbidden = /@\s*(?:[A-Za-z_]\w*\s*\.\s*)*State\b|\#\s*Preview\b/
if ARGV.delete('--self-test')
  ['@SwiftUICore.State var value = 0', '@Other.State var value = 0', '@State var value = 0', '@SwiftUI.State var value = 0', "@SwiftUI.\nState var value = 0", '#Preview { Text("x") }'].each do |source|
    abort "Missed forbidden macro: #{source}" unless forbidden.match?(source)
  end
  ['@StoredState var value = 0', '@StateObject var model', 'SwiftUI.State<Int>', 'struct Demo: PreviewProvider {}'].each do |source|
    abort "Rejected supported spelling: #{source}" if forbidden.match?(source)
  end
  puts 'SwiftUI source guard self-tests passed'
end
abort 'Usage: ruby Scripts/check_swiftui_macros.rb [--self-test]' unless ARGV.empty?
root = File.expand_path('..', __dir__)
failures = Dir.glob(File.join(root, 'Sources/**/*.swift')).flat_map do |path|
  text = File.read(path)
  text.to_enum(:scan, forbidden).map do
    line = text[0...Regexp.last_match.begin(0)].count("\n") + 1
    "#{Pathname.new(path).relative_path_from(Pathname.new(root))}:#{line}: use StoredState or PreviewProvider; SwiftUI platform macros require Xcode plugins"
  end
end
abort failures.join("\n") unless failures.empty?
puts 'No SwiftUI State or Preview macros in production sources'
