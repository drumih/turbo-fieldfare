#!/usr/bin/env ruby

require "pathname"
require "set"
require "uri"

ROOT = Pathname.new(File.expand_path("..", __dir__))

def markdown_files(arguments)
  return arguments.map { |path| ROOT.join(path).cleanpath } unless arguments.empty?

  root_files = Dir[ROOT.join("*.md").to_s].map { |path| Pathname.new(path) }
  docs_files = Dir[ROOT.join("docs/**/*.md").to_s].map { |path| Pathname.new(path) }
  (root_files + docs_files).select(&:file?).uniq.sort
end

def without_inline_code(line)
  line.gsub(/`+[^`]*`+/, "")
end

def github_heading_slug(heading)
  heading
    .gsub(/<[^>]*>/, "")
    .gsub(/!\[([^\]]*)\]\([^)]+\)/, "\\1")
    .gsub(/\[([^\]]+)\]\([^)]+\)/, "\\1")
    .gsub(/[`*_~]/, "")
    .downcase
    .gsub(/[^\p{L}\p{N}\s_-]/u, "")
    .strip
    .gsub(/\s+/, "-")
end

def anchors_for(path)
  anchors = Set.new
  slug_counts = Hash.new(0)
  in_fence = false

  path.each_line do |line|
    if line.lstrip.start_with?("```")
      in_fence = !in_fence
      next
    end
    next if in_fence

    line.scan(/\bid=["']([^"']+)["']/i).flatten.each { |id| anchors << id }

    match = line.match(/\A {0,3}\#{1,6}\s+(.+?)\s*\#*\s*\z/)
    next unless match

    base = github_heading_slug(match[1])
    next if base.empty?

    occurrence = slug_counts[base]
    anchors << (occurrence.zero? ? base : "#{base}-#{occurrence}")
    slug_counts[base] += 1
  end

  anchors
end

def local_targets(line)
  markdown = without_inline_code(line)
    .scan(/\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+["'][^"']*["'])?\)/)
    .flatten
  html = line.scan(/<(?:a|img)\b[^>]*\b(?:href|src)=["']([^"']+)["']/i).flatten
  markdown + html
end

missing = []
files = markdown_files(ARGV)
anchor_cache = {}

files.each do |source|
  in_fence = false
  source.each_line.with_index(1) do |line, line_number|
    if line.lstrip.start_with?("```")
      in_fence = !in_fence
      next
    end
    next if in_fence

    local_targets(line).each do |raw_target|
      target = raw_target.delete_prefix("<").delete_suffix(">")
      next if target.empty?
      next if target.match?(%r{\A(?:https?:|mailto:|data:)})

      path, fragment = target.split("#", 2)
      resolved = path.empty? ? source : source.dirname.join(path).cleanpath
      unless resolved.exist?
        missing << "#{source.relative_path_from(ROOT)}:#{line_number} -> #{raw_target}"
        next
      end

      next if fragment.nil? || fragment.empty? || resolved.extname.downcase != ".md"

      anchor = URI::DEFAULT_PARSER.unescape(fragment)
      anchor_cache[resolved] ||= anchors_for(resolved)
      unless anchor_cache[resolved].include?(anchor)
        missing << "#{source.relative_path_from(ROOT)}:#{line_number} -> #{raw_target} (missing anchor)"
      end
    end
  end
end

if missing.empty?
  puts "checked #{files.count} Markdown files; all local links and anchors resolve"
  exit 0
end

warn missing.join("\n")
exit 1
