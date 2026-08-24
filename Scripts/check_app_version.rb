#!/usr/bin/env ruby

# Verifies that the version the Mac app shows has not fallen behind the
# published GitHub releases.
#
# Most users build from a clone, where there is no Info.plist, so the compiled
# constant in AboutPanelPresentation is the version they see. Nothing bumps it
# automatically, so without this check it silently ages: the About panel would
# claim an old version on a current checkout.
#
# The release flow is: merge work, tag main, then merge a follow-up PR that
# bumps the constant. Between the tag and that follow-up, the constant is one
# release behind on purpose, so exactly one release of lag is allowed. Two or
# more means the bump was forgotten. Being ahead is allowed too, which is the
# state after the bump merges and before the next tag.

require "json"
require "net/http"
require "uri"

ROOT = File.expand_path("..", __dir__)
SOURCE = File.join(ROOT, "Sources/TurboFieldfareApp/MacPresentation/AboutPanelPresentation.swift")
RELEASES_URL = "https://api.github.com/repos/drumih/turbo-fieldfare/releases?per_page=100"

def compiled_version
  source = File.read(SOURCE)
  match = source[/fallbackShortVersion\s*=\s*"([^"]+)"/, 1]
  abort "could not find fallbackShortVersion in #{SOURCE}" unless match

  match
end

# Published release versions, newest first. Returns nil when GitHub cannot be
# reached, which skips the comparison rather than failing an offline run.
def published_versions
  uri = URI(RELEASES_URL)
  request = Net::HTTP::Get.new(uri)
  request["Accept"] = "application/vnd.github+json"
  request["User-Agent"] = "turbofieldfare-version-check"
  token = ENV["GITHUB_TOKEN"] || ENV["GH_TOKEN"]
  request["Authorization"] = "Bearer #{token}" if token && !token.empty?

  response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: true, open_timeout: 10, read_timeout: 10) do |http|
    http.request(request)
  end
  unless response.is_a?(Net::HTTPSuccess)
    warn "warning: could not read releases (#{response.code}); skipping the comparison"
    return nil
  end

  releases = JSON.parse(response.body)
  return [] unless releases.is_a?(Array)

  # Prereleases are rejected with drafts: an rc tag becoming "latest" pushes
  # the real latest release into the floor slot, and the check then silently
  # accepts a version two releases behind.
  releases
    .reject { |release| release["draft"] || release["prerelease"] }
    .map { |release| release["tag_name"].to_s.delete_prefix("v") }
    .reject(&:empty?)
    .sort { |left, right| compare(right, left) }
rescue StandardError => error
  warn "warning: could not reach GitHub (#{error.class}); skipping the comparison"
  nil
end

# Compares dotted numeric versions. Returns -1, 0, or 1.
def compare(left, right)
  left_parts = left.split(".").map(&:to_i)
  right_parts = right.split(".").map(&:to_i)
  length = [left_parts.length, right_parts.length].max
  length.times do |index|
    result = (left_parts[index] || 0) <=> (right_parts[index] || 0)
    return result unless result.zero?
  end
  0
end

compiled = compiled_version
published = published_versions
if published.nil?
  puts "app version #{compiled}; releases unknown"
  exit 0
end
if published.empty?
  puts "app version #{compiled}; no releases published yet"
  exit 0
end

latest = published.first
# One release of lag is the tag-then-bump window. The release before the newest
# is therefore the oldest version the constant may still report.
floor = published[1] || latest

if compare(compiled, floor) < 0
  abort <<~MESSAGE
    app version #{compiled} is more than one release behind (latest #{latest})

    Update fallbackShortVersion in
    Sources/TurboFieldfareApp/MacPresentation/AboutPanelPresentation.swift
    so a clone build reports the version it actually corresponds to.
  MESSAGE
end

case compare(compiled, latest)
when 1 then puts "app version #{compiled} is ahead of the latest release #{latest}; unreleased build"
when 0 then puts "app version #{compiled} matches the latest release"
else puts "app version #{compiled} is one release behind #{latest}; bump pending"
end
