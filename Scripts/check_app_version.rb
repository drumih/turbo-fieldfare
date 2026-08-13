#!/usr/bin/env ruby

# Verifies that the version the Mac app shows is not behind the newest published
# GitHub release.
#
# Most users build from a clone, where there is no Info.plist, so the compiled
# constant in AboutPanelPresentation is the version they see. Nothing bumps it
# automatically, so without this check it silently ages: the About panel would
# claim an old version on a current checkout.
#
# Being ahead of the newest release is allowed. That is the normal state between
# bumping the constant and publishing the release built from it.

require "json"
require "net/http"
require "uri"

ROOT = File.expand_path("..", __dir__)
SOURCE = File.join(ROOT, "Sources/TurboFieldfareApp/MacPresentation/AboutPanelPresentation.swift")
RELEASES_URL = "https://api.github.com/repos/drumih/turbo-fieldfare/releases/latest"

def compiled_version
  source = File.read(SOURCE)
  match = source[/fallbackShortVersion\s*=\s*"([^"]+)"/, 1]
  abort "could not find fallbackShortVersion in #{SOURCE}" unless match

  match
end

def published_version
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
    warn "warning: could not read the latest release (#{response.code}); skipping the comparison"
    return nil
  end

  JSON.parse(response.body)["tag_name"].to_s.delete_prefix("v")
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
published = published_version
if published.nil?
  puts "app version #{compiled}; latest release unknown"
  exit 0
end

case compare(compiled, published)
when -1
  abort <<~MESSAGE
    app version #{compiled} is behind the latest release #{published}

    Update fallbackShortVersion in
    Sources/TurboFieldfareApp/MacPresentation/AboutPanelPresentation.swift
    so a clone build reports the version it actually corresponds to.
  MESSAGE
when 0
  puts "app version #{compiled} matches the latest release"
else
  puts "app version #{compiled} is ahead of the latest release #{published}; unreleased build"
end
