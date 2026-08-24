#!/usr/bin/env ruby

# Fails when the repository tracks a symlink.
#
# This exists because a tracked symlink named after an ignored directory is a
# data-loss trap, and git gives no warning when it springs. `scratch` was once
# committed as a symlink to its own absolute path. Checking out, merging, or
# pulling that tree replaces the real `scratch/` directory - the converted
# `.gturbo` models and the download cache, tens of gigabytes - with a one-line
# symlink, exit code 0, no prompt. Recovery is a full remote repack.
#
# A directory-only ignore pattern cannot prevent the recurrence: `scratch/`
# matches the directory but not a symlink of the same name, so `git add -A`
# stages it. The anchored `/scratch` in .gitignore closes that specific hole;
# this check closes the general one, for any path.
#
# The repository has no legitimate tracked symlinks. If one is ever genuinely
# needed, add it to ALLOWED below with the reason, so the exemption is a
# deliberate edit rather than a silent pass.

ROOT = File.expand_path("..", __dir__)

ALLOWED = {}.freeze

SYMLINK_MODE = "120000".freeze

entries = `git -C #{ROOT} ls-files -s -z`
abort "could not read the git index" unless $?.success?

offenders = entries.split("\0").map do |entry|
  mode, rest = entry.split(" ", 2)
  next nil unless mode == SYMLINK_MODE

  path = rest.split("\t", 2).last
  next nil if ALLOWED.key?(path)

  target = `git -C #{ROOT} show :#{path}`.strip
  [path, target]
end.compact

if offenders.empty?
  puts "no tracked symlinks"
  exit 0
end

listing = offenders.map { |path, target| "  #{path} -> #{target}" }.join("\n")
abort <<~MESSAGE
  tracked symlink(s) found:

  #{listing}

  A tracked symlink whose name collides with a real local directory is deleted
  without warning on checkout, merge, or pull. Untrack it with

    git rm --cached <path>

  and make sure .gitignore matches the name itself, not just its children:
  an anchored `/name` covers the directory, its contents, and a symlink of the
  same name, while `name/` covers only the directory.
MESSAGE
