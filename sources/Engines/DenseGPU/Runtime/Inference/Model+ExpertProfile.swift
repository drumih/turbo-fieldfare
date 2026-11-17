import Foundation

// The persisted expert-routing profile: where it lives, and how a request's
// counts get back into it.
//
// Rebuilt from Project Maya's `expert-profile.bin` idea. Maya writes the
// counts after every answer and reads them back at the next start, where the
// most-routed experts are placed first. This engine already has the aging-LFU
// score those counts feed (`PreadExpertStreamer.expertUseCount`), but it is
// wiped at every prefill-to-decode handover and never leaves the process, so
// every start re-learns the same working set. The profile closes that gap.
//
// The file lives beside the install rather than in a user cache so it follows
// the weights it describes. `TINYTITAN_EXPERT_PROFILE=0` turns the whole
// mechanism off; any other value is used as the path instead.

extension Model {
    static func expertProfileURL(directoryURL: URL) -> URL? {
        guard let raw = ProcessInfo.processInfo.environment["TINYTITAN_EXPERT_PROFILE"]
        else {
            return directoryURL.appendingPathComponent("expert-profile.bin")
        }
        if raw == "0" { return nil }
        let path = (raw as NSString).expandingTildeInPath
        guard !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }

    /// Fold the open layers' current LFU scores into the profile file.
    ///
    /// Best effort by design: a read-only install, a full disk or a racing
    /// writer must not fail a request that has already produced tokens. The
    /// write is serialised so two concurrent generations cannot interleave one
    /// file, and it starts from the profile loaded at open so layers that were
    /// never touched keep their history.
    public func saveExpertUsageProfile() {
        guard let url = expertProfileURL else { return }
        var updated =
            expertUsageProfile
            ?? ExpertUsageProfile(
                layers: packedExpertsLayout.numLayers,
                expertsPerLayer: packedExpertsLayout.expertsPerLayer)
        let snapshots: [(Int, [Int])] = streamersQueue.sync {
            streamersBox.streamers.enumerated().compactMap { layer, streamer in
                guard let streamer else { return nil }
                let counts = streamer.expertUseCountsSnapshot()
                // The next request's handover restores this, so a long-lived
                // server keeps learning instead of resetting to the snapshot
                // it loaded at process start.
                streamer.adoptExpertUseCounts(counts)
                return (layer, counts)
            }
        }
        for (layer, counts) in snapshots {
            updated.setCounts(layer: layer, counts)
        }
        guard !updated.isEmpty else { return }
        let snapshot = updated
        profileWriteQueue.async {
            try? snapshot.write(to: url)
        }
    }
}
