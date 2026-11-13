/// The single version literal in the tree.
///
/// It used to be `CFBundleVersion` in the `Info.plist`
/// `tools/install_tinytitan.sh` wrote when it wrapped the Mac app into a
/// bundle. That app is gone, so the server carries the version instead: it is
/// printed in the ready banner, which is the one line every launch shows, and
/// `tools/release.sh` refuses to publish a tag that disagrees with it.
///
/// Preparing a release therefore still means bumping exactly one declaration;
/// `docs/release-process.md` step 1 is that step. Nothing else in the tree
/// holds a version, so nothing can drift from this one.
public enum ServerVersion {
    public static let current = "6.3"
}
