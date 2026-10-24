import Foundation

/// Native installation already stores the web payload locally. Verify the same
/// required asset list as the PWA before making any offline-ready claim.
enum OfflineAssets {
    struct Report { let count: Int; let bytes: Int }
    static func verify(root: URL) throws -> Report {
        let sw = try String(contentsOf: root.appendingPathComponent("sw.js"), encoding: .utf8)
        guard let start = sw.range(of: "const PRECACHE = ["),
              let end = sw.range(of: "];", range: start.upperBound..<sw.endIndex) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let list = String(sw[start.upperBound..<end.lowerBound])
        // Match only literal array entries, never commented-out paths.
        let regex = try NSRegularExpression(pattern: #"(?m)^\s*"(\./[^"\n]*)"\s*,?\s*$"#)
        let matches = regex.matches(in: list, range: NSRange(list.startIndex..., in: list))
        guard !matches.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
        var bytes = 0
        for match in matches {
            let relative = String(list[Range(match.range(at: 1), in: list)!]).dropFirst(2)
            guard !relative.contains("..") else { throw CocoaError(.fileReadCorruptFile) }
            let name = relative.isEmpty ? "index.html" : String(relative)
            let data = try Data(contentsOf: root.appendingPathComponent(name), options: .mappedIfSafe)
            guard !data.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
            bytes += data.count
        }
        return Report(count: matches.count, bytes: bytes)
    }
}
