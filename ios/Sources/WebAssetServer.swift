import Foundation
import Network

/// A tiny read-only HTTP/1.1 server for the bundled web payload.
///
/// Adapted from the local Aria Reader shell. SightLine's worker modules need an
/// HTTP origin, rather than an opaque `file://` origin, so module imports and
/// worker fetches can use ordinary same-origin URLs. Loopback HTTP is treated as
/// a secure context. WKWebView service-worker availability is a separate platform
/// capability; OfflineAssets verifies the native installed payload without it.
///
/// Deliberately minimal: GET only, no directory listing, no ranges, path traversal
/// rejected. It serves what is in the app bundle and nothing else.
final class WebAssetServer {

    enum ServerError: Error { case badPort, noPort }

    private let root: URL
    private let queue = DispatchQueue(label: "com.sightline.assetserver")
    private var listener: NWListener?
    private var live: [ObjectIdentifier: NWConnection] = [:]
    private(set) var port: UInt16 = 0

    private let requestedPort: UInt16
    init(root: URL, port: UInt16 = 0) { self.root = root; self.requestedPort = port }

    /// Starts listening on a free loopback port and reports the chosen port once ready.
    func start(completion: @escaping (Result<UInt16, Error>) -> Void) throws {
        guard listener == nil else { return }
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        params.requiredInterfaceType = .loopback
        guard let port = NWEndpoint.Port(rawValue: requestedPort) else { throw ServerError.badPort }

        let listener = try NWListener(using: params, on: port)
        var reported = false
        listener.stateUpdateHandler = { [weak self] state in
            guard let self, !reported else { return }
            switch state {
            case .ready:
                reported = true
                if let chosen = listener.port?.rawValue {
                    self.port = chosen
                    completion(.success(chosen))
                } else {
                    completion(.failure(ServerError.noPort))
                }
            case .failed(let error):
                reported = true
                completion(.failure(error))
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.start(queue: queue)
        self.listener = listener
    }

    func stop() {
        listener?.cancel()
        listener = nil
        queue.async { [weak self] in
            self?.live.values.forEach { $0.cancel() }
            self?.live.removeAll()
        }
    }

    // MARK: - connection handling

    private func accept(_ connection: NWConnection) {
        let key = ObjectIdentifier(connection)
        live[key] = connection
        connection.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed, .cancelled:
                self?.queue.async { self?.live[key] = nil }
            default:
                break
            }
        }
        connection.start(queue: queue)
        read(connection, buffer: Data())
    }

    private func read(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] chunk, _, isComplete, error in
            guard let self else { return }
            var buffer = buffer
            if let chunk { buffer.append(chunk) }

            if let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: buffer[..<headerEnd.lowerBound], as: UTF8.self)
                self.respond(to: head, on: connection)
                return
            }
            if error != nil || isComplete || buffer.count > 64 * 1024 {
                connection.cancel()
                return
            }
            self.read(connection, buffer: buffer)
        }
    }

    private func respond(to head: String, on connection: NWConnection) {
        let requestLine = head.split(separator: "\r\n", maxSplits: 1).first.map(String.init) ?? ""
        let parts = requestLine.split(separator: " ")
        let method = parts.count > 0 ? String(parts[0]) : "GET"
        var path = parts.count > 1 ? String(parts[1]) : "/"

        if let query = path.firstIndex(of: "?") { path = String(path[..<query]) }
        path = path.removingPercentEncoding ?? path
        if path.isEmpty || path == "/" { path = "/index.html" }

        guard method == "GET" || method == "HEAD" else {
            return send(connection, status: "405 Method Not Allowed", type: "text/plain", body: Data("only GET/HEAD\n".utf8))
        }
        guard !path.contains("..") else {
            return send(connection, status: "403 Forbidden", type: "text/plain", body: Data("path traversal rejected\n".utf8))
        }

        let fileURL = root.appendingPathComponent(String(path.drop(while: { $0 == "/" })))
        guard FileManager.default.fileExists(atPath: fileURL.path),
              let body = try? Data(contentsOf: fileURL) else {
            return send(connection, status: "404 Not Found", type: "text/plain", body: Data("not found: \(path)\n".utf8))
        }
        send(connection, status: "200 OK", type: WebAssetServer.mediaType(for: fileURL.lastPathComponent), body: body)
    }

    private func send(_ connection: NWConnection, status: String, type: String, body: Data) {
        var head = "HTTP/1.1 \(status)\r\n"
        head += "Content-Type: \(type)\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Cache-Control: no-store\r\n"
        head += "X-Content-Type-Options: nosniff\r\n"
        head += "Connection: close\r\n\r\n"
        var out = Data(head.utf8)
        out.append(body)
        connection.send(content: out, isComplete: true, completion: .contentProcessed { _ in
            connection.cancel()
        })
    }

    /// MIME types matter more than they look here: a module script served as anything
    /// other than a JavaScript type is refused by WebKit outright.
    static func mediaType(for name: String) -> String {
        let lower = name.lowercased()
        if lower.hasSuffix(".traineddata.gz") { return "application/gzip" }
        switch (lower as NSString).pathExtension {
        case "html", "htm":      return "text/html; charset=utf-8"
        case "js", "mjs":        return "text/javascript; charset=utf-8"
        case "css":              return "text/css; charset=utf-8"
        case "json":             return "application/json"
        case "webmanifest":      return "application/manifest+json"
        case "png":              return "image/png"
        case "jpg", "jpeg":      return "image/jpeg"
        case "svg":              return "image/svg+xml"
        case "gz":               return "application/gzip"
        case "wasm":             return "application/wasm"
        case "txt":              return "text/plain; charset=utf-8"
        default:                 return "application/octet-stream"
        }
    }
}
