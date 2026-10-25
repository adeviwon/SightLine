import AppKit
import WebKit
import Foundation

final class Audit: NSObject, WKNavigationDelegate {
    var web: WKWebView!
    var window: NSWindow!
    var server: WebAssetServer!
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    let output = URL(fileURLWithPath: CommandLine.arguments[3])
    var serverRestarted = false
    func start() {
        do {
            let assets = try OfflineAssets.verify(root: root)
            server = WebAssetServer(root: root)
            try server.start { result in DispatchQueue.main.async {
                switch result { case .success(let port): self.open(port, assets); case .failure(let e): self.finish(["error":String(describing:e)]) }
            }}
        } catch { finish(["error":String(describing:error)]) }
    }
    func open(_ port: UInt16, _ assets: OfflineAssets.Report) {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        let marker = "Object.defineProperty(window,'SightLineNativeOffline',{value:Object.freeze({verified:true,assetCount:\(assets.count),bytes:\(assets.bytes)})});"
        config.userContentController.addUserScript(WKUserScript(source: marker, injectionTime:.atDocumentStart, forMainFrameOnly:true))
        // Block every external HTTP(S) request. Loopback bundle serving remains.
        let rules = #"[{"trigger":{"url-filter":"^https?://"},"action":{"type":"block"}},{"trigger":{"url-filter":"^http://127\\.0\\.0\\.1:"},"action":{"type":"ignore-previous-rules"}}]"#
        WKContentRuleListStore.default().compileContentRuleList(forIdentifier:"SightLineAuditOffline", encodedContentRuleList:rules) { list, error in
            if let error { self.finish(["error":String(describing:error)]); return }
            config.userContentController.add(list!)
            self.web = WKWebView(frame:NSRect(x:0,y:0,width:390,height:844), configuration:config)
            self.web.navigationDelegate=self
            self.window = NSWindow(contentRect:NSRect(x:0,y:0,width:390,height:844),styleMask:[.titled],backing:.buffered,defer:false)
            self.window.contentView=self.web
            self.window.orderFront(nil)
            self.web.load(URLRequest(url:URL(string:"http://127.0.0.1:\(port)/index.html")!))
        }
    }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        if CommandLine.arguments.contains("--restart-server"), !serverRestarted {
            // Model the native background/foreground lifecycle with the real
            // listener, persistent page, same origin and real module graph.
            webView.callAsyncJavaScript("await window.__SightLineReaderWarmup; return true;", arguments: [:], in: nil, in: .page) { result in
                guard case .success = result else { self.finish(["error":"Reader warmup failed"]); return }
                let port = self.server.port
                self.server.stop()
                self.server = WebAssetServer(root: self.root, port: port)
                do {
                    try self.server.start { result in DispatchQueue.main.async {
                        switch result {
                        case .success(let resumedPort):
                            guard resumedPort == port else { self.finish(["error":"Origin changed after resume"]); return }
                            self.serverRestarted = true
                            self.runScanAudit(webView)
                        case .failure(let error): self.finish(["error":String(describing:error)])
                        }
                    }}
                } catch { self.finish(["error":String(describing:error)]) }
            }
            return
        }
        runScanAudit(webView)
    }
    func runScanAudit(_ webView: WKWebView) {
        let image = try! Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[2])).base64EncodedString()
        let js = """
        for(let i=0;i<200&&!window.SightLineApp;i++)await new Promise(r=>setTimeout(r,50));
        let externalBlocked=false;try{await fetch('https://example.com/offline-proof',{mode:'no-cors'})}catch(e){externalBlocked=true}
        const image=new Image();image.src='data:image/jpeg;base64,\(image)';await image.decode();
        await SightLineApp.runScan(image,'Public SROIE receipt');
        const stamp=SightLineApp.lastResult?.timestamp;
        document.getElementById('btn-help').click();
        document.querySelector('[data-open-device]').click();
        const frame=document.getElementById('device-frame');
        for(let i=0;i<200&&!frame.contentWindow?.SightLineDeviceReport;i++)await new Promise(r=>setTimeout(r,50));
        const device=frame.contentWindow?.SightLineDeviceReport;
        frame.contentDocument.getElementById('device-return').click();
        await new Promise(r=>setTimeout(r,100));
        const returned=!document.getElementById('device-dialog').open&&SightLineApp.lastResult?.timestamp===stamp;
        return JSON.stringify({serverRestarted:\(serverRestarted),device,returned,externalBlocked,secureContext:isSecureContext,serviceWorkerSupported:'serviceWorker' in navigator,badge:document.getElementById('badge').textContent,nativeOffline:window.SightLineNativeOffline,result:SightLineApp.lastResult});
        """
        webView.callAsyncJavaScript(js,arguments:[:],in:nil,in:.page) { result in
            switch result { case .success(let value):
                if let string=value as? String, let data=string.data(using:.utf8),
                   let record = try? JSONSerialization.jsonObject(with:data) as? [String:Any],
                   let result = record["result"] as? [String:Any],
                   let engine = result["engine"] as? [String:Any],
                   let device = record["device"] as? [String:Any],
                   device["installed"] as? Bool == true,
                   record["returned"] as? Bool == true,
                   record["externalBlocked"] as? Bool == true,
                   record["badge"] as? String == "Ready offline",
                   engine["ocr"] as? String == "ppocrv5",
                   result["category"] as? String == "receipt" {
                    try? data.write(to:self.output)
                    print("PASS native WKWebView: verified bundle, blocked external network, live PP-OCRv5 receipt scan, Device Check and return")
                    exit(0)
                }
                self.finish(["error":"Unexpected result"])
            case .failure(let error): self.finish(["error":String(describing:error)]) }
        }
    }
    func finish(_ value:[String:String]) {try? JSONSerialization.data(withJSONObject:value).write(to:output); print(value); exit(1)}
}
let application=NSApplication.shared
application.setActivationPolicy(.accessory)
let audit=Audit();audit.start()
DispatchQueue.main.asyncAfter(deadline:.now()+120){audit.finish(["error":"Native scan timed out"])}
application.run()
