import AppKit
import WebKit
import Foundation

final class Audit: NSObject, WKNavigationDelegate {
    var web: WKWebView!
    var window: NSWindow!
    var server: WebAssetServer!
    let root = URL(fileURLWithPath: CommandLine.arguments[1])
    let output = URL(fileURLWithPath: CommandLine.arguments[3])
    func start() {
        do {
            let assets = try OfflineAssets.verify(root: root)
            server = WebAssetServer(root: root, port: 19193)
            try server.start { result in DispatchQueue.main.async {
                switch result { case .success(let port): self.open(port, assets); case .failure(let e): self.finish(["error":String(describing:e)]) }
            }}
        } catch { finish(["error":String(describing:error)]) }
    }
    func open(_ port: UInt16, _ assets: OfflineAssets.Report) {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .default()
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
        let phase = CommandLine.arguments[2]
        let js = """
        for(let i=0;i<200&&!window.SightLineApp;i++)await new Promise(r=>setTimeout(r,50));
        const phase='\(phase)';
        if(phase==='write'){
          await SightLineHistory.clear();
          const image=new Image();image.src='data:image/jpeg;base64,\(try! Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[4])).base64EncodedString())';await image.decode();
          await SightLineApp.runScan(image,'Public receipt, native history test');
          document.getElementById('btn-save-scan').click();document.getElementById('consent-save').click();
          for(let i=0;i<200&&!(await SightLineHistory.list()).length;i++)await new Promise(r=>setTimeout(r,50));
          return JSON.stringify({phase,passed:(await SightLineHistory.list()).length===1,origin:location.origin});
        }
        window.ocrCalls=0;SightLine.runOCR=()=>{ocrCalls++;throw new Error('History must not invoke OCR')};
        document.getElementById('btn-history').click();
        for(let i=0;i<200&&!document.querySelector('.history-record button');i++)await new Promise(r=>setTimeout(r,50));
        document.querySelector('.history-record button').click();
        for(let i=0;i<200&&!SightLineApp.lastResult;i++)await new Promise(r=>setTimeout(r,50));
        const record={phase,passed:SightLineApp.lastResult?.category==='receipt'&&ocrCalls===0,origin:location.origin,offline:window.SightLineNativeOffline?.verified,engine:SightLineApp.lastResult?.engine};
        await SightLineHistory.clear();return JSON.stringify(record);
        """
        webView.callAsyncJavaScript(js,arguments:[:],in:nil,in:.page) { result in
            switch result { case .success(let value):
                if let string=value as? String, let data=string.data(using:.utf8),
                   let record=try? JSONSerialization.jsonObject(with:data) as? [String:Any], record["passed"] as? Bool == true {
                   try? data.write(to:self.output); print("PASS native history \(CommandLine.arguments[2]): stable origin and persistent WKWebView");exit(0)
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
