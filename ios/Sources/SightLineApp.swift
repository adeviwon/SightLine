import SwiftUI
import WebKit

@main
struct SightLineApp: App {
    var body: some Scene { WindowGroup { ReaderRoot() } }
}

@MainActor
final class ReaderHost: ObservableObject {
    @Published var url: URL?
    @Published var error: String?
    @Published var offlineAssets: OfflineAssets.Report?
    private var server: WebAssetServer?

    func start() {
        guard server == nil else { return }
        error = nil
        guard let root = Bundle.main.url(forResource: "app", withExtension: nil) else {
            error = "The reading files are missing. Please reinstall SightLine."
            return
        }
        do { offlineAssets = try OfflineAssets.verify(root: root) }
        catch {
            self.error = "Some offline reading files are missing. Please reinstall SightLine."
            return
        }
        let server = WebAssetServer(root: root, port: 18192)
        self.server = server
        do {
            try server.start { [weak self] result in
                DispatchQueue.main.async {
                    switch result {
                    case .success(let port): self?.url = URL(string: "http://127.0.0.1:\(port)/index.html")
                    case .failure: self?.error = "SightLine could not start. Close the app and try again."
                    }
                }
            }
        } catch { self.error = "SightLine could not start. Close the app and try again." }
    }

    func suspend() {
        // iOS can invalidate a listener while the app is suspended. Recreate it
        // on foreground instead of retaining a once-ready, dead listener.
        server?.stop()
        server = nil
    }
}

struct ReaderRoot: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var host = ReaderHost()
    var body: some View {
        Group {
            if let url = host.url { ReaderWebView(url: url, offlineAssets: host.offlineAssets) }
            else if let error = host.error {
                ContentUnavailableView("Unable to open SightLine", systemImage: "doc.text.viewfinder", description: Text(error))
            } else { ProgressView("Opening SightLine…").font(.title2) }
        }
        .task { host.start() }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { host.suspend() }
            else if phase == .active { host.start() }
        }
    }
}

struct ReaderWebView: UIViewRepresentable {
    let url: URL
    let offlineAssets: OfflineAssets.Report?
    func makeCoordinator() -> Navigation { Navigation() }
    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.allowsInlineMediaPlayback = true
        let navigation = context.coordinator
        configuration.userContentController.add(navigation.speechBridge, name: "sightlineSpeech")
        if let assets = offlineAssets {
            // Only our validated bundle and loopback server receive this marker.
            let script = "if(location.origin === '\(url.scheme!)://\(url.host!):\(url.port!)'){Object.defineProperty(window,'SightLineNativeOffline',{value:Object.freeze({verified:true,assetCount:\(assets.count),bytes:\(assets.bytes)}),writable:false,configurable:false});}"
            configuration.userContentController.addUserScript(WKUserScript(source: script, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        }
        let view = WKWebView(frame: .zero, configuration: configuration)
        navigation.webView = view
        // The native speech bridge returns voice catalogs and playback events
        // by evaluating JavaScript in this web view. Without this connection,
        // JS can send requests but never receives replies, so installed iOS
        // voices appear missing and speech appears to stall.
        navigation.speechBridge.webView = view
        view.navigationDelegate = context.coordinator
        view.allowsBackForwardNavigationGestures = true
        view.load(URLRequest(url: url))
        return view
    }
    func updateUIView(_ view: WKWebView, context: Context) {}
    static func dismantleUIView(_ view: WKWebView, coordinator: Navigation) {
        coordinator.speechBridge.stop()
        view.configuration.userContentController.removeScriptMessageHandler(forName: "sightlineSpeech")
        coordinator.webView = nil
    }

    final class Navigation: NSObject, WKNavigationDelegate {
        let speechBridge = NativeSpeechBridge()
        weak var webView: WKWebView?
        #if DEBUG
        private var speechAuditStarted = false
        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            if ProcessInfo.processInfo.arguments.contains("--speech-observe"), !speechAuditStarted {
                speechAuditStarted = true
                webView.evaluateJavaScript("""
                (()=>{
                  const send=report=>window.webkit.messageHandlers.sightlineSpeech.postMessage({action:'diagnostic',report});
                  const snapshot=()=>({screen:document.body.dataset.screen,language:document.getElementById('speak-lang')?.value,disabled:document.getElementById('btn-speak')?.disabled,category:SightLineApp.lastResult?.category,fieldCount:SightLineApp.lastResult?.fields?.length,eligible:SightLineSpeech.plan(SightLineApp.lastResult).eligible,state:TTS.getState().status,voiceReady:TTS.hasVoice(document.getElementById('speak-lang')?.value),note:document.getElementById('speech-note')?.textContent});
                  TTS.subscribe(state=>send({event:'speech',...state}));
                  document.addEventListener('pointerup',e=>{if(e.target.closest('#btn-speak')){send({event:'tap',...snapshot()});setTimeout(()=>send({event:'after-tap',...snapshot(),error:document.getElementById('error-banner')?.textContent}),500);}},true);
                  window.addEventListener('error',e=>send({event:'js-error',message:e.message}));
                  send({event:'ready'});
                })();
                """)
                return
            }
            guard ProcessInfo.processInfo.arguments.contains("--speech-audit"), !speechAuditStarted else { return }
            speechAuditStarted = true
            let script = """
            for (let i=0;i<100&&!window.TTS;i++) await new Promise(r=>setTimeout(r,50));
            if (!window.TTS) return {error:'speech-module-missing'};
            const voices=await TTS.loadVoices(5000);
            const report={native:TTS.nativeBackend(),voiceCount:voices.length,english:TTS.pickVoice('en-GB'),events:[]};
            const unsubscribe=TTS.subscribe(s=>report.events.push(s));
            try {
              if (fixture) {
                const consent=document.getElementById('save-scans');
                const wasSaving=consent.checked;
                try {
                  if(wasSaving){consent.checked=false;consent.dispatchEvent(new Event('change'));}
                  const image=new Image();image.src='data:image/jpeg;base64,'+fixture;await image.decode();
                  const result=await SightLineApp.runScan(image,'Public receipt voice check');
                  report.scan={engine:result?.engine,category:result?.category,fields:result?.fields?.length};
                  const language=document.getElementById('visible-language');language.value='en-GB';language.dispatchEvent(new Event('change'));
                  const button=document.getElementById('btn-speak');report.buttonEnabled=!button.disabled;
                  button.click();
                  for(let i=0;i<2400&&!report.events.some(s=>['ended','error'].includes(s.status));i++)await new Promise(r=>setTimeout(r,50));
                  report.playback=TTS.getState();
                } finally {if(wasSaving){consent.checked=true;consent.dispatchEvent(new Event('change'));}}
              } else report.playback=await TTS.speak('SightLine voice test. Your English voice is ready.',{lang:'en-GB'});
            }
            catch(e) { report.error=e.code||String(e); }
            unsubscribe();return report;
            """
            let fixture = ProcessInfo.processInfo.environment["SIGHTLINE_SPEECH_AUDIT_IMAGE"] ?? ""
            webView.callAsyncJavaScript(script, arguments: ["fixture": fixture], in: nil, in: .page) { result in
                switch result {
                case .success(let report):
                    if let data = try? JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]), let json = String(data: data, encoding: .utf8) { print("SIGHTLINE_SPEECH_AUDIT \(json)") }
                case .failure(let error): print("SIGHTLINE_SPEECH_AUDIT_ERROR \(error)")
                }
            }
        }
        #endif
        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            // Keep the reader inside its bundled origin. Credits are bundled too.
            let destination = action.request.url
            decisionHandler(destination?.host == "127.0.0.1" && destination?.port == webView.url?.port
                || webView.url == nil && destination?.host == "127.0.0.1" ? .allow : .cancel)
        }
    }
}
