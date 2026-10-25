import AVFoundation
import WebKit

/// Uses iOS's installed speech voices for the bundled WKWebView reader. This
/// avoids relying on WebKit's sometimes silent SpeechSynthesis audio route.
final class NativeSpeechBridge: NSObject, WKScriptMessageHandler, AVSpeechSynthesizerDelegate {
    weak var webView: WKWebView?
    private let synthesizer = AVSpeechSynthesizer()
    private var activeID: String?
    private var utteranceIDs: [ObjectIdentifier: String] = [:]

    override init() {
        super.init()
        synthesizer.usesApplicationAudioSession = true
        synthesizer.delegate = self
    }

    func stop() {
        activeID = nil
        if synthesizer.isSpeaking || synthesizer.isPaused {
            synthesizer.stopSpeaking(at: .immediate)
        }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    func userContentController(_ userContentController: WKUserContentController,
                               didReceive message: WKScriptMessage) {
        guard let body = message.body as? [String: Any],
              let action = body["action"] as? String else { return }
        switch action {
        #if DEBUG
        case "diagnostic":
            if let data = try? JSONSerialization.data(withJSONObject: body["report"] ?? [:], options: [.sortedKeys]),
               let json = String(data: data, encoding: .utf8) { print("SIGHTLINE_SPEECH_TRACE \(json)") }
        #endif
        case "voices":
            let voices = AVSpeechSynthesisVoice.speechVoices().map { voice in
                ["name": voice.name, "lang": voice.language, "localService": true,
                 "identifier": voice.identifier] as [String: Any]
            }
            post(["type": "voices", "voices": voices])
        case "stop":
            activeID = nil
            guard synthesizer.isSpeaking || synthesizer.isPaused else { return }
            synthesizer.stopSpeaking(at: .immediate)
        case "speak":
            speak(body)
        default:
            break
        }
    }

    private func speak(_ body: [String: Any]) {
        guard let id = body["id"] as? String,
              let text = body["text"] as? String,
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let language = body["lang"] as? String else {
            post(["type": "error", "id": body["id"] as? String ?? "", "code": "empty-text"])
            return
        }
        if synthesizer.isSpeaking || synthesizer.isPaused {
            activeID = nil
            synthesizer.stopSpeaking(at: .immediate)
        }

        let voices = AVSpeechSynthesisVoice.speechVoices()
        let requestedVoice = body["voice"] as? String
        let requestedLanguage = body["voiceLang"] as? String
        let voice: AVSpeechSynthesisVoice?
        if language == "zh-HK" {
            // Never substitute Mandarin (zh-CN/zh-TW) for Cantonese.
            voice = voices.first {
                ["zh-HK", "yue-HK"].contains($0.language)
                    && $0.name == requestedVoice
                    && $0.language == requestedLanguage
            }
        } else {
            voice = voices.first {
                $0.name == requestedVoice && $0.language == requestedLanguage
                    && $0.language.lowercased().hasPrefix(language.split(separator: "-").first?.lowercased() ?? "en")
            }
        }
        guard let voice else {
            post(["type": "error", "id": id,
                  "code": language == "zh-HK" ? "no-local-voice" : "voice-unavailable"])
            return
        }

        do {
            // The playback category permits speech output even when the ring
            // switch is silent. The voice itself is an installed on-device voice.
            let audio = AVAudioSession.sharedInstance()
            try audio.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try audio.setActive(true)
        } catch {
            post(["type": "error", "id": id, "code": "audio-hardware"])
            return
        }

        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        let requestedRate = body["rate"] as? Double ?? 0.95
        utterance.rate = Float(min(max(requestedRate * 0.5, 0.42), 0.58))
        utterance.volume = 1
        activeID = id
        utteranceIDs[ObjectIdentifier(utterance)] = id
        synthesizer.speak(utterance)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didStart utterance: AVSpeechUtterance) {
        guard let id = utteranceIDs[ObjectIdentifier(utterance)], activeID == id else { return }
        let audio = AVAudioSession.sharedInstance()
        let output = audio.currentRoute.outputs.first
        post(["type": "started", "id": id,
              "outputKind": output?.portType.rawValue ?? "none",
              "outputName": output?.portName ?? "",
              "outputVolume": audio.outputVolume,
              "voiceName": utterance.voice?.name ?? ""])
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didFinish utterance: AVSpeechUtterance) {
        guard let id = utteranceIDs.removeValue(forKey: ObjectIdentifier(utterance)), activeID == id else { return }
        activeID = nil
        post(["type": "finished", "id": id])
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didCancel utterance: AVSpeechUtterance) {
        guard let id = utteranceIDs.removeValue(forKey: ObjectIdentifier(utterance)), activeID == id else { return }
        activeID = nil
        post(["type": "stopped", "id": id])
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func post(_ payload: [String: Any]) {
        guard let webView,
              let data = try? JSONSerialization.data(withJSONObject: payload, options: [.fragmentsAllowed]),
              let json = String(data: data, encoding: .utf8) else { return }
        DispatchQueue.main.async { [weak webView] in
            webView?.evaluateJavaScript("window.TTS && window.TTS.nativeEvent(\(json))")
        }
    }
}
