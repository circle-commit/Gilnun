import AVFoundation

final class SpeechManager: NSObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    private var finishHandlers: [ObjectIdentifier: () -> Void] = [:]
    /// Urgency of what is being spoken now; `nil` when idle.
    private var currentUrgency: RiskLevel?
    /// Newest live guidance waiting for the current sentence to finish.
    private var pendingGuidance: (text: String, urgency: RiskLevel, createdAt: Date)?
    /// Waiting guidance older than this is dropped; the scene has likely changed.
    private let pendingGuidanceLifetime: TimeInterval = 2.0

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    /// Speaks right away, cutting off anything in progress (mode announcements, OCR results).
    func speak(_ text: String, onFinish: (() -> Void)? = nil) {
        guard !text.isEmpty else { return }

        DispatchQueue.main.async {
            self.pendingGuidance = nil
            // Only a critical live warning may cut this off.
            self.startSpeaking(text, urgency: .high, onFinish: onFinish)
        }
    }

    /// Speaks live guidance without cutting off a sentence of equal or higher urgency.
    /// Less urgent guidance waits for the current sentence; only the newest one is kept.
    func speakGuidance(_ text: String, urgency: RiskLevel) {
        guard !text.isEmpty else { return }

        DispatchQueue.main.async {
            if self.synthesizer.isSpeaking, let currentUrgency = self.currentUrgency, urgency <= currentUrgency {
                self.pendingGuidance = (text, urgency, Date())
                return
            }

            self.pendingGuidance = nil
            self.startSpeaking(text, urgency: urgency, onFinish: nil)
        }
    }

    /// Drops live guidance that is still waiting to be spoken (e.g. after a mode switch).
    func clearPendingGuidance() {
        DispatchQueue.main.async {
            self.pendingGuidance = nil
        }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        DispatchQueue.main.async {
            self.utteranceEnded(utterance)
            self.speakPendingGuidance()
        }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        DispatchQueue.main.async {
            self.utteranceEnded(utterance)
        }
    }

    private func startSpeaking(_ text: String, urgency: RiskLevel, onFinish: (() -> Void)?) {
        synthesizer.stopSpeaking(at: .immediate)

        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.48
        utterance.voice = AVSpeechSynthesisVoice(language: voiceLanguage(for: text))
        if let onFinish {
            finishHandlers[ObjectIdentifier(utterance)] = onFinish
        }
        currentUrgency = urgency
        synthesizer.speak(utterance)
    }

    private func utteranceEnded(_ utterance: AVSpeechUtterance) {
        if !synthesizer.isSpeaking {
            currentUrgency = nil
        }
        finishHandlers.removeValue(forKey: ObjectIdentifier(utterance))?()
    }

    private func speakPendingGuidance() {
        guard !synthesizer.isSpeaking, let pending = pendingGuidance else { return }

        pendingGuidance = nil
        guard Date().timeIntervalSince(pending.createdAt) <= pendingGuidanceLifetime else { return }
        startSpeaking(pending.text, urgency: pending.urgency, onFinish: nil)
    }

    private func voiceLanguage(for text: String) -> String {
        text.unicodeScalars.contains { scalar in
            (0xAC00...0xD7AF).contains(Int(scalar.value))
        } ? "ko-KR" : "en-US"
    }
}
