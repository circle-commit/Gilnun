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
    /// Chosen voice per language (identifiers), cleared when the user downloads new voices.
    private var voiceIdentifiers: [String: String] = [:]

    override init() {
        super.init()
        synthesizer.delegate = self
        configureAudioSession()
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(availableVoicesChanged),
            name: AVSpeechSynthesizer.availableVoicesDidChangeNotification,
            object: nil
        )
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
            self.deactivateAudioSessionIfIdle()
        }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        DispatchQueue.main.async {
            self.utteranceEnded(utterance)
            self.deactivateAudioSessionIfIdle()
        }
    }

    /// Behaves like a navigation app: speaks even when the ring/silent switch is set to
    /// silent, lowers music while speaking and pauses podcasts or audiobooks.
    private func configureAudioSession() {
        do {
            try AVAudioSession.sharedInstance().setCategory(
                .playback,
                mode: .voicePrompt,
                options: [.duckOthers, .interruptSpokenAudioAndMixWithOthers]
            )
        } catch {
            print("Audio session setup failed: \(error)")
        }
    }

    /// Lets other apps' audio return to normal volume between prompts.
    private func deactivateAudioSessionIfIdle() {
        guard !synthesizer.isSpeaking else { return }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func startSpeaking(_ text: String, urgency: RiskLevel, onFinish: (() -> Void)?) {
        synthesizer.stopSpeaking(at: .immediate)
        try? AVAudioSession.sharedInstance().setActive(true)

        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = 0.48
        utterance.voice = bestVoice(for: voiceLanguage(for: text))
        // With VoiceOver on, speak in the voice, speed and pitch the user chose there.
        utterance.prefersAssistiveTechnologySettings = true
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

    /// Prefers premium, then enhanced voices downloaded in Settings > Accessibility >
    /// Spoken Content > Voices; the built-in compact voice sounds noticeably robotic.
    private func bestVoice(for language: String) -> AVSpeechSynthesisVoice? {
        if let identifier = voiceIdentifiers[language], let voice = AVSpeechSynthesisVoice(identifier: identifier) {
            return voice
        }

        let voice = AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language == language && $0.quality != .default }
            .max { $0.quality.rawValue < $1.quality.rawValue }
            ?? AVSpeechSynthesisVoice(language: language)
        voiceIdentifiers[language] = voice?.identifier
        return voice
    }

    @objc private func availableVoicesChanged() {
        DispatchQueue.main.async {
            self.voiceIdentifiers.removeAll()
        }
    }

    private func voiceLanguage(for text: String) -> String {
        text.unicodeScalars.contains { scalar in
            (0xAC00...0xD7AF).contains(Int(scalar.value))
        } ? "ko-KR" : "en-US"
    }
}
