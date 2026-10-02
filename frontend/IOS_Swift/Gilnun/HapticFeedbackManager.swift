import UIKit

enum HapticEvent {
    case readableTextConfirmed
    case ocrFailed
}

enum OCRHapticPulseState {
    case searching
    case detected
    case stabilizing
}

struct OCRHapticConfiguration {
    let searchingInterval: TimeInterval
    let candidateInterval: TimeInterval
    let timeoutInterval: TimeInterval
    let timeoutIntervalAfterNoText: TimeInterval

    static let standard = OCRHapticConfiguration(
        searchingInterval: 0.9,
        candidateInterval: 0.45,
        timeoutInterval: 12.0,
        timeoutIntervalAfterNoText: 1.5
    )
}

final class HapticFeedbackManager: NSObject {
    private let configuration: OCRHapticConfiguration
    private var pulseTimer: Timer?
    private var pulseState: OCRHapticPulseState?
    private var pulseStartedAt: Date?
    private let searchingGenerator = UIImpactFeedbackGenerator(style: .heavy)
    private let candidateGenerator = UIImpactFeedbackGenerator(style: .heavy)
    private let confirmationGenerator = UIImpactFeedbackGenerator(style: .heavy)
    private let notificationGenerator = UINotificationFeedbackGenerator()

    init(configuration: OCRHapticConfiguration = .standard) {
        self.configuration = configuration
        super.init()
        prepareGenerators()
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(stopRepeatingPulses),
            name: UIApplication.didEnterBackgroundNotification,
            object: nil
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        pulseTimer?.invalidate()
    }

    func updateOCRPulseState(_ state: OCRHapticPulseState) {
        DispatchQueue.main.async {
            guard self.pulseState != state else { return }

            self.pulseState = state
            self.pulseStartedAt = Date()
            self.emitPulse(for: state)
            self.scheduleNextPulse(after: self.interval(for: state))
        }
    }

    @objc func stopRepeatingPulses() {
        guard Thread.isMainThread else {
            DispatchQueue.main.async {
                self.stopRepeatingPulses()
            }
            return
        }

        pulseTimer?.invalidate()
        pulseTimer = nil
        pulseState = nil
        pulseStartedAt = nil
    }

    func play(_ event: HapticEvent) {
        DispatchQueue.main.async {
            switch event {
            case .readableTextConfirmed:
                self.emitStrongDoublePulse()
                self.notificationGenerator.notificationOccurred(.success)
                self.prepareGenerators()
            case .ocrFailed:
                self.notificationGenerator.notificationOccurred(.error)
            }
        }
    }

    private func scheduleNextPulse(after interval: TimeInterval) {
        pulseTimer?.invalidate()
        let timer = Timer(timeInterval: interval, repeats: false) { [weak self] _ in
            self?.emitPulseAndReschedule()
        }
        RunLoop.main.add(timer, forMode: .common)
        pulseTimer = timer
    }

    private func emitPulseAndReschedule() {
        guard let pulseState else { return }

        emitPulse(for: pulseState)
        scheduleNextPulse(after: interval(for: pulseState))
    }

    private func emitPulse(for state: OCRHapticPulseState) {
        switch state {
        case .searching:
            searchingGenerator.impactOccurred(intensity: 1.0)
            searchingGenerator.prepare()
        case .detected:
            emitStrongDoublePulse()
            notificationGenerator.notificationOccurred(.warning)
            prepareGenerators()
        case .stabilizing:
            emitStrongDoublePulse()
            prepareGenerators()
        }
    }

    private func prepareGenerators() {
        searchingGenerator.prepare()
        candidateGenerator.prepare()
        confirmationGenerator.prepare()
        notificationGenerator.prepare()
    }

    private func emitStrongDoublePulse() {
        candidateGenerator.impactOccurred(intensity: 1.0)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) { [weak self] in
            self?.confirmationGenerator.impactOccurred(intensity: 1.0)
            self?.confirmationGenerator.prepare()
        }
    }

    private func interval(for state: OCRHapticPulseState) -> TimeInterval {
        if let pulseStartedAt,
           Date().timeIntervalSince(pulseStartedAt) >= configuration.timeoutInterval {
            return configuration.timeoutIntervalAfterNoText
        }

        switch state {
        case .searching:
            return configuration.searchingInterval
        case .detected, .stabilizing:
            return configuration.candidateInterval
        }
    }
}
