//
//  CameraManager.swift
//  Gilnun
//
//  Created by JoMinHui on 4/10/26.
//

import AVFoundation
import Combine
import CoreImage

final class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    enum ProcessingMode: String {
        case liveAnalyzing = "live"
        case textDescription = "text"
    }

    @Published var session = AVCaptureSession()
    @Published var latestGuide = "실시간 안내 모드가 준비되었습니다."
    @Published var latestDetectedText: String?
    @Published var liveOCRStatus: LiveOCRStatus = .searching
    @Published var latestLiveDirection = "center"
    @Published var latestLiveRiskScore = 0
    @Published var liveBoxes: [LiveGuidanceBox] = []
    @Published var liveImageSize: CGSize = .zero

    /// Every detection is visualized (classic detector-style overlay). Set above 0 to
    /// hide low-risk objects again.
    private static let liveBoxMinRisk = 0
    /// Only the highest-risk objects are drawn, to keep the preview readable for low-vision
    /// users and avoid clutter over the live guidance UI.
    private static let liveBoxMaxCount = 2

    private let output = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "videoQueue")
    private let frameAnalyzer = OCRFrameAnalyzer()
    private let stabilityTracker = TextStabilityTracker()
    private let duplicateSuppressor = DuplicateTextSuppressor()
    private let speechManager = SpeechManager()
    private let hapticManager = HapticFeedbackManager()
    private var currentMode: ProcessingMode = .liveAnalyzing
    private var lastFullOCRRequestDate: Date = .distantPast
    private let fullOCRCooldown: TimeInterval = 3.0

    // Live mode runs entirely on the video queue; these are only touched there.
    private let sceneAnalyzer = SceneAnalyzer()
    private var objectDetector: ObjectDetector?
    private var objectDetectorFailed = false
    private var lastLiveAnalysisTime: TimeInterval = 0
    private var lastLiveGuideTime: TimeInterval = 0
    /// About five detector runs per second: fast enough to track approaching objects.
    private let liveAnalysisInterval: TimeInterval = 0.2
    /// How long a spoken live message stays on the guidance card.
    private let liveGuideDisplayDuration: TimeInterval = 3.0

    override init() {
        super.init()
        checkPermissions()
        setupSession()
    }

    func setMode(_ mode: ProcessingMode) {
        currentMode = mode
        stabilityTracker.reset()
        speechManager.clearPendingGuidance()
        videoQueue.async {
            self.sceneAnalyzer.reset()
            self.lastLiveGuideTime = ProcessInfo.processInfo.systemUptime
        }

        let message: String
        let shouldAnnounceMode: Bool
        switch mode {
        case .liveAnalyzing:
            latestDetectedText = nil
            liveOCRStatus = .searching
            latestLiveDirection = "center"
            latestLiveRiskScore = 0
            liveBoxes = []
            hapticManager.stopRepeatingPulses()
            message = "실시간 보행 안내를 시작합니다."
            shouldAnnounceMode = true
        case .textDescription:
            latestDetectedText = nil
            liveOCRStatus = .searching
            latestLiveDirection = "center"
            latestLiveRiskScore = 0
            liveBoxes = []
            hapticManager.updateOCRPulseState(.searching)
            message = "문자 읽기 모드입니다. 카메라를 가까운 문자에 맞춰주세요."
            shouldAnnounceMode = false
        }

        updateResponse(voiceGuide: message, detectedText: nil, shouldSpeak: shouldAnnounceMode)
    }

    private func checkPermissions() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { _ in }
        default:
            liveOCRStatus = .unavailable
            updateResponse(voiceGuide: "이 앱을 사용하려면 카메라 권한이 필요합니다.", detectedText: nil, shouldSpeak: false)
        }
    }

    private func setupSession() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return }

        session.beginConfiguration()
        if session.canAddInput(input) {
            session.addInput(input)
        }

        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        ]
        output.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        session.commitConfiguration()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.session.startRunning()
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        switch currentMode {
        case .liveAnalyzing:
            analyzeLiveFrame(sampleBuffer)
        case .textDescription:
            analyzeTextFrame(sampleBuffer)
        }
    }

    // MARK: - Live guidance (on-device YOLO)

    private func analyzeLiveFrame(_ sampleBuffer: CMSampleBuffer) {
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastLiveAnalysisTime >= liveAnalysisInterval else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastLiveAnalysisTime = now

        // Loaded on the first live frame (off the main thread); the model is bundled with the app.
        guard let detector = loadObjectDetectorIfNeeded() else { return }

        let frame = ObjectDetector.uprightFrame(from: pixelBuffer)
        let frameSize = frame.extent.size
        let detections: [RawDetection]
        do {
            detections = try detector.detect(in: frame)
        } catch {
            // Skip this frame; the next one retries.
            return
        }

        let result = sceneAnalyzer.analyze(detections, frameSize: frameSize, now: now)
        publishLiveResult(result, frameSize: frameSize, now: now)
    }

    private func loadObjectDetectorIfNeeded() -> ObjectDetector? {
        if let objectDetector {
            return objectDetector
        }
        guard !objectDetectorFailed else { return nil }

        do {
            let detector = try ObjectDetector()
            objectDetector = detector
            return detector
        } catch {
            objectDetectorFailed = true
            print("Object detector unavailable: \(error)")
            updateResponse(voiceGuide: "비전 모델을 사용할 수 없습니다.", detectedText: nil, shouldSpeak: true)
            return nil
        }
    }

    private func publishLiveResult(_ result: LiveSceneResult, frameSize: CGSize, now: TimeInterval) {
        let hasNewGuide = !result.voiceGuide.isEmpty
        if hasNewGuide {
            lastLiveGuideTime = now
        }
        let shouldClearGuide = !hasNewGuide && now - lastLiveGuideTime > liveGuideDisplayDuration
        let boxes = makeLiveBoxes(from: result.detections, imageSize: frameSize)
        let primaryDetection = result.detections.first

        DispatchQueue.main.async {
            guard self.currentMode == .liveAnalyzing else { return }

            if hasNewGuide {
                self.latestGuide = result.voiceGuide
            } else if shouldClearGuide && !self.latestGuide.isEmpty {
                self.latestGuide = ""
            }
            self.latestLiveDirection = primaryDetection?.position.rawValue ?? "center"
            self.latestLiveRiskScore = primaryDetection?.riskScore ?? 0
            self.liveImageSize = frameSize
            self.liveBoxes = boxes
        }

        if hasNewGuide && currentMode == .liveAnalyzing {
            speechManager.speakGuidance(result.voiceGuide, urgency: result.voiceUrgency)
        }
    }

    /// Builds bounding boxes for the highest-risk detections only.
    ///
    /// - Detections arrive sorted by guidance priority (risk score first).
    /// - Nothing is drawn unless at least one object reaches `liveBoxMinRisk`, so a
    ///   calm scene keeps the camera preview clean.
    /// - At most `liveBoxMaxCount` boxes are returned (the top objects), each carrying
    ///   its own risk score so the UI can color it independently.
    private func makeLiveBoxes(from detections: [SceneDetection], imageSize: CGSize) -> [LiveGuidanceBox] {
        guard imageSize.width > 0, imageSize.height > 0 else { return [] }

        return detections
            .filter { $0.riskScore >= Self.liveBoxMinRisk }
            .prefix(Self.liveBoxMaxCount)
            .map { detection in
                LiveGuidanceBox(
                    rect: normalizedRect(for: detection.bbox, imageSize: imageSize),
                    riskScore: detection.riskScore,
                    label: detection.koreanLabel
                )
            }
    }

    /// Converts a detector bounding box (upright frame pixels) into a normalized
    /// (0...1) rect in the portrait space the camera preview is rendered in.
    private func normalizedRect(for bbox: BoundingBox, imageSize: CGSize) -> CGRect {
        CGRect(
            x: bbox.x1 / imageSize.width,
            y: bbox.y1 / imageSize.height,
            width: (bbox.x2 - bbox.x1) / imageSize.width,
            height: (bbox.y2 - bbox.y1) / imageSize.height
        )
    }

    // MARK: - Text reading (on-device Vision OCR)

    private func analyzeTextFrame(_ sampleBuffer: CMSampleBuffer) {
        frameAnalyzer.analyze(sampleBuffer: sampleBuffer) { [weak self] analysis in
            guard let self else { return }

            let decision = self.stabilityTracker.update(with: analysis)
            self.updateLiveOCRStatus(decision.status)

            guard decision.shouldRunFullOCR else { return }
            guard self.currentMode == .textDescription else { return }
            guard Date().timeIntervalSince(self.lastFullOCRRequestDate) >= self.fullOCRCooldown else {
                self.updateLiveOCRStatus(.coolingDown)
                return
            }

            self.lastFullOCRRequestDate = Date()
            self.hapticManager.stopRepeatingPulses()
            self.readText(analysis.readableText)
        }
    }

    private func readText(_ detectedText: String) {
        let voiceGuide = detectedText.isEmpty
            ? "문자 읽기 모드입니다. 현재 화면에서 읽을 수 있는 문자를 찾지 못했습니다."
            : "문자 읽기 모드입니다. 인식된 문자는 다음과 같습니다. \(detectedText)"
        let shouldSpeak = !detectedText.isEmpty && duplicateSuppressor.shouldSpeak(detectedText)

        if shouldSpeak {
            hapticManager.stopRepeatingPulses()
            hapticManager.play(.readableTextConfirmed)
        }

        updateResponse(voiceGuide: voiceGuide, detectedText: detectedText, shouldSpeak: shouldSpeak)

        DispatchQueue.main.async {
            guard self.currentMode == .textDescription else { return }

            self.liveOCRStatus = .coolingDown
            if !shouldSpeak {
                self.hapticManager.updateOCRPulseState(.searching)
            }
        }
    }

    private func updateResponse(voiceGuide: String, detectedText: String?, shouldSpeak: Bool) {
        DispatchQueue.main.async {
            self.latestGuide = voiceGuide
            self.latestDetectedText = detectedText
        }

        guard shouldSpeak else { return }
        hapticManager.stopRepeatingPulses()
        speechManager.speak(voiceGuide) { [weak self] in
            guard let self else { return }
            guard self.currentMode == .textDescription else { return }
            self.hapticManager.updateOCRPulseState(.searching)
        }
    }

    private func updateLiveOCRStatus(_ status: LiveOCRStatus) {
        DispatchQueue.main.async {
            guard self.currentMode == .textDescription else { return }

            if self.liveOCRStatus != status {
                self.liveOCRStatus = status
            }

            if self.latestDetectedText == nil || self.latestDetectedText?.isEmpty == true {
                self.latestGuide = status.rawValue
            }

            switch status {
            case .searching, .coolingDown:
                self.hapticManager.updateOCRPulseState(.searching)
            case .detected:
                self.hapticManager.updateOCRPulseState(.detected)
            case .stabilizing:
                self.hapticManager.updateOCRPulseState(.stabilizing)
            case .reading, .unavailable:
                self.hapticManager.stopRepeatingPulses()
            }
        }
    }
}
