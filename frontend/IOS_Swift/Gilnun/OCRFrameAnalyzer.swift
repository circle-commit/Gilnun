import AVFoundation
import CoreImage
import Vision

struct OCRFrameAnalysis {
    let textRegion: CGRect?
    let confidence: Float
    let blurScore: Double
    let movementScore: Double
    /// Mean luma (0-255) of the whole frame.
    let brightness: Double
    /// Share of the text region (0-1) that is blown out to white, as glare on glossy paper.
    let glareRatio: Double
    let timestamp: Date
    /// Lines confident enough to read aloud, in reading order.
    let readableLines: [String]
    /// The analyzed frame, kept so `readDetail` can read it again more closely.
    var pixelBuffer: CVPixelBuffer? = nil

    var hasText: Bool {
        textRegion != nil
    }

    var readableText: String {
        readableLines.joined(separator: " ")
    }
}

/// A close reading of the text region, made safe to speak.
struct DetailRead {
    /// Lines in reading order.
    var lines: [String]
    /// Lines Vision saw but could not read.
    var missedLines: Int
    /// Lines with digits left out because they were not certain.
    var droppedNumberLines: Int

    /// A misread digit can change a dose or a phone number, so a line with digits is kept only
    /// when `previous`, a read of the next or last frame, has the very same line.
    func confirmed(by previous: DetailRead) -> DetailRead {
        let earlier = Set(previous.lines.map(Self.normalized))
        let kept = lines.filter { !Self.hasDigit($0) || earlier.contains(Self.normalized($0)) }
        return DetailRead(lines: kept, missedLines: missedLines, droppedNumberLines: droppedNumberLines + lines.count - kept.count)
    }

    static func hasDigit(_ text: String) -> Bool {
        text.contains { $0.isNumber }
    }

    private static func normalized(_ text: String) -> String {
        text.filter { !$0.isWhitespace }
    }
}

final class OCRFrameAnalyzer {
    private let queue = DispatchQueue(label: "gilnun.ocr.frame-analyzer", qos: .userInitiated)
    private let request = VNRecognizeTextRequest()
    /// Reads only around the text, so small print gets more of the recognizer's resolution.
    private let detailRequest = VNRecognizeTextRequest()
    private var previousLumaSample: [Double]?
    private var isAnalyzing = false
    private var lastAnalysisDate: Date = .distantPast
    private let minimumFrameInterval: TimeInterval = 0.16
    /// Same cutoff the backend used for PaddleOCR (`OCR_MIN_CONFIDENCE`).
    private let minimumReadableConfidence: Float = 0.5
    /// Lines between this and `minimumReadableConfidence` are text Vision saw but could not read;
    /// lower scores are mostly texture, not text.
    private let minimumMissedConfidence: Float = 0.3
    /// Lines with digits need Vision's top score (it reports 0.5 or 1.0 for most lines).
    private let minimumNumberConfidence: Float = 0.8

    init() {
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.recognitionLanguages = preferredRecognitionLanguages(for: request)

        detailRequest.recognitionLevel = .accurate
        detailRequest.usesLanguageCorrection = true
        detailRequest.recognitionLanguages = request.recognitionLanguages
    }

    /// Reads the text region of `analysis` again at full detail. Call it from the completion of
    /// `analyze`, which runs on the analyzer's queue. Nil when the frame or region is missing.
    func readDetail(_ analysis: OCRFrameAnalysis) -> DetailRead? {
        guard let pixelBuffer = analysis.pixelBuffer, let region = analysis.textRegion else { return nil }

        detailRequest.regionOfInterest = region.insetBy(dx: -0.05, dy: -0.05).intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        guard (try? handler.perform([detailRequest])) != nil else { return nil }

        let results = detailRequest.results ?? []
        let candidates = results.compactMap { $0.topCandidates(1).first }
        let missed = candidates.filter { $0.confidence >= minimumMissedConfidence && $0.confidence < minimumReadableConfidence }
        let uncertainNumbers = candidates.filter {
            $0.confidence >= minimumReadableConfidence && $0.confidence < minimumNumberConfidence && DetailRead.hasDigit($0.string)
        }
        return DetailRead(
            lines: readableLines(from: results, minimumNumberConfidence: minimumNumberConfidence),
            missedLines: missed.count,
            droppedNumberLines: uncertainNumbers.count
        )
    }

    func analyze(sampleBuffer: CMSampleBuffer, completion: @escaping (OCRFrameAnalysis) -> Void) {
        let now = Date()
        guard now.timeIntervalSince(lastAnalysisDate) >= minimumFrameInterval else { return }
        guard !isAnalyzing else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        lastAnalysisDate = now
        isAnalyzing = true

        queue.async { [weak self] in
            guard let self else { return }

            let sample = self.lumaSample(from: pixelBuffer)
            let blurScore = self.blurScore(from: sample)
            let movementScore = self.movementScore(current: sample)
            self.previousLumaSample = sample

            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
            try? handler.perform([self.request])

            let results = self.request.results ?? []
            let observations = results.filter { $0.confidence >= 0.45 }

            let region = self.mergedBoundingBox(for: observations)
            let confidence = observations.map(\.confidence).max() ?? 0
            let analysis = OCRFrameAnalysis(
                textRegion: region,
                confidence: confidence,
                blurScore: blurScore,
                movementScore: movementScore,
                brightness: sample.isEmpty ? 0 : sample.reduce(0, +) / Double(sample.count),
                glareRatio: region.map { self.glareRatio(in: $0, sample: sample) } ?? 0,
                timestamp: now,
                readableLines: self.readableLines(from: results),
                pixelBuffer: pixelBuffer
            )

            completion(analysis)
            // After the completion, so frames wait while it reads a frame in detail.
            self.isAnalyzing = false
        }
    }

    private func preferredRecognitionLanguages(for request: VNRecognizeTextRequest) -> [String] {
        let preferred = ["ko-KR", "en-US"]

        guard let supported = try? VNRecognizeTextRequest.supportedRecognitionLanguages(
            for: request.recognitionLevel,
            revision: request.revision
        ) else {
            return preferred
        }

        let available = preferred.filter { supported.contains($0) }
        return available.isEmpty ? preferred : available
    }

    /// Orders confident lines top-to-bottom, then left-to-right within a row.
    private func readableLines(from observations: [VNRecognizedTextObservation], minimumNumberConfidence: Float = 0) -> [String] {
        let lines = observations.compactMap { observation -> (text: String, box: CGRect)? in
            guard let candidate = observation.topCandidates(1).first,
                  candidate.confidence >= minimumReadableConfidence,
                  candidate.confidence >= minimumNumberConfidence || !DetailRead.hasDigit(candidate.string) else { return nil }

            let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? nil : (text, observation.boundingBox)
        }

        // Vision's normalized coordinates have their origin at the bottom-left.
        var rows: [[(text: String, box: CGRect)]] = []
        for line in lines.sorted(by: { $0.box.midY > $1.box.midY }) {
            if let rowLine = rows.last?.first, abs(rowLine.box.midY - line.box.midY) < min(rowLine.box.height, line.box.height) / 2 {
                rows[rows.count - 1].append(line)
            } else {
                rows.append([line])
            }
        }

        return rows.flatMap { row in
            row.sorted { $0.box.minX < $1.box.minX }.map(\.text)
        }
    }

    private func mergedBoundingBox(for observations: [VNRecognizedTextObservation]) -> CGRect? {
        guard let first = observations.first else { return nil }

        return observations.dropFirst().reduce(first.boundingBox) { partial, observation in
            partial.union(observation.boundingBox)
        }
    }

    private func lumaSample(from pixelBuffer: CVPixelBuffer) -> [Double] {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let width: Int
        let height: Int
        let bytesPerRow: Int
        let baseAddress: UnsafeMutableRawPointer?

        if CVPixelBufferIsPlanar(pixelBuffer) {
            width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
            height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
            bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
            baseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
        } else {
            width = CVPixelBufferGetWidth(pixelBuffer)
            height = CVPixelBufferGetHeight(pixelBuffer)
            bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        }

        guard let baseAddress else { return [] }

        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        let columns = 24
        let rows = 32

        var sample: [Double] = []
        sample.reserveCapacity(columns * rows)

        for row in 0..<rows {
            let y = min(height - 1, max(0, row * height / rows))
            for column in 0..<columns {
                let x = min(width - 1, max(0, column * width / columns))
                sample.append(Double(buffer[y * bytesPerRow + x]))
            }
        }

        return sample
    }

    /// Share of the luma samples inside `region` (Vision coordinates of the upright image) that are saturated.
    private func glareRatio(in region: CGRect, sample: [Double]) -> Double {
        let columns = 24
        let rows = sample.count / columns
        // A little margin: glare can wash out the text next to the lines Vision still finds.
        let area = region.insetBy(dx: -0.05, dy: -0.05)
        var inside = 0
        var saturated = 0
        for (index, luma) in sample.enumerated() {
            // The buffer is the landscape sensor image; `.right` turns it upright, so a
            // sensor row becomes a column counted from the right, and a column a row from the top.
            let row = index / columns
            let column = index % columns
            let point = CGPoint(x: 1 - (Double(row) + 0.5) / Double(rows), y: 1 - (Double(column) + 0.5) / Double(columns))
            guard area.contains(point) else { continue }
            inside += 1
            if luma >= 250 { saturated += 1 }
        }
        return inside == 0 ? 0 : Double(saturated) / Double(inside)
    }

    private func blurScore(from sample: [Double]) -> Double {
        guard !sample.isEmpty else { return 0 }

        let columns = 24
        var edgeEnergy = 0.0
        var comparisons = 0.0

        for index in sample.indices {
            if index % columns != columns - 1 {
                edgeEnergy += abs(sample[index] - sample[index + 1])
                comparisons += 1
            }

            let lowerIndex = index + columns
            if lowerIndex < sample.count {
                edgeEnergy += abs(sample[index] - sample[lowerIndex])
                comparisons += 1
            }
        }

        return comparisons == 0 ? 0 : edgeEnergy / comparisons
    }

    private func movementScore(current: [Double]) -> Double {
        guard let previousLumaSample, previousLumaSample.count == current.count else { return 0 }

        let totalDifference = zip(previousLumaSample, current).reduce(0.0) { partial, pair in
            partial + abs(pair.0 - pair.1)
        }

        return totalDifference / Double(current.count)
    }
}
