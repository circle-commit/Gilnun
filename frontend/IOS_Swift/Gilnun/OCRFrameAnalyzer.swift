import AVFoundation
import CoreImage
import Vision

struct OCRFrameAnalysis {
    let textRegion: CGRect?
    let confidence: Float
    let blurScore: Double
    let movementScore: Double
    let timestamp: Date
    /// Lines confident enough to read aloud, in reading order.
    let readableLines: [String]

    var hasText: Bool {
        textRegion != nil
    }

    var readableText: String {
        readableLines.joined(separator: " ")
    }
}

final class OCRFrameAnalyzer {
    private let queue = DispatchQueue(label: "gilnun.ocr.frame-analyzer", qos: .userInitiated)
    private let request = VNRecognizeTextRequest()
    private var previousLumaSample: [Double]?
    private var isAnalyzing = false
    private var lastAnalysisDate: Date = .distantPast
    private let minimumFrameInterval: TimeInterval = 0.16
    /// Same cutoff the backend used for PaddleOCR (`OCR_MIN_CONFIDENCE`).
    private let minimumReadableConfidence: Float = 0.5

    init() {
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.recognitionLanguages = preferredRecognitionLanguages(for: request)
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
                timestamp: now,
                readableLines: self.readableLines(from: results)
            )

            self.isAnalyzing = false
            completion(analysis)
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
    private func readableLines(from observations: [VNRecognizedTextObservation]) -> [String] {
        let lines = observations.compactMap { observation -> (text: String, box: CGRect)? in
            guard let candidate = observation.topCandidates(1).first,
                  candidate.confidence >= minimumReadableConfidence else { return nil }

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
