import CoreImage
import CoreML
import CoreVideo
import Foundation

/// Where braille guide blocks (점자블록) are in an upright frame, as a coarse grid.
nonisolated struct BrailleMask: Equatable {
    /// About 24 px square cells on a 1080x1920 portrait frame.
    static let columns = 45
    static let rows = 80

    /// Row-major, origin top-left.
    var cells: [Bool]

    static let empty = BrailleMask(cells: Array(repeating: false, count: columns * rows))

    subscript(column: Int, row: Int) -> Bool {
        cells[row * Self.columns + column]
    }
}

/// Runs the bundled braille block segmentation model (`BrailleBlockModel.mlpackage`,
/// YOLO11n-seg trained on AI Hub 인도보행 surface polygons) on camera frames.
///
/// Exported with `YOLO(best.pt).export(format="coreml", imgsz=[640, 384], half=True)`, so the
/// model returns raw predictions (1x37xN: box, score, 32 mask coefficients) and mask
/// prototypes (1x32xHxW). Overlapping candidates are merged into one mask anyway, so no NMS
/// is needed. Not thread-safe: call from one queue.
nonisolated final class BrailleBlockSegmenter {
    static let modelName = "BrailleBlockModel"

    /// Same threshold the validation numbers were measured at.
    let confidenceThreshold: Float = 0.25
    /// One block yields many overlapping candidates; fewer than this missed parts of blocks.
    let maxCandidates = 50

    private let model: MLModel
    private let inputWidth: Int
    private let inputHeight: Int
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    private let padColor = CIImage(color: CIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255))
    private var inputBuffer: CVPixelBuffer?

    convenience init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(forResource: Self.modelName, withExtension: "mlmodelc") else {
            throw ObjectDetector.DetectorError.modelNotFound
        }
        try self.init(compiledModelURL: url)
    }

    init(compiledModelURL: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: compiledModelURL, configuration: configuration)
        guard let constraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else {
            throw ObjectDetector.DetectorError.unexpectedModel("missing image input")
        }
        inputWidth = constraint.pixelsWide
        inputHeight = constraint.pixelsHigh
    }

    func segment(in image: CIImage) throws -> BrailleMask {
        let frame = image.extent.size
        guard frame.width > 0, frame.height > 0 else { return .empty }

        let letterbox = Letterbox(frame: frame, modelWidth: inputWidth, modelHeight: inputHeight)
        if inputBuffer == nil {
            inputBuffer = try Letterbox.makeInputBuffer(width: inputWidth, height: inputHeight)
        }
        guard let buffer = inputBuffer else { return .empty }
        letterbox.render(image, into: buffer, padColor: padColor, context: context, colorSpace: colorSpace)

        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: buffer)]))
        // The exported output names are generated (var_1323, ...), so tell them apart by rank.
        let arrays = output.featureNames.compactMap { output.featureValue(for: $0)?.multiArrayValue }
        guard let predictions = arrays.first(where: { $0.shape.count == 3 }),
              let prototypes = arrays.first(where: { $0.shape.count == 4 }),
              predictions.dataType == .float32, prototypes.dataType == .float32 else {
            throw ObjectDetector.DetectorError.unexpectedModel("missing prediction/prototype outputs")
        }

        let maskCount = prototypes.shape[1].intValue
        let protoHeight = prototypes.shape[2].intValue
        let protoWidth = prototypes.shape[3].intValue
        guard predictions.shape[1].intValue == 5 + maskCount else {
            throw ObjectDetector.DetectorError.unexpectedModel("expected one class and \(maskCount) mask coefficients")
        }

        let candidates = Self.candidates(in: predictions, maskCount: maskCount, threshold: confidenceThreshold, limit: maxCandidates)
        guard !candidates.isEmpty else { return .empty }

        let protoStride = prototypes.strides.map(\.intValue)
        let protoScaleX = letterbox.modelWidth / Double(protoWidth)
        let protoScaleY = letterbox.modelHeight / Double(protoHeight)
        var mask = BrailleMask.empty

        prototypes.withUnsafeBufferPointer(ofType: Float.self) { proto in
            let logit = { (candidate: Candidate, protoX: Int, protoY: Int) -> Float in
                let base = protoY * protoStride[2] + protoX * protoStride[3]
                var sum: Float = 0
                for k in 0..<maskCount {
                    sum += candidate.coefficients[k] * proto[base + k * protoStride[1]]
                }
                return sum
            }
            for row in 0..<BrailleMask.rows {
                // Cell center in model input pixels.
                let y = (Double(row) + 0.5) / Double(BrailleMask.rows) * Double(frame.height) * letterbox.gain + letterbox.padY
                // Bilinear upsampling as in `F.interpolate(align_corners=False)`.
                let sourceY = min(max(y / protoScaleY - 0.5, 0), Double(protoHeight - 1))
                let y0 = Int(sourceY), y1 = min(y0 + 1, protoHeight - 1)
                let weightY = Float(sourceY - Double(y0))
                for column in 0..<BrailleMask.columns {
                    let x = (Double(column) + 0.5) / Double(BrailleMask.columns) * Double(frame.width) * letterbox.gain + letterbox.padX
                    let sourceX = min(max(x / protoScaleX - 0.5, 0), Double(protoWidth - 1))
                    let x0 = Int(sourceX), x1 = min(x0 + 1, protoWidth - 1)
                    let weightX = Float(sourceX - Double(x0))

                    // Ultralytics `process_mask`: inside the box and a positive mask logit.
                    mask.cells[row * BrailleMask.columns + column] = candidates.contains { candidate in
                        guard candidate.box.contains(CGPoint(x: x, y: y)) else { return false }
                        let top = (1 - weightX) * logit(candidate, x0, y0) + weightX * logit(candidate, x1, y0)
                        let bottom = (1 - weightX) * logit(candidate, x0, y1) + weightX * logit(candidate, x1, y1)
                        return (1 - weightY) * top + weightY * bottom > 0
                    }
                }
            }
        }
        return mask
    }

    private struct Candidate {
        let score: Float
        /// Model input pixels.
        let box: CGRect
        let coefficients: [Float]
    }

    private static func candidates(in predictions: MLMultiArray, maskCount: Int, threshold: Float, limit: Int) -> [Candidate] {
        let stride = predictions.strides.map(\.intValue)
        let anchors = predictions.shape[2].intValue
        return predictions.withUnsafeBufferPointer(ofType: Float.self) { values in
            let value = { (channel: Int, anchor: Int) in values[channel * stride[1] + anchor * stride[2]] }
            let strongest = (0..<anchors)
                .filter { value(4, $0) >= threshold }
                .sorted { value(4, $0) > value(4, $1) }
                .prefix(limit)
            return strongest.map { anchor in
                let width = Double(value(2, anchor))
                let height = Double(value(3, anchor))
                return Candidate(
                    score: value(4, anchor),
                    box: CGRect(x: Double(value(0, anchor)) - width / 2, y: Double(value(1, anchor)) - height / 2, width: width, height: height),
                    coefficients: (0..<maskCount).map { value(5 + $0, anchor) }
                )
            }
        }
    }
}

/// Decides when to say where the braille blocks are. Speaks when blocks show up in the
/// walking area (the lower part of the frame) for a few frames in a row, and again when
/// they move to another side; stays quiet while they stay put.
nonisolated final class BrailleBlockTracker {
    /// Rows below this share of the frame height are the ground ahead of the user.
    static let walkingAreaTop = 0.4
    /// Blocks must cover this share of the whole frame inside the walking area.
    static let minimumShare = 0.01
    /// A single frame wrongly shows blocks about 7% of the time on validation crops.
    static let framesToConfirm = 3
    /// About two seconds at five analyses a second before blocks count as gone.
    static let framesToLose = 10
    static let sideChangeInterval: TimeInterval = 4

    private var candidate: HorizontalPosition?
    private var candidateFrames = 0
    private var missedFrames = 0
    private var announced: HorizontalPosition?
    private var lastAnnouncement: TimeInterval = -.infinity

    func reset() {
        candidate = nil
        candidateFrames = 0
        missedFrames = 0
        announced = nil
        lastAnnouncement = -.infinity
    }

    /// - Parameter canSpeak: false while more urgent guidance is being spoken; the
    ///   announcement then waits for a later frame.
    /// - Returns: The sentence to speak, or nil.
    func update(_ mask: BrailleMask, now: TimeInterval, canSpeak: Bool) -> String? {
        guard let position = Self.position(of: mask) else {
            missedFrames += 1
            candidate = nil
            candidateFrames = 0
            if missedFrames >= Self.framesToLose {
                announced = nil
            }
            return nil
        }

        missedFrames = 0
        if position == candidate {
            candidateFrames += 1
        } else {
            candidate = position
            candidateFrames = 1
        }

        guard candidateFrames >= Self.framesToConfirm, position != announced, canSpeak else { return nil }
        if announced != nil, now - lastAnnouncement < Self.sideChangeInterval {
            return nil
        }
        announced = position
        lastAnnouncement = now
        return Self.message(for: position)
    }

    /// Side of the blocks in the walking area, or nil when there are too few there.
    static func position(of mask: BrailleMask) -> HorizontalPosition? {
        let firstRow = Int((Double(BrailleMask.rows) * walkingAreaTop).rounded(.up))
        var count = 0
        var columnSum = 0
        for row in firstRow..<BrailleMask.rows {
            for column in 0..<BrailleMask.columns where mask[column, row] {
                count += 1
                columnSum += column
            }
        }
        guard Double(count) >= minimumShare * Double(BrailleMask.columns * BrailleMask.rows) else { return nil }
        let centerX = Double(columnSum) / Double(count) + 0.5
        return HorizontalPosition(centerX: centerX, frameWidth: Double(BrailleMask.columns))
    }

    static func message(for position: HorizontalPosition) -> String {
        switch position {
        case .left: return "왼쪽에 점자블록이 있어요."
        case .center: return "앞에 점자블록이 있어요."
        case .right: return "오른쪽에 점자블록이 있어요."
        }
    }
}
