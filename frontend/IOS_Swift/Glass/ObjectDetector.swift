import CoreImage
import CoreML
import CoreVideo
import Foundation

/// Runs the bundled sidewalk YOLO model (`SidewalkDetector.mlpackage`) on camera frames.
///
/// Frames are letterboxed the way Ultralytics does it (scale to fit, centered,
/// gray padding) and boxes are mapped back with its `scale_boxes` math, so the
/// results match the PyTorch checkpoint. Not thread-safe: call from one queue.
nonisolated final class ObjectDetector {
    enum DetectorError: Error {
        case modelNotFound
        case unexpectedModel(String)
        case pixelBufferUnavailable
    }

    static let modelName = "SidewalkDetector"

    let confidenceThreshold = 0.35
    let iouThreshold = 0.5
    let maxDetections = 20

    private let model: MLModel
    private let labels: [String]
    private let inputWidth: Int
    private let inputHeight: Int
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    private let padColor: CIImage
    private var inputBuffer: CVPixelBuffer?

    convenience init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(forResource: Self.modelName, withExtension: "mlmodelc") else {
            throw DetectorError.modelNotFound
        }
        try self.init(compiledModelURL: url)
    }

    init(compiledModelURL: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: compiledModelURL, configuration: configuration)

        guard let constraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else {
            throw DetectorError.unexpectedModel("missing image input")
        }
        inputWidth = constraint.pixelsWide
        inputHeight = constraint.pixelsHigh

        labels = Self.classNames(of: model.modelDescription)
        guard !labels.isEmpty else {
            throw DetectorError.unexpectedModel("missing class names")
        }

        let gray = CIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255, alpha: 1, colorSpace: colorSpace)
            ?? CIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255)
        padColor = CIImage(color: gray)
    }

    /// Camera buffers arrive in the sensor's landscape orientation; `.right` turns them into
    /// the upright portrait frame the user sees, so boxes line up with the preview.
    static func uprightFrame(from pixelBuffer: CVPixelBuffer) -> CIImage {
        CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
    }

    /// Detects objects in an upright frame. Boxes are in the frame's pixel coordinates
    /// (origin top-left), sorted by confidence.
    func detect(in image: CIImage) throws -> [RawDetection] {
        let extent = image.extent
        let frameWidth = Double(extent.width)
        let frameHeight = Double(extent.height)
        guard frameWidth > 0, frameHeight > 0 else { return [] }

        let modelWidth = Double(inputWidth)
        let modelHeight = Double(inputHeight)
        let gain = min(modelWidth / frameWidth, modelHeight / frameHeight)
        let scaledWidth = (frameWidth * gain).rounded(.toNearestOrEven)
        let scaledHeight = (frameHeight * gain).rounded(.toNearestOrEven)
        let left = ((modelWidth - scaledWidth) / 2 - 0.1).rounded(.toNearestOrEven)
        let top = ((modelHeight - scaledHeight) / 2 - 0.1).rounded(.toNearestOrEven)

        // Core Image's origin is bottom-left, so the top padding becomes a y offset from below.
        let scaled = image
            .transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .clampedToExtent()
            .transformed(by: CGAffineTransform(scaleX: scaledWidth / frameWidth, y: scaledHeight / frameHeight))
            .cropped(to: CGRect(x: 0, y: 0, width: scaledWidth, height: scaledHeight))
            .transformed(by: CGAffineTransform(translationX: left, y: modelHeight - top - scaledHeight))
        let canvas = CGRect(x: 0, y: 0, width: modelWidth, height: modelHeight)
        let letterboxed = scaled.composited(over: padColor).cropped(to: canvas)

        let buffer = try modelInputBuffer()
        context.render(letterboxed, to: buffer, bounds: canvas, colorSpace: colorSpace)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: buffer),
            "iouThreshold": MLFeatureValue(double: iouThreshold),
            "confidenceThreshold": MLFeatureValue(double: confidenceThreshold),
        ])
        let output = try model.prediction(from: input)
        guard let confidences = output.featureValue(for: "confidence")?.multiArrayValue,
              let coordinates = output.featureValue(for: "coordinates")?.multiArrayValue,
              confidences.shape.count == 2,
              coordinates.shape.count == 2 else {
            throw DetectorError.unexpectedModel("missing confidence/coordinates outputs")
        }

        // Ultralytics `scale_boxes`: undo the padding, then the scale, then clip to the frame.
        let padX = ((modelWidth - frameWidth * gain) / 2 - 0.1).rounded(.toNearestOrEven)
        let padY = ((modelHeight - frameHeight * gain) / 2 - 0.1).rounded(.toNearestOrEven)
        let boxCount = confidences.shape[0].intValue
        let classCount = min(labels.count, confidences.shape[1].intValue)

        var detections: [RawDetection] = []
        for row in 0..<boxCount {
            var bestClass = 0
            var bestScore = -Double.infinity
            for column in 0..<classCount {
                let score = confidences[[row, column] as [NSNumber]].doubleValue
                if score > bestScore {
                    bestScore = score
                    bestClass = column
                }
            }
            guard bestScore >= confidenceThreshold else { continue }

            let centerX = coordinates[[row, 0] as [NSNumber]].doubleValue * modelWidth
            let centerY = coordinates[[row, 1] as [NSNumber]].doubleValue * modelHeight
            let width = coordinates[[row, 2] as [NSNumber]].doubleValue * modelWidth
            let height = coordinates[[row, 3] as [NSNumber]].doubleValue * modelHeight

            let bbox = BoundingBox(
                x1: clamp((centerX - width / 2 - padX) / gain, upperBound: frameWidth),
                y1: clamp((centerY - height / 2 - padY) / gain, upperBound: frameHeight),
                x2: clamp((centerX + width / 2 - padX) / gain, upperBound: frameWidth),
                y2: clamp((centerY + height / 2 - padY) / gain, upperBound: frameHeight)
            )
            detections.append(RawDetection(label: labels[bestClass], confidence: bestScore, bbox: bbox))
        }

        return Array(stableSorted(detections) { $0.confidence > $1.confidence }.prefix(maxDetections))
    }

    private func modelInputBuffer() throws -> CVPixelBuffer {
        if let inputBuffer {
            return inputBuffer
        }

        let attributes: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as [CFString: Any],
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferMetalCompatibilityKey: true,
        ]
        var buffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            inputWidth,
            inputHeight,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &buffer
        )
        guard status == kCVReturnSuccess, let buffer else {
            throw DetectorError.pixelBufferUnavailable
        }

        inputBuffer = buffer
        return buffer
    }

    private func clamp(_ value: Double, upperBound: Double) -> Double {
        max(0, min(upperBound, value))
    }

    /// Reads class names from the export metadata, e.g. "{0: 'person', 1: 'car', ...}".
    /// The NMS stage pads its label list to 80 entries, so it cannot be used directly.
    private static func classNames(of description: MLModelDescription) -> [String] {
        let metadata = description.metadata[.creatorDefinedKey] as? [String: String]
        guard let names = metadata?["names"],
              let pattern = try? NSRegularExpression(pattern: #"(\d+):\s*'([^']*)'"#) else {
            return []
        }

        let matches = pattern.matches(in: names, range: NSRange(names.startIndex..., in: names))
        let indexed = matches.compactMap { match -> (Int, String)? in
            guard let indexRange = Range(match.range(at: 1), in: names),
                  let nameRange = Range(match.range(at: 2), in: names),
                  let index = Int(names[indexRange]) else {
                return nil
            }
            return (index, String(names[nameRange]))
        }
        return indexed.sorted { $0.0 < $1.0 }.map(\.1)
    }
}
