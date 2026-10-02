import Foundation

final class DuplicateTextSuppressor {
    private var lastNormalizedText = ""
    private var lastSpokenDate: Date = .distantPast
    private let cooldown: TimeInterval = 8.0

    func shouldSpeak(_ text: String, now: Date = Date()) -> Bool {
        let normalized = normalize(text)
        guard !normalized.isEmpty else { return false }

        if isDuplicate(normalized, lastNormalizedText) {
            return false
        }

        if now.timeIntervalSince(lastSpokenDate) < cooldown {
            let previousWords = Set(lastNormalizedText.split(separator: " "))
            let newWords = Set(normalized.split(separator: " "))
            let overlap = previousWords.intersection(newWords).count
            if overlap > 0 {
                return false
            }
        }

        lastNormalizedText = normalized
        lastSpokenDate = now
        return true
    }

    private func normalize(_ text: String) -> String {
        text
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func isDuplicate(_ first: String, _ second: String) -> Bool {
        guard !first.isEmpty, !second.isEmpty else { return false }
        if first == second { return true }
        if first.contains(second) || second.contains(first) { return true }

        let firstWords = Set(first.split(separator: " "))
        let secondWords = Set(second.split(separator: " "))
        guard !firstWords.isEmpty, !secondWords.isEmpty else { return false }

        let overlap = firstWords.intersection(secondWords).count
        let union = firstWords.union(secondWords).count
        return Double(overlap) / Double(union) >= 0.84
    }
}
