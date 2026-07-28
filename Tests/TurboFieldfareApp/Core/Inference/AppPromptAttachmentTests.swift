import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppPromptAttachmentTests {
    @Test func noAttachmentsLeaveTheUserPromptByteForByteUnchanged() {
        let prompt = "  Keep leading and trailing whitespace.  \n"

        #expect(AppPromptContext.compose(
            userPrompt: prompt,
            attachments: [],
            maximumAttachmentCharacters: 1_000) == prompt)
    }

    @Test func compositionKeepsUserRequestAfterDocumentContext() {
        let prompt = AppPromptContext.compose(
            userPrompt: "Summarize the risks.",
            attachments: [
                AppPromptAttachment(
                    fileName: "report.pdf",
                    formatLabel: "PDF",
                    extractedText: "Revenue fell by 10 percent."),
            ],
            maximumAttachmentCharacters: 1_000)

        #expect(prompt.contains("[Attached document 1: report.pdf (PDF)]"))
        #expect(prompt.contains("Revenue fell by 10 percent."))
        #expect(prompt.hasSuffix("User request:\nSummarize the risks."))
    }

    @Test func contextBudgetIsSharedAcrossDocumentsAndMarksTruncation() {
        let prompt = AppPromptContext.compose(
            userPrompt: "Compare.",
            attachments: [
                AppPromptAttachment(
                    fileName: "one.docx",
                    formatLabel: "Word",
                    extractedText: "1234567890"),
                AppPromptAttachment(
                    fileName: "two.pptx",
                    formatLabel: "PowerPoint",
                    extractedText: "abcdefghij"),
            ],
            maximumAttachmentCharacters: 10)

        #expect(prompt.contains("12345"))
        #expect(prompt.contains("abcde"))
        #expect(prompt.components(separatedBy: "was truncated").count - 1 == 2)
    }

    @Test func unusedBudgetFromAShortDocumentFlowsToLaterDocuments() {
        let prompt = AppPromptContext.compose(
            userPrompt: "Compare.",
            attachments: [
                AppPromptAttachment(
                    fileName: "short.pdf",
                    formatLabel: "PDF",
                    extractedText: "abc"),
                AppPromptAttachment(
                    fileName: "long.docx",
                    formatLabel: "Word",
                    extractedText: "0123456789"),
            ],
            maximumAttachmentCharacters: 10)

        #expect(prompt.contains("\nabc\n[End attached document 1]"))
        #expect(prompt.contains("\n0123456\n"))
        #expect(!prompt.contains("01234567"))
        #expect(prompt.components(separatedBy: "was truncated").count - 1 == 1)
    }

    @Test func extractionTruncationIsReportedEvenWhenContextHasRoom() {
        let prompt = AppPromptContext.compose(
            userPrompt: "Read it.",
            attachments: [
                AppPromptAttachment(
                    fileName: "large.xlsx",
                    formatLabel: "Excel",
                    extractedText: "all extracted text",
                    wasTruncatedDuringExtraction: true),
            ],
            maximumAttachmentCharacters: 10_000)

        #expect(prompt.contains("all extracted text"))
        #expect(prompt.components(separatedBy: "was truncated").count - 1 == 1)
    }

    @Test func zeroAndNegativeBudgetsExposeMetadataButNoDocumentText() {
        for budget in [0, -100] {
            let prompt = AppPromptContext.compose(
                userPrompt: "Question",
                attachments: [
                    AppPromptAttachment(
                        fileName: "secret.pptx",
                        formatLabel: "PowerPoint",
                        extractedText: "must not fit"),
                ],
                maximumAttachmentCharacters: budget)

            #expect(prompt.contains(
                "[Attached document 1: secret.pptx (PowerPoint)]"))
            #expect(!prompt.contains("must not fit"))
            #expect(prompt.contains("was truncated"))
            #expect(prompt.hasSuffix("User request:\nQuestion"))
        }
    }

    @Test func characterCountUsesSwiftCharactersAndMetadataRoundTrips() throws {
        let attachment = AppPromptAttachment(
            fileName: "emoji.pdf",
            formatLabel: "PDF",
            extractedText: "A🦝e\u{301}",
            wasTruncatedDuringExtraction: true)

        #expect(attachment.characterCount == 3)
        let decoded = try JSONDecoder().decode(
            AppPromptAttachment.self,
            from: JSONEncoder().encode(attachment))
        #expect(decoded == attachment)
    }
}
