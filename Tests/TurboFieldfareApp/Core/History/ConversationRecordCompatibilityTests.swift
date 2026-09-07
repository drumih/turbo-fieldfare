import Foundation
import Testing
import TurboFieldfare
@testable import TurboFieldfareAppCore

/// The transcript has to stay readable in both directions across a build that
/// adds a field or a record kind, because both worktrees on this machine share
/// one store through the `scratch` symlink and either can be running.
///
/// Every case here is a way that could stop being true: a key a reader does not
/// know, a key a writer did not write, or a whole record type from a later
/// build. The first two passed before the projection landed and are pinned so
/// they keep passing; the third pins the downgrade direction against a copy of
/// the decoder as it was.
@Suite struct ConversationRecordCompatibilityTests {
    private static func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    private static func encoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    /// A later build adding a field to a turn must not make the line
    /// unreadable: `open` treats a line it cannot parse anywhere but the end as
    /// two writers or in-place damage, and refuses the whole conversation.
    @Test func aTurnLineWithAKeyThisBuildDoesNotKnowStillDecodes() throws {
        let line = """
        {"type":"turn","turn":{"id":"E621E1F8-C36C-495A-93FC-0C247A3E6E5F",\
        "role":"user","at":"2026-09-02T10:00:00Z","text":"hello","images":[],\
        "tokens":[1,2,3],"summary":"written by a later build",\
        "moodRing":{"colour":"amber"}}}
        """
        let record = try Self.decoder().decode(
            TranscriptRecord.self, from: Data(line.utf8))
        guard case .turn(let turn) = record else {
            Issue.record("the line decoded as something other than a turn")
            return
        }
        #expect(turn.text == "hello")
        #expect(turn.tokens == [1, 2, 3])
        #expect(turn.sampling == nil)
        #expect(turn.boundary == nil)
    }

    /// A header written before the creation facts moved into it.
    ///
    /// Every transcript in the developer store has one, and refusing them would
    /// hide every conversation written before this build.
    @Test func aHeaderWithoutTheCreationFactsStillDecodes() throws {
        let line = """
        {"type":"header","header":{"version":1,\
        "id":"E621E1F8-C36C-495A-93FC-0C247A3E6E5F",\
        "createdAt":"2026-09-01T09:00:00Z"}}
        """
        let record = try Self.decoder().decode(
            TranscriptRecord.self, from: Data(line.utf8))
        guard case .header(let header) = record else {
            Issue.record("the line decoded as something other than a header")
            return
        }
        #expect(header.id == UUID(uuidString: "E621E1F8-C36C-495A-93FC-0C247A3E6E5F"))
        #expect(header.identity == nil)
        #expect(header.session == nil)
        #expect(header.sampling == nil)
    }

    /// The extra header fields survive a round trip through this build.
    @Test func aHeaderCarryingTheCreationFactsRoundTrips() throws {
        let header = ConversationHeaderRecord(
            id: UUID(), createdAt: Date(timeIntervalSince1970: 1_756_800_000),
            identity: Self.identity, session: Self.session,
            sampling: Self.sampling)
        let data = try Self.encoder().encode(TranscriptRecord.header(header))
        let decoded = try Self.decoder().decode(TranscriptRecord.self, from: data)
        #expect(decoded == .header(header))
    }

    /// The downgrade direction: a build without the `origin` case reads the
    /// line as an unknown record and carries on.
    ///
    /// Decoded here through a copy of the enum's `CodingKeys` and `init` as
    /// they were before `origin` existed, because the point is what the *older*
    /// build does — asking the current one would prove nothing.
    @Test func anOriginLineReadsAsUnknownToABuildWithoutIt() throws {
        let origin = ConversationOriginRecord(
            identity: Self.identity, session: Self.session,
            sampling: Self.sampling, boundary: ConversationBoundary(),
            at: Date(timeIntervalSince1970: 1_756_800_000))
        let data = try Self.encoder().encode(TranscriptRecord.origin(origin))

        let asOlderBuildReadsIt = try Self.decoder().decode(
            PreOriginTranscriptRecord.self, from: data)
        #expect(asOlderBuildReadsIt == .unknown(type: "origin"))

        // And this build still reads it as itself.
        #expect(try Self.decoder().decode(TranscriptRecord.self, from: data)
            == .origin(origin))
    }

    /// `TranscriptRecord` as it decoded before the `origin` case existed.
    private enum PreOriginTranscriptRecord: Equatable, Decodable {
        case header(ConversationHeaderRecord)
        case turn(ConversationTurnRecord)
        case partial(ConversationTurnRecord)
        case title(ConversationTitleRecord)
        case unknown(type: String)

        private enum CodingKeys: String, CodingKey {
            case type, header, turn, partial, title
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type)
            switch type {
            case "header":
                self = .header(try container.decode(
                    ConversationHeaderRecord.self, forKey: .header))
            case "turn":
                self = .turn(try container.decode(
                    ConversationTurnRecord.self, forKey: .turn))
            case "partial":
                self = .partial(try container.decode(
                    ConversationTurnRecord.self, forKey: .partial))
            case "title":
                self = .title(try container.decode(
                    ConversationTitleRecord.self, forKey: .title))
            default:
                self = .unknown(type: type)
            }
        }
    }

    private static let identity = ConversationIdentity(
        modelID: "google/gemma-4-26B-A4B-it",
        sourceSnapshotHash: "0d77464e",
        templateIdentity: GFTokenizer.chatTemplateIdentity,
        imageProcessingVersion: VisionImageProcessing.version)

    private static let session = ConversationSessionSettings(
        contextTokens: 8_192, expertCacheSlots: 16,
        visionResidencyPolicy: "onDemand")

    private static let sampling = ConversationSampling(
        temperature: 0.2, topKEnabled: true, topK: 64,
        topPEnabled: true, topP: 0.95, maxNewTokens: 4_096)
}
