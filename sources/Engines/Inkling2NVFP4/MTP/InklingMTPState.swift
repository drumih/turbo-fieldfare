import Foundation
import Metal
import TinyTitanCore

/// GPU buffers for Inkling 2's DSpark MTP draft model and target verify
/// rollback snapshots (TT-129).
final class InklingMTPState {
    let config: InklingMTPConfig
    let maxContext: Int
    let maxVerifyTokens: Int
    let capacities: [Int]
    let kCaches: [MTLBuffer]
    let vCaches: [MTLBuffer]

    // Target tap capture buffer: [maxChunk, taps, hidden] FP32
    let targetTaps: MTLBuffer
    // Per-visit conv state snapshots for verify rollback: [48][maxVerifyTokens * convChannels * 3]
    let convSnapshots: [MTLBuffer]
    let convSnapshotStrideFloats: Int
    // Per-token verify logits & argmax for up to maxVerifyTokens
    let verifyNextTokens: MTLBuffer
    let verifyFinalHidden: MTLBuffer

    // Draft context KV projection scratch (sized for maxChunk tokens)
    let ctxMixed: MTLBuffer
    let ctxK: MTLBuffer
    let ctxV: MTLBuffer

    // Draft block-B scratch
    let anchorToken: MTLBuffer
    let blockX: MTLBuffer
    let blockResidual: MTLBuffer
    let blockNormed: MTLBuffer
    let blockQ: MTLBuffer
    let blockK: MTLBuffer
    let blockV: MTLBuffer
    let blockAttn: MTLBuffer
    let blockProj: MTLBuffer
    let blockGate: MTLBuffer
    let blockUp: MTLBuffer
    let blockAct: MTLBuffer
    let blockMLPOut: MTLBuffer
    let headHidden: MTLBuffer

    // Markov RNN-hidden head scratch
    let fRnn: MTLBuffer
    let prevTokenBuf: MTLBuffer
    let prevEmbed: MTLBuffer
    let rnnInput: MTLBuffer
    let rnnRaw: MTLBuffer
    let rnnStateA: MTLBuffer
    let rnnStateB: MTLBuffer
    let rnnLatent: MTLBuffer
    let hiddenDelta: MTLBuffer
    let stepHidden: MTLBuffer
    let draftLogits: MTLBuffer
    let argmaxPartials: MTLBuffer
    let draftNextToken: MTLBuffer

    init(
        config: InklingMTPConfig, manifest: InklingQ4Manifest, device: MTLDevice,
        maxContext: Int, maxChunk: Int
    ) throws {
        func buf(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared)
            else { throw MetalError.bufferAllocationFailed("Inkling MTP \(label)") }
            memset(b.contents(), 0, b.length)
            return b
        }
        self.config = config
        self.maxContext = maxContext
        let b = config.blockSize
        self.maxVerifyTokens = b
        let h = config.hiddenSize
        let taps = config.targetLayerIDs.count

        var caps: [Int] = []
        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        for i in 0..<config.numLayers {
            let local = config.localLayerIDs.contains(i)
            let cap = local ? min(config.slidingWindow + maxChunk, maxContext) : maxContext
            caps.append(cap)
            ks.append(try buf(cap * config.kvWidth * 2, "draft K \(i)"))
            vs.append(try buf(cap * config.kvWidth * 2, "draft V \(i)"))
        }
        self.capacities = caps
        self.kCaches = ks
        self.vCaches = vs

        let kvWidthMain = manifest.headDimension * manifest.layers[0].kvHeads
        let convChannels = 2 * kvWidthMain + 2 * manifest.hidden
        self.convSnapshotStrideFloats = convChannels * 3
        self.convSnapshots = try (0..<manifest.unrolledLayers).map { visit in
            try buf(b * convChannels * 3 * 4, "conv snap \(visit)")
        }
        self.targetTaps = try buf(maxChunk * taps * h * 4, "target taps")
        self.verifyNextTokens = try buf(b * 4, "verify next tokens")
        self.verifyFinalHidden = try buf(maxChunk * h * 4, "verify final hidden")

        self.ctxMixed = try buf(maxChunk * h * 4, "ctx mixed")
        self.ctxK = try buf(maxChunk * config.kvWidth * 4, "ctx K")
        self.ctxV = try buf(maxChunk * config.kvWidth * 4, "ctx V")

        self.anchorToken = try buf(16, "anchor token")
        self.blockX = try buf(b * h * 4, "block x")
        self.blockResidual = try buf(b * h * 4, "block residual")
        self.blockNormed = try buf(b * h * 4, "block normed")
        self.blockQ = try buf(b * config.qWidth * 4, "block q")
        self.blockK = try buf(b * config.kvWidth * 4, "block k")
        self.blockV = try buf(b * config.kvWidth * 4, "block v")
        self.blockAttn = try buf(b * config.qWidth * 4, "block attn")
        self.blockProj = try buf(b * h * 4, "block proj")
        self.blockGate = try buf(b * config.intermediateSize * 4, "block gate")
        self.blockUp = try buf(b * config.intermediateSize * 4, "block up")
        self.blockAct = try buf(b * config.intermediateSize * 4, "block act")
        self.blockMLPOut = try buf(b * h * 4, "block mlp out")
        self.headHidden = try buf(b * h * 4, "head hidden")

        let r = config.markovRank
        self.fRnn = try buf(h * 4, "f_rnn")
        self.prevTokenBuf = try buf(16, "prev token")
        self.prevEmbed = try buf(r * 4, "prev embed")
        self.rnnInput = try buf((2 * r + 2 * h) * 4, "rnn input")
        self.rnnRaw = try buf(3 * r * 4, "rnn raw")
        self.rnnStateA = try buf(r * 4, "rnn state A")
        self.rnnStateB = try buf(r * 4, "rnn state B")
        self.rnnLatent = try buf(r * 4, "rnn latent")
        self.hiddenDelta = try buf(h * 4, "hidden delta")
        self.stepHidden = try buf(h * 4, "step hidden")
        self.draftLogits = try buf(manifest.paddedVocabulary * 4, "draft logits")
        self.argmaxPartials = try buf(InklingQ4State.argmaxGroups * 8, "draft argmax partials")
        self.draftNextToken = try buf(16, "draft next token")
    }

    func reset() {
        for b in kCaches + vCaches {
            memset(b.contents(), 0, b.length)
        }
    }
}
