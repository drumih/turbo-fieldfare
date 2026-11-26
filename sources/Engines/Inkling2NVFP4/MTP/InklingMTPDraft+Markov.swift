import Foundation
import Metal
import TinyTitanCore

extension InklingMTPDraft {
    func runMarkovChain(anchorToken: Int) throws -> [Int] {
        let cfg = weights.config
        let h = cfg.hiddenSize
        let r = cfg.markovRank
        let nSpec = cfg.numSpeculativeTokens
        memset(state.rnnStateA.contents(), 0, r * 4)
        var prev = anchorToken
        var drafted: [Int] = []
        drafted.reserveCapacity(nSpec)

        let headPtr = state.headHidden.contents().bindMemory(
            to: Float.self, capacity: cfg.blockSize * h)
        let fRnnPtr = state.fRnn.contents().bindMemory(to: Float.self, capacity: h)
        let w1Ptr = (weights.markovW1.operand.buffer.contents() + weights.markovW1.operand.offset)
            .bindMemory(to: UInt16.self, capacity: weights.markovW1.rows * r)
        let rnnInPtr = state.rnnInput.contents().bindMemory(to: Float.self, capacity: 2 * r + 2 * h)

        // rnnInput layout matches vLLM Inkling2MarkovHead: [state(r), markov_w1[prev](r), hidden(h), f_rnn(h)]
        memcpy(rnnInPtr + 2 * r + h, fRnnPtr, h * 4)

        for step in 0..<nSpec {
            let curStateBuf = (step % 2 == 0) ? state.rnnStateA : state.rnnStateB
            let nextStateBuf = (step % 2 == 0) ? state.rnnStateB : state.rnnStateA
            let statePtr = curStateBuf.contents().bindMemory(to: Float.self, capacity: r)
            memcpy(rnnInPtr, statePtr, r * 4)
            let clampedPrev = max(0, min(prev, weights.markovW1.rows - 1))
            let rowStart = clampedPrev * r
            for c in 0..<r {
                rnnInPtr[r + c] = Float(bitPattern: UInt32(w1Ptr[rowStart + c]) << 16)
            }
            // Rows 1..<blockSize hold the nSpec draft positions (row 0 is anchor).
            let stepRowBase = (step + 1) * h
            memcpy(rnnInPtr + 2 * r, headPtr + stepRowBase, h * 4)

            guard let cmd = context.queue.makeCommandBuffer(),
                let enc = cmd.makeComputeCommandEncoder(dispatchType: .concurrent)
            else { throw MetalError.commandEncoderFailed }

            try ops.gemv(
                weights.rnnProjWeight, x: .init(state.rnnInput), y: .init(state.rnnRaw), tokens: 1,
                encoder: enc)
            var rawWidth = UInt32(3 * r)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try pipeline("draft_add_bf16_bias"))
            enc.setBuffer(state.rnnRaw, offset: 0, index: 0)
            enc.setBuffer(
                weights.rnnProjBias.operand.buffer, offset: weights.rnnProjBias.operand.offset,
                index: 1)
            enc.setBytes(&rawWidth, length: 4, index: 2)
            enc.dispatchThreads(
                MTLSize(width: 3 * r, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            var rankU32 = UInt32(r)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try pipeline("draft_markov_gate"))
            enc.setBuffer(state.rnnRaw, offset: 0, index: 0)
            enc.setBuffer(curStateBuf, offset: 0, index: 1)
            enc.setBuffer(nextStateBuf, offset: 0, index: 2)
            enc.setBuffer(state.rnnLatent, offset: 0, index: 3)
            enc.setBytes(&rankU32, length: 4, index: 4)
            enc.dispatchThreads(
                MTLSize(width: r, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, r), height: 1, depth: 1))

            try ops.gemv(
                weights.hiddenOutWeight, x: .init(state.rnnLatent), y: .init(state.hiddenDelta),
                tokens: 1, encoder: enc)
            var hU32 = UInt32(h)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(try pipeline("draft_add_bf16_bias"))
            enc.setBuffer(state.hiddenDelta, offset: 0, index: 0)
            enc.setBuffer(
                weights.hiddenOutBias.operand.buffer, offset: weights.hiddenOutBias.operand.offset,
                index: 1)
            enc.setBytes(&hU32, length: 4, index: 2)
            enc.dispatchThreads(
                MTLSize(width: h, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            if let err = cmd.error { throw err }

            let deltaPtr = state.hiddenDelta.contents().bindMemory(to: Float.self, capacity: h)
            let stepPtr = state.stepHidden.contents().bindMemory(to: Float.self, capacity: h)
            for c in 0..<h { stepPtr[c] = headPtr[stepRowBase + c] + deltaPtr[c] }

            guard let cmd2 = context.queue.makeCommandBuffer(),
                let enc2 = cmd2.makeComputeCommandEncoder(dispatchType: .concurrent)
            else { throw MetalError.commandEncoderFailed }
            try ops.gemv(
                targetWeights.output, x: .init(state.stepHidden), y: .init(state.draftLogits),
                tokens: 1, encoder: enc2)
            try ops.argmax(
                logits: .init(state.draftLogits), count: manifest.vocabulary,
                partials: .init(state.argmaxPartials), groupCount: InklingQ4State.argmaxGroups,
                token: .init(state.draftNextToken), encoder: enc2)
            enc2.endEncoding()
            cmd2.commit()
            cmd2.waitUntilCompleted()
            if let err = cmd2.error { throw err }

            prev = Int(state.draftNextToken.contents().load(as: Int32.self))
            drafted.append(prev)
        }
        return drafted
    }

    func encodeCaptureTap(
        src: Operand, width: Int, taps: Int, tapIndex: Int, tokens: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        var widthU32 = UInt32(width)
        var tapsU32 = UInt32(taps)
        var tapIdxU32 = UInt32(tapIndex)
        encoder.memoryBarrier(scope: .buffers)
        encoder.setComputePipelineState(try pipeline("draft_capture_tap"))
        encoder.setBuffer(src.buffer, offset: src.offset, index: 0)
        encoder.setBuffer(state.targetTaps, offset: 0, index: 1)
        encoder.setBytes(&widthU32, length: 4, index: 2)
        encoder.setBytes(&tapsU32, length: 4, index: 3)
        encoder.setBytes(&tapIdxU32, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: width, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, width), height: 1, depth: 1))
    }

    func encodeConvSnapshot(
        x: Operand, convState: Operand, routeWeights: Operand?, visit: Int,
        channels: Int, slots: Int, tokens: Int, channelOffset: Int,
        encoder: MTLComputeCommandEncoder
    ) throws {
        var args = ConvSnapshotArgs(
            channels: UInt32(channels), slots: UInt32(slots), tokens: UInt32(tokens),
            snapStrideFloats: UInt32(state.convSnapshotStrideFloats),
            channelOffset: UInt32(channelOffset))
        encoder.memoryBarrier(scope: .buffers)
        encoder.setComputePipelineState(try pipeline("draft_conv_snapshots"))
        encoder.setBuffer(x.buffer, offset: x.offset, index: 0)
        encoder.setBuffer(convState.buffer, offset: convState.offset, index: 1)
        let rw = routeWeights ?? convState
        encoder.setBuffer(rw.buffer, offset: rw.offset, index: 2)
        encoder.setBuffer(state.convSnapshots[visit], offset: 0, index: 3)
        encoder.setBytes(&args, length: MemoryLayout<ConvSnapshotArgs>.stride, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: channels, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, channels), height: 1, depth: 1))
    }
}
