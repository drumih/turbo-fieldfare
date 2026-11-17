import Metal
import Testing
import TinyTitanValidationSupport

@testable import TinyTitan
@testable import TinyTitanCore

@Suite struct KVCacheQuantizedAttentionTests {
    @Test(arguments: [TinyTitanCore.KVCachePrecision.int8, .int4])
    func quantizedDecodeAttentionTracksFP16Reference(_ precision: TinyTitanCore.KVCachePrecision)
        throws
    {
        let config = ArchConfig.qwen36_35B_A3B
        let headDim = config.fullHeadDim
        let numQHeads = config.numHeads
        let numKVHeads = config.numFullKVHeads
        let seqLen = 8
        let qCount = numQHeads * headDim
        let rowElements = numKVHeads * headDim
        let kvCount = seqLen * rowElements
        var rng = SeedTree(0x4B56).key("kv-cache-\(precision.rawValue)")
        let q = (0..<qCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let k = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let v = (0..<kvCount).map { _ in Float16(rng.uniform(-0.5, 0.5)) }

        let context = try MetalContext()
        let cache = try KVCacheManager(
            device: context.device,
            config: config,
            maxContext: seqLen,
            precision: precision)
        let quantizer = try KVCacheQuantizer(context: context)
        let attention = try Attention(context: context)
        guard let qBuffer = Fp16Buffer.make(context.device, halves: q),
            let kBuffer = Fp16Buffer.make(context.device, halves: k),
            let vBuffer = Fp16Buffer.make(context.device, halves: v),
            let output = Fp16Buffer.make(context.device, count: qCount),
            let commandBuffer = context.queue.makeCommandBuffer()
        else {
            Issue.record("buffer allocation failed")
            return
        }
        let keyView = cache.keyView(layer: 3, validTokenCount: seqLen)
        let valueView = cache.valueView(layer: 3, validTokenCount: seqLen)
        try quantizer.encode(
            commandBuffer: commandBuffer,
            source: kBuffer,
            sourceTokenStrideElements: rowElements,
            destination: keyView,
            tokenCount: seqLen,
            elementCount: rowElements)
        try quantizer.encode(
            commandBuffer: commandBuffer,
            source: vBuffer,
            sourceTokenStrideElements: rowElements,
            destination: valueView,
            tokenCount: seqLen,
            elementCount: rowElements)
        try attention.encodeFull(
            commandBuffer: commandBuffer,
            q: qBuffer,
            k: keyView.buffer,
            v: valueView.buffer,
            out: output,
            headDim: UInt32(headDim),
            numQHeads: UInt32(numQHeads),
            numKVHeads: UInt32(numKVHeads),
            seqLen: UInt32(seqLen),
            kvFormat: keyView)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let reference = AttentionRef.apply(
            q: q.map(Float.init), k: k.map(Float.init), v: v.map(Float.init),
            headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
            seqLen: seqLen)
        let actual = Fp16Buffer.read(output, count: qCount)
        let relativeError = RelError.compute(actual: actual, reference: reference)
        let tolerance: Float = precision == .int8 ? 0.02 : 0.20
        #expect(
            relativeError < tolerance,
            "\(precision.label) KV attention rel=\(relativeError)")
    }

    @Test func rowLayoutMatchesExpectedMemoryReduction() throws {
        let context = try MetalContext()
        let config = ArchConfig.qwen36_35B_A3B
        let fp16 = try KVCacheManager(
            device: context.device, config: config,
            maxContext: 1, precision: .fp16)
        let int8 = try KVCacheManager(
            device: context.device, config: config,
            maxContext: 1, precision: .int8)
        let int4 = try KVCacheManager(
            device: context.device, config: config,
            maxContext: 1, precision: .int4)
        #expect(fp16.stride(layer: 3) == 1_024)
        #expect(int8.stride(layer: 3) == 544)
        #expect(int4.stride(layer: 3) == 288)
    }

    @Test(arguments: [TinyTitanCore.KVCachePrecision.int8, .int4])
    func quantizedChunkedPrefillTracksFP16Kernel(_ precision: TinyTitanCore.KVCachePrecision) throws
    {
        let config = ArchConfig.qwen36_35B_A3B
        let headDim = config.fullHeadDim
        let numQHeads = config.numHeads
        let numKVHeads = config.numFullKVHeads
        let queryCount = 4
        let startPosition = 4
        let sequenceLength = startPosition + queryCount
        let qRow = numQHeads * headDim
        let kvRow = numKVHeads * headDim
        var rng = SeedTree(0x5052).key("prefill-kv-\(precision.rawValue)")
        let q = (0..<(queryCount * qRow)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let k = (0..<(sequenceLength * kvRow)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let v = (0..<(sequenceLength * kvRow)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let context = try MetalContext()
        let cache = try KVCacheManager(
            device: context.device, config: config,
            maxContext: sequenceLength, precision: precision)
        let quantizer = try KVCacheQuantizer(context: context)
        let attention = try PrefillAttention(context: context)
        guard let qBuffer = Fp16Buffer.make(context.device, halves: q),
            let kBuffer = Fp16Buffer.make(context.device, halves: k),
            let vBuffer = Fp16Buffer.make(context.device, halves: v),
            let fp16Output = Fp16Buffer.make(context.device, count: queryCount * qRow),
            let quantizedOutput = Fp16Buffer.make(context.device, count: queryCount * qRow),
            let commandBuffer = context.queue.makeCommandBuffer()
        else {
            Issue.record("buffer allocation failed")
            return
        }
        let keyView = cache.keyView(layer: 3, validTokenCount: sequenceLength)
        let valueView = cache.valueView(layer: 3, validTokenCount: sequenceLength)
        try quantizer.encode(
            commandBuffer: commandBuffer, source: kBuffer,
            sourceTokenStrideElements: kvRow,
            destination: keyView, tokenCount: sequenceLength,
            elementCount: kvRow)
        try quantizer.encode(
            commandBuffer: commandBuffer, source: vBuffer,
            sourceTokenStrideElements: kvRow,
            destination: valueView, tokenCount: sequenceLength,
            elementCount: kvRow)
        let baseParams = PrefillAttentionParams(
            startPosition: UInt32(startPosition), queryCount: UInt32(queryCount),
            headDim: UInt32(headDim), numQHeads: UInt32(numQHeads),
            numKVHeads: UInt32(numKVHeads), kvValidCount: UInt32(sequenceLength),
            slidingWindow: 0, kvTokenStrideElements: UInt32(kvRow),
            qTokenStrideElements: UInt32(qRow), oTokenStrideElements: UInt32(qRow),
            scale: Attention.defaultScale(headDim: UInt32(headDim)))
        try attention.encodeCausal(
            commandBuffer: commandBuffer, q: qBuffer,
            k: kBuffer, v: vBuffer, out: fp16Output,
            params: baseParams, path: .causalTiled)
        var quantizedParams = baseParams
        quantizedParams.kvBits = UInt32(precision.rawValue)
        quantizedParams.kvTokenStrideBytes = UInt32(keyView.stride)
        quantizedParams.kvValueBytes = UInt32(keyView.valueBytes)
        quantizedParams.kvGroupSize = UInt32(keyView.groupSize)
        try attention.encodeCausal(
            commandBuffer: commandBuffer, q: qBuffer,
            k: keyView.buffer, v: valueView.buffer,
            out: quantizedOutput, params: quantizedParams,
            path: .causalTiled)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        let fp16 = Fp16Buffer.read(fp16Output, count: queryCount * qRow)
        let quantized = Fp16Buffer.read(quantizedOutput, count: queryCount * qRow)
        let relativeError = RelError.compute(actual: quantized, reference: fp16)
        let tolerance: Float = precision == .int8 ? 0.025 : 0.20
        #expect(
            relativeError < tolerance,
            "\(precision.label) prefill KV rel=\(relativeError)")
    }
}
