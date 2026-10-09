// Created only for explicit benchmarks; ordinary frames do not resolve queries
// or map buffers. Browsers may quantize timestamps, so short passes can read 0.
export default class GpuTiming {
  constructor(device, passCount) {
    this.count = passCount * 2;
    this.querySet = device.createQuerySet({ type: 'timestamp', count: this.count });
    this.resolveBuffer = device.createBuffer({ size: this.count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    this.readBuffer = device.createBuffer({ size: this.count * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }

  writes(passIndex = 0) {
    return { querySet: this.querySet, beginningOfPassWriteIndex: passIndex * 2, endOfPassWriteIndex: passIndex * 2 + 1 };
  }

  resolve(encoder) {
    encoder.resolveQuerySet(this.querySet, 0, this.count, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, this.readBuffer, 0, this.count * 8);
  }

  async read() {
    try {
      await this.readBuffer.mapAsync(GPUMapMode.READ);
      const timestamps = new BigUint64Array(this.readBuffer.getMappedRange());
      let nanoseconds = 0n;
      for (let i = 0; i < this.count; i += 2) {
        if (timestamps[i + 1] >= timestamps[i]) nanoseconds += timestamps[i + 1] - timestamps[i];
      }
      return Number(nanoseconds) / 1e6;
    } finally {
      this.readBuffer.unmap();
      this.readBuffer.destroy();
      this.resolveBuffer.destroy();
      this.querySet.destroy();
    }
  }
}
