// Separable sliding extrema: O(width * height), independent of window radius.
// Invalid neighbors are ignored; invalid centers have zero relief.
export function computeLocalRelief(heights, width, height, radius = 2, mask = null) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || heights.length !== width * height
      || (mask && mask.length !== heights.length) || !Number.isFinite(radius)) throw new Error('Invalid relief raster dimensions or radius.');
  radius = Math.max(0, Math.floor(radius));
  const low = new Float32Array(heights.length), high = new Float32Array(heights.length), result = new Float32Array(heights.length);
  const minQueue = new Int32Array(Math.max(width, height)), maxQueue = new Int32Array(minQueue.length);
  for (let y = 0; y < height; y++) {
    let a = 0, b = 0, c = 0, d = 0, next = 0;
    for (let x = 0; x < width; x++) {
      const end = Math.min(width - 1, x + radius);
      while (next <= end) {
        const i = y * width + next, v = heights[i];
        if ((!mask || mask[i]) && Number.isFinite(v)) {
          while (b > a && heights[y * width + minQueue[b - 1]] >= v) b--;
          while (d > c && heights[y * width + maxQueue[d - 1]] <= v) d--;
          minQueue[b++] = next; maxQueue[d++] = next;
        }
        next++;
      }
      while (b > a && minQueue[a] < x - radius) a++;
      while (d > c && maxQueue[c] < x - radius) c++;
      low[y * width + x] = b > a ? heights[y * width + minQueue[a]] : Infinity;
      high[y * width + x] = d > c ? heights[y * width + maxQueue[c]] : -Infinity;
    }
  }
  for (let x = 0; x < width; x++) {
    let a = 0, b = 0, c = 0, d = 0, next = 0;
    for (let y = 0; y < height; y++) {
      const end = Math.min(height - 1, y + radius);
      while (next <= end) {
        const i = next * width + x;
        while (b > a && low[minQueue[b - 1] * width + x] >= low[i]) b--;
        while (d > c && high[maxQueue[d - 1] * width + x] <= high[i]) d--;
        minQueue[b++] = next; maxQueue[d++] = next; next++;
      }
      while (b > a && minQueue[a] < y - radius) a++;
      while (d > c && maxQueue[c] < y - radius) c++;
      const i = y * width + x;
      if ((!mask || mask[i]) && Number.isFinite(heights[i])) result[i] = high[maxQueue[c] * width + x] - low[minQueue[a] * width + x];
    }
  }
  return result;
}
