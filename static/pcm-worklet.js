// AudioWorklet that downmixes to mono and posts int16 PCM at the input
// sample rate. Resampling to 16 kHz is done on the main thread.
class PCMWorklet extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channel = input[0];
    if (!channel) return true;

    const out = new Int16Array(channel.length);
    let sumSquares = 0;
    for (let i = 0; i < channel.length; i++) {
      const s = Math.max(-1, Math.min(1, channel[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      sumSquares += s * s;
    }
    const rms = Math.sqrt(sumSquares / channel.length);
    this.port.postMessage({ pcm: out.buffer, rms }, [out.buffer]);
    return true;
  }
}
registerProcessor("pcm-worklet", PCMWorklet);
