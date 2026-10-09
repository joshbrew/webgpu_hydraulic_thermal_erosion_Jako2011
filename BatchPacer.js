// Use existing completion fences for feedback; this adds no GPU readback.
export default class BatchPacer {
  constructor({ budgetMs = 8, maximumSteps = 64 } = {}) {
    this.budgetMs = budgetMs;
    this.maximumSteps = maximumSteps;
    this.reset();
  }

  reset(steps = 4) {
    this.steps = Math.max(1, Math.min(this.maximumSteps, Math.floor(Number(steps)) || 1));
  }

  observe(submittedSteps, completionMs) {
    if (!Number.isFinite(completionMs) || completionMs <= 0) return;
    // Grow cautiously and back off promptly when the queue or GPU is busy.
    const scale = Math.max(0.5, Math.min(1.25, this.budgetMs / Math.max(completionMs, 0.25)));
    const next = Math.round(submittedSteps * scale);
    this.reset(scale >= 1.2 ? Math.max(submittedSteps + 1, next) : next);
  }
}
