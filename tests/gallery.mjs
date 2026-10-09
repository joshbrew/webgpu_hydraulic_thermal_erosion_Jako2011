import assert from 'node:assert/strict';

export async function checkGallery(page, url, count, { full = true } = {}) {
  const errors = [];
  const onError = error => errors.push(error.message);
  page.on('pageerror', onError);
  try {
    await page.goto(url);
    assert.equal(await page.getByRole('button', { name: 'Run sim', exact: true }).count(), count, 'every result can run');
    const volcano = page.locator('article[data-terrain="volcano"]');
    if (full) {
      await volcano.evaluate(card => { card.querySelector('.run-sim').click(); [...card.querySelectorAll('.live-panel button')].find(b => b.textContent === 'Close').click(); });
      await page.waitForFunction(() => !document.querySelector('article[data-terrain="volcano"] .run-sim').disabled, undefined, { timeout: 60000 });
      assert.equal(await page.locator('.live-panel').isVisible(), false, 'closing during preparation cancels the preview');
    }
    await volcano.getByRole('button', { name: 'Run sim', exact: true }).click();
    await page.waitForFunction(() => Number(document.querySelector('.live-panel')?.dataset.time) > 0.2, undefined, { timeout: 60000 });
    const panel = page.locator('.live-panel');
    assert.equal(await panel.getByRole('combobox', { name: 'Live view' }).inputValue(), '8', 'slope relief is the default');
    assert.equal(await panel.locator('.layer-legend span').count(), 4, 'volcano has four live layers');
    await panel.getByRole('button', { name: 'Pause', exact: true }).click();
    const paused = Number(await panel.getAttribute('data-time'));
    await page.waitForTimeout(120);
    assert.equal(Number(await panel.getAttribute('data-time')), paused, 'pause stops the live preview');
    await panel.getByRole('combobox', { name: 'Live view' }).selectOption('8');
    await panel.getByRole('combobox', { name: 'Live view' }).selectOption('9');
    await panel.getByRole('combobox', { name: 'Live view' }).selectOption('8');
    if (full) {
      await panel.getByRole('button', { name: 'View result', exact: true }).click();
      await page.waitForFunction(() => Math.abs(Number(document.querySelector('.live-panel')?.dataset.time) - 120) < 0.001
        && ![...document.querySelectorAll('.live-panel button')].find(b => b.textContent === 'Reset')?.disabled, undefined, { timeout: 60000 });
      await panel.getByRole('button', { name: 'Run', exact: true }).click();
      await page.waitForFunction(() => Number(document.querySelector('.live-panel')?.dataset.time) > 180.1, undefined, { timeout: 60000 });
      assert.equal(await panel.getAttribute('data-running'), 'true', 'Run continues beyond the pictured time and former 60-second extension');
      await panel.getByRole('button', { name: 'Pause', exact: true }).click();
      await panel.getByRole('button', { name: 'Reset', exact: true }).click();
      await page.waitForFunction(() => document.querySelector('.live-panel')?.dataset.time === '0'
        && ![...document.querySelectorAll('.live-panel button')].find(b => b.textContent === 'Reset')?.disabled);
      const delta = page.locator('article[data-terrain="river_delta"]');
      await delta.getByRole('button', { name: 'Run sim', exact: true }).click();
      await page.waitForFunction(() => document.querySelector('article[data-terrain="river_delta"] .live-panel')?.dataset.running === 'true');
      assert.equal(await page.locator('canvas').count(), 1, 'scene switches reuse one viewport');
      assert.equal(await panel.locator('.layer-legend span').count(), 3, 'delta restores its own layers');
      assert.equal(await panel.getByRole('spinbutton', { name: 'River sediment load' }).inputValue(), '0.12', 'delta starts with a sediment-fed source');
      await panel.getByRole('spinbutton', { name: 'River sediment load' }).fill('0.16');
      await panel.getByRole('button', { name: 'Pause', exact: true }).click();
    }
    await panel.getByRole('button', { name: 'Close', exact: true }).click();
    assert.equal(await panel.isVisible(), false, 'close hides the stopped preview');
    assert.deepEqual(errors, [], 'live gallery browser errors');
    console.log(`Live gallery (${new URL(url).protocol}): playback, layers, pause, relief view${full ? ', result replay, reset, switching' : ''} passed.`);
  } finally { page.off('pageerror', onError); }
}
