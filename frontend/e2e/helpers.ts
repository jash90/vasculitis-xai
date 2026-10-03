import { expect, type Page } from '@playwright/test';

/** Fail the test on uncaught page errors / console errors (ignores aborted requests we cause on purpose). */
export function trackConsoleErrors(page: Page) {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (msg) => {
    if (msg.type() === 'error' && !/Failed to load resource|ERR_FAILED|net::/.test(msg.text())) errors.push(msg.text());
  });
  return () => expect(errors, `console errors: ${errors.join('\n')}`).toEqual([]);
}

export async function openSingle(page: Page) {
  await page.goto('/');
  await expect(page.getByRole('tab', { name: 'Pojedynczy pacjent' })).toHaveAttribute('aria-selected', 'true');
}

export async function analyzeExample(page: Page) {
  await openSingle(page);
  await page.getByRole('button', { name: 'Wstaw przykładowego pacjenta' }).click();
  await page.getByRole('button', { name: 'Analizuj' }).click();
  await expect(page.getByRole('heading', { name: 'Wynik analizy' })).toBeVisible();
}
