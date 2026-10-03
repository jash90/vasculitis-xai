import { test, expect } from '@playwright/test';
import { trackConsoleErrors } from './helpers';

test('agent collects 16 answers (incl. "nie wiem"), predicts, survives tab switch and restarts', async ({ page }) => {
  const assertNoErrors = trackConsoleErrors(page);
  await page.goto('/');
  await page.getByRole('tab', { name: 'Asystent AI' }).click();
  await page.getByRole('button', { name: 'Rozpocznij rozmowę' }).click();

  for (let step = 1; step <= 16; step++) {
    await expect(page.getByText(`[${step}/16]`)).toBeVisible();
    const confirm = page.getByRole('button', { name: /^Potwierdź/ });
    if (await confirm.isVisible()) {
      // number question: answer "nie wiem" for the delay (step 2), confirm the default otherwise
      if (step === 2) await page.getByRole('button', { name: 'Nie wiem' }).click();
      else await confirm.click();
    } else {
      await page.getByRole('button', { name: step % 3 === 0 ? 'nie wiem' : step % 2 ? 'tak' : 'nie', exact: true }).click();
    }
  }
  await expect(page.getByText('Zebrałem wszystkie dane')).toBeVisible();
  await expect(page.getByText('Opóźnienie rozpoznania: nie wiem')).toBeVisible();

  // conversation is kept when switching tabs
  await page.getByRole('tab', { name: 'Pojedynczy pacjent' }).click();
  await page.getByRole('tab', { name: 'Asystent AI' }).click();
  await expect(page.getByText('Zebrałem wszystkie dane')).toBeVisible();

  await page.getByRole('button', { name: 'Jakie czynniki wpływają na wynik?' }).click();
  await expect(page.getByText('Szczegółowe zestawienie czynników')).toBeVisible();

  await page.getByRole('button', { name: 'Zacznij od nowa' }).click();
  await expect(page.getByText('[1/16]')).toBeVisible();
  await expect(page.getByText('Zebrałem wszystkie dane')).toHaveCount(0);
  assertNoErrors();
});
