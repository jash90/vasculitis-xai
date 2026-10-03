import { test, expect } from '@playwright/test';
import { trackConsoleErrors } from './helpers';

test('navigation is an accessible tablist and every view renders', async ({ page }) => {
  const assertNoErrors = trackConsoleErrors(page);
  await page.goto('/');
  const tabs = page.getByRole('tablist', { name: 'Tryb pracy' }).getByRole('tab');
  await expect(tabs).toHaveCount(4);

  await page.getByRole('tab', { name: 'Pojedynczy pacjent' }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Asystent AI' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('button', { name: 'Rozpocznij rozmowę' })).toBeVisible();

  await page.getByRole('tab', { name: 'Analiza masowa' }).click();
  await expect(page.getByRole('button', { name: /Przeciągnij plik/ })).toBeVisible();

  await page.getByRole('tab', { name: 'O modelach' }).click();
  await expect(page.getByRole('heading', { name: 'O modelach' })).toBeVisible();
  await expect(page.getByText('Ograniczenia')).toBeVisible();
  await expect(page.getByRole('table').getByRole('row')).toHaveCount(4); // header + 3 models
  assertNoErrors();
});

test('welcome page shows measured (not hard-coded) model quality', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText(/Jakość na odłożonym zbiorze testowym: AUC 0\.\d\d/)).toBeVisible();
  await expect(page.getByText('AUC 0.81')).toHaveCount(0);
  await expect(page.getByText('719 pacjentow')).toHaveCount(0);
});
