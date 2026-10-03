import { test, expect } from '@playwright/test';
import { analyzeExample, openSingle, trackConsoleErrors } from './helpers';

test('form validation blocks submission without age', async ({ page }) => {
  await openSingle(page);
  const age = page.getByLabel('Wiek w chwili rozpoznania (lata)');
  await age.fill('');
  await page.getByRole('button', { name: 'Analizuj' }).click();
  await expect(page.getByText('Podaj wiek w chwili rozpoznania')).toBeVisible();
  await expect(age).toHaveAttribute('aria-invalid', 'true');
  await age.fill('130');
  await page.getByRole('button', { name: 'Analizuj' }).click();
  await expect(page.getByText('Wiek nie może przekraczać 120 lat')).toBeVisible();
});

test('organ count follows the selected manifestations', async ({ page }) => {
  await openSingle(page);
  await page.getByRole('switch', { name: 'Nerki' }).check({ force: true });
  await page.getByRole('switch', { name: 'Układ oddechowy' }).check({ force: true });
  await expect(page.getByRole('slider', { name: /Liczba zajętych narządów/ })).toHaveValue('2');
});

test('full single-patient analysis: models, survival, factors, every XAI method', async ({ page }) => {
  const assertNoErrors = trackConsoleErrors(page);
  await analyzeExample(page);

  for (const model of ['XGBoost', 'Random Forest', 'LightGBM']) {
    await expect(page.getByText(model, { exact: true }).first()).toBeVisible();
  }
  await expect(page.getByText('Średnia trzech modeli')).toBeVisible();

  const survival = page.getByRole('region', { name: 'Ryzyko zgonu w czasie' });
  await expect(survival.getByText('Po 5 latach')).toBeVisible();
  await expect(survival.getByText(/C-index/)).toBeVisible();

  const factors = page.getByRole('region', { name: /Najważniejsze czynniki/ });
  await expect(factors.getByRole('listitem').first()).toBeVisible();
  await expect(factors).not.toContainText('_'); // Polish labels, no raw backend names

  const panel = page.locator('#xai-panel');
  await expect(panel.getByText('Średnie ryzyko w danych')).toBeVisible();

  await page.getByRole('tab', { name: 'LIME' }).click();
  await expect(panel.getByText(/predykcja lokalnego modelu LIME/)).toBeVisible();

  await page.getByRole('tab', { name: 'DALEX' }).click();
  await expect(panel.getByText('Rozkład predykcji na cechy')).toBeVisible();
  await expect(panel.getByText('Globalna ważność cech (permutacja)')).toBeVisible();

  await page.getByRole('tab', { name: 'EBM' }).click();
  await expect(panel.getByText('Predykcja EBM')).toBeVisible({ timeout: 90_000 });

  await page.getByRole('tab', { name: 'Porównanie' }).click();
  await expect(panel.getByRole('heading', { name: 'Ranking SHAP' })).toBeVisible();
  await expect(panel.getByText(/Korelacja rang Spearmana/)).toBeVisible();

  // switching models re-explains; going back is served from cache
  await page.getByRole('tab', { name: 'SHAP' }).click();
  await page.getByRole('button', { name: 'Random Forest' }).click();
  await expect(page.getByRole('button', { name: 'Random Forest' })).toHaveAttribute('aria-pressed', 'true');
  await expect(panel.getByText('Średnie ryzyko w danych')).toBeVisible({ timeout: 90_000 });

  assertNoErrors();
});

test('edit data keeps values, new patient clears them', async ({ page }) => {
  await analyzeExample(page);
  await page.getByRole('button', { name: /Edytuj dane/ }).click();
  await expect(page.getByLabel('Wiek w chwili rozpoznania (lata)')).toHaveValue('71');
  await expect(page.getByLabel('Kreatynina (μmol/L)')).toHaveValue('380');

  await page.getByRole('button', { name: 'Analizuj' }).click();
  await page.getByRole('button', { name: 'Nowy pacjent' }).click();
  await expect(page.getByLabel('Kreatynina (μmol/L)')).toHaveValue('');
});

test('chat answers a suggested question with Polish feature names', async ({ page }) => {
  await analyzeExample(page);
  await page.getByRole('button', { name: 'Jakie czynniki najbardziej wpływają na ryzyko?' }).click();
  const chat = page.getByRole('region', { name: 'Rozmowa o wynikach' });
  await expect(chat.getByText('Piszę odpowiedź…')).toBeHidden({ timeout: 60_000 });
  await expect(chat.locator('.bg-gray-700').first()).toContainText(/ryzyk/i);
  await expect(chat).not.toContainText('Manifestacja_');
});

test('API failure shows an error with retry (no fake result)', async ({ page }) => {
  await openSingle(page);
  await page.route('**/predict/all', (route) => route.abort());
  await page.getByRole('button', { name: 'Analizuj' }).click();
  const alert = page.getByRole('alert');
  await expect(alert).toContainText('Nie udało się uzyskać predykcji');
  await expect(page.getByText('Średnia trzech modeli')).toHaveCount(0);

  await page.unroute('**/predict/all');
  await alert.getByRole('button', { name: 'Spróbuj ponownie' }).click();
  await expect(page.getByText('Średnia trzech modeli')).toBeVisible();
});
