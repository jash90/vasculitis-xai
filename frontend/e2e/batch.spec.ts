import { test, expect } from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs';

const fixture = (name: string) => path.join(import.meta.dirname, 'fixtures', name);

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.getByRole('tab', { name: 'Analiza masowa' }).click();
});

test('sample CSV has exactly the model columns and analyses cleanly', async ({ page }) => {
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Pobierz przykładowy plik CSV' }).click();
  const content = fs.readFileSync(await (await download).path());
  const header = content.toString('utf-8').split('\n')[0];
  expect(header.split(',')).toHaveLength(21);
  expect(header).toContain('max_crp');
  expect(header).not.toContain('dializa');

  await page.locator('input[type=file]').setInputFiles({ name: 'przykladowi_pacjenci.csv', mimeType: 'text/csv', buffer: content });
  await expect(page.getByRole('heading', { name: 'Szczegółowe wyniki' })).toBeVisible();
  await expect(page.getByText('Uwagi do pliku')).toHaveCount(0);
  await expect(page.getByRole('table').getByRole('row')).toHaveCount(7); // header + 6 patients
});

test('file with issues: report, unique ids, per-patient factors and export', async ({ page }) => {
  await page.locator('input[type=file]').setInputFiles(fixture('patients_with_issues.csv'));
  const report = page.getByRole('status').filter({ hasText: 'Uwagi do pliku' });
  await expect(report).toContainText('plec, dializa');
  await expect(report).toContainText('Kreatynina: „abc”');

  const table = page.getByRole('table');
  await expect(table.getByRole('row')).toHaveCount(6);
  await expect(table.getByRole('cell', { name: 'A1 (5)' })).toBeVisible();
  await expect(table).not.toContainText('_');

  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Pobierz wyniki (CSV)' }).click();
  const csv = fs.readFileSync(await (await download).path(), 'utf-8');
  expect(csv.split('\n')).toHaveLength(6);
  expect(csv).toContain('"A1 (5)"');

  await page.getByRole('button', { name: 'Wgraj kolejny plik' }).click();
  await expect(page.getByRole('button', { name: /Przeciągnij plik/ })).toBeVisible();
});

test('unsupported file type shows an inline error', async ({ page }) => {
  await page.locator('input[type=file]').setInputFiles(fixture('notes.txt'));
  await expect(page.getByRole('alert')).toContainText('Nieobsługiwany format');
});

test('API failure: explicit error and retry, never silent demo predictions', async ({ page }) => {
  await page.route('**/predict/batch', (route) => route.abort());
  await page.locator('input[type=file]').setInputFiles(fixture('patients_with_issues.csv'));
  const alert = page.getByRole('alert');
  await expect(alert).toContainText('Analiza przerwana po 0 z 5 pacjentów');
  await expect(page.getByRole('table')).toHaveCount(0);

  await page.unroute('**/predict/batch');
  await alert.getByRole('button', { name: 'Spróbuj ponownie' }).click();
  await expect(page.getByRole('table').getByRole('row')).toHaveCount(6);
});
