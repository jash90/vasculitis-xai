import type { BatchResultRow } from '../api/types';

const quote = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`;

export function exportToCSV(results: BatchResultRow[]): string {
  const headers = ['ID pacjenta', 'Wiek w chwili rozpoznania', 'Liczba zajętych narządów', 'Ryzyko (%)', 'Poziom ryzyka', 'Główne czynniki'];
  const rows = results.map((r) => [r.patient_id, r.wiek_rozpoznania, r.liczba_narzadow, r.probability_pct, r.risk_level_pl, r.top_factors]);
  return '\uFEFF' + [headers, ...rows].map((row) => row.map(quote).join(',')).join('\n');
}

export function exportToJSON(results: BatchResultRow[]): string {
  return JSON.stringify(
    {
      analysis_date: new Date().toISOString(),
      total_patients: results.length,
      summary: {
        low_risk: results.filter((r) => r.risk_level === 'low').length,
        moderate_risk: results.filter((r) => r.risk_level === 'moderate').length,
        high_risk: results.filter((r) => r.risk_level === 'high').length,
        avg_probability: results.reduce((a, r) => a + r.probability, 0) / Math.max(results.length, 1),
      },
      patients: results,
    },
    null,
    2,
  );
}

export function downloadBlob(content: string, filename: string, mime: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
