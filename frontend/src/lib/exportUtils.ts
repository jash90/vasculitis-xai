import type { BatchResultRow } from '../api/types';

export function exportToCSV(results: BatchResultRow[]): string {
  const headers = ['ID Pacjenta', 'Wiek rozpoznania', 'Liczba narządów', 'Ryzyko (%)', 'Poziom ryzyka', 'Główne czynniki'];
  const rows = results.map((r) => [
    r.patient_id,
    r.wiek_rozpoznania,
    r.liczba_narzadow,
    r.probability_pct,
    r.risk_level_pl,
    r.top_factors,
  ]);

  const csv = [headers.join(','), ...rows.map((row) => row.map((v) => `"${v}"`).join(','))].join('\n');
  return '\uFEFF' + csv; // BOM for UTF-8
}

export function exportToJSON(results: BatchResultRow[]): string {
  const data = {
    analysis_date: new Date().toISOString(),
    total_patients: results.length,
    summary: {
      low_risk: results.filter((r) => r.risk_level === 'low').length,
      moderate_risk: results.filter((r) => r.risk_level === 'moderate').length,
      high_risk: results.filter((r) => r.risk_level === 'high').length,
      avg_probability: results.reduce((a, r) => a + r.probability, 0) / results.length,
    },
    patients: results,
  };
  return JSON.stringify(data, null, 2);
}

export function downloadBlob(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
