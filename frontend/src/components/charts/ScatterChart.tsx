import Plot from './Plot';
import type { BatchResultRow, RiskLevel } from '../../api/types';

export function AgeRiskScatter({ results }: { results: BatchResultRow[] }) {
  const colors: Record<RiskLevel, string> = { low: '#28a745', moderate: '#ffc107', high: '#dc3545' };
  const labels: Record<RiskLevel, string> = { low: 'Niskie', moderate: 'Umiarkowane', high: 'Wysokie' };

  const traces = (['low', 'moderate', 'high'] as RiskLevel[]).map((level) => {
    const filtered = results.filter((r) => r.risk_level === level);
    return {
      type: 'scatter' as const,
      mode: 'markers' as const,
      x: filtered.map((r) => r.wiek_rozpoznania),
      y: filtered.map((r) => r.probability * 100),
      marker: { size: 8, color: colors[level], opacity: 0.7 },
      name: labels[level],
      text: filtered.map((r) => r.patient_id),
      hovertemplate: '<b>%{text}</b><br>Wiek w chwili rozpoznania: %{x}<br>Ryzyko: %{y:.1f}%<extra></extra>',
    };
  });

  return (
    <Plot
      data={traces}
      layout={{
        
        xaxis: {
          title: { text: 'Wiek w chwili rozpoznania (lata)', font: { color: '#ffffff' } },
          tickfont: { color: '#ffffff' },
          gridcolor: '#444444',
        },
        yaxis: {
          title: { text: 'Prawdopodobieństwo (%)', font: { color: '#ffffff' } },
          tickfont: { color: '#ffffff' },
          gridcolor: '#444444',
        },
        font: { color: '#ffffff' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        template: 'plotly_dark' as unknown as undefined,
        height: 380,
        margin: { l: 60, r: 20, t: 20, b: 50 },
        legend: { font: { color: '#ffffff' } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      className="w-full"
    />
  );
}
