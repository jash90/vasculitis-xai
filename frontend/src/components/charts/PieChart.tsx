import Plot from './Plot';
import type { BatchResultRow } from '../../api/types';

export function RiskPieChart({ results }: { results: BatchResultRow[] }) {
  const counts = { low: 0, moderate: 0, high: 0 };
  results.forEach((r) => {
    if (r.risk_level in counts) counts[r.risk_level as keyof typeof counts]++;
  });

  const all = [
    { label: 'Niskie', value: counts.low, color: '#22c55e' },
    { label: 'Umiarkowane', value: counts.moderate, color: '#eab308' },
    { label: 'Wysokie', value: counts.high, color: '#ef4444' },
  ].filter((c) => c.value > 0);
  const labels = all.map((c) => c.label);
  const values = all.map((c) => c.value);
  const colors = all.map((c) => c.color);

  return (
    <Plot
      data={[
        {
          type: 'pie',
          labels,
          values,
          marker: { colors },
          hole: 0.4,
          textinfo: 'percent',
          hovertemplate: '%{label}: %{value} (%{percent})<extra></extra>',
          textfont: { size: 14, color: 'white' },
        },
      ]}
      layout={{
        title: { text: 'Rozkład poziomów ryzyka', font: { size: 18, color: '#ffffff' } },
        font: { color: '#ffffff' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        template: 'plotly_dark' as unknown as undefined,
        height: 350,
        showlegend: true,
        legend: { font: { color: '#ffffff' } },
      }}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      className="w-full"
    />
  );
}
