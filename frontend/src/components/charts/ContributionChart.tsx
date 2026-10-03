import Plot from './Plot';
import { AXIS, COLORS, DARK_LAYOUT, PLOT_CONFIG } from './chartTheme';
import type { Contribution } from '../../lib/featureLabels';

interface ContributionChartProps {
  factors: Contribution[];
  title?: string;
  maxItems?: number;
  xTitle?: string;
}

/** Signed per-feature contributions (red raises risk, green lowers it), largest on top. */
export function ContributionChart({ factors, title = '', maxItems = 12, xTitle = 'Wpływ na ryzyko' }: ContributionChartProps) {
  const top = [...factors]
    .filter((f) => Number.isFinite(f.contribution) && Math.abs(f.contribution) >= 5e-4)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, maxItems)
    .reverse();
  if (!top.length) return <p className="text-sm text-gray-500">Brak cech o istotnym wkładzie.</p>;
  const values = top.map((f) => f.contribution);
  const maxAbs = Math.max(...values.map(Math.abs), 1e-6);

  return (
    <Plot
      data={[
        {
          type: 'bar',
          orientation: 'h',
          y: top.map((f) => f.feature),
          x: values,
          marker: { color: values.map((v) => (v > 0 ? COLORS.risk : COLORS.protective)) },
          text: values.map((v) => (v >= 0 ? `+${v.toFixed(3)}` : v.toFixed(3))),
          textposition: 'outside',
          cliponaxis: false,
          hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
        },
      ]}
      layout={{
        ...DARK_LAYOUT,
        title: title ? { text: title, font: { size: 15 } } : undefined,
        xaxis: { ...AXIS, title: { text: xTitle }, range: [-maxAbs * 1.35, maxAbs * 1.35] },
        yaxis: { ...AXIS, automargin: true },
        height: Math.max(220, 34 * top.length + 70),
        margin: { l: 10, r: 20, t: title ? 40 : 10, b: 45 },
        bargap: 0.25,
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      className="w-full"
    />
  );
}
