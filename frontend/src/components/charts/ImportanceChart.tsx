import Plot from './Plot';
import { AXIS, COLORS, DARK_LAYOUT, PLOT_CONFIG } from './chartTheme';
import type { Contribution } from '../../lib/featureLabels';

interface ImportanceChartProps {
  items: Contribution[];
  title?: string;
  xTitle?: string;
  maxItems?: number;
}

/** Unsigned global importance (no direction), largest on top. */
export function ImportanceChart({ items, title = '', xTitle = 'Ważność', maxItems = 12 }: ImportanceChartProps) {
  if (!items.length) return null;
  const top = [...items].sort((a, b) => b.contribution - a.contribution).slice(0, maxItems).reverse();
  return (
    <Plot
      data={[
        {
          type: 'bar',
          orientation: 'h',
          y: top.map((f) => f.feature),
          x: top.map((f) => f.contribution),
          marker: { color: COLORS.neutral },
          text: top.map((f) => f.contribution.toFixed(3)),
          textposition: 'outside',
          cliponaxis: false,
          hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
        },
      ]}
      layout={{
        ...DARK_LAYOUT,
        title: title ? { text: title, font: { size: 15 } } : undefined,
        xaxis: { ...AXIS, title: { text: xTitle } },
        yaxis: { ...AXIS, automargin: true },
        height: Math.max(220, 30 * top.length + 70),
        margin: { l: 10, r: 50, t: title ? 40 : 10, b: 45 },
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      className="w-full"
    />
  );
}
