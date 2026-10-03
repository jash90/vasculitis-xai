import Plot from './Plot';
import { AXIS, COLORS, DARK_LAYOUT, PLOT_CONFIG } from './chartTheme';
import type { SurvivalPoint } from '../../api/types';

/** Cumulative risk of death over time since diagnosis (1 - S(t)). */
export function SurvivalCurveChart({ curve }: { curve: SurvivalPoint[] }) {
  return (
    <Plot
      data={[
        {
          type: 'scatter',
          mode: 'lines',
          line: { shape: 'hv', color: COLORS.risk, width: 3 },
          x: curve.map((p) => p.time_years),
          y: curve.map((p) => (1 - p.survival) * 100),
          fill: 'tozeroy',
          fillcolor: 'rgba(239,68,68,0.12)',
          hovertemplate: '%{x:.1f} r.: %{y:.1f}%<extra></extra>',
        },
      ]}
      layout={{
        ...DARK_LAYOUT,
        xaxis: { ...AXIS, title: { text: 'Lata od rozpoznania' }, dtick: 1 },
        yaxis: { ...AXIS, title: { text: 'Skumulowane ryzyko zgonu (%)' }, rangemode: 'tozero' },
        height: 280,
        margin: { l: 60, r: 20, t: 10, b: 45 },
        shapes: [1, 3, 5].map((x) => ({
          type: 'line', x0: x, x1: x, y0: 0, y1: 1, yref: 'paper', line: { color: '#6b7280', dash: 'dot', width: 1 },
        })),
      }}
      config={PLOT_CONFIG}
      useResizeHandler
      className="w-full"
    />
  );
}
