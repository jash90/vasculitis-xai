import Plot from './Plot';
import { DARK_LAYOUT, PLOT_CONFIG } from './chartTheme';

interface GaugeChartProps {
  probability: number;
  title?: string;
  height?: number;
}

/** Risk gauge; bands match the API thresholds (30% / 70%). */
export function GaugeChart({ probability, title = '', height = 220 }: GaugeChartProps) {
  const color = probability < 0.3 ? '#22c55e' : probability < 0.7 ? '#eab308' : '#ef4444';
  return (
    <Plot
      data={[
        {
          type: 'indicator',
          mode: 'gauge+number',
          value: Math.round(probability * 1000) / 10,
          title: title ? { text: title, font: { size: 14 } } : undefined,
          number: { suffix: '%', font: { size: 34, color: '#ffffff' } },
          gauge: {
            axis: { range: [0, 100], tickvals: [0, 30, 70, 100], tickfont: { color: '#d1d5db', size: 11 } },
            bar: { color, thickness: 0.3 },
            bgcolor: '#1f2937',
            borderwidth: 0,
            steps: [
              { range: [0, 30], color: 'rgba(34,197,94,0.18)' },
              { range: [30, 70], color: 'rgba(234,179,8,0.18)' },
              { range: [70, 100], color: 'rgba(239,68,68,0.18)' },
            ],
          },
        },
      ]}
      layout={{ ...DARK_LAYOUT, height, margin: { l: 30, r: 30, t: title ? 40 : 20, b: 10 } }}
      config={PLOT_CONFIG}
      useResizeHandler
      className="w-full"
    />
  );
}
