/** Shared Plotly layout pieces for the dark theme. */
export const DARK_LAYOUT = {
  font: { color: '#e5e7eb', size: 12 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
} as const;

export const AXIS = { tickfont: { color: '#d1d5db' }, gridcolor: '#374151', zerolinecolor: '#9ca3af' } as const;

export const PLOT_CONFIG = { displayModeBar: false, responsive: true } as const;

export const COLORS = { risk: '#ef4444', protective: '#22c55e', neutral: '#3b82f6' } as const;
