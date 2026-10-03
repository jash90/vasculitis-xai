import type { ComponentType } from 'react';
import factoryModule from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';

// Handle CJS/ESM interop: factory may be { default: fn } or fn directly
const createPlotlyComponent =
  typeof factoryModule === 'function'
    ? factoryModule
    : (factoryModule as { default: typeof factoryModule }).default;

const Plot = createPlotlyComponent(Plotly as object) as unknown as ComponentType<Record<string, unknown>>;
export default Plot;
