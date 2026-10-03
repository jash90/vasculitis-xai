import { lazy, Suspense, type ComponentType } from 'react';

// Plotly is ~4 MB: load it on demand, in a separate chunk.
const PlotImpl = lazy(() => import('./PlotImpl')) as unknown as ComponentType<Record<string, unknown>>;

export default function Plot(props: Record<string, unknown>) {
  const layout = props.layout as { height?: number } | undefined;
  return (
    <Suspense
      fallback={
        <div
          className="flex w-full animate-pulse items-center justify-center rounded-lg bg-gray-800/40 text-xs text-gray-500"
          style={{ height: layout?.height ?? 300 }}
        >
          Ładowanie wykresu…
        </div>
      }
    >
      <PlotImpl {...props} />
    </Suspense>
  );
}
