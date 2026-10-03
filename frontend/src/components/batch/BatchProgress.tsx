export function BatchProgress({ progress }: { progress: number }) {
  const pct = Math.round(progress * 100);
  return (
    <div className="space-y-4" role="status">
      <h3 className="text-lg font-semibold text-gray-200">Przetwarzanie pacjentów…</h3>
      <div className="h-4 overflow-hidden rounded-full bg-gray-700" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div
          className="h-full rounded-full bg-blue-500 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-sm text-gray-400">{pct}% ukończone</p>
    </div>
  );
}
