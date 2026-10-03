export function LoadingSkeleton() {
  return (
    <div role="status" aria-label="Trwa analiza" className="animate-pulse space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <div className="h-60 rounded-lg bg-gray-800" />
        <div className="h-60 rounded-lg bg-gray-800" />
        <div className="h-60 rounded-lg bg-gray-800" />
      </div>
      <div className="h-24 rounded-lg bg-gray-800" />
      <div className="h-64 rounded-lg bg-gray-800" />
    </div>
  );
}
