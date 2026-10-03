import type { ReactNode } from 'react';
import { errorMessage } from '../../lib/errors';

interface QueryStateProps {
  isLoading: boolean;
  error: unknown;
  onRetry: () => void;
  loadingText: string;
  children: ReactNode;
  minHeight?: number;
}

/** One consistent loading / error (with retry) / content state for API-backed panels. */
export function QueryState({ isLoading, error, onRetry, loadingText, children, minHeight = 240 }: QueryStateProps) {
  if (isLoading) {
    return (
      <div role="status" className="flex items-center justify-center gap-3 text-sm text-gray-400" style={{ minHeight }}>
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-gray-500 border-t-blue-400" aria-hidden />
        {loadingText}
      </div>
    );
  }
  if (error) {
    return (
      <div role="alert" className="rounded-lg border border-red-500/40 bg-red-900/15 p-4 text-sm text-red-200">
        <p>{errorMessage(error)}</p>
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-md border border-red-400/50 px-3 py-1 text-xs font-medium text-red-100 hover:bg-red-900/30"
        >
          Spróbuj ponownie
        </button>
      </div>
    );
  }
  return <>{children}</>;
}
