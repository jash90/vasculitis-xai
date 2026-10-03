import { useDemoMode } from '../../hooks/useApi';

export function DemoModeIndicator() {
  const { data, isError } = useDemoMode();
  const mode = isError ? 'unavailable' : data?.current_mode;
  if (!mode || mode === 'api') return null;
  const isDemo = mode === 'demo';
  return (
    <span role="status" title={isDemo ? 'Model nie jest załadowany — wyniki są poglądowe' : 'Brak połączenia z serwerem API'}
      className={`rounded-full px-3 py-0.5 text-xs font-bold text-white ${isDemo ? 'bg-yellow-600' : 'bg-red-600'}`}>
      {isDemo ? 'Tryb demonstracyjny' : 'API niedostępne'}
    </span>
  );
}
