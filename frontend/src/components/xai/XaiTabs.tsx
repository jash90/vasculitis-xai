import { useState } from 'react';
import { ShapTab } from './ShapTab';
import { LimeTab } from './LimeTab';
import { ComparisonTab } from './ComparisonTab';
import { DalexTab } from './DalexTab';
import { EbmTab } from './EbmTab';
import { ChatTab } from './ChatTab';
import { Tabs } from '../common/Tabs';
import { pl } from '../../i18n/pl';
import type { PatientInput } from '../../api/types';

const TABS = [
  { id: 'shap', label: pl.xai.shap },
  { id: 'lime', label: pl.xai.lime },
  { id: 'dalex', label: pl.xai.dalex },
  { id: 'ebm', label: pl.xai.ebm },
  { id: 'comparison', label: pl.xai.comparison },
] as const;

type TabId = (typeof TABS)[number]['id'];

export function XaiTabs({ patient }: { patient: PatientInput }) {
  const [activeTab, setActiveTab] = useState<TabId>('shap');

  return (
    <div className="space-y-10">
      <section aria-label={pl.xai.label}>
        <div className="border-b border-gray-700">
          <Tabs items={TABS} value={activeTab} onChange={setActiveTab} label={pl.xai.label} idPrefix="xai" size="sm" />
        </div>
        <div id="xai-panel" role="tabpanel" aria-labelledby={`xai-tab-${activeTab}`} className="mt-4">
          {activeTab === 'shap' && <ShapTab patient={patient} />}
          {activeTab === 'lime' && <LimeTab patient={patient} />}
          {activeTab === 'dalex' && <DalexTab patient={patient} />}
          {activeTab === 'ebm' && <EbmTab patient={patient} />}
          {activeTab === 'comparison' && <ComparisonTab patient={patient} />}
        </div>
      </section>

      <hr className="border-gray-700/50" />
      <ChatTab patient={patient} />
    </div>
  );
}
