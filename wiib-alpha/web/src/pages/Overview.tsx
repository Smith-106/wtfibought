import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Eye, History, Library, Lock, RefreshCw } from 'lucide-react';
import { alpha, CANDIDATE_STATUSES, type Overview, type StatusCounts } from '../api';
import { localTime } from '../format';
import { useFail } from '../toast';
import { RunBadge, StatusBadge } from '../components/FactorCard';
import { Button, Card, Empty, Spinner, Stat } from '../components/ui';

/** 挖掘页顶上的三个数：只读和管理员都有 */
export function OverviewStats({ trials, factors, observing }: { trials: number; factors: number; observing: number }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap gap-x-12 gap-y-3 border border-line bg-panel px-4 py-3">
      <Stat label={t('overview.trials')} hint={t('overview.trialsHint', { fp: (trials * 0.05).toFixed(1) })}>{trials}</Stat>
      <Stat label={t('overview.observing')}>{observing}</Stat>
      <Stat label={t('overview.factors')}>{factors}</Stat>
    </div>
  );
}

/** 一轮各状态几条，按流程先后排 */
export function StatusCountBadges({ counts }: { counts: StatusCounts }) {
  return (
    <div className="flex flex-wrap gap-1">
      {CANDIDATE_STATUSES.filter(s => counts[s]).map(s => <StatusBadge key={s} status={s} count={counts[s]} />)}
    </div>
  );
}

function AdminOnly() {
  const { t } = useTranslation();
  return (
    <p className="flex items-center gap-1.5 text-[11.5px] text-muted">
      <Lock className="w-3 h-3 inline-flex" />{t('overview.adminOnly')}
    </p>
  );
}

function useOverview(active: boolean) {
  const fail = useFail();
  const [overview, setOverview] = useState<Overview | null>(null);
  const load = useCallback(() => alpha.overview().then(setOverview).catch(fail), [fail]);
  useEffect(() => {
    if (active) void load();
  }, [active, load]);
  return { overview, load };
}

/** 只读的挖掘页：只有数字，每轮各状态候选几条 */
export function GuestMine({ active }: { active: boolean }) {
  const { t } = useTranslation();
  const { overview, load } = useOverview(active);
  if (!overview) return <Spinner />;
  return (
    <div className="space-y-4">
      <OverviewStats trials={overview.trials} factors={overview.factors} observing={overview.observing} />
      <Card title={t('run.list')} icon={History}
            actions={<Button variant="ghost" icon={RefreshCw} onClick={() => void load()} className="px-2" />}>
        {overview.runs.length === 0 ? <Empty>{t('run.noRun')}</Empty> : (
          <div className="overflow-x-auto -mx-4">
            <table className="tbl">
              <thead>
                <tr>
                  <th>{t('run.col.time')}</th>
                  <th>{t('run.col.status')}</th>
                  <th className="r">{t('run.col.k')}</th>
                  <th>{t('overview.candidates')}</th>
                </tr>
              </thead>
              <tbody>
                {overview.runs.map(r => (
                  <tr key={r.id}>
                    <td className="num whitespace-nowrap" title={r.id}>{localTime(r.created_at)}</td>
                    <td><RunBadge status={r.status} /></td>
                    <td className="r num">{r.k}</td>
                    <td><StatusCountBadges counts={r.counts} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="mt-3"><AdminOnly /></div>
      </Card>
    </div>
  );
}

/** 只读的因子库页：只有因子数 */
export function GuestLibrary({ active }: { active: boolean }) {
  const { t } = useTranslation();
  const { overview } = useOverview(active);
  return (
    <Card title={t('library.title')} icon={Library}>
      {overview ? (
        <div className="space-y-3">
          <Stat label={t('overview.factors')}>{overview.factors}</Stat>
          <AdminOnly />
        </div>
      ) : <Spinner />}
    </Card>
  );
}

/** 只读的观察区页：只有条数 */
export function GuestObservation({ active }: { active: boolean }) {
  const { t } = useTranslation();
  const { overview } = useOverview(active);
  return (
    <Card title={t('observe.title')} icon={Eye}>
      {overview ? (
        <div className="space-y-3">
          <Stat label={t('overview.observing')}>{overview.observing}</Stat>
          <AdminOnly />
        </div>
      ) : <Spinner />}
    </Card>
  );
}
