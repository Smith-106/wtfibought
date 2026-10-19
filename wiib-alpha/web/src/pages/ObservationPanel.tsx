import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Archive, Eye, RefreshCw, ScanSearch, Trash2, TriangleAlert } from 'lucide-react';
import { alpha, type ObservationItem, type Reeval, type ReevalJob } from '../api';
import { cx, localTime, num } from '../format';
import { usePoll } from '../poll';
import { useFail, useToast } from '../toast';
import { BriefTable, CandidateCard, DirectionLine, GateList, PassBadge } from '../components/FactorCard';
import { Button, Card, Empty, Progress, Spinner } from '../components/ui';

// G6 的阈值：截面 |ρ| 到这个数就算同一个信号
const REDUNDANT = 0.7;

/** 管理员的观察区：过了全部门、评审和同批 BH 的候选，逐条入库或丢弃；可以用当前评估器整批复评 */
export function ObservationPanel({ active }: { active: boolean }) {
  const { t } = useTranslation();
  const toast = useToast();
  const fail = useFail();
  const [items, setItems] = useState<ObservationItem[] | null>(null);
  const [job, setJob] = useState<ReevalJob | null>(null);
  const [starting, setStarting] = useState(false);
  // 正在处理的那一条与动作：按钮转圈，别的都禁掉
  const [busy, setBusy] = useState<{ id: string; approve: boolean } | null>(null);

  const load = useCallback(() => alpha.observation().then(r => {
    setItems(r.items);
    setJob(r.reeval);
  }).catch(fail), [fail]);

  useEffect(() => {
    if (active) void load();
  }, [active, load]);

  const running = job?.running ?? false;
  usePoll(active && running, load, 3000);

  const reeval = async () => {
    setStarting(true);
    try {
      await alpha.reeval();
      await load();
    } catch (err) {
      fail(err);
    } finally {
      setStarting(false);
    }
  };

  const decide = async (c: ObservationItem, approve: boolean) => {
    setBusy({ id: c.id, approve });
    try {
      if (approve) {
        const { name } = await alpha.approve(c.id);
        toast(t('observe.approved', { name }), 'success');
      } else {
        await alpha.decline(c.id);
        toast(t('observe.declined', { name: c.name }), 'success');
      }
      setItems(list => list && list.filter(x => x.id !== c.id));
    } catch (err) {
      fail(err);
      // 409 是别的标签页已经处理过了，重拉对齐
      void load();
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-4">
      <Card title={t('observe.title')} icon={Eye} hint={t('observe.hint')}
            actions={<Button variant="ghost" icon={RefreshCw} onClick={() => void load()} className="px-2" />}>
        {!items ? <Spinner /> : items.length === 0
          ? <Empty>{t('observe.empty')}</Empty>
          : (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-xs text-muted">{t('observe.count', { n: items.length })}</p>
                <Button icon={ScanSearch} loading={starting || running} disabled={busy !== null}
                        onClick={() => void reeval()} className="ml-auto">
                  {running ? t('observe.reevalRunning') : t('observe.reeval')}
                </Button>
              </div>
              <p className="text-[11px] text-muted">{t('observe.reevalHint')}</p>
              {job && <ReevalProgress job={job} />}
            </div>
          )}
      </Card>
      {items?.map(c => (
        <CandidateCard key={c.id} c={c} meta={t('observe.from', { run: c.run_id, time: localTime(c.created_at) })}>
          {c.reeval && <ReevalBlock r={c.reeval} />}
          <div className="flex items-center gap-2 border-t border-line pt-2.5">
            <Button variant="gain" icon={Archive} loading={busy?.id === c.id && busy.approve}
                    disabled={busy !== null || running} onClick={() => void decide(c, true)}>
              {t('observe.approve')}
            </Button>
            <Button icon={Trash2} loading={busy?.id === c.id && !busy.approve} disabled={busy !== null || running}
                    onClick={() => void decide(c, false)}>
              {t('observe.decline')}
            </Button>
          </div>
        </CandidateCard>
      ))}
    </div>
  );
}

function ReevalProgress({ job }: { job: ReevalJob }) {
  const { t } = useTranslation();
  if (!job.startedAt) return null;
  return (
    <div className="space-y-1.5 border-t border-line pt-2">
      {job.running && (
        <>
          <div className="flex items-center gap-3 text-xs">
            <span className="num">{t('observe.reevalProgress', { done: job.done ?? 0, total: job.total ?? 0 })}</span>
            {job.current && <span className="text-muted break-all">{job.current}</span>}
          </div>
          <Progress done={job.done ?? 0} total={job.total ?? 0} />
        </>
      )}
      {!job.running && job.finishedAt && (
        <p className="text-[11.5px] text-muted">{t('observe.reevalFinished', { time: localTime(job.finishedAt) })}</p>
      )}
      {job.error && (
        <p className="flex items-start gap-1.5 text-xs text-loss break-words">
          <TriangleAlert className="w-3.5 h-3.5 shrink-0 mt-0.5" />{t('observe.reevalError', { msg: job.error })}
        </p>
      )}
    </div>
  );
}

/** 一条候选的复评：声明的方向、结论、当前门与数字、和观察区其它候选的截面相关 */
function ReevalBlock({ r }: { r: Reeval }) {
  const { t } = useTranslation();
  const peers = [...r.peers].sort((a, b) => b[1] - a[1]);
  return (
    <section className="border-t border-line pt-2.5 space-y-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="microlabel">{t('observe.reevalTitle')}</span>
        <PassBadge passed={r.passed} />
        <span className="ml-auto text-[11px] text-muted num">{localTime(r.at)}</span>
      </div>
      <DirectionLine direction={r.direction} asked={r.asked} />
      {r.reason && <p className="text-[11.5px] text-loss leading-relaxed">{t('cand.reason')}：{r.reason}</p>}
      <div className="grid gap-3 xl:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <GateList gates={r.gates} />
        <BriefTable brief={r.brief} />
      </div>
      {peers.length > 0 && (
        <p className="flex flex-wrap gap-x-3 gap-y-0.5 px-2 text-[11px] text-muted num">
          <span>{t('observe.peers')}</span>
          {peers.map(([name, v], i) => (
            <span key={i} className="break-all">
              {name} <span className={cx(v >= REDUNDANT ? 'text-loss' : 'text-fg')}>{num(v, 2)}</span>
            </span>
          ))}
        </p>
      )}
    </section>
  );
}
