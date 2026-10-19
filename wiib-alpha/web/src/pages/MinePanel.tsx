import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Ban, ChevronDown, ChevronRight, CircleStop, History, Lightbulb, Pickaxe, Play, Radio, RefreshCw, ScrollText,
  StickyNote, Repeat, type LucideIcon,
} from 'lucide-react';
import {
  alpha, type Campaign, type CampaignDetail, type Candidate, type LessonKind, type Run, type RunEvent, type RunRow,
  type Stage,
} from '../api';
import { cx, localTime } from '../format';
import { usePoll } from '../poll';
import { useFail, useToast } from '../toast';
import { CampaignBadge, CandidateCard, RunBadge } from '../components/FactorCard';
import { ModelSettings } from '../components/ModelSettings';
import { Button, Card, Empty, Progress, Spinner } from '../components/ui';
import { OverviewStats, StatusCountBadges } from './Overview';

/** 当前看的那一轮：刷新页接着看，关标签页就算了 */
const RUN_KEY = 'wiib-alpha-run';
const MAX_LOG = 500;
const MAX_K = 10;
const MAX_ROUNDS = 50;
// 列表里不属于 campaign 的老轮次那一组
const SOLO = 'solo';

type LogLine =
  | { kind: 'log'; text: string }
  | { kind: 'lessons'; n: number }
  | { kind: 'done'; observing: number }
  | { kind: 'error'; message: string };

const LESSON_ICON: Record<LessonKind, { icon: LucideIcon; tone: string }> = {
  avoid: { icon: Ban, tone: 'text-loss' },
  try: { icon: Lightbulb, tone: 'text-gain' },
  note: { icon: StickyNote, tone: 'text-muted' },
};

const byId = (list: Candidate[]) => Object.fromEntries(list.map(c => [c.id, c]));

/** 还在跑：running 或点了停、这一轮还没跑完 */
const live = (c: Campaign) => c.status === 'running' || c.status === 'stopping';

/** 正整数；空着是 null（不限） */
const optionalInt = (v: string) => (v.trim() ? Math.max(1, Math.round(Number(v)) || 1) : null);

/** 管理员的挖掘页：开始 campaign、进行中的进度与停止、campaign 与轮次、经验、事件流与候选 */
export function MinePanel({ active, loaded }: { active: boolean; loaded: boolean }) {
  const { t } = useTranslation();
  const fail = useFail();

  const [runs, setRuns] = useState<RunRow[]>([]);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [stats, setStats] = useState({ trials: 0, factors: 0, observing: 0 });
  // 两个角色的模型都配了才能开始
  const [modelsReady, setModelsReady] = useState(false);

  // 在跑的那个 campaign 的详情，每 3 秒拉一次；另外点开看的旧 campaign 拉一次就够
  const [liveDetail, setLiveDetail] = useState<CampaignDetail | null>(null);
  const [pickedDetail, setPickedDetail] = useState<CampaignDetail | null>(null);
  // 列表里展开的那一组：campaign id、SOLO 或 ''（全收起）；null 是还没点过
  const [picked, setPicked] = useState<string | null>(null);

  const [run, setRun] = useState<Run | null>(null);
  const [cands, setCands] = useState<Record<string, Candidate>>({});
  const [logs, setLogs] = useState<LogLine[]>([]);
  // 最近一条 status 事件的阶段
  const [stage, setStage] = useState<Stage | null>(null);
  const [streaming, setStreaming] = useState(false);
  // 一轮一个 controller，管快照请求和事件流；换一轮或卸载就 abort
  const abortRef = useRef<AbortController | null>(null);
  // 收到的最后一帧 id：断线重连从这里接着收
  const lastIdRef = useRef(0);
  const logRef = useRef<HTMLDivElement>(null);
  // 在跑的 campaign 上一次看到的最新一轮：正看着它的时候换了轮就跟过去
  const latestRef = useRef<string | null>(null);

  // 列表连同顶上的数一起拉：一轮结束后试验数、观察区条数会变
  const loadLists = useCallback(() => Promise.all([alpha.runs(), alpha.campaigns(), alpha.overview()])
    .then(([r, c, o]) => {
      setRuns(r.runs);
      setCampaigns(c.campaigns);
      setStats({ trials: o.trials, factors: o.factors, observing: o.observing });
    })
    .catch(fail), [fail]);

  const running = campaigns.find(live);
  const runningId = running?.id;

  const loadLive = useCallback(() => (runningId ? alpha.campaign(runningId)
    .then(d => {
      setLiveDetail(d);
      // 停下来了：列表里的状态、顶上的数都刷一遍
      if (!live(d.campaign)) void loadLists();
    })
    .catch(fail) : Promise.resolve()), [runningId, loadLists, fail]);

  useEffect(() => {
    void loadLive();
  }, [loadLive]);
  usePoll(active && !!runningId, loadLive, 3000);
  // 只认还在跑的那个的轮询结果；停了以后列表和详情都重新拉，轮询的那份可能停在最后一轮之前
  const liveNow = liveDetail && liveDetail.campaign.id === runningId ? liveDetail : null;

  // 展开的那一组：点了哪个看哪个，没点过就看在跑的，再没有就看最新的
  const shownId = picked ?? runningId ?? campaigns[0]?.id ?? '';
  useEffect(() => {
    if (!shownId || shownId === SOLO || shownId === runningId) return;
    alpha.campaign(shownId).then(setPickedDetail).catch(fail);
  }, [shownId, runningId, fail]);
  const shown = shownId === runningId ? liveNow : pickedDetail?.campaign.id === shownId ? pickedDetail : null;

  const onEvent = useCallback((e: RunEvent) => {
    lastIdRef.current = e.id;
    switch (e.event) {
      case 'status':
        setStage(e.data.stage);
        break;
      case 'log': {
        const line: LogLine = { kind: 'log', text: e.data.text };
        setLogs(l => [...l.slice(1 - MAX_LOG), line]);
        break;
      }
      case 'lessons': {
        const line: LogLine = { kind: 'lessons', n: e.data.items.length };
        setLogs(l => [...l, line]);
        break;
      }
      case 'candidate': {
        const c = e.data;
        setCands(m => ({ ...m, [c.id]: c }));
        break;
      }
      case 'done': {
        const line: LogLine = { kind: 'done', observing: e.data.observing };
        setLogs(l => [...l, line]);
        break;
      }
      case 'error': {
        const line: LogLine = { kind: 'error', message: e.data.message };
        setLogs(l => [...l, line]);
        break;
      }
      case 'end': {
        const status = e.data.status;
        setRun(r => (r ? { ...r, status } : r));
        void loadLists();
        break;
      }
    }
  }, [loadLists]);

  const listen = useCallback((id: string, after: number, ctrl: AbortController) => {
    setStreaming(true);
    alpha.events(id, after, onEvent, ctrl.signal)
      .catch(err => { if (!ctrl.signal.aborted) fail(err); })
      .finally(() => { if (!ctrl.signal.aborted) setStreaming(false); });
  }, [onEvent, fail]);

  const renew = useCallback(() => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    return ctrl;
  }, []);

  /** 打开一轮：先拿快照，再从头回放事件；不在跑的轮次回放完就以 end 结束 */
  const openRun = useCallback((id: string) => {
    const ctrl = renew();
    alpha.run(id, ctrl.signal)
      .then(s => {
        setRun(s.run);
        setStats(v => ({ ...v, trials: s.trials }));
        setCands(byId(s.candidates));
        setLogs([]);
        setStage(null);
        lastIdRef.current = 0;
        sessionStorage.setItem(RUN_KEY, id);
        listen(id, 0, ctrl);
      })
      .catch(err => {
        if (ctrl.signal.aborted) return;
        sessionStorage.removeItem(RUN_KEY);
        setStreaming(false);
        fail(err);
      });
  }, [renew, listen, fail]);

  useEffect(() => {
    const saved = sessionStorage.getItem(RUN_KEY);
    if (saved) openRun(saved);
    return () => abortRef.current?.abort();
  }, [openRun]);

  useEffect(() => {
    if (active) void loadLists();
  }, [active, loadLists]);

  // campaign 起了下一轮：正看着上一轮就跟到新的一轮
  const latest = liveNow?.runs[0]?.id ?? null;
  const openId = run?.id;
  useEffect(() => {
    if (latest && latestRef.current && latest !== latestRef.current && openId === latestRef.current) openRun(latest);
    latestRef.current = latest;
  }, [latest, openId, openRun]);

  useEffect(() => {
    logRef.current!.scrollTop = logRef.current!.scrollHeight;
  }, [logs]);

  // 列表按 campaign 分组；在跑的那个用轮询来的，最新
  const groups = useMemo(() => {
    const out = new Map<string | null, RunRow[]>();
    for (const r of runs) out.set(r.campaign_id, [...(out.get(r.campaign_id) ?? []), r]);
    if (liveNow) out.set(liveNow.campaign.id, liveNow.runs);
    return out;
  }, [runs, liveNow]);

  const ordered = useMemo(() => Object.values(cands).sort((a, b) => a.seq - b.seq), [cands]);

  // 展示用的阶段：不在跑看轮次状态，在跑再看事件推到哪了
  const shownStage = !run ? null
    : run.status !== 'running' ? run.status
    : stage && stage !== 'proposing' ? stage
    : ordered.some(c => c.status !== 'proposed') ? 'working' : 'proposing';

  const reconnect = () => {
    if (run) listen(run.id, lastIdRef.current, renew());
  };

  const started = (runId: string) => {
    setPicked(null);
    void loadLists();
    openRun(runId);
  };

  const toggle = (key: string) => setPicked(shownId === key ? '' : key);

  return (
    <div className="grid gap-4 lg:grid-cols-[22rem_minmax(0,1fr)] items-start">
      <div className="lg:col-span-2"><OverviewStats {...stats} /></div>
      <div className="space-y-4">
        <ModelSettings onReady={setModelsReady} />

        {running
          ? <LiveCampaign campaign={liveNow?.campaign ?? running}
                          runs={groups.get(running.id) ?? []} openId={run?.id} onOpen={openRun} onStopped={loadLive} />
          : <StartCampaign ready={loaded && modelsReady} loaded={loaded} onStarted={started} />}

        <Card title={t('campaign.list')} icon={History}
              actions={<Button variant="ghost" icon={RefreshCw} onClick={() => void loadLists()} className="px-2" />}>
          {campaigns.length === 0 && !groups.get(null) ? <Empty>{t('campaign.none')}</Empty> : (
            <div className="max-h-[26rem] overflow-y-auto -mx-4 divide-y divide-line border-y border-line">
              {campaigns.map(c => (
                <CampaignGroup key={c.id} campaign={c.id === liveNow?.campaign.id ? liveNow.campaign : c}
                               runs={groups.get(c.id) ?? []} open={c.id === shownId} openId={run?.id}
                               onToggle={() => toggle(c.id)} onOpen={openRun} />
              ))}
              {groups.get(null) && (
                <CampaignGroup runs={groups.get(null)!} open={shownId === SOLO} openId={run?.id}
                               onToggle={() => toggle(SOLO)} onOpen={openRun} />
              )}
            </div>
          )}
        </Card>

        <Lessons detail={shown} />

        <Card title={t('detail.log')} icon={ScrollText}>
          <div ref={logRef} className="max-h-80 overflow-y-auto space-y-1 text-[11.5px] leading-relaxed">
            {logs.length === 0 && <Empty>{t('detail.logEmpty')}</Empty>}
            {logs.map((line, i) => (
              <p key={i} className={cx(line.kind === 'done' && 'text-gain', line.kind === 'error' && 'text-loss',
                line.kind === 'lessons' && 'text-info')}>
                {line.kind === 'log' ? line.text
                  : line.kind === 'lessons' ? t('detail.lessons', { n: line.n })
                  : line.kind === 'done' ? t('detail.done', { n: line.observing })
                  : t('detail.error', { msg: line.message })}
              </p>
            ))}
          </div>
        </Card>
      </div>

      <div className="space-y-4 min-w-0">
        {!run ? (
          <Card title={t('cand.title')} icon={Pickaxe}><Empty>{t('detail.pick')}</Empty></Card>
        ) : (
          <>
            <Card>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                <span className="num text-sm font-bold">{run.id}</span>
                <RunBadge status={run.status} />
                {shownStage && <span className="text-xs text-muted">{t(`stage.${shownStage}`)}</span>}
                {streaming && (
                  <span className="inline-flex items-center gap-1 text-[11px] text-info">
                    <Radio className="w-3 h-3 inline-flex animate-pulse" />{t('detail.live')}
                  </span>
                )}
                {!streaming && run.status === 'running' && (
                  <Button icon={RefreshCw} onClick={reconnect}>{t('detail.reconnect')}</Button>
                )}
                <span className="ml-auto text-[11px] text-muted num">
                  {localTime(run.created_at)} · K={run.k}
                </span>
              </div>
              {run.note && <p className="mt-2 text-xs text-muted">{t('run.col.note')}：{run.note}</p>}
              {run.error && <p className="mt-2 text-xs text-loss break-words">{run.error}</p>}
            </Card>

            {ordered.length === 0 ? (
              <Card title={t('cand.title')} icon={Pickaxe}>
                {streaming ? <Spinner /> : <Empty>{t('cand.empty')}</Empty>}
              </Card>
            ) : ordered.map(c => <CandidateCard key={c.id} c={c} />)}
          </>
        )}
      </div>
    </div>
  );
}

/** 没有在跑的 campaign：轮数、每轮条数、方向，可选幸存者上限与时长 */
function StartCampaign({ ready, loaded, onStarted }: { ready: boolean; loaded: boolean; onStarted: (runId: string) => void }) {
  const { t } = useTranslation();
  const toast = useToast();
  const fail = useFail();
  const [rounds, setRounds] = useState(10);
  const [k, setK] = useState(5);
  const [note, setNote] = useState('');
  // 两个可空的上限按字符串存，空着就是不限
  const [maxSurvivors, setMaxSurvivors] = useState('');
  const [minutes, setMinutes] = useState('');
  const [creating, setCreating] = useState(false);

  const start = async () => {
    setCreating(true);
    try {
      const { id, runId } = await alpha.createCampaign({
        rounds, k, note: note.trim(), maxSurvivors: optionalInt(maxSurvivors), deadlineMinutes: optionalInt(minutes),
      });
      toast(t('campaign.created', { id }), 'success');
      setNote('');
      onStarted(runId);
    } catch (err) {
      fail(err);
    } finally {
      setCreating(false);
    }
  };

  const intInput = (value: number, set: (v: number) => void, max: number) => (
    <input type="number" min={1} max={max} value={value} className="input w-full num"
           onChange={e => set(Math.min(max, Math.max(1, Math.round(Number(e.target.value)) || 1)))} />
  );

  return (
    <Card title={t('campaign.title')} icon={Play} hint={t('campaign.hint')}>
      <div className="space-y-2.5">
        <div className="grid grid-cols-2 gap-2">
          <label className="space-y-1">
            <span className="block text-[11px] text-muted">{t('campaign.rounds')}</span>
            {intInput(rounds, setRounds, MAX_ROUNDS)}
          </label>
          <label className="space-y-1">
            <span className="block text-[11px] text-muted">{t('campaign.k')}</span>
            {intInput(k, setK, MAX_K)}
          </label>
          <label className="space-y-1">
            <span className="block text-[11px] text-muted">{t('campaign.maxSurvivors')}</span>
            <input type="number" min={1} value={maxSurvivors} onChange={e => setMaxSurvivors(e.target.value)}
                   placeholder={t('campaign.unlimited')} className="input w-full num" />
          </label>
          <label className="space-y-1">
            <span className="block text-[11px] text-muted">{t('campaign.minutes')}</span>
            <input type="number" min={1} value={minutes} onChange={e => setMinutes(e.target.value)}
                   placeholder={t('campaign.unlimited')} className="input w-full num" />
          </label>
        </div>
        <label className="block space-y-1">
          <span className="block text-[11px] text-muted">{t('campaign.note')}</span>
          <input value={note} onChange={e => setNote(e.target.value)} placeholder={t('campaign.notePh')}
                 className="input w-full" />
        </label>
        <Button variant="primary" icon={Play} loading={creating} disabled={!ready} onClick={() => void start()}
                className="w-full">
          {t('campaign.start')}
        </Button>
        {!loaded && <p className="text-[11px] text-warn">{t('common.needData')}</p>}
        {loaded && !ready && <p className="text-[11px] text-warn">{t('models.needBoth')}</p>}
      </div>
    </Card>
  );
}

/** 在跑的 campaign：第几轮、幸存者、每轮各状态条数、跑完这轮停 */
function LiveCampaign({ campaign: c, runs, openId, onOpen, onStopped }: {
  campaign: Campaign;
  runs: RunRow[];
  openId?: string;
  onOpen: (id: string) => void;
  onStopped: () => Promise<unknown>;
}) {
  const { t } = useTranslation();
  const fail = useFail();
  const [stopping, setStopping] = useState(false);

  const stop = async () => {
    setStopping(true);
    try {
      await alpha.stopCampaign(c.id);
      await onStopped();
    } catch (err) {
      fail(err);
    } finally {
      setStopping(false);
    }
  };

  // 正在跑的这一轮也算进去
  const current = Math.min(c.rounds, c.done_rounds + 1);
  return (
    <Card title={t('campaign.running')} icon={Repeat} actions={<CampaignBadge status={c.status} />}>
      <div className="space-y-2.5">
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
          <span className="text-sm font-bold num">{t('campaign.round', { n: current, total: c.rounds })}</span>
          <span className="text-xs num">
            {c.max_survivors != null ? t('campaign.survivorsOf', { n: c.survivors, max: c.max_survivors })
              : t('campaign.survivors', { n: c.survivors })}
          </span>
          {c.deadline_ms != null && (
            <span className="text-[11px] text-muted num">{t('campaign.deadlineAt', { time: localTime(c.deadline_ms) })}</span>
          )}
        </div>
        <Progress done={c.done_rounds} total={c.rounds} />
        <p className="text-[11px] text-muted num">K={c.k}{c.note && ` · ${c.note}`}</p>
        <RoundList runs={runs} openId={openId} onOpen={onOpen} />
        {c.status === 'stopping'
          ? <p className="text-[11px] text-warn">{t('campaign.stopping')}</p>
          : (
            <Button icon={CircleStop} loading={stopping} onClick={() => void stop()} className="w-full">
              {t('campaign.stop')}
            </Button>
          )}
      </div>
    </Card>
  );
}

/** 一个 campaign 的各轮，最新在前；轮号从 1 数 */
function RoundList({ runs, openId, onOpen }: { runs: RunRow[]; openId?: string; onOpen: (id: string) => void }) {
  const { t } = useTranslation();
  return (
    <ul className="max-h-72 overflow-y-auto divide-y divide-line border border-line">
      {runs.map((r, i) => (
        <li key={r.id} onClick={() => onOpen(r.id)}
            className={cx('flex flex-wrap items-center gap-x-2 gap-y-1 px-2.5 py-1.5 cursor-pointer hover:bg-panel-2',
              openId === r.id && 'bg-primary/8')}>
          <span className="text-[11px] font-semibold num w-12 shrink-0">{t('campaign.roundNo', { n: runs.length - i })}</span>
          <RunBadge status={r.status} />
          <span className="text-[11px] text-muted num" title={r.id}>{localTime(r.created_at)}</span>
          <div className="basis-full"><StatusCountBadges counts={r.counts} /></div>
        </li>
      ))}
    </ul>
  );
}

/** 列表里的一个 campaign：一行摘要，展开是它的各轮；campaign 不给是不属于 campaign 的老轮次 */
function CampaignGroup({ campaign: c, runs, open, openId, onToggle, onOpen }: {
  campaign?: Campaign;
  runs: RunRow[];
  open: boolean;
  openId?: string;
  onToggle: () => void;
  onOpen: (id: string) => void;
}) {
  const { t } = useTranslation();
  const Chevron = open ? ChevronDown : ChevronRight;
  return (
    <div>
      <button onClick={onToggle}
              className={cx('flex w-full flex-wrap items-center gap-x-2 gap-y-1 px-4 py-2 text-left cursor-pointer hover:bg-panel-2',
                open && 'bg-panel-2')}>
        <Chevron className="w-3 h-3 inline-flex shrink-0 text-muted" />
        {c ? (
          <>
            <span className="text-[11.5px] num whitespace-nowrap" title={c.id}>{localTime(c.created_at)}</span>
            <CampaignBadge status={c.status} />
            <span className="text-[11px] text-muted num">
              {t('campaign.progress', { done: c.done_rounds, total: c.rounds })} · {t('campaign.survivors', { n: c.survivors })}
            </span>
            {c.note && <span className="basis-full truncate pl-5 text-[11px] text-muted" title={c.note}>{c.note}</span>}
          </>
        ) : <span className="text-[11.5px] text-muted">{t('campaign.solo', { n: runs.length })}</span>}
      </button>
      {open && (
        <div className="space-y-1.5 px-4 pb-3">
          {c?.stop_reason && <p className="text-[11px] text-muted">{t('campaign.stopReason', { reason: c.stop_reason })}</p>}
          {c?.error && <p className="text-[11px] text-loss break-words">{c.error}</p>}
          {runs.length === 0 ? <Empty>{t('run.noRun')}</Empty> : <RoundList runs={runs} openId={openId} onOpen={onOpen} />}
        </div>
      )}
    </div>
  );
}

/** 展开的 campaign 写下的经验，最新在前 */
function Lessons({ detail }: { detail: CampaignDetail | null }) {
  const { t } = useTranslation();
  // 经验是哪一轮写的：轮号从 1 数
  const roundOf = useMemo(() => new Map(detail?.runs.map((r, i) => [r.id, detail.runs.length - i])), [detail]);
  return (
    <Card title={t('lessons.title')} icon={Lightbulb} hint={t('lessons.hint')}>
      {!detail || detail.lessons.length === 0 ? <Empty>{t('lessons.empty')}</Empty> : (
        <ol className="max-h-[22rem] overflow-y-auto space-y-2">
          {detail.lessons.map(x => {
            const { icon: Icon, tone } = LESSON_ICON[x.kind];
            return (
              <li key={x.id} className="flex items-start gap-2 text-[11.5px] leading-relaxed">
                <Icon className={cx('w-3 h-3 inline-flex shrink-0 mt-[3px]', tone)} aria-label={t(`lessons.kind.${x.kind}`)} />
                <div className="min-w-0">
                  <p className="break-words">{x.text}</p>
                  <p className="text-[10.5px] text-muted num">
                    {t(`lessons.kind.${x.kind}`)} · {t('lessons.meta', { n: roundOf.get(x.run_id) ?? '—', time: localTime(x.created_at) })}
                  </p>
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </Card>
  );
}
