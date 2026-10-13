import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Check, ChevronDown, ChevronLeft, ChevronUp, FlaskConical, Loader2, Play, RefreshCw, X } from 'lucide-react';
import { alphaApi } from '../api';
import { Badge } from '../components/ui/badge';
import { useToast } from '../components/ui/use-toast';
import { cn } from '../lib/utils';
import { useUserStore } from '../stores/userStore';
import type {
  AlphaBrief, AlphaBriefCore, AlphaCandidate, AlphaCandidateStatus, AlphaCombo, AlphaDataStatus, AlphaEvalResult,
  AlphaEvent, AlphaFactor, AlphaGate, AlphaHoldoutHorizon, AlphaReference, AlphaRun,
} from '../types';

/** 当前看的那一轮存这儿：刷新页接着看，关标签页就算了 */
const RUN_KEY = 'wiib-alpha-run';
const MAX_LOG = 200;
const HORIZONS = ['6h', '12h', '24h'] as const;

const BTN = 'border border-border hover:bg-surface-hover rounded-lg px-3 py-1.5 text-[11px] font-bold inline-flex items-center gap-1.5 disabled:opacity-50';
const INPUT = 'h-9 rounded-lg border border-border bg-card-2 px-2.5 text-xs';

const fmt = (v: number | null | undefined, digits = 4) => (v == null ? '—' : v.toFixed(digits));
const day = (ms: number | null | undefined) => (ms == null ? '—' : new Date(ms).toLocaleDateString());
const when = (ms: number) => new Date(ms).toLocaleString();

const STATUS_VARIANT: Record<AlphaCandidateStatus, 'gain' | 'loss' | 'outline' | 'secondary' | 'warning'> = {
  proposed: 'secondary', evaluating: 'secondary', reviewing: 'secondary', revise: 'warning',
  passed: 'gain', approved: 'gain',
  rejected_gates: 'loss', rejected_review: 'loss', rejected_fdr: 'loss', failed: 'loss',
  declined: 'outline',
};

/**
 * 因子挖掘管理页：刷数据 → 开一轮看日志与候选 → 过了门的逐个入库或否决 → 因子库与入库后复查。
 * 后端是 py 服务 wiib-alpha（/api/alpha），一轮的过程走订阅型 SSE，其余走普通 JSON 接口。只有管理员能进。
 */
export function Alpha() {
  const user = useUserStore(s => s.user);
  const { t } = useTranslation('alpha');
  const { toast } = useToast();

  const [data, setData] = useState<AlphaDataStatus | null>(null);
  const [runs, setRuns] = useState<AlphaRun[]>([]);
  const [trials, setTrials] = useState(0);
  const [library, setLibrary] = useState<AlphaFactor[]>([]);
  const [combo, setCombo] = useState<AlphaCombo | null>(null);
  const [reference, setReference] = useState<AlphaReference | null>(null);

  const [k, setK] = useState(3);
  const [note, setNote] = useState('');
  const [run, setRun] = useState<AlphaRun | null>(null);
  const [cands, setCands] = useState<Record<string, AlphaCandidate>>({});
  const [logs, setLogs] = useState<string[]>([]);
  const [stage, setStage] = useState<string | null>(null);
  const [decisions, setDecisions] = useState<Record<string, boolean>>({});
  const [lastId, setLastId] = useState(0);
  const [live, setLive] = useState(false);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const [detail, setDetail] = useState<AlphaFactor | null>(null);
  const [expr, setExpr] = useState('');
  const [evalResult, setEvalResult] = useState<AlphaEvalResult | null>(null);
  const [evaluating, setEvaluating] = useState(false);

  const fail = useCallback((err: unknown) => toast((err as Error).message, 'error'), [toast]);
  const loadData = useCallback(() => alphaApi.dataStatus().then(setData).catch(fail), [fail]);
  const loadRuns = useCallback(() => alphaApi.runs().then(r => { setRuns(r.runs); setTrials(r.trials); }).catch(fail), [fail]);
  const loadLibrary = useCallback(() => alphaApi.library().then(r => { setLibrary(r.factors); setCombo(r.combo); }).catch(fail), [fail]);
  const loadReference = useCallback(() => alphaApi.reference().then(setReference).catch(fail), [fail]);

  /** 订阅一轮：after=0 是从头回放（日志清掉重来），否则只收后面的 */
  const subscribe = useCallback((id: string, after: number) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    if (after === 0) setLogs([]);
    setLive(true);
    const onEvent = (e: AlphaEvent) => {
      switch (e.type) {
        case 'status':
          setStage(e.stage);
          break;
        case 'log':
          setLogs(l => [...l.slice(-(MAX_LOG - 1)), e.text]);
          break;
        case 'candidate': {
          const { type, ...c } = e;
          void type;
          setCands(m => ({ ...m, [c.id]: c }));
          break;
        }
        case 'awaiting':
          setStage('awaiting');
          setDecisions(Object.fromEntries(e.candidates.map(c => [c.id, false])));
          break;
        case 'done':
          setStage('done');
          void loadLibrary();
          break;
        case 'error':
          setStage('failed');
          toast(e.message, 'error');
          break;
        case 'end':
          setLastId(e.lastId);
          setRun(r => (r ? { ...r, status: e.status } : r));
          if (e.status === 'awaiting') setStage('awaiting');
          void loadRuns();
          break;
      }
    };
    alphaApi.events(id, after, onEvent, ctrl.signal)
      .catch(err => { if (!ctrl.signal.aborted) fail(err); })
      .finally(() => { if (abortRef.current === ctrl) setLive(false); });
  }, [fail, loadLibrary, loadRuns, toast]);

  /** 打开一轮：先拿快照，再从头回放事件；不在跑的轮次回放完就结束 */
  const openRun = useCallback(async (id: string) => {
    try {
      const s = await alphaApi.run(id);
      setRun(s.run);
      setTrials(s.trials);
      setCands(Object.fromEntries(s.candidates.map(c => [c.id, c])));
      setDecisions(Object.fromEntries(s.candidates.filter(c => c.status === 'passed').map(c => [c.id, false])));
      setStage(s.run.status === 'running' ? 'working' : s.run.status);
      sessionStorage.setItem(RUN_KEY, id);
      subscribe(id, 0);
    } catch (err) {
      sessionStorage.removeItem(RUN_KEY);
      fail(err);
    }
  }, [fail, subscribe]);

  useEffect(() => {
    void loadData();
    void loadRuns();
    void loadLibrary();
    void loadReference();
    const saved = sessionStorage.getItem(RUN_KEY);
    if (saved) void openRun(saved);
    return () => abortRef.current?.abort();
  }, [loadData, loadRuns, loadLibrary, loadReference, openRun]);

  // 刷新数据、参照集跑着的时候定时看一眼进度
  useEffect(() => {
    if (!data?.refresh.running) return;
    const id = setInterval(() => { void loadData(); }, 3000);
    return () => clearInterval(id);
  }, [data?.refresh.running, loadData]);
  useEffect(() => {
    if (!reference?.job.running) return;
    const id = setInterval(() => { void loadReference(); }, 5000);
    return () => clearInterval(id);
  }, [reference?.job.running, loadReference]);

  const startRun = async () => {
    setBusy(true);
    try {
      const { id } = await alphaApi.createRun({ k, note: note.trim() });
      toast(t('run.created', { id }), 'success');
      await openRun(id);
      await loadRuns();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const submit = async () => {
    if (!run) return;
    setBusy(true);
    try {
      await alphaApi.decide(run.id, decisions);
      setRun(r => (r ? { ...r, status: 'running' } : r));
      setStage('working');
      subscribe(run.id, lastId);
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const refresh = () => alphaApi.dataRefresh().then(loadData).catch(fail);

  const recheck = (id: string) => alphaApi.recheck(id)
    .then(async () => {
      await loadLibrary();
      if (detail?.id === id) setDetail(await alphaApi.factor(id));
    })
    .catch(fail);

  const toggleDetail = (id: string) => {
    if (detail?.id === id) { setDetail(null); return; }
    alphaApi.factor(id).then(setDetail).catch(fail);
  };

  const evaluate = () => {
    if (!expr.trim()) return;
    setEvaluating(true);
    alphaApi.evalExpr(expr.trim()).then(setEvalResult).catch(fail).finally(() => setEvaluating(false));
  };

  const referenceRun = () => alphaApi.referenceRun().then(loadReference).catch(fail);

  const ordered = useMemo(() => Object.values(cands).sort((a, b) => a.seq - b.seq), [cands]);
  const awaiting = run?.status === 'awaiting';
  const referenceTop = useMemo(() => (reference?.items ?? [])
    .slice()
    .sort((a, b) => Math.abs(b.brief.horizons['12h']?.meanIc ?? 0) - Math.abs(a.brief.horizons['12h']?.meanIc ?? 0))
    .slice(0, 10), [reference]);

  // 外层守卫保证 token 存在，user 为 null 只是还没回来；不是管理员直接回首页
  if (!user) return null;
  if (user.id !== 1) return <Navigate to="/" replace />;

  return (
    <div className="page-shell p-4 md:p-6 space-y-4 max-w-6xl">
      <div className="flex items-center gap-2.5">
        <Link to="/admin" aria-label={t('back')}
              className="border border-border hover:bg-surface-hover w-8 h-8 rounded-lg flex items-center justify-center text-muted-foreground hover:text-primary">
          <ChevronLeft className="w-4 h-4" />
        </Link>
        <FlaskConical className="w-5 h-5 text-primary" />
        <h1 className="text-lg font-black">{t('title')}</h1>
        <span className="ml-auto inline-flex items-center gap-1.5 text-[11px] font-bold text-muted-foreground">
          {live && <Loader2 className="w-3 h-3 inline-flex animate-spin" />}
          {stage ? t(`stage.${stage}`, { defaultValue: stage }) : ''}
        </span>
      </div>
      <p className="text-[11px] leading-relaxed text-muted-foreground">{t('desc')}</p>

      {/* 行情数据 */}
      <div className="rounded-lg pt-card p-4 space-y-2.5">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="microlabel">{t('data.title')}</span>
          {data?.loaded && data.facts ? (
            <span className="text-[11px] num">
              {t('data.range', { from: data.facts.from, to: data.facts.to, rows: data.facts.rows })}
              {' · '}
              {data.facts.liqFrom ? t('data.liqFrom', { date: data.facts.liqFrom }) : t('data.noLiq')}
              {' · '}
              {t('data.holdout', { from: data.facts.holdoutFrom })}
            </span>
          ) : (
            <span className="text-[11px] text-muted-foreground">{t('data.none')}</span>
          )}
          <button onClick={refresh} disabled={data?.refresh.running} className={cn(BTN, 'ml-auto text-primary')}>
            {data?.refresh.running ? <Loader2 className="w-3 h-3 inline-flex animate-spin" /> : <RefreshCw className="w-3 h-3 inline-flex" />}
            {data?.refresh.running ? t('data.refreshing') : t('data.refresh')}
          </button>
        </div>
        {data?.refresh.error && <p className="text-[11px] text-loss">{t('data.refreshError', { msg: data.refresh.error })}</p>}
        {data?.loaded && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground num">
            {Object.entries(data.symbols).map(([s, v]) => <span key={s}>{s} {t('data.rows', { n: v.rows })}</span>)}
          </div>
        )}
        <p className="text-[11px] text-muted-foreground num">{t('trials', { n: trials, fp: (trials * 0.05).toFixed(1) })}</p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[22rem_minmax(0,1fr)]">
        {/* 左列：发起一轮 + 轮次列表 + 日志 */}
        <div className="space-y-4">
          <div className="rounded-lg pt-card p-4 space-y-3">
            <span className="microlabel block">{t('run.title')}</span>
            <div className="flex items-end gap-2">
              <label className="space-y-1">
                <span className="text-[10px] text-muted-foreground block">{t('run.k')}</span>
                <input type="number" min={1} max={10} value={k}
                       onChange={e => setK(Math.min(10, Math.max(1, Number(e.target.value) || 1)))}
                       className={cn(INPUT, 'w-16 num')} />
              </label>
              <button onClick={startRun} disabled={busy || !data?.loaded || run?.status === 'running'}
                      className={cn(BTN, 'text-primary h-9 px-4')}>
                {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                {t('run.start')}
              </button>
            </div>
            <label className="space-y-1 block">
              <span className="text-[10px] text-muted-foreground block">{t('run.note')}</span>
              <input value={note} onChange={e => setNote(e.target.value)} placeholder={t('run.notePh')}
                     className={cn(INPUT, 'w-full')} />
            </label>
            {!data?.loaded && <p className="text-[10px] text-muted-foreground">{t('run.needData')}</p>}
          </div>

          <div className="rounded-lg pt-card p-4 space-y-2">
            <span className="microlabel block">{t('run.recent')}</span>
            {runs.length === 0 ? (
              <p className="text-[11px] text-muted-foreground">{t('run.noRun')}</p>
            ) : (
              <select value={run?.id ?? ''} onChange={e => { if (e.target.value) void openRun(e.target.value); }}
                      className={cn(INPUT, 'w-full')}>
                <option value="">{t('run.pick')}</option>
                {runs.map(r => (
                  <option key={r.id} value={r.id}>
                    {when(r.created_at)} · k={r.k} · {t(`runStatus.${r.status}`)}{r.note ? ` · ${r.note}` : ''}
                  </option>
                ))}
              </select>
            )}
            {run?.error && <p className="text-[11px] text-loss">{run.error}</p>}
          </div>

          <div className="rounded-lg pt-card p-4 space-y-2">
            <span className="microlabel block">{t('run.log')}</span>
            <ul className="max-h-72 overflow-auto space-y-1 text-[11px] leading-relaxed text-muted-foreground">
              {logs.map((line, i) => <li key={i}>{line}</li>)}
            </ul>
          </div>
        </div>

        {/* 右列：候选卡片 */}
        <div className="rounded-lg pt-card p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <span className="microlabel">{t('cand.title')}</span>
            {run && !live && run.status === 'running' && (
              <button onClick={() => subscribe(run.id, 0)} className={cn(BTN, 'text-primary')}>
                <RefreshCw className="w-3 h-3 inline-flex" />{t('cand.resubscribe')}
              </button>
            )}
          </div>
          {ordered.length === 0 ? (
            <p className="text-[11px] text-muted-foreground">{t('cand.empty')}</p>
          ) : ordered.map(c => (
            <CandidateCard key={c.id} c={c} awaiting={awaiting} decision={decisions[c.id]}
                           onDecide={v => setDecisions(d => ({ ...d, [c.id]: v }))} />
          ))}
          {awaiting && (
            <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-border">
              <button onClick={submit} disabled={busy} className={cn(BTN, 'text-primary')}>
                <Check className="w-3 h-3 inline-flex" />{t('cand.submit')}
              </button>
              <span className="text-[10px] text-muted-foreground">{t('cand.submitHint')}</span>
            </div>
          )}
        </div>
      </div>

      {/* 因子库：顶上是全部因子合成的组合，下面一条条因子 */}
      <div className="rounded-lg pt-card p-4 space-y-3">
        <span className="microlabel block">{t('library.title')}</span>
        {combo && (
          <div className="rounded-lg border border-border p-3 space-y-2">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="text-xs font-black">{t('library.combo', { n: combo.weights.length })}</span>
              <span className="text-[10px] text-muted-foreground">{t('library.comboHint')}</span>
            </div>
            <BriefTable brief={combo.brief} />
            <p className="text-[11px] text-muted-foreground num">
              {t('library.weights')}：{combo.weights.map(w => `${w.name} ${fmt(w.weight, 3)}`).join(' · ')}
            </p>
            <p className="text-[11px] text-muted-foreground num">
              {t('library.latest', { time: when(combo.asOf) })}：{Object.entries(combo.latest).map(([s, v]) => `${s} ${fmt(v, 2)}`).join(' · ')}
            </p>
          </div>
        )}
        {library.length === 0 ? (
          <p className="text-[11px] text-muted-foreground">{t('library.empty')}</p>
        ) : library.map(f => (
          <div key={f.id} className="rounded-lg border border-border p-3 space-y-2">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="text-xs font-black">{f.name}</span>
              <code className="text-[11px] text-muted-foreground break-all">{f.expr}</code>
              <span className="text-[11px] num ml-auto">
                {t('library.ic12')} {fmt(f.brief.horizons['12h']?.meanIc)}
                {' · '}{t('library.approvedAt')} {day(f.approved_at)}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <span className="text-muted-foreground num">
                {f.recheck
                  ? `${t('library.recheck')} ${t('library.ic12')} ${fmt(f.recheck.horizons['48']?.mean)}（${t('library.recheckHint', { n: f.recheck.horizons['48']?.samples ?? 0 })}）`
                  : t('library.noRecheck')}
              </span>
              <button onClick={() => recheck(f.id)} className={cn(BTN, 'ml-auto text-primary')}>
                <RefreshCw className="w-3 h-3 inline-flex" />{t('library.recheck')}
              </button>
              <button onClick={() => toggleDetail(f.id)} className={cn(BTN, 'text-muted-foreground')}>
                {detail?.id === f.id ? <ChevronUp className="w-3 h-3 inline-flex" /> : <ChevronDown className="w-3 h-3 inline-flex" />}
                {detail?.id === f.id ? t('library.close') : t('library.detail')}
              </button>
            </div>
            {detail?.id === f.id && (
              <div className="space-y-2 pt-2 border-t border-border">
                <p className="text-xs leading-relaxed">{detail.hypothesis}</p>
                {detail.review && (
                  <p className="text-[11px] text-muted-foreground">
                    {t('cand.review')}：{t(`cand.verdict.${detail.review.verdict}`)}，{detail.review.comment}
                  </p>
                )}
                {detail.gates && <GateList gates={detail.gates} />}
                <BriefTable brief={detail.brief} />
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* 手填表达式 */}
        <div className="rounded-lg pt-card p-4 space-y-3">
          <span className="microlabel block">{t('eval.title')}</span>
          <div className="flex gap-2">
            <input value={expr} onChange={e => setExpr(e.target.value)} placeholder={t('eval.ph')}
                   onKeyDown={e => { if (e.key === 'Enter') evaluate(); }} className={cn(INPUT, 'w-full')} />
            <button onClick={evaluate} disabled={evaluating || !expr.trim() || !data?.loaded} className={cn(BTN, 'text-primary shrink-0')}>
              {evaluating ? <Loader2 className="w-3 h-3 inline-flex animate-spin" /> : <Play className="w-3 h-3 inline-flex" />}
              {t('eval.run')}
            </button>
          </div>
          <p className="text-[10px] text-muted-foreground">{t('eval.hint')}</p>
          {evalResult && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <code className="text-[11px] break-all">{evalResult.expr}</code>
                <Badge variant={evalResult.passed ? 'gain' : 'loss'}>{evalResult.passed ? t('eval.passed') : t('eval.failed')}</Badge>
              </div>
              <GateList gates={evalResult.gates} />
              <BriefTable brief={evalResult.brief} />
            </div>
          )}
        </div>

        {/* 参照集 */}
        <div className="rounded-lg pt-card p-4 space-y-3">
          <div className="flex items-center gap-2">
            <span className="microlabel">{t('reference.title')}</span>
            <button onClick={referenceRun} disabled={!!reference?.job.running || !data?.loaded} className={cn(BTN, 'ml-auto text-primary')}>
              {reference?.job.running ? <Loader2 className="w-3 h-3 inline-flex animate-spin" /> : <Play className="w-3 h-3 inline-flex" />}
              {reference?.job.running
                ? t('reference.running', { done: reference.job.done ?? 0, total: reference.job.total ?? 0 })
                : t('reference.run')}
            </button>
          </div>
          <p className="text-[10px] text-muted-foreground">{t('reference.hint')}</p>
          {reference?.job.error && <p className="text-[11px] text-loss">{reference.job.error}</p>}
          {!reference || reference.items.length === 0 ? (
            <p className="text-[11px] text-muted-foreground">{t('reference.empty')}</p>
          ) : (
            <div className="space-y-1">
              <p className="text-[11px] num">
                {t('reference.passed', { n: reference.items.filter(i => i.passed).length, total: reference.items.length })}
              </p>
              <ul className="space-y-0.5 text-[11px]">
                {referenceTop.map(item => (
                  <li key={item.name} className="flex items-center gap-2">
                    {item.passed ? <Check className="w-3 h-3 inline-flex text-gain" /> : <X className="w-3 h-3 inline-flex text-muted-foreground" />}
                    <span className="font-bold">{item.name}</span>
                    <code className="text-muted-foreground truncate">{item.expr}</code>
                    <span className="ml-auto num">{fmt(item.brief.horizons['12h']?.meanIc)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function CandidateCard({ c, awaiting, decision, onDecide }: {
  c: AlphaCandidate;
  awaiting: boolean;
  decision: boolean | undefined;
  onDecide: (approve: boolean) => void;
}) {
  const { t } = useTranslation('alpha');
  return (
    <div className="rounded-lg border border-border p-3 space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-black">{c.seq}. {c.name}</span>
        {c.family && <span className="text-[10px] text-muted-foreground">{c.family}</span>}
        <Badge variant={STATUS_VARIANT[c.status]}>{t(`status.${c.status}`)}</Badge>
        <span className="ml-auto text-[10px] text-muted-foreground num">{t('cand.attempts', { n: c.attempts })}</span>
      </div>
      <p className="text-xs leading-relaxed">{c.hypothesis}</p>
      {c.expr && (
        <div className="text-[11px]">
          <span className="text-muted-foreground">{t('cand.expr')}：</span>
          <code className="break-all">{c.expr}</code>
          {c.note && <span className="text-muted-foreground">（{c.note}）</span>}
        </div>
      )}
      {c.reason && <p className="text-[11px] text-loss leading-relaxed">{t('cand.reason')}：{c.reason}</p>}
      {c.review && (
        <p className="text-[11px] text-muted-foreground leading-relaxed">
          {t('cand.review')}：{t(`cand.verdict.${c.review.verdict}`)}，{c.review.comment}
          {c.review.suggestion ? `；${c.review.suggestion}` : ''}
        </p>
      )}
      {c.gates && <GateList gates={c.gates} />}
      {c.brief && <BriefTable brief={c.brief} />}
      {awaiting && c.status === 'passed' && (
        <div className="flex items-center gap-2 pt-1">
          <button onClick={() => onDecide(true)}
                  className={cn(BTN, decision ? 'border-gain bg-gain/10 text-gain' : 'text-muted-foreground')}>
            <Check className="w-3 h-3 inline-flex" />{t('cand.approve')}
          </button>
          <button onClick={() => onDecide(false)}
                  className={cn(BTN, decision === false ? 'border-loss bg-loss/10 text-loss' : 'text-muted-foreground')}>
            <X className="w-3 h-3 inline-flex" />{t('cand.decline')}
          </button>
        </div>
      )}
    </div>
  );
}

/** 每门一行：过没过 + 一句带数字的原因 */
function GateList({ gates }: { gates: AlphaGate[] }) {
  return (
    <ul className="space-y-0.5 text-[11px] leading-relaxed">
      {gates.map(g => (
        <li key={g.id} className="flex items-start gap-1.5">
          {g.passed
            ? <Check className="w-3 h-3 inline-flex shrink-0 mt-0.5 text-gain" />
            : <X className="w-3 h-3 inline-flex shrink-0 mt-0.5 text-loss" />}
          <span className="font-bold shrink-0 num">{g.id}</span>
          <span className={cn(g.passed ? 'text-muted-foreground' : 'text-foreground')}>{g.note}</span>
        </li>
      ))}
    </ul>
  );
}

/** 三档并排的数字；单因子再加组合增量一行和替身、冗余、复杂度一行；算过 holdout 的再来一张同样三档的小表 */
function BriefTable({ brief }: { brief: AlphaBriefCore & Partial<Pick<AlphaBrief, 'surrogate' | 'redundancy' | 'complexity'>> }) {
  const { t } = useTranslation('alpha');
  const rows: { key: string; get: (h: AlphaBrief['horizons'][string]) => string }[] = [
    { key: 'meanIc', get: h => fmt(h.meanIc) },
    { key: 'icir', get: h => fmt(h.icir, 2) },
    { key: 'sameSign', get: h => String(h.sameSignSymbols) },
    { key: 'net', get: h => fmt(h.netReturn, 3) },
    { key: 'perm', get: h => fmt(h.permPct, 3) },
    { key: 'hit', get: h => fmt(h.hitRate, 3) },
    { key: 'exposure', get: h => fmt(h.exposure, 2) },
    { key: 'turnover', get: h => fmt(h.turnover, 2) },
  ];
  const holdoutRows: { key: string; get: (h: AlphaHoldoutHorizon) => string }[] = [
    { key: 'meanIc', get: h => fmt(h.meanIc) },
    { key: 'icir', get: h => fmt(h.icir, 2) },
    { key: 'sameSign', get: h => String(h.sameSignSymbols) },
    { key: 'samples', get: h => String(h.samples) },
  ];
  const { holdout, gain, surrogate, redundancy, complexity } = brief;
  return (
    <div className="space-y-1">
      <table className="text-[11px] num w-full max-w-md">
        <thead>
          <tr className="text-muted-foreground">
            <th className="text-left font-normal">{t('gates.horizon')}</th>
            {HORIZONS.map(h => <th key={h} className="text-right font-normal">{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.key}>
              <td className="text-muted-foreground">{t(`gates.${r.key}`)}</td>
              {HORIZONS.map(h => <td key={h} className="text-right">{brief.horizons[h] ? r.get(brief.horizons[h]) : '—'}</td>)}
            </tr>
          ))}
          {gain && (
            <tr>
              <td className="text-muted-foreground">{t('gates.gain')}</td>
              {HORIZONS.map(h => <td key={h} className="text-right">{fmt(gain[h])}</td>)}
            </tr>
          )}
        </tbody>
      </table>
      {surrogate && redundancy && complexity && (
        <p className="text-[10px] text-muted-foreground num">
          {t('gates.surrogate', { p: fmt(surrogate.p, 3), mean: fmt(surrogate.nullMean), k: surrogate.k ?? 0 })}
          {' · '}
          {redundancy.with
            ? t('gates.redundancy', { name: redundancy.with, r: fmt(redundancy.maxAbs, 2) })
            : t('gates.noRedundancy')}
          {' · '}
          {t('gates.complexity', complexity)}
        </p>
      )}
      {holdout && (
        <table className="text-[11px] num w-full max-w-md">
          <thead>
            <tr className="text-muted-foreground">
              <th className="text-left font-normal">{t('gates.holdout', { from: day(holdout.since) })}</th>
              {HORIZONS.map(h => <th key={h} className="text-right font-normal">{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {holdoutRows.map(r => (
              <tr key={r.key}>
                <td className="text-muted-foreground">{t(`gates.${r.key}`)}</td>
                {HORIZONS.map(h => <td key={h} className="text-right">{holdout.horizons[h] ? r.get(holdout.horizons[h]) : '—'}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
