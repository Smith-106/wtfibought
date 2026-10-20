import { Fragment, useCallback, useEffect, useMemo, useState, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';
import {
  ChartCandlestick, ChevronDown, ChevronRight, CircleCheck, CircleX, GitCompare, ListChecks, RefreshCw, Square,
  SquareCheck, TrendingUp, TriangleAlert, Wallet,
} from 'lucide-react';
import {
  alpha, TRADE_LEGS, type CompareJob, type EquityPoint, type SchemeResult, type TradeRunBrief, type TradeRunDetail,
  type TradeStatus, type TraderAccount, type TraderFactor, type TraderJob, type TraderPosition,
} from '../api';
import { cx, DASH, localTime, num, pct, signedPct, utcMinute } from '../format';
import { usePoll } from '../poll';
import { useFail } from '../toast';
import { Badge, Button, Card, Empty, Progress, Spinner, Stat, type Tone } from '../components/ui';

const STATUS_TONE: Record<TradeStatus, Tone> = { done: 'gain', partial: 'warn', failed: 'loss' };
const STATUS_ICON = { done: CircleCheck, partial: TriangleAlert, failed: CircleX };
const METRICS: (keyof SchemeResult)[] = ['netReturn', 'sharpe', 'maxDrawdown', 'turnover', 'carryFundingBps', 'periods'];

/** 金额两位小数 */
const money = (v: number | null | undefined) => num(v, 2);

/** 管理员的交易员页：账户、净值曲线、持仓、调仓记录、交易因子的勾选与对比 */
export function TraderPanel({ active, loaded }: { active: boolean; loaded: boolean }) {
  const { t } = useTranslation();
  const fail = useFail();
  const [account, setAccount] = useState<TraderAccount | null>(null);
  const [positions, setPositions] = useState<{ items: TraderPosition[]; day: string | null } | null>(null);
  const [curve, setCurve] = useState<{ initialBalance: number; items: EquityPoint[] } | null>(null);
  const [runs, setRuns] = useState<TradeRunBrief[] | null>(null);
  const [factors, setFactors] = useState<TraderFactor[] | null>(null);
  const [compare, setCompare] = useState<CompareJob | null>(null);

  // 账户和持仓要问 sim，sim 不通是 502，各自报错，不拖累别的
  const loadTrader = useCallback(() => Promise.all([
    alpha.traderAccount().then(setAccount).catch(fail),
    alpha.traderPositions().then(setPositions).catch(fail),
    alpha.traderEquity().then(setCurve).catch(fail),
    alpha.traderRuns().then(r => setRuns(r.items)).catch(fail),
  ]), [fail]);

  const loadFactors = useCallback(() => alpha.traderFactors()
    .then(r => { setFactors(r.items); setCompare(r.compare); })
    .catch(fail), [fail]);

  useEffect(() => {
    if (!active) return;
    void loadTrader();
    void loadFactors();
  }, [active, loadTrader, loadFactors]);

  usePoll(active && (account?.job.running ?? false), loadTrader, 5000);
  usePoll(active && (compare?.running ?? false), loadFactors, 3000);

  return (
    <div className="space-y-4">
      <AccountCard account={account} curve={curve} lastRun={runs?.[0] ?? null} onReload={loadTrader} />
      <Card title={t('trader.curve')} icon={TrendingUp} hint={t('trader.curveHint')}>
        {!curve ? <Spinner /> : <EquityChart points={curve.items} initial={curve.initialBalance} />}
      </Card>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <PositionsCard positions={positions} />
        <FactorsCard factors={factors} compare={compare} loaded={loaded} onChanged={loadFactors} onList={setFactors} />
      </div>
      <RunsCard runs={runs} />
    </div>
  );
}

function AccountCard({ account, curve, lastRun, onReload }: {
  account: TraderAccount | null;
  curve: { initialBalance: number; items: EquityPoint[] } | null;
  lastRun: TradeRunBrief | null;
  onReload: () => Promise<unknown>;
}) {
  const { t } = useTranslation();
  // 最大回撤按每小时快照算
  const drawdown = useMemo(() => {
    if (!curve?.items.length) return null;
    let peak = curve.initialBalance;
    let worst = 0;
    for (const p of curve.items) {
      peak = Math.max(peak, p.equity);
      worst = Math.max(worst, 1 - p.equity / peak);
    }
    return worst;
  }, [curve]);
  return (
    <Card title={t('trader.title')} icon={Wallet} hint={t('trader.hint')}
          actions={<Button variant="ghost" icon={RefreshCw} onClick={() => void onReload()} className="px-2" />}>
      {!account ? <Spinner /> : (
        <div className="space-y-3">
          {account.userId == null ? <Empty>{t('trader.noAccount')}</Empty> : (
            <div className="flex flex-wrap gap-x-10 gap-y-3">
              <Stat label={t('trader.equity')}>{money(account.equity)}</Stat>
              <Stat label={t('trader.totalReturn')} hint={t('trader.totalReturnHint', { initial: account.initialBalance })}>
                <span className={tone(account.equity == null ? null : account.equity - account.initialBalance)}>
                  {account.equity == null ? DASH : signedPct(account.equity / account.initialBalance - 1, 2)}
                </span>
              </Stat>
              <Stat label={t('trader.drawdown')} hint={t('trader.drawdownHint')}>{pct(drawdown, 2)}</Stat>
              <Stat label={t('trader.balance')}>{money(account.balance)}</Stat>
              <Stat label={t('trader.available')}>{money(account.available)}</Stat>
              <Stat label={t('trader.userId')}>{account.userId}</Stat>
            </div>
          )}
          <JobLine job={account.job} lastRun={lastRun} />
        </div>
      )}
    </Card>
  );
}

function JobLine({ job, lastRun }: { job: TraderJob; lastRun: TradeRunBrief | null }) {
  const { t } = useTranslation();
  return (
    <div className="space-y-1.5 border-t border-line pt-2 text-[11.5px]">
      <div className="flex flex-wrap items-center gap-2 text-muted">
        <span>{t('trader.schedule')}</span>
        {lastRun && <StatusBadge status={lastRun.status} />}
        {lastRun && <span className="num">{lastRun.day}</span>}
      </div>
      {job.running && (
        <>
          <div className="flex items-center gap-3">
            <Badge tone="info" icon={RefreshCw}>{t('trader.running', { stage: t(`trader.stage.${job.stage ?? 'prepare'}`) })}</Badge>
            {job.stage === 'live' && <span className="num text-muted">{job.done ?? 0} / {job.total ?? 0} {job.symbol ?? ''}</span>}
          </div>
          {job.stage === 'live' && <Progress done={job.done ?? 0} total={job.total ?? 0} />}
        </>
      )}
      {!job.running && job.error && (
        <p className="flex items-start gap-1.5 text-loss break-words">
          <TriangleAlert className="w-3.5 h-3.5 shrink-0 mt-0.5" />{t('trader.lastError', { msg: job.error })}
        </p>
      )}
    </div>
  );
}

// ========== 净值曲线 ==========

const W = 800;
const H = 200;

/** 一条净值线 + 初始资金虚线；悬停看那一刻的净值。线宽不随拉伸变，点和提示用 HTML 叠在上面 */
function EquityChart({ points, initial }: { points: EquityPoint[]; initial: number }) {
  const { t } = useTranslation();
  const [hover, setHover] = useState<number | null>(null);
  if (points.length < 2) return <Empty>{t('trader.curveEmpty')}</Empty>;
  const values = points.map(p => p.equity);
  const lo = Math.min(initial, ...values);
  const hi = Math.max(initial, ...values);
  const pad = (hi - lo) * 0.08 || initial * 0.005;
  const [y0, y1] = [lo - pad, hi + pad];
  const [t0, t1] = [points[0].ts, points[points.length - 1].ts];
  const x = (ts: number) => ((ts - t0) / (t1 - t0)) * W;
  const y = (v: number) => H - ((v - y0) / (y1 - y0)) * H;
  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.ts).toFixed(1)},${y(p.equity).toFixed(1)}`).join('');

  const onMove = (e: MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const ts = t0 + ((e.clientX - box.left) / box.width) * (t1 - t0);
    // 离鼠标最近的那个快照
    const i = points.findIndex(p => p.ts >= ts);
    if (i === -1) setHover(points.length - 1);
    else if (i === 0) setHover(0);
    else setHover(ts - points[i - 1].ts < points[i].ts - ts ? i - 1 : i);
  };
  const h = hover == null ? null : points[hover];
  const left = h ? (x(h.ts) / W) * 100 : 0;
  return (
    <div className="space-y-1">
      <div className="relative h-48">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full cursor-crosshair"
             onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label={t('trader.curve')}>
          <line x1={0} x2={W} y1={y(initial)} y2={y(initial)} stroke="var(--color-muted)" strokeWidth={1}
                strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
          <path d={line} fill="none" stroke="var(--color-primary)" strokeWidth={2} strokeLinejoin="round"
                vectorEffect="non-scaling-stroke" />
          {h && <line x1={x(h.ts)} x2={x(h.ts)} y1={0} y2={H} stroke="var(--color-line)" strokeWidth={1}
                      vectorEffect="non-scaling-stroke" />}
        </svg>
        <span className="pointer-events-none absolute left-1 top-0 text-[10.5px] text-muted num">{money(y1)}</span>
        <span className="pointer-events-none absolute left-1 bottom-0 text-[10.5px] text-muted num">{money(y0)}</span>
        <span className="pointer-events-none absolute right-1 text-[10.5px] text-muted num -translate-y-full"
              style={{ top: `${(y(initial) / H) * 100}%` }}>{t('trader.initial', { v: initial })}</span>
        {h && (
          <>
            <span className="pointer-events-none absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary ring-2 ring-panel"
                  style={{ left: `${left}%`, top: `${(y(h.equity) / H) * 100}%` }} />
            <div className={cx('pointer-events-none absolute top-2 border border-line bg-panel px-2 py-1 text-[11px] shadow-sm num',
              left > 70 ? '-translate-x-[calc(100%+8px)]' : 'translate-x-2')} style={{ left: `${left}%` }}>
              <div className="text-muted">{localTime(h.ts)}</div>
              <div className="font-semibold">{money(h.equity)}</div>
              <div className={tone(h.equity - initial)}>{signedPct(h.equity / initial - 1, 2)}</div>
            </div>
          </>
        )}
      </div>
      <div className="flex justify-between text-[10.5px] text-muted num">
        <span>{utcMinute(t0)}</span>
        <span>{utcMinute(t1)}</span>
      </div>
    </div>
  );
}

// ========== 持仓 ==========

function PositionsCard({ positions }: { positions: { items: TraderPosition[]; day: string | null } | null }) {
  const { t } = useTranslation();
  const items = useMemo(() => [...(positions?.items ?? [])].sort((a, b) => b.notional - a.notional), [positions]);
  return (
    <Card title={t('trader.positions')} icon={ChartCandlestick}
          hint={positions?.day ? t('trader.positionsHint', { day: positions.day }) : undefined}>
      {!positions ? <Spinner /> : items.length === 0 ? <Empty>{t('trader.positionsEmpty')}</Empty> : (
        <div className="max-h-[32rem] overflow-auto -mx-4">
          <table className="tbl tbl-sticky">
            <thead>
              <tr>
                <th>{t('trader.col.symbol')}</th>
                <th>{t('trader.col.side')}</th>
                <th className="r">{t('trader.col.quantity')}</th>
                <th className="r">{t('trader.col.notional')}</th>
                {TRADE_LEGS.map(leg => <th key={leg} className="r">{t(`trader.leg.${leg}`)}</th>)}
                <th className="r">{t('trader.col.mark')}</th>
                <th className="r">{t('trader.col.pnl')}</th>
              </tr>
            </thead>
            <tbody>
              {items.map(p => (
                <tr key={`${p.symbol}-${p.side}`}>
                  <td className="num font-semibold">{p.symbol}</td>
                  <td><SideBadge side={p.side} /></td>
                  <td className="r num">{p.quantity}</td>
                  <td className="r num">{money(p.notional)}</td>
                  {TRADE_LEGS.map(leg => <td key={leg} className="r num">{money(p.legs?.[leg])}</td>)}
                  <td className="r num" title={`${t('trader.col.entry')} ${p.entryPrice}`}>{p.markPrice}</td>
                  <td className={cx('r num', tone(p.unrealizedPnl))}>{money(p.unrealizedPnl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

// ========== 交易因子：勾选与对比 ==========

function FactorsCard({ factors, compare, loaded, onChanged, onList }: {
  factors: TraderFactor[] | null;
  compare: CompareJob | null;
  loaded: boolean;
  onChanged: () => Promise<unknown>;
  onList: (list: TraderFactor[]) => void;
}) {
  const { t } = useTranslation();
  const fail = useFail();
  const [busy, setBusy] = useState<string | null>(null);
  const running = compare?.running ?? false;

  const toggle = async (f: TraderFactor) => {
    setBusy(f.id);
    try {
      const r = await alpha.selectTraderFactor(f.id, !f.selected);
      onList((factors ?? []).map(x => (x.id === f.id ? { ...x, selected: r.selected } : x)));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(null);
    }
  };

  const start = async (f: TraderFactor) => {
    setBusy(f.id);
    try {
      await alpha.compareTraderFactor(f.id);
      await onChanged();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Card title={t('trader.factors')} icon={ListChecks} hint={t('trader.factorsHint')}>
      {!factors ? <Spinner /> : factors.length === 0 ? <Empty>{t('trader.factorsEmpty')}</Empty> : (
        <div className="space-y-3">
          <ul className="divide-y divide-line border-y border-line">
            {factors.map(f => (
              <li key={f.id} className="flex items-start gap-2 py-2">
                <button onClick={() => void toggle(f)} disabled={busy !== null || (!f.selected && f.missing.length > 0)}
                        title={f.selected ? t('trader.selected') : t('trader.unselected')}
                        className="mt-0.5 inline-flex cursor-pointer text-primary disabled:cursor-not-allowed disabled:opacity-40">
                  {f.selected ? <SquareCheck className="w-4 h-4" /> : <Square className="w-4 h-4" />}
                </button>
                <div className="min-w-0 flex-1 space-y-0.5">
                  <div className="text-xs font-semibold">{f.name}</div>
                  <code className="block truncate text-[11px] text-muted" title={f.expr}>{f.expr}</code>
                  {f.missing.length > 0 && (
                    <p className="flex items-center gap-1 text-[11px] text-warn">
                      <TriangleAlert className="w-3 h-3 inline-flex" />
                      {t('trader.missing', { cols: f.missing.map(c => `$${c}`).join(', ') })}
                    </p>
                  )}
                </div>
                <Button icon={GitCompare} loading={busy === f.id || (running && compare?.factorId === f.id)}
                        disabled={busy !== null || running || !loaded} onClick={() => void start(f)}>
                  {t('trader.compare')}
                </Button>
              </li>
            ))}
          </ul>
          {!loaded && <p className="text-[11px] text-warn">{t('common.needData')}</p>}
          {compare && <CompareView job={compare} />}
        </div>
      )}
    </Card>
  );
}

function CompareView({ job }: { job: CompareJob }) {
  const { t } = useTranslation();
  if (!job.startedAt) return null;
  if (job.running) {
    return (
      <p className="flex items-center gap-2 text-[11.5px] text-muted">
        <Spinner />{t('trader.comparing', { name: job.name, stage: t(`trader.compareStage.${job.stage ?? 'library'}`) })}
      </p>
    );
  }
  if (job.error) {
    return (
      <p className="flex items-start gap-1.5 text-xs text-loss break-words">
        <TriangleAlert className="w-3.5 h-3.5 shrink-0 mt-0.5" />{t('trader.compareError', { msg: job.error })}
      </p>
    );
  }
  const r = job.result;
  if (!r) return null;
  const action = t(`trader.compareAction.${r.action}`);
  const columns = [
    { label: `${t('trader.compareBase')} · ${t('trader.search')}`, v: r.base.search },
    { label: `${t('trader.compareAlt', { action })} · ${t('trader.search')}`, v: r.alt.search },
    ...(r.base.holdout && r.alt.holdout ? [
      { label: `${t('trader.compareBase')} · ${t('trader.holdout')}`, v: r.base.holdout },
      { label: `${t('trader.compareAlt', { action })} · ${t('trader.holdout')}`, v: r.alt.holdout },
    ] : []),
  ];
  return (
    <section className="space-y-2 border border-line bg-panel p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="microlabel">{t('trader.compareTitle', { action, name: job.name })}</span>
        <span className="ml-auto text-[11px] text-muted num">{t('trader.compareAt', { time: localTime(job.finishedAt) })}</span>
      </div>
      <p className="text-[11px] text-muted">
        {t('trader.compareSets', {
          base: r.base.factors.join(', ') || t('trader.noneFactors'), action,
          alt: r.alt.factors.join(', ') || t('trader.noneFactors'),
        })}
      </p>
      <div className="overflow-x-auto">
        <table className="tbl">
          <thead>
            <tr>
              <th>{t('brief.metric')}</th>
              {columns.map(c => <th key={c.label} className="r">{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {METRICS.map(k => (
              <tr key={k}>
                <td className="text-muted">{t(`trader.metric.${k}`)}</td>
                {columns.map(c => <td key={c.label} className="r num">{schemeValue(k, c.v[k])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={cx('text-[11px]', r.better ? 'text-gain' : 'text-muted')}>{t(r.better ? 'trader.betterYes' : 'trader.betterNo')}</p>
    </section>
  );
}

function schemeValue(k: keyof SchemeResult, v: number | null): string {
  if (k === 'netReturn') return signedPct(v, 1);
  if (k === 'maxDrawdown') return pct(v, 1);
  if (k === 'carryFundingBps') return v == null ? DASH : `${num(v, 2)} bps`;
  if (k === 'periods') return v == null ? DASH : String(v);
  return num(v, k === 'sharpe' ? 2 : 3);
}

// ========== 调仓记录 ==========

function RunsCard({ runs }: { runs: TradeRunBrief[] | null }) {
  const { t } = useTranslation();
  const fail = useFail();
  const [open, setOpen] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, TradeRunDetail>>({});

  const toggle = (day: string) => {
    setOpen(o => (o === day ? null : day));
    if (!details[day]) {
      alpha.traderRun(day).then(d => setDetails(all => ({ ...all, [day]: d }))).catch(fail);
    }
  };

  return (
    <Card title={t('trader.runs')} icon={ListChecks} hint={t('trader.runsHint')}>
      {!runs ? <Spinner /> : runs.length === 0 ? <Empty>{t('trader.runsEmpty')}</Empty> : (
        <div className="overflow-x-auto -mx-4">
          <table className="tbl">
            <thead>
              <tr>
                <th className="w-6" />
                <th>{t('trader.runCol.day')}</th>
                <th>{t('trader.runCol.status')}</th>
                <th>{t('trader.runCol.factors')}</th>
                <th className="r">{t('trader.runCol.equity')}</th>
                <th className="r">{t('trader.runCol.orders')}</th>
              </tr>
            </thead>
            <tbody>
              {runs.map(r => (
                <Fragment key={r.day}>
                  <tr onClick={() => toggle(r.day)} className="cursor-pointer">
                    <td>{open === r.day ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}</td>
                    <td className="num whitespace-nowrap">{r.day}</td>
                    <td><StatusBadge status={r.status} /></td>
                    <td className="text-muted">{r.factorNames.join(', ') || DASH}</td>
                    <td className="r num whitespace-nowrap">{money(r.equityBefore)} → {money(r.equityAfter)}</td>
                    <td className="r num whitespace-nowrap">
                      {t('trader.orderCount', { n: r.orderCount })}
                      {r.failedCount > 0 && <span className="ml-1.5 text-loss">{t('trader.orderFailed', { n: r.failedCount })}</span>}
                    </td>
                  </tr>
                  {open === r.day && (
                    <tr>
                      <td colSpan={6} className="bg-panel-2/40">
                        {details[r.day] ? <RunDetail run={details[r.day]} /> : <Spinner />}
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function RunDetail({ run }: { run: TradeRunDetail }) {
  const { t } = useTranslation();
  const coins = run.targets?.coins ?? [];
  return (
    <div className="space-y-3 py-2">
      <p className="text-[11px] text-muted num">
        {t('trader.signalAt', { time: utcMinute(run.signalMs + 1) })} · {t('trader.ranAt', {
          start: localTime(run.startedAt), end: localTime(run.finishedAt) })}
      </p>
      {run.error && (
        <p className="flex items-start gap-1.5 text-xs text-loss break-words">
          <TriangleAlert className="w-3.5 h-3.5 shrink-0 mt-0.5" />{run.error}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
        <span className="text-muted mr-1">{t('trader.usedFactors')}</span>
        {run.factors.length === 0 ? <span className="text-muted">{t('trader.noFactors')}</span>
          : run.factors.map(f => (
            <span key={f.id} title={f.expr}><Badge>{f.name} {num(f.weight, 4)}</Badge></span>
          ))}
      </div>
      {run.legs && TRADE_LEGS.map(leg => (
        <div key={leg} className="flex flex-wrap items-center gap-1 text-[11px]">
          <span className="text-muted mr-1 w-16">{t(`trader.leg.${leg}`)}</span>
          {run.legs![leg].long.map(s => <Badge key={`l-${s}`} tone="gain">{s}</Badge>)}
          {run.legs![leg].short.map(s => <Badge key={`s-${s}`} tone="loss">{s}</Badge>)}
        </div>
      ))}
      {run.targets && run.targets.notTrading.length > 0 && (
        <p className="text-[11px] text-muted">{t('trader.notTrading')}{run.targets.notTrading.join(', ')}</p>
      )}
      {coins.length > 0 && (
        <section className="space-y-1">
          <div className="microlabel">{t('trader.targets')}</div>
          <div className="max-h-[28rem] overflow-auto border border-line">
            <table className="tbl tbl-sticky">
              <thead>
                <tr>
                  <th>{t('trader.col.symbol')}</th>
                  <th className="r">{t('trader.coinCol.weight')}</th>
                  {TRADE_LEGS.map(leg => <th key={leg} className="r">{t(`trader.leg.${leg}`)}</th>)}
                  <th className="r">{t('trader.coinCol.notional')}</th>
                  <th className="r">{t('trader.coinCol.current')}</th>
                  <th className="r">{t('trader.coinCol.target')}</th>
                  <th className="r">{t('trader.coinCol.price')}</th>
                  <th className="r">{t('trader.coinCol.ref')}</th>
                  <th>{t('trader.coinCol.note')}</th>
                </tr>
              </thead>
              <tbody>
                {coins.map(c => (
                  <tr key={c.symbol}>
                    <td className="num font-semibold">{c.symbol}</td>
                    <td className="r num">{signedPct(c.weight, 2)}</td>
                    {TRADE_LEGS.map(leg => <td key={leg} className="r num">{signedPct(c[leg], 2)}</td>)}
                    <td className="r num">{money(c.notional)}</td>
                    <td className="r num">{c.current}</td>
                    <td className="r num">{c.target ?? DASH}</td>
                    <td className="r num">{c.price ?? DASH}</td>
                    <td className="r num">{c.refPrice ?? DASH}</td>
                    <td className={cx('text-[11px]', c.failed ? 'text-loss' : 'text-muted')}>{c.note ?? ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      <section className="space-y-1">
        <div className="microlabel">{t('trader.orders')}</div>
        {run.orders.length === 0 ? <Empty>{t('trader.ordersEmpty')}</Empty> : (
          <div className="max-h-[28rem] overflow-auto border border-line">
            <table className="tbl tbl-sticky">
              <thead>
                <tr>
                  <th>{t('trader.col.symbol')}</th>
                  <th>{t('trader.orderCol.action')}</th>
                  <th className="r">{t('trader.orderCol.quantity')}</th>
                  <th className="r">{t('trader.orderCol.fill')}</th>
                  <th>{t('trader.orderCol.result')}</th>
                </tr>
              </thead>
              <tbody>
                {run.orders.map(o => (
                  <tr key={o.requestId}>
                    <td className="num font-semibold">{o.symbol}</td>
                    <td className="whitespace-nowrap">
                      <span className="mr-1.5">{t(`trader.action.${o.action}`)}</span><SideBadge side={o.side} />
                    </td>
                    <td className="r num">{o.quantity ?? t('trader.closeAll')}</td>
                    <td className="r num">{o.price ?? DASH}</td>
                    <td className="text-[11px]">
                      {o.ok ? <Badge tone="gain" icon={CircleCheck}>{t('trader.filled')} {o.filled}</Badge>
                        : <span className="text-loss break-words">{o.error}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {run.targets && (
        <details className="group border border-line">
          <summary className="flex items-center gap-1.5 px-3 py-2 text-xs text-muted cursor-pointer select-none hover:text-fg">
            {t('trader.signals')}
            <ChevronDown className="w-3.5 h-3.5 ml-auto transition-transform group-open:rotate-180" />
          </summary>
          <div className="grid gap-3 border-t border-line p-3 lg:grid-cols-2">
            {TRADE_LEGS.map(leg => (
              <div key={leg} className="space-y-1">
                <div className="microlabel">{t(`trader.leg.${leg}`)}</div>
                <div className="max-h-[24rem] overflow-auto border border-line">
                  <table className="tbl tbl-sticky">
                    <thead>
                      <tr>
                        <th>{t('trader.col.symbol')}</th>
                        <th className="r">{t('trader.signalCol.rank')}</th>
                        <th className="r">{t('trader.signalCol.signal')}</th>
                        <th className="r">{t('trader.signalCol.vol')}</th>
                        <th className="r">{t('trader.signalCol.weight')}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...run.targets!.signals[leg]].sort((a, b) => a.rank - b.rank).map(c => (
                        <tr key={c.symbol}>
                          <td className="num">{c.symbol}</td>
                          <td className="r num">{c.rank}</td>
                          <td className="r num">{num(c.signal, 5)}</td>
                          <td className="r num">{num(c.vol, 5)}</td>
                          <td className={cx('r num', tone(c.weight))}>{c.weight ? signedPct(c.weight, 2) : ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

// ========== 小件 ==========

function StatusBadge({ status }: { status: TradeStatus }) {
  const { t } = useTranslation();
  return <Badge tone={STATUS_TONE[status]} icon={STATUS_ICON[status]}>{t(`trader.status.${status}`)}</Badge>;
}

function SideBadge({ side }: { side: 'LONG' | 'SHORT' }) {
  const { t } = useTranslation();
  return <Badge tone={side === 'LONG' ? 'gain' : 'loss'}>{t(`trader.side.${side}`)}</Badge>;
}

/** 正负着色：赚是 gain、亏是 loss */
function tone(v: number | null | undefined): string | undefined {
  if (v == null || v === 0) return undefined;
  return v > 0 ? 'text-gain' : 'text-loss';
}
