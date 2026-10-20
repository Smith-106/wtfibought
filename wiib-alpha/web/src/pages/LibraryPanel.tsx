import { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Library, RefreshCw, ScanSearch, Sigma } from 'lucide-react';
import { alpha, type ComboReport, type Factor, type FactorDetail, type Recheck, type ReevalJob } from '../api';
import { cx, HORIZON_BARS, HORIZONS, localTime, num, primaryOf, utcMinute } from '../format';
import { usePoll } from '../poll';
import { useSort } from '../sort';
import { useFail } from '../toast';
import { BriefTable, ExprBlock, GateList, ReviewLine } from '../components/FactorCard';
import { Badge, Button, Card, Drawer, Empty, RawJson, SortTh, Spinner } from '../components/ui';
import { ReevalBlock, ReevalProgress } from './ObservationPanel';

type FactorKey = 'name' | 'approvedAt' | 'ic' | 'icir' | 'holdout' | 'recheck' | 'reeval';

// 表里的主档数字：搜索段、holdout、入库后复查、复评；主档是这条入库时判门用的那一档，复评的是现在的主档
const primaryIc = (f: Factor) => f.brief.horizons[primaryOf(f.brief)]?.meanIc;
const primaryIcir = (f: Factor) => f.brief.horizons[primaryOf(f.brief)]?.icir;
const holdoutIc = (f: Factor) => f.brief.holdout?.horizons[primaryOf(f.brief)]?.meanIc;
const recheckIc = (f: Factor) =>
  f.recheck?.horizons[HORIZON_BARS[primaryOf(f.brief) as (typeof HORIZONS)[number]]]?.mean;
const reevalIc = (f: Factor) => f.reeval?.brief.horizons[primaryOf(f.reeval.brief)]?.meanIc;

const FACTOR_VALUES: Record<FactorKey, (f: Factor) => string | number | null | undefined> = {
  name: f => f.name,
  approvedAt: f => f.approved_at,
  ic: primaryIc,
  icir: primaryIcir,
  holdout: holdoutIc,
  recheck: recheckIc,
  reeval: reevalIc,
};

/** 管理员的因子库：列表、组合报告、详情与复查；可以用当前评估器整批复评 */
export function LibraryPanel({ active, loaded }: { active: boolean; loaded: boolean }) {
  const { t } = useTranslation();
  const fail = useFail();
  const [factors, setFactors] = useState<Factor[] | null>(null);
  const [combo, setCombo] = useState<ComboReport | null>(null);
  const [job, setJob] = useState<ReevalJob | null>(null);
  const [detail, setDetail] = useState<FactorDetail | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [rechecking, setRechecking] = useState(false);
  const [starting, setStarting] = useState(false);

  const load = useCallback(() => alpha.library()
    .then(r => { setFactors(r.factors); setCombo(r.combo); setJob(r.reeval); })
    .catch(fail), [fail]);

  useEffect(() => {
    if (active) void load();
  }, [active, load]);

  const running = job?.running ?? false;
  usePoll(active && running, load, 3000);

  const reeval = async () => {
    setStarting(true);
    try {
      await alpha.libraryReeval();
      await load();
    } catch (err) {
      fail(err);
    } finally {
      setStarting(false);
    }
  };

  const { sorted, sort, toggle } = useSort(factors ?? [], FACTOR_VALUES, { key: 'approvedAt', dir: 'desc' });

  const open = (id: string) => {
    setOpenId(id);
    setDetail(null);
    alpha.factor(id).then(setDetail).catch(err => { fail(err); setOpenId(null); });
  };

  const close = useCallback(() => setOpenId(null), []);

  const recheck = async (id: string) => {
    setRechecking(true);
    try {
      const r = await alpha.recheck(id);
      setDetail(d => (d && d.id === id ? { ...d, recheck: r } : d));
      setFactors(list => list && list.map(f => (f.id === id ? { ...f, recheck: r } : f)));
    } catch (err) {
      fail(err);
    } finally {
      setRechecking(false);
    }
  };

  const th = (k: FactorKey, label: string, right = true) =>
    <SortTh key={k} k={k} sort={sort} onSort={toggle} right={right}>{label}</SortTh>;

  return (
    <div className="space-y-4">
      {combo ? <ComboCard combo={combo} /> : factors && factors.length > 0 && !loaded && (
        <Card><Empty>{t('library.comboNone')}</Empty></Card>
      )}

      <Card title={t('library.title')} icon={Library}
            actions={<Button variant="ghost" icon={RefreshCw} onClick={() => void load()} className="px-2" />}>
        {!factors ? <Spinner /> : factors.length === 0 ? <Empty>{t('library.empty')}</Empty> : (
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-[11px] text-muted">{t('library.reevalHint')}</p>
              <Button icon={ScanSearch} loading={starting || running} disabled={!loaded}
                      onClick={() => void reeval()} className="ml-auto">
                {running ? t('library.reevalRunning') : t('library.reeval')}
              </Button>
            </div>
            {job && <ReevalProgress job={job} />}
            <div className="overflow-x-auto -mx-4">
              <table className="tbl">
                <thead>
                  <tr>
                    {th('name', t('library.col.name'), false)}
                    <th>{t('library.col.expr')}</th>
                    {th('approvedAt', t('library.col.approvedAt'))}
                    {th('ic', t('library.col.ic'))}
                    {th('icir', t('library.col.icir'))}
                    {th('holdout', t('library.col.holdout'))}
                    {th('recheck', t('library.col.recheck'))}
                    {th('reeval', t('library.col.reeval'))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map(f => (
                    <tr key={f.id} onClick={() => open(f.id)} className="cursor-pointer">
                      <td className="font-semibold whitespace-nowrap">{f.name}</td>
                      <td className="max-w-[28rem]"><code className="block truncate text-[11.5px] text-muted" title={f.expr}>{f.expr}</code></td>
                      <td className="r num whitespace-nowrap">{localTime(f.approved_at)}</td>
                      <td className="r num">{num(primaryIc(f))}</td>
                      <td className="r num">{num(primaryIcir(f), 2)}</td>
                      <td className="r num">{num(holdoutIc(f))}</td>
                      <td className="r num">{num(recheckIc(f))}</td>
                      <td className={cx('r num', f.reeval && !f.reeval.passed && 'text-loss')}>{num(reevalIc(f))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card>

      <Drawer open={openId !== null} onClose={close}
              title={detail
                ? <div className="flex items-center gap-2"><span className="font-bold">{detail.name}</span><Badge tone="gain">{t('status.approved')}</Badge></div>
                : <Spinner />}>
        {detail && (
          <FactorDetailView f={detail} loaded={loaded} rechecking={rechecking} onRecheck={() => void recheck(detail.id)} />
        )}
      </Drawer>
    </div>
  );
}

function FactorDetailView({ f, loaded, rechecking, onRecheck }: {
  f: FactorDetail;
  loaded: boolean;
  rechecking: boolean;
  onRecheck: () => void;
}) {
  const { t } = useTranslation();
  return (
    <>
      <ExprBlock expr={f.expr} />
      <dl className="grid gap-x-6 gap-y-2 text-[11.5px] sm:grid-cols-2">
        <div className="sm:col-span-2">
          <dt className="text-muted">{t('library.canonical')}</dt>
          <dd><code className="break-all">{f.canonical}</code></dd>
        </div>
        <div>
          <dt className="text-muted">{t('library.columns')}</dt>
          <dd className="flex flex-wrap gap-1 pt-0.5">{f.columns.map(c => <Badge key={c}>${c}</Badge>)}</dd>
        </div>
        <div>
          <dt className="text-muted">{t('library.approvedAt')} · {t('library.runId')}</dt>
          <dd className="num">{localTime(f.approved_at)} · {f.run_id}</dd>
        </div>
      </dl>
      <div className="space-y-1">
        <div className="microlabel">{t('library.hypothesis')}</div>
        <p className="text-xs leading-relaxed">{f.hypothesis}</p>
      </div>
      {f.review && <ReviewLine review={f.review} />}
      <GateList gates={f.gates} />
      <BriefTable brief={f.brief} />
      {f.reeval && <ReevalBlock r={f.reeval} />}
      <section className="border border-line bg-panel p-3 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="microlabel">{t('library.recheck')}</span>
          <span className="text-[11px] text-muted">{t('library.recheckHint')}</span>
          <Button icon={RefreshCw} loading={rechecking} disabled={!loaded} onClick={onRecheck} className="ml-auto">
            {t('library.recheckRun')}
          </Button>
        </div>
        {!loaded && <p className="text-[11px] text-warn">{t('common.needData')}</p>}
        {f.recheck ? <RecheckView recheck={f.recheck} primary={primaryOf(f.brief)} /> : <Empty>{t('library.recheckNone')}</Empty>}
      </section>
      <RawJson value={f.eval} />
    </>
  );
}

/** 复查结果：三档截面 IC，primary 是这条判门用的那一档 */
function RecheckView({ recheck, primary }: { recheck: Recheck; primary: string }) {
  const { t } = useTranslation();
  const cell = (h: (typeof HORIZONS)[number]) => recheck.horizons[HORIZON_BARS[h]];
  const rows: { key: string; get: (h: Recheck['horizons'][string]) => string | number }[] = [
    { key: 'meanIc', get: h => num(h.mean) },
    { key: 'icir', get: h => num(h.icir, 2) },
    { key: 'months', get: h => h.months },
    { key: 'monotonic', get: h => num(h.monotonic, 2) },
    { key: 'samples', get: h => h.samples },
  ];
  return (
    <div className="space-y-2">
      <p className="text-[11px] text-muted num">{t('library.recheckAt', { since: utcMinute(recheck.since), at: localTime(recheck.at) })}</p>
      <table className="tbl">
        <thead>
          <tr>
            <th>{t('brief.metric')}</th>
            {HORIZONS.map(h => <th key={h} className={cx('r', h === primary && 'text-primary')}>{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.key}>
              <td className="text-muted">{t(`brief.${r.key}`)}</td>
              {HORIZONS.map(h => <td key={h} className="r num">{cell(h) ? r.get(cell(h)) : '—'}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** 正负对着中线画的小条 */
function Diverging({ v, max }: { v: number | null; max: number }) {
  const width = v == null || max === 0 ? 0 : Math.min(1, Math.abs(v) / max) * 50;
  return (
    <span className="relative block h-2 w-28 bg-panel-2">
      <span className="absolute inset-y-0 left-1/2 w-px bg-line" />
      {v != null && (
        <span className={cx('absolute inset-y-0', v >= 0 ? 'left-1/2 bg-gain/70' : 'right-1/2 bg-loss/70')}
              style={{ width: `${width}%` }} />
      )}
    </span>
  );
}

function ComboCard({ combo }: { combo: ComboReport }) {
  const { t } = useTranslation();
  const maxWeight = Math.max(...combo.weights.map(w => Math.abs(w.weight)));
  // 截面信号按从强到弱排：排头的是组合看多的币
  const latest = useMemo(() => Object.entries(combo.latest).sort(([, a], [, b]) => (b ?? -Infinity) - (a ?? -Infinity)),
    [combo.latest]);
  const maxLatest = Math.max(0, ...latest.map(([, v]) => Math.abs(v ?? 0)));
  return (
    <Card title={t('library.combo', { n: combo.weights.length })} icon={Sigma} hint={t('library.comboHint')}>
      <div className="grid gap-5 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <BriefTable brief={combo.brief} />
        <div className="space-y-4">
          <table className="tbl">
            <thead>
              <tr><th colSpan={3}>{t('library.weights')}</th></tr>
            </thead>
            <tbody>
              {combo.weights.map(w => (
                <tr key={w.name}>
                  <td className="font-semibold">{w.name}</td>
                  <td className="r num">{num(w.weight, 3)}</td>
                  <td className="w-32"><Diverging v={w.weight} max={maxWeight} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="max-h-[28rem] overflow-auto border border-line">
            <table className="tbl tbl-sticky">
              <thead>
                <tr><th colSpan={3}>{t('library.latest', { time: utcMinute(combo.asOf) })}</th></tr>
              </thead>
              <tbody>
                {latest.map(([s, v]) => (
                  <tr key={s}>
                    <td className="num">{s}</td>
                    <td className="r num">{num(v, 3)}</td>
                    <td className="w-32"><Diverging v={v} max={maxLatest} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Card>
  );
}
