import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Play, TriangleAlert } from 'lucide-react';
import { alpha, type EvalJob } from '../api';
import { usePoll } from '../poll';
import { useFail } from '../toast';
import { FactorCard, PassBadge } from '../components/FactorCard';
import { Button, Card, RawJson, Spinner } from '../components/ui';

/** 管理员的手填评估：后台跑，跑着就轮询 */
export function EvalPanel({ loaded }: { loaded: boolean }) {
  const { t } = useTranslation();
  const fail = useFail();
  const [expr, setExpr] = useState('');
  const [job, setJob] = useState<EvalJob | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => alpha.evalJob().then(setJob).catch(fail), [fail]);
  useEffect(() => { void load(); }, [load]);

  const running = job?.running ?? false;
  usePoll(running, load, 2000);

  const run = async () => {
    setBusy(true);
    try {
      await alpha.evaluate(expr.trim());
      await load();
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const ready = loaded && expr.trim() !== '' && !busy && !running;
  const result = running ? null : job?.result;

  return (
    <div className="space-y-14">
      <Card title={t('eval.title')} hint={t('eval.hint')}>
        <div className="space-y-3">
          <textarea value={expr} onChange={e => setExpr(e.target.value)} placeholder={t('eval.ph')} rows={3}
                    onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && ready) void run(); }}
                    className="input h-auto w-full py-2.5 font-mono text-[13px] leading-relaxed resize-y" />
          <div className="flex items-center gap-3">
            {!loaded && <span className="text-[12px] text-warn">{t('common.needData')}</span>}
            {running && (
              <span className="flex items-center gap-2 min-w-0 text-[12.5px] text-muted">
                <Spinner /><span className="truncate">{t(`eval.${job?.stage ?? 'queued'}`, { expr: job?.expr })}</span>
              </span>
            )}
            <Button variant="primary" icon={Play} loading={busy || running} disabled={!ready} onClick={() => void run()}
                    className="ml-auto">
              {t('eval.run')}
            </Button>
          </div>
          {!running && job?.error && (
            <p className="flex items-start gap-1.5 text-[13px] text-loss break-words">
              <TriangleAlert className="w-3.5 h-3.5 shrink-0 mt-0.5" />{t('eval.error', { msg: job.error })}
            </p>
          )}
        </div>
      </Card>

      {result && (
        <FactorCard title={<code>{result.expr}</code>} badges={<PassBadge passed={result.passed} />}
                    gates={result.gates} brief={result.brief}>
          <p className="text-[12.5px] text-muted break-all">{t('eval.canonical')}：<code>{result.canonical}</code></p>
          <RawJson value={result.eval} />
        </FactorCard>
      )}
    </div>
  );
}
