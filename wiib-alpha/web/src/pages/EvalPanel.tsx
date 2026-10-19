import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Lock, Play, SquareFunction } from 'lucide-react';
import { alpha, type EvalResult } from '../api';
import { useFail } from '../toast';
import { FactorCard, PassBadge } from '../components/FactorCard';
import { Button, Card, RawJson } from '../components/ui';

export function EvalPanel({ admin, loaded }: { admin: boolean; loaded: boolean }) {
  const { t } = useTranslation();
  const fail = useFail();
  const [expr, setExpr] = useState('');
  const [result, setResult] = useState<EvalResult | null>(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    try {
      setResult(await alpha.evaluate(expr.trim()));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const ready = admin && loaded && expr.trim() !== '' && !busy;

  return (
    <div className="space-y-4">
      <Card title={t('eval.title')} icon={SquareFunction} hint={t('eval.hint')}>
        <div className="space-y-2">
          <textarea value={expr} onChange={e => setExpr(e.target.value)} placeholder={t('eval.ph')} rows={3} disabled={!admin}
                    onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey) && ready) void run(); }}
                    className="input h-auto w-full py-2 font-mono text-[12.5px] leading-relaxed resize-y disabled:opacity-50" />
          <div className="flex items-center gap-3">
            {!admin && <span className="inline-flex items-center gap-1 text-[11px] text-warn"><Lock className="w-3 h-3 inline-flex" />{t('eval.needAdmin')}</span>}
            {admin && !loaded && <span className="text-[11px] text-warn">{t('common.needData')}</span>}
            <Button variant="primary" icon={Play} loading={busy} disabled={!ready} onClick={() => void run()} className="ml-auto">
              {t('eval.run')}
            </Button>
          </div>
        </div>
      </Card>

      {result && (
        <FactorCard title={<code>{result.expr}</code>} badges={<PassBadge passed={result.passed} />}
                    gates={result.gates} brief={result.brief}>
          <p className="text-[11.5px] text-muted break-all">{t('eval.canonical')}：<code>{result.canonical}</code></p>
          <RawJson value={result.eval} />
        </FactorCard>
      )}
    </div>
  );
}
