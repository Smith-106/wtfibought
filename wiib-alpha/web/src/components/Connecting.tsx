import { useTranslation } from 'react-i18next';
import { FlaskConical, RefreshCw, WifiOff } from 'lucide-react';
import { Button, Spinner } from './ui';

/** 头一次拉数据状态：连上之前转圈，连不上给重试 */
export function Connecting({ error, onRetry }: { error: string | null; onRetry: () => void }) {
  const { t } = useTranslation();
  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-sm border border-line bg-panel p-6 space-y-4">
        <div className="flex items-center gap-2 text-sm font-bold">
          <FlaskConical className="w-4 h-4 text-primary" />
          {t('title')}
          <span className="text-muted font-normal num">wiib-alpha</span>
        </div>
        {error ? (
          <div className="space-y-3">
            <p className="flex items-start gap-2 text-xs text-loss break-words">
              <WifiOff className="w-4 h-4 shrink-0" />{t('auth.down', { msg: error })}
            </p>
            <Button icon={RefreshCw} onClick={onRetry}>{t('auth.retry')}</Button>
          </div>
        ) : (
          <p className="flex items-center gap-2 text-xs text-muted"><Spinner />{t('auth.checking')}</p>
        )}
      </div>
    </div>
  );
}
