import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Bot, Database, Eye, FlaskConical, Languages, Library, Lock, LogOut, Pickaxe, RefreshCw, Ruler, ShieldCheck,
  SquareFunction, TriangleAlert,
} from 'lucide-react';
import { alpha, ApiError, clearToken, getToken, type DataStatus } from './api';
import i18n, { currentLang, setLang } from './i18n';
import { cx } from './format';
import { usePoll } from './poll';
import { AuthErrorContext, useFail, useToast } from './toast';
import { AdminLogin } from './components/AdminLogin';
import { Connecting } from './components/Connecting';
import { Badge, Button, Card } from './components/ui';
import { DataPanel } from './pages/DataPanel';
import { MinePanel } from './pages/MinePanel';
import { ObservationPanel } from './pages/ObservationPanel';
import { GuestLibrary, GuestMine, GuestObservation } from './pages/Overview';
import { LibraryPanel } from './pages/LibraryPanel';
import { ReferencePanel } from './pages/ReferencePanel';
import { EvalPanel } from './pages/EvalPanel';
import { TraderPanel } from './pages/TraderPanel';

const TABS = [
  { id: 'data', icon: Database },
  { id: 'mine', icon: Pickaxe },
  { id: 'observe', icon: Eye },
  { id: 'library', icon: Library },
  { id: 'reference', icon: Ruler },
  { id: 'eval', icon: SquareFunction },
  { id: 'trader', icon: Bot },
] as const;

type Tab = (typeof TABS)[number]['id'];

/** 当前分区记在地址的 # 上，刷新页还在原处 */
const tabFromHash = (): Tab => TABS.find(t => `#${t.id}` === location.hash)?.id ?? 'mine';

export function App() {
  const { t } = useTranslation();
  const toast = useToast();
  const fail = useFail();
  // 默认只读；/me 认了手里的 token 才进管理员模式
  const [admin, setAdmin] = useState(false);
  const [status, setStatus] = useState<DataStatus | null>(null);
  // 头一次拉状态失败的原因，显示在连接页；拉到过之后的失败只弹提示
  const [down, setDown] = useState<string | null>(null);
  const connected = useRef(false);
  const [tab, setTab] = useState<Tab>(tabFromHash);

  // 启动时看手里的 token：401 当失效清掉，403 是主站的普通用户，静默留在只读，别打扰读者
  useEffect(() => {
    if (!getToken()) return;
    const ctrl = new AbortController();
    alpha.me(ctrl.signal)
      .then(() => setAdmin(true))
      .catch((err: Error) => {
        if (ctrl.signal.aborted) return;
        if (err instanceof ApiError && err.code === 401) clearToken();
        else if (!(err instanceof ApiError && err.code === 403)) toast(err.message, 'error');
      });
    return () => ctrl.abort();
  }, [toast]);

  // 动手的接口中途 401/403：退回只读，401 的 token 清掉
  const onAuthError = useCallback((code: 401 | 403) => {
    if (code === 401) clearToken();
    setAdmin(false);
    toast(i18n.t(code === 401 ? 'auth.expired' : 'auth.notAdmin'), 'error');
  }, [toast]);

  const logout = () => {
    clearToken();
    setAdmin(false);
  };

  const loadStatus = useCallback(() => alpha.dataStatus()
    .then(s => { connected.current = true; setStatus(s); setDown(null); })
    .catch((err: Error) => { if (connected.current) fail(err); else setDown(err.message); }), [fail]);

  useEffect(() => {
    void loadStatus();
  }, [loadStatus]);

  // 刷新数据或补算参照网格跑着的时候每 3 秒看一眼进度
  const refreshing = (status?.refresh.running || status?.pool.grid.running) ?? false;
  usePoll(refreshing, loadStatus, 3000);

  const go = (id: Tab) => {
    setTab(id);
    history.replaceState(null, '', `#${id}`);
  };

  if (!status) return <Connecting error={down} onRetry={() => void loadStatus()} />;

  return (
    <AuthErrorContext.Provider value={onAuthError}>
      <div className="min-h-screen">
        <header className="sticky top-0 z-30 border-b border-line bg-panel/95 backdrop-blur">
          <div className="mx-auto flex h-12 max-w-[1440px] items-center gap-5 px-4 md:px-6">
            <span className="inline-flex shrink-0 items-center gap-2 text-sm font-bold">
              <FlaskConical className="w-4 h-4 text-primary" />
              {t('title')}
            </span>
            <nav className="flex h-full min-w-0 items-stretch overflow-x-auto">
              {TABS.map(({ id, icon: Icon }) => (
                <button key={id} onClick={() => go(id)}
                        className={cx('inline-flex shrink-0 items-center gap-1.5 border-b-2 px-3 text-xs font-semibold cursor-pointer',
                          tab === id ? 'border-primary text-fg' : 'border-transparent text-muted hover:text-fg')}>
                  <Icon className="w-3.5 h-3.5" />
                  {t(`tabs.${id}`)}
                </button>
              ))}
            </nav>
            <div className="ml-auto flex shrink-0 items-center gap-3">
              {refreshing && <Badge tone="info" icon={RefreshCw}>{t('data.refreshing')}</Badge>}
              {!status.loaded && <Badge tone="warn" icon={TriangleAlert}>{t('data.none')}</Badge>}
              <button onClick={() => setLang(currentLang() === 'zh' ? 'en' : 'zh')}
                      className="inline-flex items-center gap-1 text-xs text-muted hover:text-fg cursor-pointer">
                <Languages className="w-3.5 h-3.5" />{t('lang')}
              </button>
              {admin ? (
                <>
                  <Badge tone="gain" icon={ShieldCheck}>{t('auth.admin')}</Badge>
                  <Button variant="ghost" icon={LogOut} onClick={logout}>{t('auth.logout')}</Button>
                </>
              ) : <AdminLogin onDone={() => setAdmin(true)} />}
            </div>
          </div>
        </header>

        {/* 七块都常驻，切走只是藏起来：挖掘那块的事件流不断。
            挖掘、观察区、因子库两种模式是两套组件，切模式就整块换掉重拉，不留另一种模式的内容 */}
        <main className="mx-auto max-w-[1440px] px-4 py-5 md:px-6">
          <div hidden={tab !== 'data'}><DataPanel status={status} admin={admin} onReload={loadStatus} /></div>
          <div hidden={tab !== 'mine'}>
            {admin ? <MinePanel active={tab === 'mine'} loaded={status.loaded} /> : <GuestMine active={tab === 'mine'} />}
          </div>
          <div hidden={tab !== 'observe'}>
            {admin ? <ObservationPanel active={tab === 'observe'} /> : <GuestObservation active={tab === 'observe'} />}
          </div>
          <div hidden={tab !== 'library'}>
            {admin ? <LibraryPanel active={tab === 'library'} loaded={status.loaded} /> : <GuestLibrary active={tab === 'library'} />}
          </div>
          <div hidden={tab !== 'reference'}><ReferencePanel active={tab === 'reference'} admin={admin} loaded={status.loaded} /></div>
          <div hidden={tab !== 'eval'}><EvalPanel key={String(admin)} admin={admin} loaded={status.loaded} /></div>
          {/* 交易员整块只给管理员 */}
          <div hidden={tab !== 'trader'}>
            {admin ? <TraderPanel active={tab === 'trader'} loaded={status.loaded} /> : (
              <Card>
                <p className="flex items-center gap-1.5 text-[11.5px] text-muted">
                  <Lock className="w-3 h-3 inline-flex" />{t('overview.adminOnly')}
                </p>
              </Card>
            )}
          </div>
        </main>
      </div>
    </AuthErrorContext.Provider>
  );
}
