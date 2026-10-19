import { useEffect, useState, type FormEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { LogIn } from 'lucide-react';
import { alpha, ApiError, clearToken, saveToken, sim, type AuthMode } from '../api';
import { Button, Spinner } from './ui';

/** 顶栏的"管理员登录"：点开一个小弹层，登 sim 拿 token，再用 /me 确认是管理员 */
export function AdminLogin({ onDone }: { onDone: () => void }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <Button icon={LogIn} onClick={() => setOpen(o => !o)}>{t('auth.login')}</Button>
      {open && <LoginForm onClose={() => setOpen(false)} onDone={onDone} />}
    </div>
  );
}

function LoginForm({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const { t } = useTranslation();
  const [mode, setMode] = useState<AuthMode | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  useEffect(() => {
    sim.mode().then(setMode).catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  // token 按 wiib-web 的格式写回；/me 不认就不算登上，401 的 token 顺手清掉
  const login = async (get: () => Promise<string>) => {
    setBusy(true);
    setError('');
    try {
      saveToken(await get());
      await alpha.me();
      onDone();
    } catch (err) {
      if (err instanceof ApiError && err.code === 401) clearToken();
      setError(err instanceof ApiError && err.code === 403 ? t('auth.notAdmin') : (err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void login(() => sim.passwordLogin(username.trim(), password));
  };

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div className="absolute right-0 top-full z-50 mt-2 w-72 border border-line bg-panel p-4 shadow-xl space-y-3">
        <div className="text-xs font-bold">{t('auth.loginTitle')}</div>
        {!mode && !error && <Spinner />}
        {mode?.passwordLoginEnabled && (
          <form onSubmit={submit} className="space-y-2">
            <input className="input w-full" value={username} onChange={e => setUsername(e.target.value)}
                   placeholder={t('auth.username')} autoComplete="username" autoFocus required />
            <input className="input w-full" type="password" value={password} onChange={e => setPassword(e.target.value)}
                   placeholder={t('auth.password')} autoComplete="current-password" required />
            <Button type="submit" variant="primary" icon={LogIn} loading={busy} className="w-full">{t('auth.submit')}</Button>
          </form>
        )}
        {mode && !mode.passwordLoginEnabled && !mode.linuxDoEnabled && (
          <div className="space-y-2">
            <p className="text-[11.5px] text-muted">{t('auth.localHint')}</p>
            <Button variant="primary" icon={LogIn} loading={busy} className="w-full" onClick={() => void login(sim.localLogin)}>
              {t('auth.local')}
            </Button>
          </div>
        )}
        {mode && !mode.passwordLoginEnabled && mode.linuxDoEnabled && (
          <p className="text-[11.5px] text-muted">{t('auth.noPassword')}</p>
        )}
        {error && <p className="text-[11.5px] text-loss break-words">{error}</p>}
      </div>
    </>
  );
}
