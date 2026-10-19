import { useEffect, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { ArrowDown, ArrowUp, ArrowUpDown, Braces, ChevronDown, LoaderCircle, X, type LucideIcon } from 'lucide-react';
import { cx } from '../format';
import type { SortState } from '../sort';

export function Card({ title, icon: Icon, actions, hint, children, className }: {
  title?: ReactNode;
  icon?: LucideIcon;
  actions?: ReactNode;
  hint?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <section className={cx('border border-line bg-panel', className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 pt-3">
          {title && (
            <h2 className="microlabel inline-flex items-center gap-1.5">
              {Icon && <Icon className="w-3.5 h-3.5 text-primary" />}
              {title}
            </h2>
          )}
          {actions && <div className="ml-auto flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      {hint && <p className="px-4 pt-1 text-[11.5px] text-muted leading-relaxed">{hint}</p>}
      <div className="p-4 pt-3">{children}</div>
    </section>
  );
}

type Variant = 'primary' | 'outline' | 'ghost' | 'gain' | 'loss';

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-primary text-primary-fg border-primary hover:brightness-110',
  outline: 'border-line bg-panel hover:bg-panel-2',
  ghost: 'border-transparent hover:bg-panel-2 text-muted hover:text-fg',
  gain: 'border-gain bg-gain/10 text-gain',
  loss: 'border-loss bg-loss/10 text-loss',
};

export function Button({ variant = 'outline', icon: Icon, loading, className, children, disabled, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; icon?: LucideIcon; loading?: boolean }) {
  return (
    <button {...rest} disabled={disabled || loading}
            className={cx('inline-flex items-center justify-center gap-1.5 h-8 px-3 border text-xs font-semibold',
              'transition-colors disabled:opacity-45 disabled:cursor-not-allowed cursor-pointer whitespace-nowrap',
              VARIANTS[variant], className)}>
      {loading ? <LoaderCircle className="w-3.5 h-3.5 animate-spin" /> : Icon && <Icon className="w-3.5 h-3.5" />}
      {children}
    </button>
  );
}

export type Tone = 'gain' | 'loss' | 'warn' | 'info' | 'muted' | 'primary';

const TONES: Record<Tone, string> = {
  gain: 'text-gain border-gain/40 bg-gain/10',
  loss: 'text-loss border-loss/40 bg-loss/10',
  warn: 'text-warn border-warn/40 bg-warn/10',
  info: 'text-info border-info/40 bg-info/10',
  muted: 'text-muted border-line bg-panel-2',
  primary: 'text-primary border-primary/40 bg-primary/10',
};

export function Badge({ tone = 'muted', icon: Icon, children, className }: {
  tone?: Tone;
  icon?: LucideIcon;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span className={cx('inline-flex items-center gap-1 border px-1.5 py-px text-[10.5px] font-semibold whitespace-nowrap',
      TONES[tone], className)}>
      {Icon && <Icon className="w-3 h-3 inline-flex" />}
      {children}
    </span>
  );
}

export function Progress({ done, total }: { done: number; total: number }) {
  const ratio = total > 0 ? Math.min(1, done / total) : 0;
  return (
    <div className="h-1.5 w-full bg-panel-2 overflow-hidden">
      <div className="h-full bg-primary transition-[width] duration-500" style={{ width: `${ratio * 100}%` }} />
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <LoaderCircle className={cx('w-3.5 h-3.5 animate-spin text-muted', className)} />;
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="text-xs text-muted py-2">{children}</p>;
}

/** 一格统计：小标签 + 值 */
export function Stat({ label, children, hint }: { label: ReactNode; children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-[10.5px] text-muted">{label}</div>
      <div className="text-[13px] font-semibold num">{children}</div>
      {hint && <div className="text-[10.5px] text-muted leading-snug">{hint}</div>}
    </div>
  );
}

/** 可排序表头 */
export function SortTh<K extends string>({ k, sort, onSort, children, right = true, title }: {
  k: K;
  sort: SortState<K>;
  onSort: (k: K) => void;
  children: ReactNode;
  right?: boolean;
  title?: string;
}) {
  const active = sort.key === k;
  const Arrow = !active ? ArrowUpDown : sort.dir === 'asc' ? ArrowUp : ArrowDown;
  return (
    <th className={right ? 'r' : undefined} title={title}>
      <button onClick={() => onSort(k)}
              className={cx('inline-flex items-center gap-1 cursor-pointer hover:text-fg', active && 'text-fg')}>
        {children}
        <Arrow className={cx('w-3 h-3 inline-flex', !active && 'opacity-30')} />
      </button>
    </th>
  );
}

/** 折叠的原始 JSON */
export function RawJson({ value }: { value: unknown }) {
  const { t } = useTranslation();
  return (
    <details className="group border border-line">
      <summary className="flex items-center gap-1.5 px-3 py-2 text-xs text-muted cursor-pointer select-none hover:text-fg">
        <Braces className="w-3.5 h-3.5" />
        {t('common.raw')}
        <ChevronDown className="w-3.5 h-3.5 ml-auto transition-transform group-open:rotate-180" />
      </summary>
      <pre className="max-h-[28rem] overflow-auto border-t border-line bg-panel-2 p-3 text-[11px] leading-relaxed">
        {JSON.stringify(value, null, 2)}
      </pre>
    </details>
  );
}

/** 右侧抽屉：点遮罩或按 Esc 关 */
export function Drawer({ open, onClose, title, children }: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
}) {
  const { t } = useTranslation();
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <aside className="relative h-full w-full max-w-4xl overflow-y-auto border-l border-line bg-bg shadow-2xl">
        <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-line bg-panel px-5 py-3">
          <div className="min-w-0 flex-1">{title}</div>
          <Button variant="ghost" icon={X} onClick={onClose} aria-label={t('common.close')} className="px-2" />
        </header>
        <div className="p-5 space-y-4">{children}</div>
      </aside>
    </div>
  );
}
