import { useCallback, useState, type ReactNode } from 'react';
import { CircleAlert, CircleCheck, X } from 'lucide-react';
import { ToastContext, type ToastKind } from '../toast';
import { cx } from '../format';

interface Item {
  id: number;
  message: string;
  kind: ToastKind;
  ms: number;
}

let seq = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Item[]>([]);

  const dismiss = useCallback((id: number) => setItems(list => list.filter(i => i.id !== id)), []);

  const toast = useCallback((message: string, kind: ToastKind = 'error') => {
    const id = ++seq;
    const ms = kind === 'error' ? 6000 : 3000;
    setItems(list => [...list.slice(-3), { id, message, kind, ms }]);
    setTimeout(() => dismiss(id), ms);
  }, [dismiss]);

  return (
    <ToastContext.Provider value={toast}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 w-[min(24rem,calc(100vw-2rem))]">
        {items.map(i => (
          <div key={i.id} className="toast">
            {i.kind === 'error'
              ? <CircleAlert className="w-4 h-4 shrink-0 mt-0.5 text-loss" />
              : <CircleCheck className="w-4 h-4 shrink-0 mt-0.5 text-gain" />}
            <span className="flex-1 break-words">{i.message}</span>
            <button onClick={() => dismiss(i.id)} className="text-muted transition-colors hover:text-fg cursor-pointer" aria-label="close">
              <X className="w-3.5 h-3.5" />
            </button>
            {/* 底边细线倒计时，走完就收 */}
            <i className={cx('toast-bar', i.kind === 'error' ? 'bg-loss' : 'bg-gain')} style={{ animationDuration: `${i.ms}ms` }} />
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
