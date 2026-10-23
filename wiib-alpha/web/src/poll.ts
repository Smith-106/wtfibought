import { useEffect } from 'react';

/** active 时轮询：上一次回来 ms 后再发下一次，不叠着发 */
export function usePoll(active: boolean, load: () => Promise<unknown>, ms: number) {
  useEffect(() => {
    if (!active) return;
    let id = 0;
    let stopped = false;
    const tick = () => {
      void load().finally(() => { if (!stopped) id = window.setTimeout(tick, ms); });
    };
    id = window.setTimeout(tick, ms);
    return () => { stopped = true; clearTimeout(id); };
  }, [active, load, ms]);
}
