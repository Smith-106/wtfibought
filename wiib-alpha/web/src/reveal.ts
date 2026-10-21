import { createContext, useContext, useEffect, useRef, useState } from 'react';

const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** 开屏裂开了才放行；没放行之前进了视口的也先不揭开 */
export const RevealGate = createContext(true);

/** 进视口挂 in，入场动效从这里起；关了动画就一开始是终态 */
export function useReveal<T extends Element>() {
  const ref = useRef<T>(null);
  const open = useContext(RevealGate);
  const [shown, setShown] = useState(REDUCED_MOTION);
  useEffect(() => {
    if (shown || !open) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        setShown(true);
        io.disconnect();
      }
    }, { rootMargin: '0px 0px -8% 0px' });
    io.observe(ref.current!);
    return () => io.disconnect();
  }, [shown, open]);
  return [ref, shown] as const;
}
