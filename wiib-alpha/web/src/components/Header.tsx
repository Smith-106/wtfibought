import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { Moon, ShieldCheck, Sun, type LucideIcon } from 'lucide-react';
import { currentLang, setLang } from '../i18n';
import { cx } from '../format';
import { Badge } from './ui';

/** 展开态是整词，往下滚折叠成 WIIB.：留 W(hat) I(f) I B(ought)，其余字母挂 go 淡掉（同主站） */
const WORDMARK = 'WHATIFIBOUGHT';
const WORDMARK_KEEP = new Set([0, 4, 6, 7]);

/** 往下滚过这么多 px 顶栏就折叠，回到这以内展开；只是防顶部回弹抖动的阈值 */
const COLLAPSE_AT = 8;

/** 折叠过渡的时长（CSS 里 .hdr 的 --dur）再多留一点，期间每帧跟着摆橙方块 */
const FOLD_MS = 620;

export interface NavItem<T extends string> {
  id: T;
  icon: LucideIcon;
  label: string;
}

/**
 * 顶栏：字标 + 分区导航 + 工具，底下 2px 墨线。
 * 折叠、橙方块、阅读进度全部直接操作 DOM，滚动不走 React 状态：
 * 折叠是给 header 切 collapsed，动画本身是 CSS 过渡；橙方块停在当前项左侧 13px，悬停到别的项滑过去，移开滑回
 */
export function Header<T extends string>({ items, active, onGo, admin, badges }: {
  items: NavItem<T>[];
  active: T;
  onGo: (id: T) => void;
  admin: boolean;
  badges: ReactNode;
}) {
  const { t, i18n } = useTranslation();
  const hdrRef = useRef<HTMLElement>(null);
  const navRef = useRef<HTMLElement>(null);
  const dotRef = useRef<HTMLSpanElement>(null);
  const headRef = useRef<HTMLElement>(null);
  // 当前项、语言、项数变了时的归位：重量宽、方块滑过去
  const homeRef = useRef<(() => void) | null>(null);
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'));

  // 首帧画出来之前量好各处宽度
  useLayoutEffect(() => {
    const hdr = hdrRef.current!, nav = navRef.current!, dot = dotRef.current!, head = headRef.current!;

    // 折叠动画要知道各处自然宽：要淡掉的字母写在自己身上，导航项量文案层写在项上，ALPHA 的字宽写 --tw
    const measure = () => {
      hdr.querySelectorAll<HTMLElement>('.logo .go').forEach(el =>
        el.style.setProperty('--w', `${el.getBoundingClientRect().width}px`));
      nav.querySelectorAll<HTMLElement>('.item').forEach(el =>
        el.style.setProperty('--w', `${el.querySelector('.lab')!.getBoundingClientRect().width}px`));
      hdr.style.setProperty('--tw', `${hdr.querySelector('.tag .tf')!.getBoundingClientRect().width}px`);
    };

    let hover: HTMLElement | null = null;
    const place = (animate: boolean) => {
      const el = hover ?? nav.querySelector<HTMLElement>('.item.on');
      if (!el) { dot.style.opacity = '0'; return; }
      dot.style.transition = animate ? '' : 'none';
      dot.style.left = `${el.getBoundingClientRect().left - nav.getBoundingClientRect().left + nav.scrollLeft - 13}px`;
      dot.style.opacity = '1';
    };

    // 阅读进度：底线上的橙方块按滚了多少从左走到右；一帧最多算一次
    let progressRaf = 0;
    const progress = () => {
      if (progressRaf) return;
      progressRaf = requestAnimationFrame(() => {
        progressRaf = 0;
        const max = document.documentElement.scrollHeight - window.innerHeight;
        const p = max > 0 ? Math.min(1, window.scrollY / max) : 0;
        head.style.transform = `translateX(${p * (head.parentElement!.clientWidth - 8)}px)`;
      });
    };

    // 折叠过渡期间导航项在变窄，每帧把方块摆到当前项上
    let raf = 0;
    let until = 0;
    const track = () => {
      place(false);
      raf = performance.now() < until ? requestAnimationFrame(track) : 0;
    };
    const onScroll = () => {
      progress();
      const want = window.scrollY > COLLAPSE_AT;
      if (want === hdr.classList.contains('collapsed')) return;
      hdr.classList.toggle('collapsed', want);
      until = performance.now() + FOLD_MS;
      if (!raf) raf = requestAnimationFrame(track);
    };

    const over = (e: MouseEvent) => {
      const item = (e.target as HTMLElement).closest<HTMLElement>('.item');
      if (item && item !== hover) { hover = item; place(true); }
    };
    const leave = () => { hover = null; place(true); };
    const relayout = () => { measure(); place(false); progress(); };
    homeRef.current = () => { measure(); place(true); progress(); };

    measure();
    hdr.classList.toggle('collapsed', window.scrollY > COLLAPSE_AT);
    place(false);
    progress();
    // 先按位摆好，下一帧再挂 ready 开过渡
    const readyRaf = requestAnimationFrame(() => hdr.classList.add('ready'));

    nav.addEventListener('mouseover', over);
    nav.addEventListener('mouseleave', leave);
    window.addEventListener('resize', relayout);
    window.addEventListener('scroll', onScroll, { passive: true });
    // 页面变高变矮（数据到了、展开收起）也重算阅读进度
    const ro = new ResizeObserver(progress);
    ro.observe(document.body);
    // 字体就位后再量一次
    void document.fonts.ready.then(relayout);
    return () => {
      nav.removeEventListener('mouseover', over);
      nav.removeEventListener('mouseleave', leave);
      window.removeEventListener('resize', relayout);
      window.removeEventListener('scroll', onScroll);
      ro.disconnect();
      cancelAnimationFrame(raf);
      cancelAnimationFrame(readyRaf);
      cancelAnimationFrame(progressRaf);
    };
  }, []);

  useEffect(() => { homeRef.current!(); }, [active, items.length, i18n.language]);

  // 深浅色存在 localStorage 的 theme，和主站共用
  const toggleTheme = () => {
    const next = !document.documentElement.classList.contains('dark');
    document.documentElement.classList.toggle('dark', next);
    localStorage.setItem('theme', next ? 'dark' : 'light');
    setDark(next);
  };

  return (
    <header ref={hdrRef} className="hdr">
      <div className="wrap top">
        <a className="logo" href="https://wtfibought.com" aria-label="WhatIfIBought">
          <span className="word" aria-hidden="true">
            {[...WORDMARK].map((c, i) => <span key={i} className={WORDMARK_KEEP.has(i) ? undefined : 'go'}>{c}</span>)}
            <i>.</i>
          </span>
        </a>
        <span className="tag"><span className="tf">ALPHA</span><span className="ta" aria-hidden="true">α</span></span>
        <nav ref={navRef} className="nav">
          <span ref={dotRef} className="nav-dot" />
          {items.map(({ id, icon: Icon, label }) => (
            <button key={id} type="button" onClick={() => onGo(id)} data-label={label}
                    aria-current={active === id ? 'page' : undefined} className={cx('item', active === id && 'on')}>
              <span className="ico" aria-hidden="true"><Icon /></span>
              <span className="lab">{label}</span>
            </button>
          ))}
        </nav>
        <div className="tools">
          {badges}
          <button type="button" className="tool" onClick={() => setLang(currentLang() === 'zh' ? 'en' : 'zh')}
                  aria-label={t('header.lang')}>
            中 / EN
          </button>
          <button type="button" className="tool" onClick={toggleTheme} aria-label={t(dark ? 'header.toLight' : 'header.toDark')}>
            {dark ? <Sun /> : <Moon />}
          </button>
          <a className="tool" href="https://github.com/mamawai/wtfibought" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" /></svg>
          </a>
          {admin && <Badge tone="gain" icon={ShieldCheck}>{t('auth.admin')}</Badge>}
        </div>
      </div>
      <div className="rule"><i ref={headRef} className="rule-head" /></div>
    </header>
  );
}
