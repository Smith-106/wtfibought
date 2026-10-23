/** 数字格式：后端把 NaN/inf 清成了 null，统一显示成横线 */

export const DASH = '—';

export const num = (v: number | null | undefined, digits = 4): string => (v == null ? DASH : v.toFixed(digits));

export const pct = (v: number | null | undefined, digits = 1): string =>
  v == null ? DASH : `${(v * 100).toFixed(digits)}%`;

export const signedPct = (v: number | null | undefined, digits = 1): string =>
  v == null ? DASH : `${v > 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`;

const pad = (n: number) => String(n).padStart(2, '0');

/** UTC 日期：数据层的时间都按 UTC 算 */
export const utcDay = (ms: number | null | undefined): string =>
  ms == null ? DASH : new Date(ms).toISOString().slice(0, 10);

export const utcMinute = (ms: number | null | undefined): string =>
  ms == null ? DASH : new Date(ms).toISOString().slice(0, 16).replace('T', ' ');

/** 本地时间：操作发生在什么时候 */
export const localTime = (ms: number | null | undefined): string => {
  if (ms == null) return DASH;
  const d = new Date(ms);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
};

/** 本地时分秒：实时数是什么时候的 */
export const localClock = (ms: number | null | undefined): string => {
  if (ms == null) return DASH;
  const d = new Date(ms);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
};

/** 6h/12h/24h 三档 */
export const HORIZONS = ['6h', '12h', '24h'] as const;
/** 判门用的那一档：现在是 24h，之前落库的老结果没记、是 12h */
export const primaryOf = (brief: { primary?: string }): string => brief.primary ?? '12h';
/** 复查结果按 bar 数做键 */
export const HORIZON_BARS: Record<(typeof HORIZONS)[number], string> = { '6h': '24', '12h': '48', '24h': '96' };

export const cx = (...parts: (string | false | null | undefined)[]): string => parts.filter(Boolean).join(' ');
