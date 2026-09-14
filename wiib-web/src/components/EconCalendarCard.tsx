import { Fragment, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Skeleton } from './ui/skeleton';
import { CountryFlag } from './CountryFlag';
import { quantApi, type EconCalendarEvent, type EconCalendarView } from '../api';
import { cn, fmtDateTime } from '../lib/utils';

/** 采集 4 小时一次、实际值由公布时刻的等待闸补齐，轮询只是让分界与实际值跟着时间走 */
const POLL_MS = 300_000;

/**
 * 首页财经日历：TradingView 全球 High 级事件，左栏过去 3 天已公布、右栏未来一周即将公布，各最多 6 条。
 * 标题是接口英文原文，两种语言都照原样显示。
 */
export function EconCalendarCard() {
  const { t } = useTranslation('home');
  const [data, setData] = useState<EconCalendarView | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => quantApi.econCalendar()
      .then(v => { if (alive) setData(v); })
      .catch(() => { if (alive) setData(prev => prev ?? { past: [], upcoming: [] }); });
    load();
    const timer = setInterval(load, POLL_MS);
    return () => { alive = false; clearInterval(timer); };
  }, []);

  return (
    <>
      <div className="sec-h">
        <h2>{t('calendar.title')}<small>{t('calendar.note')}</small></h2>
      </div>
      <div className="g12">
        <Column title={t('calendar.published')} rows={data?.past} empty={t('calendar.emptyPast')} />
        {/* 即将公布的头一条是最近要撞上的，加粗 */}
        <Column title={t('calendar.upcoming')} rows={data?.upcoming} empty={t('calendar.emptyUpcoming')} boldFirst />
      </div>
    </>
  );
}

function Column({ title, rows, empty, boldFirst }: {
  title: string;
  /** undefined=还没拉回来 */
  rows?: EconCalendarEvent[];
  empty: string;
  boldFirst?: boolean;
}) {
  return (
    <div className="col-span-12 xl:col-span-6">
      <div className="microlabel uppercase pb-2 border-b border-foreground">{title}</div>
      {rows == null ? (
        <div className="space-y-3 pt-3">
          {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-9" />)}
        </div>
      ) : rows.length === 0 ? (
        <div className="py-8 text-center text-sm text-muted-foreground">{empty}</div>
      ) : rows.map((r, i) => {
        // 手机（<640px）一行塞不下三列：时间列缩成上下两行、数值挪到标题下一行；sm 起恢复三列单行
        const [date, time] = fmtDateTime(r.eventTime).split(' ');
        return (
          <div key={`${r.eventTime}-${r.country}-${r.title}`}
               className={cn('grid grid-cols-[44px_1fr] sm:grid-cols-[92px_1fr_auto] gap-x-3 gap-y-1 items-baseline py-2.5 border-b border-border text-[14px]',
                 boldFirst && i === 0 && 'font-semibold')}>
            <span className="num text-[13px] text-muted-foreground">
              <span className="block sm:inline">{date}</span>{' '}
              <span className="block sm:inline">{time}</span>
            </span>
            <span className="min-w-0 flex flex-wrap sm:flex-nowrap items-baseline gap-x-2">
              <CountryFlag code={r.country} className="self-center" />
              <b className="shrink-0 text-[12px] font-bold">{r.country} / {r.currency}</b>
              {/* 手机上跟国旗同行放不下就整个掉到下一行，不切半截出省略号 */}
              <span className="min-w-0 flex-auto sm:truncate">{r.title}</span>
            </span>
            <Values row={r} className="col-start-2 sm:col-start-auto" />
          </div>
        );
      })}
    </div>
  );
}

/** 实际 · 预测 · 前值，有哪个显示哪个，实际值加深；讲话类三个都没有就不占位 */
function Values({ row, className }: { row: EconCalendarEvent; className?: string }) {
  const { t } = useTranslation('home');
  const parts = ([['actual', row.actual], ['forecast', row.forecast], ['previous', row.previous]] as const)
    .filter(([, v]) => v);
  if (!parts.length) return null;
  return (
    // 手机上只允许在 · 处折行，"实际 3.1%" 这种标签和数值不拆开
    <span className={cn('num sm:whitespace-nowrap text-[12px] text-muted-foreground', className)}>
      {parts.map(([k, v], i) => (
        <Fragment key={k}>
          {i > 0 && ' · '}
          <span className={cn('whitespace-nowrap', k === 'actual' && 'text-foreground font-bold')}>
            {t(`calendar.${k}`)} {v}
          </span>
        </Fragment>
      ))}
    </span>
  );
}
