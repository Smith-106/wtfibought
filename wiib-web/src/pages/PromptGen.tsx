import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ArrowRight, Check, ChevronLeft, CircleCheck, Copy, Loader2, Play, RefreshCw, RotateCcw, SkipForward, Sparkles } from 'lucide-react';
import { llmEndpointApi, promptGenApi, userApi } from '../api';
import { LlmEndpointSelect } from '../components/LlmEndpointSelect';
import { FluidOrb } from '../components/ui/fluid-orb';
import { StepPlayer } from '../components/ui/step-player';
import { useToast } from '../components/ui/use-toast';
import { useIsDark } from '../hooks/useIsDark';
import { cn } from '../lib/utils';
import { SUPPORTED_LANGS, type Lang } from '../i18n';
import type { GenEvent, GenQuestion, GenThreadState, GenVersion, LlmEndpointView } from '../types';

/** 线程 id 存这儿：刷新页面接着答，关标签页就算了 */
const THREAD_KEY = 'wiib-promptgen-thread';
/** 指令正文上限，与后端 TraderService 同数 */
const MAX_CHARS = 4000;
/** 球的两套颜色，取 index.css 的 --color-primary 与 --color-card-2 */
const ORB_LIGHT = { color: '#f25f0a', base: '#f0f0ec' };
const ORB_DARK = { color: '#f97316', base: '#1a1b1f' };

type Stage = 'idle' | 'generating' | 'asking' | 'done';

const BTN = 'border border-border hover:bg-surface-hover rounded-lg px-3 py-1.5 text-[11px] font-bold inline-flex items-center gap-1.5 disabled:opacity-50';

/**
 * 交易指令生成器：选端点与指令语言 → 开始 → 一屏一屏答题 → 每轮出一版正文 → 复制回我的 Trader。
 * 后端是 py 服务 wiib-gen（/api/gen），答题走 SSE，回退与刷新恢复走普通 JSON 接口。
 */
export function PromptGen() {
  const { t } = useTranslation(['promptGen', 'ai']);
  const { toast } = useToast();
  const orb = useIsDark() ? ORB_DARK : ORB_LIGHT;

  const [endpoints, setEndpoints] = useState<LlmEndpointView[]>([]);
  const [endpointId, setEndpointId] = useState<number | null>(null);
  const [outputLang, setOutputLang] = useState<Lang>('zh');

  const [threadId, setThreadId] = useState<string | null>(null);
  const [versions, setVersions] = useState<GenVersion[]>([]);
  const [question, setQuestion] = useState<GenQuestion | null>(null);
  const [done, setDone] = useState(false);
  /** 后端图还停在半路：没题可答时给「继续生成」，专治生成中途断流 */
  const [pending, setPending] = useState(false);
  const [stage, setStage] = useState<Stage>('idle');
  /** 非 null = 这一版正在流式写，正文区显示它而不是已落版的正文 */
  const [streaming, setStreaming] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  /** 本屏答案：chips 选的与手打的，同一题两者互斥 */
  const [picked, setPicked] = useState<Record<string, string>>({});
  const [typed, setTyped] = useState<Record<string, string>>({});

  /** 一份线程快照落成页面状态：刷新恢复与回退都用它 */
  const apply = useCallback((s: GenThreadState) => {
    setVersions(s.versions);
    setQuestion(s.question);
    setDone(s.done);
    setPending(s.pending);
    setStage(s.done ? 'done' : s.question ? 'asking' : 'idle');
    setPicked({});
    setTyped({});
    setStreaming(null);
  }, []);

  const reset = useCallback(() => {
    sessionStorage.removeItem(THREAD_KEY);
    setThreadId(null);
    setVersions([]);
    setQuestion(null);
    setDone(false);
    setPending(false);
    setStage('idle');
    setPicked({});
    setTyped({});
    setStreaming(null);
  }, []);

  useEffect(() => { llmEndpointApi.list().then(setEndpoints).catch(() => setEndpoints([])); }, []);
  // 指令语言默认跟 agent 语言（问答语言另走界面语言，请求头 X-Lang 里带）
  useEffect(() => { userApi.getLang().then(setOutputLang).catch(() => { /* 拿不到就留 zh */ }); }, []);
  // 上次那条线还在就接着答；线程没了（换账号、服务重建库）就清掉重开
  useEffect(() => {
    const id = sessionStorage.getItem(THREAD_KEY);
    if (!id) return;
    promptGenApi.state(id)
      .then(s => { setThreadId(id); apply(s); })
      .catch(() => sessionStorage.removeItem(THREAD_KEY));
  }, [apply]);

  const onEvent = useCallback((e: GenEvent) => {
    switch (e.type) {
      case 'thread':
        setThreadId(e.id);
        sessionStorage.setItem(THREAD_KEY, e.id);
        break;
      case 'status':
        setStage(e.stage);
        if (e.stage === 'generating') setStreaming('');
        break;
      case 'token':
        setStreaming(s => (s ?? '') + e.text);
        break;
      case 'version':
        // 回退后重生成的那版号会重复，按版号覆盖
        setVersions(vs => [...vs.slice(0, e.no - 1), { no: e.no, text: e.text, notes: e.notes }]);
        setStreaming(null);
        break;
      case 'question':
        setQuestion({ kind: e.kind, items: e.items, allow_skip: e.allow_skip, allow_finish: e.allow_finish });
        setPending(true);
        setPicked({});
        setTyped({});
        break;
      case 'done':
        setDone(true);
        setQuestion(null);
        setPending(false);
        setStage('done');
        break;
      case 'error':
        // 流断在哪儿，题还留着，原样再提交一次就是重试；没题的（首版断了）给「继续生成」
        setPending(true);
        toast(e.message, 'error');
        break;
    }
  }, [toast]);

  const run = useCallback(async (call: (on: (e: GenEvent) => void) => Promise<void>) => {
    setBusy(true);
    try {
      await call(onEvent);
    } catch (err) {
      toast((err as Error).message, 'error');
    } finally {
      setBusy(false);
      setStreaming(null);
    }
  }, [onEvent, toast]);

  const start = () => {
    // 「跟随默认」时把默认那条的 id 显式发给 py（它只认具体端点）
    const id = endpointId ?? endpoints.find(e => e.isDefault)?.id ?? endpoints[0]?.id;
    if (id == null) return;
    reset();
    void run(on => promptGenApi.start({ endpointId: id, outputLang }, on));
  };

  const send = (body: { answer?: Record<string, string>; skip?: boolean; finish?: boolean }) => {
    if (!threadId) return;
    void run(on => promptGenApi.resume(threadId, body, on));
  };

  const answerOf = (key: string) => (typed[key] ?? '').trim() || picked[key] || '';
  const ready = question != null && question.items.every(it => answerOf(it.key) !== '');

  const submit = () => {
    if (!question || !ready) return;
    const answer: Record<string, string> = {};
    for (const it of question.items) answer[it.key] = answerOf(it.key);
    send({ answer });
  };

  const pick = (key: string, option: string) => {
    setPicked(p => ({ ...p, [key]: option }));
    setTyped(p => ({ ...p, [key]: '' }));
  };
  const write = (key: string, value: string) => {
    setTyped(p => ({ ...p, [key]: value }));
    if (value.trim()) setPicked(p => ({ ...p, [key]: '' }));
  };

  /** 点版本条：点最后一版没意义，点前面的要先把后面的作废 */
  const seek = (index: number) => {
    const no = index + 1;
    if (!threadId || busy || no >= versions.length) return;
    if (!window.confirm(t('rollbackConfirm', { no }))) return;
    setBusy(true);
    promptGenApi.rollback(threadId, no)
      .then(apply)
      .catch(err => toast((err as Error).message, 'error'))
      .finally(() => setBusy(false));
  };

  const last = versions.length ? versions[versions.length - 1] : null;
  const body = streaming ?? last?.text ?? '';
  const steps = useMemo(() => versions.map(v => ({ label: t('version', { no: v.no }) })), [versions, t]);

  const copy = () => {
    navigator.clipboard.writeText(body)
      .then(() => toast(t('copied'), 'success'))
      .catch(() => toast(t('copyFailed'), 'error'));
  };

  return (
    <div className="page-shell p-4 md:p-6 space-y-4 max-w-6xl">
      <div className="flex items-center gap-2.5">
        <Link to="/my-trader" aria-label={t('back')}
              className="border border-border hover:bg-surface-hover w-8 h-8 rounded-lg flex items-center justify-center text-muted-foreground hover:text-primary">
          <ChevronLeft className="w-4 h-4" />
        </Link>
        <Sparkles className="w-5 h-5 text-primary" />
        <h1 className="text-lg font-black">{t('title')}</h1>
        <span className="ml-auto inline-flex items-center gap-1.5 text-[11px] font-bold text-muted-foreground">
          {busy && <Loader2 className="w-3 h-3 inline-flex animate-spin" />}
          {t(`stage.${stage}`)}
        </span>
      </div>

      {/* 开场：端点 + 指令语言，开始之后就锁住 */}
      <div className="rounded-lg pt-card p-4 space-y-3">
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <div className="space-y-1.5">
            <span className="microlabel block">{t('endpoint')}</span>
            <LlmEndpointSelect endpoints={endpoints} value={endpointId} onChange={setEndpointId}
                               disabled={threadId != null} className="w-full sm:min-w-[280px]" />
          </div>
          <div className="space-y-1.5">
            <span className="microlabel block">{t('outputLang')}</span>
            <div className="flex h-9 border border-border divide-x divide-border">
              {SUPPORTED_LANGS.map(l => (
                <button key={l} type="button" disabled={threadId != null} onClick={() => setOutputLang(l)}
                        className={cn('px-3 text-xs font-bold disabled:opacity-60',
                          outputLang === l
                            ? 'bg-card-2 text-foreground shadow-[inset_0_2px_0_var(--color-primary)]'
                            : 'text-muted-foreground')}>
                  {t(`lang.${l}`)}
                </button>
              ))}
            </div>
          </div>
          {/* 有线程了就把这个位置让给「再来一份」，随时能丢掉重开 */}
          {threadId == null ? (
            <button onClick={start} disabled={busy || endpoints.length === 0}
                    className="border border-border hover:bg-surface-hover rounded-lg px-4 py-2 text-xs font-black text-primary inline-flex items-center gap-1.5 disabled:opacity-50">
              {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
              {t('start')}
            </button>
          ) : (
            <button onClick={reset} disabled={busy}
                    className="border border-border hover:bg-surface-hover rounded-lg px-4 py-2 text-xs font-black text-primary inline-flex items-center gap-1.5 disabled:opacity-50">
              <RotateCcw className="w-3.5 h-3.5" />
              {t('restart')}
            </button>
          )}
        </div>
        <p className="text-[10px] leading-relaxed text-muted-foreground/70">
          {endpoints.length === 0 ? t('noEndpoint') : t('startHint')}
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[22rem_minmax(0,1fr)]">
        <div className="space-y-4">
          {/* 版本条：点前面的版本就是回退 */}
          <div className="rounded-lg pt-card p-4 space-y-2.5">
            <div className="flex items-baseline justify-between gap-2">
              <span className="microlabel">{t('versions')}</span>
              <span className="text-[11px] font-bold num">
                {last ? t('version', { no: last.no }) : t('noVersion')}
              </span>
            </div>
            {versions.length > 0 && (
              <StepPlayer steps={steps} value={versions.length - 1} onValueChange={seek}
                          seekable showControl={false} playing={false} size={26} className="w-full" />
            )}
          </div>

          {/* 问答区 */}
          <div className="rounded-lg pt-card p-4 space-y-3">
            <span className="microlabel block">{t('questionTitle')}</span>
            {done ? (
              <div className="space-y-3">
                <div className="flex items-center gap-1.5 text-xs font-black text-gain">
                  <CircleCheck className="w-3 h-3 inline-flex" />{t('doneTitle')}
                </div>
                <p className="text-[11px] leading-relaxed text-muted-foreground">{t('doneHint')}</p>
              </div>
            ) : question ? (
              <div className={cn('space-y-4', busy && 'opacity-60 pointer-events-none')}>
                {question.items.map(it => (
                  <div key={it.key} className="space-y-2">
                    <p className="text-xs font-bold leading-relaxed">{it.text}</p>
                    {it.options.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {it.options.map(op => (
                          <button key={op} type="button" onClick={() => pick(it.key, op)}
                                  className={cn('rounded-lg border px-2.5 py-1 text-[11px] font-bold',
                                    picked[it.key] === op
                                      ? 'border-primary bg-primary/10 text-primary'
                                      : 'border-border text-muted-foreground hover:bg-surface-hover')}>
                            {op}
                          </button>
                        ))}
                      </div>
                    )}
                    {it.allow_free && (
                      <input value={typed[it.key] ?? ''} onChange={e => write(it.key, e.target.value)}
                             placeholder={t('freePh')}
                             className="w-full h-9 rounded-lg border border-border bg-card-2 px-2.5 text-xs" />
                    )}
                  </div>
                ))}
                <div className="flex flex-wrap items-center gap-2">
                  <button onClick={submit} disabled={!ready || busy} className={cn(BTN, 'text-primary')}>
                    <ArrowRight className="w-3 h-3 inline-flex" />{t('submit')}
                  </button>
                  {question.allow_skip && (
                    <button onClick={() => send({ skip: true })} disabled={busy} className={cn(BTN, 'text-muted-foreground')}>
                      <SkipForward className="w-3 h-3 inline-flex" />{t('skip')}
                    </button>
                  )}
                  {question.allow_finish && (
                    <button onClick={() => send({ finish: true })} disabled={busy} className={cn(BTN, 'text-muted-foreground')}>
                      <Check className="w-3 h-3 inline-flex" />{t('finish')}
                    </button>
                  )}
                </div>
                {question.kind === 'base' && !ready && (
                  <p className="text-[10px] text-muted-foreground">{t('needAll')}</p>
                )}
              </div>
            ) : pending && !busy ? (
              // 没题可答但图还没跑完 = 上次生成断在半路，空 resume 把它接着跑完
              <button onClick={() => send({})} className={cn(BTN, 'text-primary')}>
                <RefreshCw className="w-3 h-3 inline-flex" />{t('resume')}
              </button>
            ) : (
              <p className="text-[11px] leading-relaxed text-muted-foreground">{t('desc')}</p>
            )}
          </div>
        </div>

        {/* 右列：当前版本正文 */}
        <div className="rounded-lg pt-card p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <span className="microlabel">{t('body')}</span>
            <div className="flex items-center gap-3">
              <span className={cn('text-[11px] num', body.length > MAX_CHARS ? 'text-loss' : 'text-muted-foreground')}>
                {t('chars', { n: body.length, max: MAX_CHARS })}
              </span>
              <button onClick={copy} disabled={!body}
                      className="inline-flex items-center gap-1 text-[11px] font-bold text-primary hover:underline disabled:opacity-50">
                <Copy className="w-3 h-3 inline-flex" />{t('copy')}
              </button>
            </div>
          </div>

          {stage === 'generating' && (
            <div className="flex justify-center py-2">
              <FluidOrb size={120} color={orb.color} base={orb.base} />
            </div>
          )}

          {body
            ? <pre className="whitespace-pre-wrap font-sans text-xs leading-relaxed">{body}</pre>
            : <p className="text-[11px] leading-relaxed text-muted-foreground">{t('empty')}</p>}

          {last?.notes && streaming == null && (
            <details className="rounded-lg border border-border bg-card-2/50">
              <summary className="px-3 py-2 text-[11px] font-bold text-muted-foreground cursor-pointer">{t('notes')}</summary>
              <pre className="px-3 pb-3 text-[11px] leading-relaxed text-muted-foreground whitespace-pre-wrap font-sans">{last.notes}</pre>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}
