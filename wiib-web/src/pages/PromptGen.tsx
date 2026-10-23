import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ArrowRight, BadgeCheck, ChevronLeft, Copy, FileText, Loader2, Play, RotateCcw, Send, SkipForward, Sparkles, ThumbsUp, Undo2 } from 'lucide-react';
import { llmEndpointApi, promptGenApi, userApi } from '../api';
import { LlmEndpointSelect } from '../components/LlmEndpointSelect';
import { FluidOrb } from '../components/ui/fluid-orb';
import { useToast } from '../components/ui/use-toast';
import { useIsDark } from '../hooks/useIsDark';
import { cn } from '../lib/utils';
import { SUPPORTED_LANGS, type Lang } from '../i18n';
import type { LlmEndpointView, PromptGenEvent, PromptGenItem, PromptGenView } from '../types';

/** 指令正文上限，与后端 TraderService.MAX_CUSTOM_PROMPT 同数 */
const MAX_CHARS = 4000;
/** 球的两套颜色，取 index.css 的 --color-primary 与 --color-card-2 */
const ORB_LIGHT = { color: '#f25f0a', base: '#f0f0ec' };
const ORB_DARK = { color: '#f97316', base: '#1a1b1f' };

type Stage = 'thinking' | 'reading' | 'writing';

const BTN = 'border border-border hover:bg-surface-hover rounded-lg px-3 py-1.5 text-[11px] font-bold inline-flex items-center gap-1.5 disabled:opacity-50';
const CHIP = 'rounded-lg border px-2.5 py-1 text-[11px] font-bold';
const CHIP_ON = 'border-primary bg-primary/10 text-primary';
const CHIP_OFF = 'border-border text-muted-foreground hover:bg-surface-hover';

/**
 * 交易指令生成器：agent 先看 trader 数据，再一题一题问，信息够了写一版；用户提修改它接着问，直到它认为写完了。
 * 后端在 agent（/api/ai/prompt-gen），每轮走 SSE，轮结束重新拉一次 session；最近 3 版可以回到当时接着聊。
 */
export function PromptGen() {
  const { t } = useTranslation(['promptGen', 'ai']);
  const { toast } = useToast();
  const orb = useIsDark() ? ORB_DARK : ORB_LIGHT;

  const [endpoints, setEndpoints] = useState<LlmEndpointView[]>([]);
  const [endpointId, setEndpointId] = useState<number | null>(null);
  const [outputLang, setOutputLang] = useState<Lang>('zh');

  const [view, setView] = useState<PromptGenView | null>(null);
  /** 点了「重新开始」：先回到开场设置，点开始才真覆盖旧对话 */
  const [restarting, setRestarting] = useState(false);
  /** 非 null = 正在跑一轮 */
  const [stage, setStage] = useState<Stage | null>(null);
  /** 这一轮里 agent 说的话（一次模型调用一段）、用户刚发的话；轮结束拉回 session 后由 items 接手 */
  const [liveTexts, setLiveTexts] = useState<string[]>([]);
  const [sent, setSent] = useState<string | null>(null);
  /** 状态切过 = 后面的字是下一次模型调用说的，另起一段 */
  const newSegmentRef = useRef(false);
  /** 非 null = 这一版正在流式写 */
  const [draft, setDraft] = useState<string | null>(null);
  /** 右栏在看哪一版；null 跟着当前版 */
  const [viewing, setViewing] = useState<number | null>(null);
  /** 题卡：点选的与手写的二者互斥 */
  const [picked, setPicked] = useState('');
  const [typed, setTyped] = useState('');
  const [message, setMessage] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  /** 在跑的那一轮；离开页面就掐断，后端随之停掉这一轮 */
  const abortRef = useRef<AbortController | null>(null);

  const busy = stage !== null;
  const active = !!view?.active && !restarting;

  const refresh = useCallback(() => promptGenApi.session().then(v => {
    setView(v);
    setViewing(null);
    setPicked('');
    setTyped('');
    return v;
  }), []);

  useEffect(() => { llmEndpointApi.list().then(setEndpoints).catch(() => setEndpoints([])); }, []);
  useEffect(() => () => abortRef.current?.abort(), []);
  // 有对话就接着上次的；没有的话指令语言默认跟 agent 语言
  useEffect(() => {
    refresh()
      .then(v => {
        if (v.active) {
          setEndpointId(v.endpointId);
          setOutputLang(v.outputLang as Lang);
        } else {
          userApi.getLang().then(setOutputLang).catch(() => { /* 拿不到就留 zh */ });
        }
      })
      .catch(err => toast((err as Error).message, 'error'));
  }, [refresh, toast]);

  // 对话往下长时滚到底
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [view, liveTexts, sent]);

  const onEvent = useCallback((e: PromptGenEvent) => {
    switch (e.type) {
      case 'status':
        setStage(e.stage);
        // writing：超长重写时会再来一次，首稿清掉从头接；切到别的阶段 = 写稿收场了（没写成也一样），右栏回到当前版
        setDraft(e.stage === 'writing' ? '' : null);
        newSegmentRef.current = true;
        break;
      case 'text': {
        const fresh = newSegmentRef.current;
        newSegmentRef.current = false;
        setLiveTexts(s => fresh || s.length === 0 ? [...s, e.text] : [...s.slice(0, -1), s[s.length - 1] + e.text]);
        break;
      }
      case 'draft':
        setDraft(d => (d ?? '') + e.text);
        break;
      case 'version':
        // 刚写好的这版先放进列表，轮结束拉 session 再对齐
        setView(v => v && {
          ...v,
          currentNo: e.no,
          versions: [{ no: e.no, body: e.body, notes: e.notes, finalVersion: e.finalVersion }, ...v.versions].slice(0, 3),
        });
        setViewing(null);
        setDraft(null);
        break;
      case 'error':
        toast(e.message, 'error');
        break;
      default:
        // done：轮结束统一拉 session
        break;
    }
  }, [toast]);

  /**
   * @param restore 后端没接下这一轮（一帧都没收到，比如名额被占）时把用户的输入放回去：这句没落库，
   *                跟着 refresh 清掉就丢了。接下了的话成败后端都会把它落库，轮后在对话里看得到
   */
  const run = async (call: (on: (e: PromptGenEvent) => void, signal: AbortSignal) => Promise<void>,
                     said: string | null, restore?: () => void) => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setStage('thinking');
    setLiveTexts([]);
    setDraft(null);
    setSent(said);
    let accepted = false;
    try {
      await call(e => {
        accepted = true;
        onEvent(e);
      }, ctrl.signal);
    } catch (err) {
      if (!ctrl.signal.aborted) toast((err as Error).message, 'error');
    }
    // 页面已经离开，不用再收尾
    if (ctrl.signal.aborted) return;
    await refresh().catch(() => { /* 拉不到就留着现场 */ });
    if (!accepted) restore?.();
    setStage(null);
    setLiveTexts([]);
    setDraft(null);
    setSent(null);
  };

  const start = () => {
    // 「跟随默认」时把默认那条的 id 显式发出去，后端只认具体端点
    const id = endpointId ?? endpoints.find(e => e.isDefault)?.id ?? endpoints[0]?.id;
    if (id == null) return;
    setRestarting(false);
    setView(v => v && { ...v, active: true, items: [], question: null, currentNo: null, done: false });
    void run((on, signal) => promptGenApi.start({ endpointId: id, outputLang }, on, signal), null);
  };

  const reply = (body: { text?: string; skip?: boolean }, said: string, restore?: () => void) =>
    void run((on, signal) => promptGenApi.reply(body, on, signal), said, restore);

  const answer = () => {
    const text = typed.trim() || picked;
    if (!text) return;
    const [wasPicked, wasTyped] = [picked, typed];
    reply({ text }, text, () => {
      setPicked(wasPicked);
      setTyped(wasTyped);
    });
  };

  const say = () => {
    const text = message.trim();
    if (!text) return;
    setMessage('');
    reply({ text }, text, () => setMessage(text));
  };

  const continueFrom = (no: number) => {
    if (busy || !window.confirm(t('rollbackConfirm', { no }))) return;
    promptGenApi.rollback(no)
      .then(v => {
        setView(v);
        setViewing(null);
        setRestarting(false);
        // 回到另一个对话写的版本时，指令语言跟着那一版
        setOutputLang(v.outputLang as Lang);
      })
      .catch(err => toast((err as Error).message, 'error'));
  };

  const pick = (option: string) => {
    setPicked(option);
    setTyped('');
  };

  /** 旧的在左 */
  const versions = useMemo(() => [...(view?.versions ?? [])].reverse(), [view]);
  const shownNo = viewing ?? view?.currentNo ?? view?.versions[0]?.no ?? null;
  const shown = view?.versions.find(v => v.no === shownNo) ?? null;
  const body = draft ?? shown?.body ?? '';
  const question = active && !busy ? view?.question ?? null : null;
  const stageKey = stage ?? (!active ? 'idle' : view?.done ? 'done' : question ? 'asking' : 'yourTurn');

  const copy = () => {
    navigator.clipboard.writeText(body)
      .then(() => toast(t('copied'), 'success'))
      .catch(() => toast(t('copyFailed'), 'error'));
  };

  const renderItem = (item: PromptGenItem, index: number) => {
    switch (item.type) {
      case 'agent':
        return <AgentLine key={index} text={item.text ?? ''} />;
      case 'user':
        return <UserLine key={index} text={item.text ?? ''} />;
      case 'question':
        // 还没答的那道在下面的题卡里
        if (item.answer == null || !item.question) return null;
        return (
          <div key={index} className="space-y-2">
            <AgentLine text={item.question.question} />
            <UserLine text={item.answer} />
          </div>
        );
      case 'version':
        // 只留最近 3 版：被挤掉的那版打不开了，照实写出来
        return view?.versions.some(v => v.no === item.no) ? (
          <button key={index} type="button" onClick={() => item.no != null && setViewing(item.no)}
                  className="w-full flex items-center justify-center gap-1.5 text-[10px] font-bold text-muted-foreground hover:text-primary">
            <FileText className="w-3 h-3 inline-flex" />{t('wrote', { no: item.no })}
          </button>
        ) : (
          <p key={index} className="flex items-center justify-center gap-1.5 text-[10px] font-bold text-muted-foreground/60">
            <FileText className="w-3 h-3 inline-flex" />{t('wrotePruned', { no: item.no })}
          </p>
        );
    }
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
          {t(`stage.${stageKey}`)}
        </span>
      </div>

      {/* 开场：端点 + 指令语言；对话开着时锁住，要换就重新开始 */}
      <div className="rounded-lg pt-card p-4 space-y-3">
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <div className="space-y-1.5">
            <span className="microlabel block">{t('endpoint')}</span>
            <LlmEndpointSelect endpoints={endpoints} value={endpointId} onChange={setEndpointId}
                               disabled={active || busy} className="w-full sm:min-w-[280px]" />
          </div>
          <div className="space-y-1.5">
            <span className="microlabel block">{t('outputLang')}</span>
            <div className="flex h-9 border border-border divide-x divide-border">
              {SUPPORTED_LANGS.map(l => (
                <button key={l} type="button" disabled={active || busy} onClick={() => setOutputLang(l)}
                        className={cn('px-3 text-xs font-bold disabled:opacity-60',
                          outputLang === l
                            ? 'bg-card-2 text-foreground shadow-[inset_0_2px_0_var(--color-primary)]'
                            : 'text-muted-foreground')}>
                  {t(`lang.${l}`)}
                </button>
              ))}
            </div>
          </div>
          {active ? (
            <button onClick={() => setRestarting(true)} disabled={busy}
                    className="border border-border hover:bg-surface-hover rounded-lg px-4 py-2 text-xs font-black text-primary inline-flex items-center gap-1.5 disabled:opacity-50">
              <RotateCcw className="w-3.5 h-3.5" />
              {t('restart')}
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <button onClick={start} disabled={busy || endpoints.length === 0}
                      className="border border-border hover:bg-surface-hover rounded-lg px-4 py-2 text-xs font-black text-primary inline-flex items-center gap-1.5 disabled:opacity-50">
                {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                {t('start')}
              </button>
              {restarting && (
                <button onClick={() => setRestarting(false)} className={cn(BTN, 'text-muted-foreground')}>
                  {t('cancelRestart')}
                </button>
              )}
            </div>
          )}
        </div>
        <p className="text-[10px] leading-relaxed text-muted-foreground/70">
          {endpoints.length === 0 ? t('noEndpoint') : restarting ? t('restartHint') : t('startHint')}
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* 左列：对话 */}
        <div className="rounded-lg pt-card p-4 space-y-3">
          <span className="microlabel block">{t('dialog')}</span>
          {!active && !busy ? (
            <p className="text-[11px] leading-relaxed text-muted-foreground">{t('desc')}</p>
          ) : (
            <>
              <div ref={scrollRef} className="space-y-2.5 max-h-[55vh] overflow-y-auto pr-1">
                {(view?.items ?? []).map(renderItem)}
                {sent && <UserLine text={sent} />}
                {liveTexts.map((text, i) => <AgentLine key={`live-${i}`} text={text} />)}
              </div>

              {question && (
                <div className="rounded-lg border border-border p-3 space-y-3">
                  <p className="text-xs font-bold leading-relaxed">{question.question}</p>
                  <button type="button" onClick={() => pick(question.recommended)}
                          className={cn(CHIP, 'w-full text-left space-y-1', picked === question.recommended ? CHIP_ON : CHIP_OFF)}>
                    <span className="inline-flex items-center gap-1 text-[10px] font-black text-primary">
                      <ThumbsUp className="w-3 h-3 inline-flex" />{t('recommended')}
                    </span>
                    <span className="block text-foreground">{question.recommended}</span>
                  </button>
                  {question.reason && (
                    <p className="text-[10px] leading-relaxed text-muted-foreground">{question.reason}</p>
                  )}
                  {question.options.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {question.options.map(op => (
                        <button key={op} type="button" onClick={() => pick(op)}
                                className={cn(CHIP, picked === op ? CHIP_ON : CHIP_OFF)}>
                          {op}
                        </button>
                      ))}
                    </div>
                  )}
                  <input value={typed} placeholder={t('freePh')}
                         onChange={e => {
                           setTyped(e.target.value);
                           if (e.target.value.trim()) setPicked('');
                         }}
                         className="w-full h-9 rounded-lg border border-border bg-card-2 px-2.5 text-xs" />
                  <div className="flex flex-wrap items-center gap-2">
                    <button onClick={answer} disabled={!(typed.trim() || picked)} className={cn(BTN, 'text-primary')}>
                      <ArrowRight className="w-3 h-3 inline-flex" />{t('submit')}
                    </button>
                    {question.required ? (
                      <span className="text-[10px] text-muted-foreground">{t('requiredHint')}</span>
                    ) : (
                      <button onClick={() => reply({ skip: true }, t('skipped'))} className={cn(BTN, 'text-muted-foreground')}>
                        <SkipForward className="w-3 h-3 inline-flex" />{t('skip')}
                      </button>
                    )}
                  </div>
                </div>
              )}

              {active && !busy && !question && (
                <div className="flex gap-2">
                  <input value={message} onChange={e => setMessage(e.target.value)}
                         onKeyDown={e => { if (e.key === 'Enter' && !e.nativeEvent.isComposing) say(); }}
                         placeholder={view?.done ? t('doneHint') : t('composerPh')}
                         className="flex-1 min-w-0 h-9 rounded-lg border border-border bg-card-2 px-2.5 text-xs" />
                  <button onClick={say} disabled={!message.trim()} className={cn(BTN, 'text-primary')}>
                    <Send className="w-3 h-3 inline-flex" />{t('send')}
                  </button>
                </div>
              )}
            </>
          )}
        </div>

        {/* 右列：版本与正文 */}
        <div className="rounded-lg pt-card p-4 space-y-3">
          <div className="flex items-baseline justify-between gap-2">
            <span className="microlabel">{t('versions')}</span>
            <span className="text-[10px] text-muted-foreground">{t('keepHint')}</span>
          </div>
          {versions.length === 0 ? (
            <p className="text-[11px] text-muted-foreground">{t('noVersion')}</p>
          ) : (
            <div className="flex flex-wrap items-center gap-1.5">
              {versions.map(v => (
                <button key={v.no} type="button" onClick={() => setViewing(v.no)}
                        className={cn(CHIP, 'num inline-flex items-center gap-1.5',
                          v.no === shownNo && draft == null ? CHIP_ON : CHIP_OFF)}>
                  {t('version', { no: v.no })}
                  {/* 当前对话停在这一版 */}
                  {v.no === view?.currentNo && <span className="w-1.5 h-1.5 rounded-full bg-primary" />}
                </button>
              ))}
              {view?.active && shown && shown.no !== view.currentNo && !busy && draft == null && (
                <button onClick={() => continueFrom(shown.no)} className={cn(BTN, 'text-primary ml-auto')}>
                  <Undo2 className="w-3 h-3 inline-flex" />{t('continueFrom')}
                </button>
              )}
            </div>
          )}

          <div className="flex items-center justify-between gap-2 pt-1">
            <span className="microlabel">{t('body')}</span>
            <div className="flex items-center gap-3">
              {shown?.finalVersion && draft == null && (
                <span className="inline-flex items-center gap-1 text-[10px] font-black text-gain">
                  <BadgeCheck className="w-3 h-3 inline-flex" />{t('final')}
                </span>
              )}
              <span className={cn('text-[11px] num', body.length > MAX_CHARS ? 'text-loss' : 'text-muted-foreground')}>
                {t('chars', { n: body.length, max: MAX_CHARS })}
              </span>
              <button onClick={copy} disabled={!body || draft != null}
                      className="inline-flex items-center gap-1 text-[11px] font-bold text-primary hover:underline disabled:opacity-50">
                <Copy className="w-3 h-3 inline-flex" />{t('copy')}
              </button>
            </div>
          </div>

          {stage === 'writing' && !draft && (
            <div className="flex justify-center py-2">
              <FluidOrb size={120} color={orb.color} base={orb.base} />
            </div>
          )}

          {body
            ? <pre className="whitespace-pre-wrap font-sans text-xs leading-relaxed">{body}</pre>
            : <p className="text-[11px] leading-relaxed text-muted-foreground">{t('empty')}</p>}

          {shown?.notes && draft == null && (
            <details className="rounded-lg border border-border bg-card-2/50">
              <summary className="px-3 py-2 text-[11px] font-bold text-muted-foreground cursor-pointer">{t('notes')}</summary>
              <pre className="px-3 pb-3 text-[11px] leading-relaxed text-muted-foreground whitespace-pre-wrap font-sans">{shown.notes}</pre>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}

function AgentLine({ text }: { text: string }) {
  return (
    <div className="mr-8 rounded-lg bg-card-2 px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap">{text}</div>
  );
}

function UserLine({ text }: { text: string }) {
  return (
    <div className="ml-8 rounded-lg border border-primary/30 bg-primary/5 px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap">{text}</div>
  );
}
