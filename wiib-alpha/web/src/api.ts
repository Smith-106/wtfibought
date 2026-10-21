import i18n, { currentLang } from './i18n';

// ========== 类型：对着 wiib-alpha 的 main.py / prompts.py / graph.py / panel.py 写 ==========

/** describe() 报覆盖率的列，与 panel.COVERAGE_COLUMNS 同 */
export const COVERAGE_COLUMNS = ['funding_rate', 'premium', 'oi', 'ls_acct', 'liq_count'] as const;
export type CoverageColumn = (typeof COVERAGE_COLUMNS)[number];

/** 一张面板的统计（panel.describe），时间都是 bar 收盘时刻 ms */
export interface PanelStatus {
  rows: number;
  from: number | null;
  to: number | null;
  liqFrom: number | null;
  /** 相邻两根不是正好 15m 的地方 */
  gaps: number;
  coverage: Partial<Record<CoverageColumn, number>>;
  /** 面板带 in_pool 列才有 */
  poolMonths?: number;
}

/** 给提示词和页面的数据事实（main._facts），日期是 UTC 的 YYYY-MM-DD；symbols 是产品币 */
export interface Facts {
  symbols: string[];
  from: string;
  to: string;
  rows: number;
  liqFrom: string | null;
  holdoutFrom: string;
  /** 多空腿单边成本 bps：当月排名 1~20 / 21~50 / 其余 */
  costTiers: number[];
  /** 每月进池的名额 */
  poolSize: number;
  /** 装进评估器的池子币数 */
  poolSymbols: number;
}

export type RefreshStage = 'symbols' | 'daily' | 'panel' | 'reference';

/** 启动时补算参照网格的进度；ready 是网格已经能给 G6 用 */
export interface GridStatus {
  ready: boolean;
  running: boolean;
  startedAt?: number;
  finishedAt?: number | null;
  total?: number;
  done?: number;
  symbol?: string | null;
  error?: string | null;
}

/** 刷新任务的进度字典：开始时只重置 running/startedAt/error/finishedAt，其余是 refresh() 一路写进来的 */
export interface RefreshJob {
  running: boolean;
  startedAt?: number;
  finishedAt?: number | null;
  error?: string | null;
  stage?: RefreshStage;
  total?: number;
  done?: number;
  symbol?: string | null;
  underlyingTypes?: string[];
  candidates?: number;
  members?: number;
  /** 费率或基差覆盖偏低的币 → 哪列覆盖多少；照常在池，缺的那几段是空值 */
  thin?: Record<string, string>;
}

export interface DataStatus {
  loaded: boolean;
  loadedAt: number | null;
  facts: Facts | null;
  /** 装进内存的产品币面板 */
  symbols: Record<string, PanelStatus>;
  pool: {
    /** YYYY-MM → 这个月池子里几个币 */
    months: Record<string, number>;
    /** 历史上进过池的币数 */
    symbols: number;
    /** 刷新时落的 status.json：全部面板 */
    status: Record<string, PanelStatus>;
    /** 装进评估器的骨架：币数、月数、6h 网格点数、(时刻, 币) 行数；没装载是 null */
    loaded: { symbols: number; months: number; gridPoints: number; rows: number } | null;
    grid: GridStatus;
  };
  referenceCount: number;
  refresh: RefreshJob;
}

export type RunStatus = 'running' | 'done' | 'failed';

export interface Run {
  id: string;
  created_at: number;
  status: RunStatus;
  k: number;
  note: string;
  lang: string;
  error: string | null;
  /** 不属于 campaign 的老轮次是 null */
  campaign_id: string | null;
}

/** 候选状态，按流程先后排；passed 是轮内中间态，commit 时变成 observing */
export const CANDIDATE_STATUSES = [
  'proposed', 'evaluating', 'rewrite', 'reviewing', 'revise', 'passed',
  'rejected_gates', 'rejected_review', 'rejected_fdr', 'failed', 'observing', 'approved', 'declined',
] as const;
export type CandidateStatus = (typeof CANDIDATE_STATUSES)[number];

/** 候选 status → 条数，没有的状态不出键 */
export type StatusCounts = Partial<Record<CandidateStatus, number>>;

/** 列表里的轮次带各状态候选数 */
export interface RunRow extends Run {
  counts: StatusCounts;
}

/** stopping 是点了停、当前这一轮跑完就停 */
export type CampaignStatus = 'running' | 'stopping' | 'stopped' | 'done' | 'failed';

export interface Campaign {
  id: string;
  created_at: number;
  status: CampaignStatus;
  rounds: number;
  k: number;
  note: string;
  lang: string;
  max_survivors: number | null;
  /** 截止时刻 ms，不限时是 null */
  deadline_ms: number | null;
  done_rounds: number;
  /** 累计进观察区的条数 */
  survivors: number;
  stop_reason: string | null;
  error: string | null;
}

export interface CampaignCreate {
  rounds: number;
  k: number;
  note: string;
  maxSurvivors: number | null;
  deadlineMinutes: number | null;
}

export type LessonKind = 'avoid' | 'try' | 'note';

/** 轮后复盘写的一条经验 */
export interface Lesson {
  id: number;
  run_id: string;
  campaign_id: string | null;
  kind: LessonKind;
  text: string;
  created_at: number;
}

/** campaign 详情：各轮与经验都是最新在前 */
export interface CampaignDetail {
  campaign: Campaign;
  runs: RunRow[];
  lessons: Lesson[];
}

export interface Review {
  verdict: 'pass' | 'revise' | 'reject';
  comment: string;
  suggestion: string;
}

/** brief 里的门：只有 id、过没过、一句原因 */
export interface GateBrief {
  id: string;
  passed: boolean;
  note: string;
}

/** 候选摘要里的门多一个中文门名 */
export interface CandidateGate extends GateBrief {
  name: string | null;
}

/** 完整的门（手填评估、因子详情）：带数字与阈值 */
export interface Gate extends GateBrief {
  name: string;
  value: unknown;
  threshold: unknown;
}

/** 一档的精简数字（prompts.horizons_brief），键是 6h/12h/24h；全是截面口径 */
export interface HorizonBrief {
  meanIc: number | null;
  icir: number | null;
  months: number;
  /** 分组单调性：三组平均收益与组序的 Spearman ρ */
  monotonic: number | null;
  /** 顶组减底组的平均收益：每期两头各截掉 5% 再减当期均值（之前落库的老结果是原始算术收益） */
  spread: number | null;
  /** 顶组减底组每期之差的 t，G2 要它和 IC 同向且 ≥ 2；老结果是 null */
  spreadT: number | null;
  /** 三组的形状，按 IC 方向从弱的一头看到强的一头；只做标签，老结果是 null */
  shape: 'monotone' | 'weakEnd' | 'strongEnd' | 'flat' | 'midHigh' | 'midLow' | null;
  segments: (number | null)[];
  netReturn: number | null;
  /** 排列检验只做主档，别的档是 null */
  permPct: number | null;
  hitRate: number | null;
  exposure: number | null;
  /** 每期 Σ|Δw| / 4：两条腿整个换掉是 1 */
  turnover: number | null;
  /** 多空腿每期平均资金费 bps，正数是收、负数是付；已经算在净收益里。还没算资金费时的老结果是 null */
  fundingBps: number | null;
  /** 盈亏平衡单边成本 bps：Σ毛收益（含资金费）/ Σ换手量；没换过手是 null */
  breakEvenBps: number | null;
}

export interface HoldoutHorizon {
  meanIc: number | null;
  icir: number | null;
  /** 月度 IC 均值的标准误；之前算的 holdout 没有 */
  se: number | null;
  months: number;
  monotonic: number | null;
  samples: number;
}

/** holdout 段的 IC（prompts.holdout_brief） */
export interface HoldoutBrief {
  since: number;
  horizons: Record<string, HoldoutHorizon>;
}

/** 产品币的时序报告（prompts.timeseries_brief），不做门 */
export interface TimeseriesBrief {
  horizons: Record<string, { meanIc: number | null; icir: number | null; sameSignSymbols: number; netReturn: number | null }>;
  /** 主档每币的时序 IC */
  bySymbol: Record<string, number | null>;
}

/** prompts.eval_brief：算过增量的带 gainSharpe，算过 holdout 的带 holdout，有产品币面板的带时序报告 */
export interface EvalBrief {
  /** 判门用的那一档（6h/12h/24h）；之前落库的老结果没有，是 12h */
  primary?: string;
  horizons: Record<string, HorizonBrief>;
  /** ready = false 是参照网格还没算好 */
  redundancy: { maxAbs: number | null; with: string | null; ready: boolean };
  complexity: { nodes: number; windows: number; constants: number; cross: number };
  gates: GateBrief[];
  /** 去老效应标签（不是门）：回归掉 against 条老效应后的主档 IC、占原 IC 的比例、t；网格没算好或老结果没有 */
  neutral?: { meanIc: number | null; icir: number | null; t: number | null; retained: number | null; against: number };
  /** 老结果的 G9：加进库组合后三档 IC 的变化；现在的结果没有 */
  gain?: Record<string, number | null>;
  /** G9：交易方案（库里因子的组合 + 基差 carry）搜索段的年化净 Sharpe，加进去前、后、增量；老结果是库组合多空腿的 */
  gainSharpe?: { base: number | null; with: number | null; gain: number | null };
  holdout?: HoldoutBrief;
  /** 复评时才有：发现它的那一轮切点比现在的 holdout 切点早，从那个切点起它都没见过 */
  unseen?: HoldoutBrief;
  timeseries?: TimeseriesBrief;
}

/** graph.brief：页面用的候选摘要 */
export interface Candidate {
  id: string;
  seq: number;
  name: string;
  hypothesis: string;
  family: string | null;
  expr: string | null;
  note: string | null;
  /** 按假设声明的方向：1 因子值越大越强、-1 越弱；声明方向之前的老候选是 null */
  direction: 1 | -1 | null;
  status: CandidateStatus;
  attempts: number;
  reason: string | null;
  review: Review | null;
  gates?: CandidateGate[];
  brief?: EvalBrief;
}

/** 观察区复评（main._reeval_view）：当前评估器的结果，原来的 eval 不动 */
export interface Reeval {
  at: number;
  direction: 1 | -1;
  /** 老候选没声明方向，复评时问生成模型补的 */
  asked: boolean;
  /** 单条全部门都过、同批查重也留下了 */
  passed: boolean;
  reason: string | null;
  /** 和观察区其它候选的截面 |ρ|：[名字, 值]，名字可能重复 */
  peers: [string, number][];
  gates: CandidateGate[];
  brief: EvalBrief;
}

/** 观察区的一条：created_at 是它那一轮开始的时间，updated_at 是进观察区的时间 */
export interface ObservationItem extends Candidate {
  run_id: string;
  created_at: number;
  updated_at: number;
  reeval: Reeval | null;
}

export interface ReevalJob {
  running: boolean;
  startedAt?: number;
  finishedAt?: number | null;
  total?: number;
  done?: number;
  /** 正在评估的候选名 */
  current?: string | null;
  error?: string | null;
}

export interface RunState {
  run: Run;
  candidates: Candidate[];
  trials: number;
}

/** 轮内阶段；working 是页面自己按候选推断的 */
export type Stage = 'proposing' | 'working' | 'summarizing' | 'reflecting' | 'committing';

/** 一轮的 SSE 帧；end 是流尾，带 lastId 给断线重连 */
export type RunEvent = { id: number } & (
  | { event: 'status'; data: { stage: Exclude<Stage, 'working'> } }
  | { event: 'log'; data: { text: string } }
  | { event: 'candidate'; data: Candidate }
  | { event: 'lessons'; data: { items: Pick<Lesson, 'kind' | 'text'>[] } }
  | { event: 'done'; data: { observing: number } }
  | { event: 'error'; data: { message: string } }
  | { event: 'end'; data: { status: RunStatus; lastId: number } }
);

export interface RecheckHorizon {
  mean: number | null;
  icir: number | null;
  months: number;
  samples: number;
  monotonic: number | null;
}

/** 入库后复查（eval.recheck）：键是 bar 数 24/48/96 */
export interface Recheck {
  since: number;
  at: number;
  horizons: Record<string, RecheckHorizon>;
}

/** 因子列表项（main._factor_view full=False） */
export interface Factor {
  id: string;
  name: string;
  hypothesis: string;
  expr: string;
  canonical: string;
  columns: string[];
  approved_at: number;
  run_id: string;
  review: Review | null;
  recheck: Recheck | null;
  /** 用当前评估器复评的结果；没有同批，asked 恒 false、peers 空 */
  reeval: Reeval | null;
  brief: EvalBrief;
}

/** 因子详情多带整份 eval 和完整的门 */
export interface FactorDetail extends Factor {
  eval: Record<string, unknown>;
  gates: Gate[];
}

/** 库里全部因子的组合（main.build_library 的 report） */
export interface ComboReport {
  brief: { primary?: string; horizons: Record<string, HorizonBrief>; holdout: HoldoutBrief };
  /** latest 对应的时刻：最后一个 6h 网格点 */
  asOf: number;
  weights: { name: string; weight: number }[];
  /** 那一刻每个在池币的组合信号（截面 z 的加权和） */
  latest: Record<string, number | null>;
}

export interface LibraryList {
  factors: Factor[];
  /** 库空或没装载数据时是 null */
  combo: ComboReport | null;
  /** 库里因子复评任务的进度 */
  reeval: ReevalJob;
}

export interface ReferenceJob {
  running: boolean;
  startedAt?: number;
  total?: number;
  done?: number;
  error?: string | null;
  finishedAt?: number;
}

export interface ReferenceItem {
  name: string;
  expr: string;
  passed: boolean;
  updatedAt: number;
  brief: EvalBrief;
}

export interface ReferenceList {
  job: ReferenceJob;
  items: ReferenceItem[];
}

export interface EvalResult {
  expr: string;
  canonical: string;
  passed: boolean;
  brief: EvalBrief;
  gates: Gate[];
  eval: Record<string, unknown>;
}

/** 手填评估是后台任务；queued 是 campaign 正在评估的两条占着，排队 */
export interface EvalJob {
  running: boolean;
  startedAt?: number;
  finishedAt?: number | null;
  expr?: string;
  stage?: 'queued' | 'evaluating' | null;
  error?: string | null;
  result?: EvalResult | null;
}

/** 公开概览：每轮只有状态和各状态候选数，没有假设、公式和过程 */
export interface Overview {
  runs: {
    id: string;
    created_at: number;
    status: RunStatus;
    k: number;
    counts: StatusCounts;
  }[];
  trials: number;
  /** 库里因子数 */
  factors: number;
  /** 观察区条数 */
  observing: number;
}

/** 挖掘用的两个角色：gen 提假设写表达式，review 评审 */
export type ModelRole = 'gen' | 'review';
export const MODEL_ROLES: ModelRole[] = ['gen', 'review'];
export const PROTOCOLS = ['openai', 'responses', 'anthropic', 'gemini'] as const;
export type Protocol = (typeof PROTOCOLS)[number];

/** 存着的一条模型端点；key 只回尾 4 位 */
export interface ModelView {
  protocol: Protocol;
  baseUrl: string;
  model: string;
  effort: string;
  apiKeyTail: string;
  updatedAt: number;
}

/** 保存与测试共用；apiKey 留空 = 沿用已存的 */
export interface ModelSave {
  protocol: Protocol;
  baseUrl: string;
  model: string;
  effort: string;
  apiKey: string;
}

export type Models = Record<ModelRole, ModelView | null>;

// ========== 交易员：对着 main.py 的 /trader 接口与 trade/ 下的模块写 ==========

/** 调仓任务的进度（scheduler.rebalance）；live 阶段 done/total/symbol 是补尾巴的进度 */
export interface TraderJob {
  running: boolean;
  startedAt?: number;
  finishedAt?: number | null;
  /** 这次调仓的 UTC 日期 */
  day?: string;
  stage?: 'prepare' | 'account' | 'live' | 'targets' | 'waiting' | 'orders' | null;
  total?: number;
  done?: number;
  symbol?: string | null;
  error?: string | null;
}

/** 账户概况；还没调过仓（还没建账户）的都是 null */
export interface TraderAccount {
  userId: number | null;
  /** 全仓净值 = 余额 + Σ浮盈亏 */
  equity: number | null;
  balance: number | null;
  /** 全仓可用 = 净值 − 占用保证金 − 挂单占用 */
  available: number | null;
  initialBalance: number;
  job: TraderJob;
}

export type TradeLeg = 'combo' | 'carry';
export const TRADE_LEGS: TradeLeg[] = ['combo', 'carry'];

export interface TraderPosition {
  symbol: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  /** 数量 × 标记价 */
  notional: number;
  entryPrice: number;
  markPrice: number;
  unrealizedPnl: number;
  /** 最近一次出了目标的调仓里两条腿各给这个币的目标名义（带正负）；那次没这个币是 null */
  legs: Record<TradeLeg, number> | null;
}

/** 每小时一次的净值快照；long/short 是多空持仓名义 */
export interface EquityPoint {
  ts: number;
  equity: number;
  balance: number;
  long: number;
  short: number;
}

/** done 全成；partial 有币失败或停在半路但有成交；failed 一笔没成 */
export type TradeStatus = 'done' | 'partial' | 'failed';

export interface TradeRunBrief {
  day: string;
  status: TradeStatus;
  /** 出信号那根（UTC 0 点那根 15m）的收盘时刻 */
  signalMs: number;
  startedAt: number;
  finishedAt: number;
  equityBefore: number | null;
  equityAfter: number | null;
  error: string | null;
  factorNames: string[];
  orderCount: number;
  failedCount: number;
}

/** 一个币的目标与现有（executor.plan 的明细）：数量带正负，多是正 */
export interface TradeCoin {
  symbol: string;
  /** 净权重 = combo + carry，乘净值是目标名义 */
  weight: number;
  combo: number;
  carry: number;
  notional: number;
  price: number | null;
  /** 00:15 那根的收盘价：回测按它成交 */
  refPrice: number | null;
  current: number;
  target: number | null;
  /** 没下单的原因：差额太小、要加的量太小、缺价格 */
  note: string | null;
  failed: boolean;
}

/** 一条腿里一个币这一期的信号、名次（信号从大到小数）、波动、腿内权重 */
export interface LegSignal {
  symbol: string;
  signal: number;
  rank: number;
  vol: number;
  weight: number;
}

export interface TradeOrder {
  symbol: string;
  action: 'open' | 'close';
  side: 'LONG' | 'SHORT';
  /** null 是全平 */
  quantity: number | null;
  requestId: string;
  ok: boolean;
  orderId?: number;
  filled?: number;
  price?: number;
  commission?: number;
  error?: string;
}

export interface TradeRunDetail extends TradeRunBrief {
  userId: number | null;
  factors: { id: string; name: string; expr: string; weight: number }[];
  legs: Record<TradeLeg, { long: string[]; short: string[] }> | null;
  targets: { coins: TradeCoin[]; signals: Record<TradeLeg, LegSignal[]>; notTrading: string[] } | null;
  orders: TradeOrder[];
}

/** 库里的因子与交易员的勾选；missing 是 live 尾巴补不出来的列，有就不能勾 */
export interface TraderFactor {
  id: string;
  name: string;
  expr: string;
  selected: boolean;
  missing: string[];
}

/** 交易方案回测（scheme.backtest） */
export interface SchemeResult {
  periods: number;
  netReturn: number | null;
  sharpe: number | null;
  maxDrawdown: number | null;
  /** 每期 Σ|Δw| / 4：两条腿各自整个换掉是 1 */
  turnover: number | null;
  /** carry 腿每期平均资金费 bps，正数是收 */
  carryFundingBps: number | null;
}

export interface CompareSide {
  factors: string[];
  search: SchemeResult;
  /** 加减之后搜索段更好才有 */
  holdout?: SchemeResult;
}

export interface CompareJob {
  running: boolean;
  startedAt?: number;
  finishedAt?: number | null;
  factorId?: string;
  name?: string;
  stage?: 'library' | 'inputs' | 'search' | 'holdout' | null;
  error?: string | null;
  result?: { action: 'add' | 'remove'; better: boolean; base: CompareSide; alt: CompareSide } | null;
}

// ========== 鉴权 ==========

export class ApiError extends Error {
  readonly code: number;

  constructor(message: string, code: number) {
    super(message);
    this.code = code;
  }
}

// 与 wiib-web 同 key 同格式（zustand persist）：同域部署，主站登录了这里就带着同一个 token
const USER_KEY = 'wiib-user';

// 本地 vite dev 和主站不同源，手里没有主站的 token 时用 .env.development.local 里的 VITE_DEV_TOKEN（文件不进库）
const DEV_TOKEN: string | undefined = import.meta.env.DEV ? import.meta.env.VITE_DEV_TOKEN : undefined;

export function getToken(): string | null {
  const raw = localStorage.getItem(USER_KEY);
  const stored = raw ? (JSON.parse(raw) as { state?: { token?: string | null } }).state?.token : null;
  return stored ?? DEV_TOKEN ?? null;
}

export function clearToken(): void {
  localStorage.removeItem(USER_KEY);
}

function headers(json: boolean): Record<string, string> {
  const token = getToken();
  return {
    'X-Lang': currentLang(),
    ...(token ? { satoken: token } : {}),
    ...(json ? { 'Content-Type': 'application/json' } : {}),
  };
}

/** 返回体 {code,msg,data}：401 走 HTTP 200 + code 401，别的错误 HTTP 状态与 code 同值 */
async function unwrap<T>(res: Response): Promise<T> {
  const json = (res.headers.get('content-type') ?? '').includes('application/json');
  const body = json ? (await res.json() as { code?: number; msg?: string; data?: T }) : null;
  if (body?.code === 0) return body.data as T;
  throw new ApiError(body?.msg || i18n.t('errors.http', { status: res.status }), body?.code ?? res.status);
}

async function call<T>(method: 'GET' | 'POST' | 'PUT', path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: headers(body !== undefined),
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  return unwrap<T>(res);
}

// ========== SSE ==========

/** 一帧 "id: / event: / data:"；只有注释行（": ping"）的帧返回 null */
function parseFrame(block: string): RunEvent | null {
  const fields: Record<string, string> = {};
  for (const line of block.split('\n')) {
    if (line.startsWith(':')) continue;
    const i = line.indexOf(':');
    const value = line.slice(i + 1);
    fields[line.slice(0, i)] = value.startsWith(' ') ? value.slice(1) : value;
  }
  if (fields.data === undefined) return null;
  return { id: Number(fields.id), event: fields.event, data: JSON.parse(fields.data) } as RunEvent;
}

/** fetch 流读 SSE：帧以空行分隔；准入失败时后端回的是 JSON 错误体 */
async function stream(path: string, onEvent: (e: RunEvent) => void, signal: AbortSignal): Promise<void> {
  const res = await fetch(path, { headers: headers(false), signal });
  if (!(res.headers.get('content-type') ?? '').startsWith('text/event-stream')) {
    await unwrap(res);
    return;
  }
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    buffer += decoder.decode(value, { stream: true });
    let cut = buffer.indexOf('\n\n');
    while (cut >= 0) {
      const frame = parseFrame(buffer.slice(0, cut));
      buffer = buffer.slice(cut + 2);
      if (frame) onEvent(frame);
      cut = buffer.indexOf('\n\n');
    }
  }
}

// ========== 接口 ==========

const ALPHA = '/api/alpha';

/** 公开的只有 overview、data/status、reference 和交易员的 account、positions、equity；其余要管理员 token，没 token 是 code 401，不是管理员是 403 */
export const alpha = {
  /** 手里的 token 是不是管理员 */
  me: (signal?: AbortSignal) => call<{ admin: boolean; lang: string }>('GET', `${ALPHA}/me`, undefined, signal),
  overview: () => call<Overview>('GET', `${ALPHA}/overview`),
  dataStatus: () => call<DataStatus>('GET', `${ALPHA}/data/status`),
  /** 后台刷新，进度看 dataStatus().refresh */
  refresh: () => call<{ started: boolean }>('POST', `${ALPHA}/data/refresh`),
  campaigns: () => call<{ campaigns: Campaign[] }>('GET', `${ALPHA}/campaigns`),
  campaign: (id: string) => call<CampaignDetail>('GET', `${ALPHA}/campaigns/${id}`),
  /** 建 campaign 并起第一轮；已经有在跑的是 409 */
  createCampaign: (body: CampaignCreate) => call<{ id: string; runId: string }>('POST', `${ALPHA}/campaigns`, body),
  /** 当前这一轮跑完就停 */
  stopCampaign: (id: string) => call<{ stopping: boolean }>('POST', `${ALPHA}/campaigns/${id}/stop`),
  /** 最近的轮次带各状态候选数；trials 是全局累计试验数 */
  runs: () => call<{ runs: RunRow[]; trials: number }>('GET', `${ALPHA}/runs`),
  run: (id: string, signal?: AbortSignal) => call<RunState>('GET', `${ALPHA}/runs/${id}`, undefined, signal),
  /** 先回放库里 id > after 的，再收在线的；这一轮不在 running 时以 end 结束 */
  events: (id: string, after: number, onEvent: (e: RunEvent) => void, signal: AbortSignal) =>
    stream(`${ALPHA}/runs/${id}/events?after=${after}`, onEvent, signal),
  observation: () => call<{ items: ObservationItem[]; reeval: ReevalJob }>('GET', `${ALPHA}/observation`),
  /** 观察区全部候选用当前评估器重算，后台跑，进度看 observation().reeval */
  reeval: () => call<{ started: boolean; total: number }>('POST', `${ALPHA}/observation/reeval`),
  /** 入库；名字和库里撞了会加序号，回来的 name 是入库用的名字 */
  approve: (id: string) => call<{ id: string; name: string }>('POST', `${ALPHA}/candidates/${id}/approve`),
  decline: (id: string) => call<{ id: string }>('POST', `${ALPHA}/candidates/${id}/decline`),
  library: () => call<LibraryList>('GET', `${ALPHA}/library`),
  factor: (id: string) => call<FactorDetail>('GET', `${ALPHA}/library/${id}`),
  recheck: (id: string) => call<Recheck>('POST', `${ALPHA}/library/${id}/recheck`),
  /** 库里全部因子用当前评估器重算，后台跑，进度看 library().reeval */
  libraryReeval: () => call<{ started: boolean; total: number }>('POST', `${ALPHA}/library/reeval`),
  /** 后台评估，进度和结果看 evalJob() */
  evaluate: (expr: string) => call<{ started: boolean }>('POST', `${ALPHA}/eval`, { expr }),
  evalJob: () => call<EvalJob>('GET', `${ALPHA}/eval`),
  reference: () => call<ReferenceList>('GET', `${ALPHA}/reference`),
  /** windows 为 null 跑全部窗口 */
  referenceRun: (windows: number[] | null) =>
    call<{ started: boolean; total: number }>('POST', `${ALPHA}/reference/run`, { windows }),
  models: () => call<Models>('GET', `${ALPHA}/models`),
  saveModel: (role: ModelRole, body: ModelSave) => call<ModelView>('PUT', `${ALPHA}/models/${role}`, body),
  /** 真发一句话，回答与耗时；上游报错会以 400 带回来 */
  testModel: (role: ModelRole, body: ModelSave) =>
    call<{ reply: string; ms: number }>('POST', `${ALPHA}/models/${role}/test`, body),
  /** 交易员：sim 不通时账户和持仓是 502 */
  traderAccount: () => call<TraderAccount>('GET', `${ALPHA}/trader/account`),
  traderPositions: () => call<{ items: TraderPosition[]; day: string | null }>('GET', `${ALPHA}/trader/positions`),
  traderEquity: () => call<{ initialBalance: number; items: EquityPoint[] }>('GET', `${ALPHA}/trader/equity`),
  traderRuns: () => call<{ items: TradeRunBrief[]; job: TraderJob }>('GET', `${ALPHA}/trader/runs`),
  traderRun: (day: string) => call<TradeRunDetail>('GET', `${ALPHA}/trader/runs/${day}`),
  traderFactors: () => call<{ items: TraderFactor[]; compare: CompareJob }>('GET', `${ALPHA}/trader/factors`),
  /** 勾上或取消，下一次调仓起生效；live 补不出来的列是 409 */
  selectTraderFactor: (id: string, selected: boolean) =>
    call<{ id: string; selected: boolean }>('PUT', `${ALPHA}/trader/factors/${id}`, { selected }),
  /** 后台比"当前勾选"和"加上 / 去掉这条"，进度看 traderFactors().compare */
  compareTraderFactor: (id: string) => call<{ started: boolean }>('POST', `${ALPHA}/trader/compare/${id}`),
};
