package com.mawai.wiibsim.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.mawai.wiibcommon.cache.CacheService;
import com.mawai.wiibcommon.entity.FuturesPosition;
import com.mawai.wiibcommon.market.BinanceRestClient;
import com.mawai.wiibsim.mapper.FuturesPositionMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.node.ArrayNode;

import java.math.BigDecimal;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongSupplier;
import java.util.stream.Collectors;

import static com.mawai.wiibcommon.util.JsonUtils.MAPPER;

/**
 * 资金费：全部币按各自的结算时刻收，结算时刻从 feed 写的 markPrice 推送来（当期费率 r、下次结算时刻 T）。
 * 每分钟一次：只看有未平仓位的币，Redis 记"上次看到的下次结算时刻"；到点了、feed 给的 T 也挪了，就是记下的那一期结算了——
 * 去 /fapi/v1/fundingRate 取这一期的实际费率（记录没出来下一分钟再试），给结算时刻之前开的仓收一次；先记已结算再收，不重收。
 * 还没到点 T 就变了，是 Binance 改了这个币的结算间隔，只换记下的时刻。
 * sim 停机跨过一次结算，起来后 T 已经挪了，照样补上；跨过更多的只收最早那一期，其余打日志。
 * 前端查费率也读 feed 写的，不打 Binance。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class FundingRateService {

    /** 每个币上次看到的下次结算时刻（毫秒）；3 天没仓位就作废，再开仓从 feed 给的 T 重新记 */
    private static final String NEXT_KEY_PREFIX = "futures:funding:next:";
    private static final Duration NEXT_TTL = Duration.ofDays(3);
    /** 结算记录的 fundingTime 比整点多几毫秒，差这么多以内算同一个结算时刻 */
    static final long FUNDING_TIME_MATCH_MS = 60_000L;
    /** 一次取最近几条结算记录 */
    private static final int FUNDING_HISTORY_LIMIT = 10;
    /** 连着取结算记录时每次之间隔开 */
    private static final long FUNDING_REQUEST_GAP_MS = 300L;

    private final BinanceRestClient restClient;
    private final CacheService cacheService;
    private final FuturesPositionMapper positionMapper;
    private final FuturesSettlementService settlementService;

    /** 墙钟注入点 */
    LongSupplier nowMs = System::currentTimeMillis;
    private final ReentrantLock settling = new ReentrantLock();

    /** 前端查询的返回体；rate 是小数（0.0001 = 0.01%/期），fetchedAt 是 feed 写入时刻，nextTime 是下次结算时刻 */
    public record FundingRateView(String symbol, BigDecimal rate, long fetchedAt, long nextTime) {}

    /** 前端查询：只读 feed 写的；没有合约行情的标的（bStock 等）、feed 停了都返回 null */
    public FundingRateView query(String symbol) {
        CacheService.FundingRate f = symbol == null ? null : cacheService.getFundingRate(symbol);
        return f == null ? null : new FundingRateView(symbol, f.rate(), f.updatedAt(), f.nextTime());
    }

    /** 每分钟一次；上一次还没跑完就跳过 */
    public void settleDue() {
        if (!settling.tryLock()) return;
        try {
            long now = nowMs.getAsLong();
            Map<String, List<FuturesPosition>> open = positionMapper.selectList(new LambdaQueryWrapper<FuturesPosition>()
                            .eq(FuturesPosition::getStatus, "OPEN"))
                    .stream().collect(Collectors.groupingBy(FuturesPosition::getSymbol));
            boolean requested = false;
            for (Map.Entry<String, List<FuturesPosition>> entry : open.entrySet()) {
                String symbol = entry.getKey();
                CacheService.FundingRate funding = cacheService.getFundingRate(symbol);
                if (funding == null) continue;
                long next = funding.nextTime();
                Long seen = seenNext(symbol);
                if (seen == null) {
                    putSeenNext(symbol, next);
                    continue;
                }
                if (next == seen) continue;
                if (now < seen - FUNDING_TIME_MATCH_MS) {
                    // 还没到记下的时刻 T 就变了：Binance 改了这个币的结算间隔，不是结算
                    putSeenNext(symbol, next);
                    continue;
                }
                // 到点了、T 也挪了：seen 这一期结算了。没有要收的仓位就不取记录
                List<FuturesPosition> positions = openedBefore(entry.getValue(), seen);
                if (positions.isEmpty()) {
                    putSeenNext(symbol, next);
                    continue;
                }
                if (requested) {
                    try {
                        Thread.sleep(FUNDING_REQUEST_GAP_MS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
                requested = true;
                TreeMap<Long, BigDecimal> history = parseHistory(
                        restClient.getFundingRateHistory(symbol, FUNDING_HISTORY_LIMIT), symbol);
                if (history == null) continue;
                BigDecimal rate = rateAt(history, seen);
                if (rate == null) {
                    // 记录还没出来：下一分钟再试。已经有更晚的记录却没有这一条：它不会再出来了，这一期不收
                    if (history.isEmpty() || history.lastKey() <= seen + FUNDING_TIME_MATCH_MS) continue;
                    log.warn("资金费结算记录已经取不到，这一期不收 symbol={} 结算时刻={}", symbol, seen);
                    putSeenNext(symbol, next);
                    continue;
                }
                List<Long> missed = List.copyOf(history.subMap(seen + FUNDING_TIME_MATCH_MS,
                        Math.max(seen + FUNDING_TIME_MATCH_MS, next - FUNDING_TIME_MATCH_MS)).keySet());
                if (!missed.isEmpty()) {
                    log.warn("资金费漏收（停机期间错过，不补）symbol={} 结算时刻={}", symbol, missed);
                }
                // 先记已结算再收，收到一半出错也不会重收
                putSeenNext(symbol, next);
                settlementService.chargeFundingFee(symbol, seen, rate, positions);
            }
        } finally {
            settling.unlock();
        }
    }

    private Long seenNext(String symbol) {
        String v = cacheService.get(NEXT_KEY_PREFIX + symbol);
        return v == null ? null : Long.valueOf(v);
    }

    private void putSeenNext(String symbol, long next) {
        cacheService.set(NEXT_KEY_PREFIX + symbol, String.valueOf(next), NEXT_TTL);
    }

    /** 结算时刻之前开的仓（createdAt 是按本机时区填的） */
    static List<FuturesPosition> openedBefore(List<FuturesPosition> positions, long fundingTime) {
        LocalDateTime cutoff = LocalDateTime.ofInstant(Instant.ofEpochMilli(fundingTime), ZoneId.systemDefault());
        return positions.stream().filter(p -> p.getCreatedAt().isBefore(cutoff)).toList();
    }

    /** 结算记录按 fundingTime 排好；请求失败、解析失败回 null */
    static TreeMap<Long, BigDecimal> parseHistory(String json, String symbol) {
        if (json == null) return null;
        TreeMap<Long, BigDecimal> history = new TreeMap<>();
        try {
            for (JsonNode r : MAPPER.readValue(json, ArrayNode.class)) {
                history.put(r.path("fundingTime").asLong(0L), r.path("fundingRate").asDecimal(null));
            }
        } catch (Exception e) {
            log.warn("解析资金费结算记录失败 symbol={}: {}", symbol, e.toString());
            return null;
        }
        return history;
    }

    /** 和结算时刻差 60 秒以内的那条记录的费率；没有回 null */
    static BigDecimal rateAt(TreeMap<Long, BigDecimal> history, long fundingTime) {
        Map.Entry<Long, BigDecimal> e = history.ceilingEntry(fundingTime - FUNDING_TIME_MATCH_MS);
        return e != null && e.getKey() < fundingTime + FUNDING_TIME_MATCH_MS ? e.getValue() : null;
    }
}
