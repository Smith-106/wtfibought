package com.mawai.wiibsim.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.mawai.wiibcommon.cache.CacheService;
import com.mawai.wiibcommon.config.BinanceProperties;
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
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongSupplier;
import java.util.stream.Collectors;

import static com.mawai.wiibcommon.util.JsonUtils.MAPPER;

/**
 * 低频币：feed 没订阅、每分钟按标记价更新的 USDT 永续。
 * <p>
 * 每分钟拉一次全量 premiumIndex，给这些币写标记价；最新价也写标记价（没有逐笔成交价，市价单按标记价成交）。
 * 价格和 feed 写的一样常驻，下架的币停在最后价格上。feed 订阅的币一个都不写。
 * 价格不发撮合频道，全仓强平靠 30 秒一次的兜底巡检。
 * <p>
 * 资金费按每个币自己的结算时刻收：上一轮记下的 nextFundingTime 到点了就记成待结算，
 * 费率用 Binance 的实际结算记录，记录没出来就留到下一分钟。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class LowFreqFuturesService {

    private static final String MARK_PRICE_KEY_PREFIX = "market:markprice:";
    private static final String FUTURES_PRICE_KEY_PREFIX = "market:futures-price:";
    /** 每个币资金费已经结算到哪个时刻（毫秒） */
    private static final String FUNDING_SETTLED_KEY_PREFIX = "futures:funding:settled:";
    private static final Duration FUNDING_SETTLED_TTL = Duration.ofDays(3);
    /** 结算记录的 fundingTime 比整点多几毫秒，差这么多以内算同一个结算时刻 */
    private static final long FUNDING_TIME_MATCH_MS = 60_000L;
    /** 一次取最近几条结算记录 */
    private static final int FUNDING_HISTORY_LIMIT = 10;
    /** 连着取结算记录时每次之间隔开 */
    private static final long FUNDING_REQUEST_GAP_MS = 300L;

    private final BinanceRestClient restClient;
    private final BinanceProperties binanceProperties;
    private final CacheService cacheService;
    private final FuturesPositionMapper positionMapper;
    private final FuturesSettlementService settlementService;

    /** 墙钟注入点 */
    LongSupplier nowMs = System::currentTimeMillis;

    /** 一个低频币最新一条 premiumIndex */
    record Quote(BigDecimal markPrice, long nextFundingTime) {}

    /** 上一轮拉到的低频币，每轮按最新响应整体替换 */
    private volatile Map<String, Quote> quotes = Map.of();
    /** 到点待结算的资金费：symbol → 结算时刻 */
    private final Map<String, Long> dueFunding = new ConcurrentHashMap<>();
    private final ReentrantLock settling = new ReentrantLock();

    /** 拉一次全量、写价格、记下到点的结算时刻。拉不到就跳过这一轮，什么都不动 */
    public void refresh() {
        Map<String, Quote> latest = parseQuotes(restClient.getPremiumIndexAll(),
                binanceProperties.getAllFuturesSymbols());
        if (latest.isEmpty()) {
            log.warn("低频币 premiumIndex 拉取失败或为空，本轮跳过");
            return;
        }
        latest.forEach((symbol, q) -> {
            String price = q.markPrice().toPlainString();
            cacheService.set(MARK_PRICE_KEY_PREFIX + symbol, price);
            cacheService.set(FUTURES_PRICE_KEY_PREFIX + symbol, price);
        });
        long now = nowMs.getAsLong();
        quotes.forEach((symbol, prev) -> {
            if (prev.nextFundingTime() <= 0 || prev.nextFundingTime() > now) return;
            Long skipped = dueFunding.put(symbol, prev.nextFundingTime());
            if (skipped != null && skipped != prev.nextFundingTime()) {
                log.warn("低频币资金费一直没取到结算记录，跳过 symbol={} 结算时刻={}", symbol, skipped);
            }
        });
        quotes = latest;
        log.debug("低频币价格刷新 {}个", latest.size());
    }

    /**
     * 收到点的资金费：只给结算时刻之前开、现在还开着的仓位收；没有这样的仓位就不取结算记录。
     * 记录没出来或请求失败留到下一分钟；上一次还没跑完就跳过这一次。
     */
    public void settleDueFunding() {
        if (dueFunding.isEmpty() || !settling.tryLock()) return;
        try {
            Map<String, List<FuturesPosition>> open = positionMapper.selectList(new LambdaQueryWrapper<FuturesPosition>()
                            .eq(FuturesPosition::getStatus, "OPEN"))
                    .stream().collect(Collectors.groupingBy(FuturesPosition::getSymbol));
            boolean requested = false;
            for (Map.Entry<String, Long> due : dueFunding.entrySet()) {
                String symbol = due.getKey();
                long fundingTime = due.getValue();
                List<FuturesPosition> positions = openedBefore(open.getOrDefault(symbol, List.of()), fundingTime);
                if (positions.isEmpty()) {
                    dueFunding.remove(symbol, fundingTime);
                    continue;
                }
                Long settled = settledTime(symbol);
                if (settled != null && settled >= fundingTime) {
                    dueFunding.remove(symbol, fundingTime);
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
                BigDecimal rate = settledRate(restClient.getFundingRateHistory(symbol, FUNDING_HISTORY_LIMIT),
                        symbol, fundingTime, settled);
                if (rate == null) continue;
                // 先记已结算再收，收到一半出错也不会重收
                cacheService.set(FUNDING_SETTLED_KEY_PREFIX + symbol, String.valueOf(fundingTime), FUNDING_SETTLED_TTL);
                dueFunding.remove(symbol, fundingTime);
                settlementService.chargeFundingFee(symbol, fundingTime, rate, positions);
            }
        } finally {
            settling.unlock();
        }
    }

    private Long settledTime(String symbol) {
        String v = cacheService.get(FUNDING_SETTLED_KEY_PREFIX + symbol);
        return v == null ? null : Long.parseLong(v);
    }

    /** 结算时刻之前开的仓（createdAt 是按本机时区填的） */
    private static List<FuturesPosition> openedBefore(List<FuturesPosition> positions, long fundingTime) {
        LocalDateTime cutoff = LocalDateTime.ofInstant(Instant.ofEpochMilli(fundingTime), ZoneId.systemDefault());
        return positions.stream().filter(p -> p.getCreatedAt().isBefore(cutoff)).toList();
    }

    /**
     * 从结算记录里取这个结算时刻的费率；请求失败、记录还没出来返回 null。
     * 上次结算之后、这次之前还有记录的，是停机期间错过的，只打日志不补收。
     */
    static BigDecimal settledRate(String json, String symbol, long fundingTime, Long settled) {
        if (json == null) return null;
        BigDecimal rate = null;
        List<Long> missed = new ArrayList<>();
        try {
            for (JsonNode r : MAPPER.readValue(json, ArrayNode.class)) {
                long t = r.path("fundingTime").asLong(0L);
                if (Math.abs(t - fundingTime) < FUNDING_TIME_MATCH_MS) {
                    rate = r.path("fundingRate").asDecimal(null);
                } else if (settled != null && t > settled + FUNDING_TIME_MATCH_MS && t < fundingTime) {
                    missed.add(t);
                }
            }
        } catch (Exception e) {
            log.warn("解析资金费结算记录失败 symbol={}: {}", symbol, e.toString());
            return null;
        }
        if (rate != null && !missed.isEmpty()) {
            log.warn("低频币资金费漏收（停机期间错过，不补）symbol={} 结算时刻={}", symbol, missed);
        }
        return rate;
    }

    /** USDT 永续：USDT 结尾、不带下划线（带下划线的是交割合约） */
    public static boolean isUsdtPerpetual(String symbol) {
        return symbol != null && symbol.endsWith("USDT") && symbol.indexOf('_') < 0;
    }

    /** 解析全量 premiumIndex：只留 USDT 永续，去掉 feed 订阅的；坏响应给空表 */
    static Map<String, Quote> parseQuotes(String json, Collection<String> feedSymbols) {
        if (json == null || json.isBlank()) return Map.of();
        Set<String> feed = new HashSet<>(feedSymbols);
        Map<String, Quote> result = new HashMap<>();
        try {
            ArrayNode arr = MAPPER.readValue(json, ArrayNode.class);
            for (JsonNode item : arr) {
                String symbol = item.path("symbol").asString(null);
                if (!isUsdtPerpetual(symbol) || feed.contains(symbol)) continue;
                BigDecimal mark = item.path("markPrice").asDecimal(null);
                if (mark == null) continue;
                result.put(symbol, new Quote(mark, item.path("nextFundingTime").asLong(0L)));
            }
        } catch (Exception e) {
            log.warn("解析 premiumIndex 全量响应失败: {}", e.toString());
            return Map.of();
        }
        return result;
    }
}
