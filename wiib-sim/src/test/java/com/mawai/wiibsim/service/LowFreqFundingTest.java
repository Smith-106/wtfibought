package com.mawai.wiibsim.service;

import com.mawai.wiibcommon.cache.CacheService;
import com.mawai.wiibcommon.config.BinanceProperties;
import com.mawai.wiibcommon.entity.FuturesPosition;
import com.mawai.wiibcommon.market.BinanceRestClient;
import com.mawai.wiibsim.mapper.FuturesPositionMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

/**
 * 低频币资金费：上一轮记下的 nextFundingTime 到点就结算，费率用 Binance 结算记录里对上这个时刻的那条，
 * 只收结算时刻之前开的仓；每个时刻只收一次（Redis 记着已结算到哪），记录没出来下一分钟再试。
 * <p>
 * 时间轴从 T0 = 2026-10-20 00:00 UTC 起；结算记录的 fundingTime 比整点多 3ms，费率 = 第几小时 × 0.00001。
 */
class LowFreqFundingTest {

    private static final long H = 3_600_000L;
    private static final long T0 = Instant.parse("2026-10-20T00:00:00Z").toEpochMilli();

    private final BinanceRestClient client = mock(BinanceRestClient.class);
    private final CacheService cache = mock(CacheService.class);
    private final FuturesPositionMapper positionMapper = mock(FuturesPositionMapper.class);
    private final FuturesSettlementService settlement = mock(FuturesSettlementService.class);
    /** 假 Redis，重启前后共用 */
    private final Map<String, String> redis = new HashMap<>();
    /** 这一轮 premiumIndex 里各币的 nextFundingTime */
    private final Map<String, Long> next = new HashMap<>();
    /** Binance 已经出了的结算记录：symbol → 结算时刻 */
    private final Map<String, List<Long>> published = new HashMap<>();
    private final List<FuturesPosition> positions = new ArrayList<>();
    /** 收过的：symbol@第几小时:费率 */
    private final List<String> charges = new ArrayList<>();
    private boolean premiumDown;
    private boolean historyDown;
    private LowFreqFuturesService service;

    @BeforeEach
    void setUp() {
        doAnswer(inv -> redis.put(inv.getArgument(0), inv.getArgument(1)))
                .when(cache).set(anyString(), anyString(), any(Duration.class));
        when(cache.get(anyString())).thenAnswer(inv -> redis.get(inv.<String>getArgument(0)));
        when(client.getPremiumIndexAll()).thenAnswer(inv -> premiumDown ? null : premiumIndex());
        when(client.getFundingRateHistory(anyString(), anyInt()))
                .thenAnswer(inv -> historyDown ? null : history(inv.getArgument(0)));
        when(positionMapper.selectList(any())).thenAnswer(inv -> positions);
        doAnswer(inv -> charges.add(charge(inv.getArgument(0), inv.<Long>getArgument(1), inv.getArgument(2))))
                .when(settlement).chargeFundingFee(anyString(), anyLong(), any(), anyList());
        service = newService();
    }

    @Test
    void 一小时四小时八小时的币各按自己的结算时刻收() {
        positions.addAll(List.of(position(1, "AUSDT", T0 - 60_000), position(2, "BUSDT", T0 - 60_000),
                position(3, "CUSDT", T0 - 60_000)));
        for (int k = 0; k <= 8; k++) {
            next.put("AUSDT", T0 + (k + 1) * H);
            next.put("BUSDT", T0 + (k / 4 + 1) * 4 * H);
            next.put("CUSDT", T0 + (k / 8 + 1) * 8 * H);
            publish("AUSDT", k);
            if (k % 4 == 0) publish("BUSDT", k);
            if (k % 8 == 0) publish("CUSDT", k);
            round(T0 + k * H + 30_000);
        }

        assertThat(charges).containsExactlyInAnyOrder(
                expected("AUSDT", 1), expected("AUSDT", 2), expected("AUSDT", 3), expected("AUSDT", 4),
                expected("AUSDT", 5), expected("AUSDT", 6), expected("AUSDT", 7), expected("AUSDT", 8),
                expected("BUSDT", 4), expected("BUSDT", 8),
                expected("CUSDT", 8));
    }

    @Test
    void 同一结算时刻只收一次_重启后也不重收() {
        positions.add(position(1, "AUSDT", T0));
        publish("AUSDT", 1);
        next.put("AUSDT", T0 + H);
        round(T0 + 30_000);
        // 到点了 nextFundingTime 还没往后跳：收一次
        round(T0 + H + 30_000);

        service = newService();
        round(T0 + H + 60_000);
        next.put("AUSDT", T0 + 2 * H);
        // 1h 又被记成待结算，Redis 里已经结算到 1h：不收，也不取记录
        round(T0 + H + 90_000);

        assertThat(charges).containsExactly(expected("AUSDT", 1));
        verify(client, times(1)).getFundingRateHistory(eq("AUSDT"), anyInt());
    }

    @Test
    void 结算时刻之后开的仓不收_没有这样的仓位不取记录() {
        long t = T0 + H;
        FuturesPosition before = position(1, "AUSDT", t - 60_000);
        positions.addAll(List.of(before, position(2, "AUSDT", t + 10_000), position(3, "BUSDT", t + 10_000)));
        publish("AUSDT", 1);
        publish("BUSDT", 1);
        next.put("AUSDT", t);
        next.put("BUSDT", t);
        round(T0 + 30_000);
        next.put("AUSDT", t + H);
        next.put("BUSDT", t + H);
        round(t + 30_000);

        verify(settlement).chargeFundingFee(eq("AUSDT"), eq(t), any(), eq(List.of(before)));
        verify(settlement, never()).chargeFundingFee(eq("BUSDT"), anyLong(), any(), anyList());
        verify(client, never()).getFundingRateHistory(eq("BUSDT"), anyInt());
    }

    @Test
    void 结算记录没出来先不收_下一分钟再收() {
        long t = T0 + H;
        positions.add(position(1, "AUSDT", T0));
        publish("AUSDT", 0);
        next.put("AUSDT", t);
        round(T0 + 30_000);
        next.put("AUSDT", t + H);

        historyDown = true;
        round(t + 30_000);
        historyDown = false;
        // 只有上一期 0h 的记录，不能拿它顶
        round(t + 90_000);
        assertThat(charges).isEmpty();

        publish("AUSDT", 1);
        round(t + 150_000);
        assertThat(charges).containsExactly(expected("AUSDT", 1));
    }

    @Test
    void 行情拉取失败那一分钟_不写价格_上一轮记下的结算时刻还在() {
        long t = T0 + H;
        positions.add(position(1, "AUSDT", T0));
        publish("AUSDT", 1);
        next.put("AUSDT", t);
        round(T0 + 30_000);

        premiumDown = true;
        clearInvocations(cache);
        round(t + 30_000);
        verify(cache, never()).set(anyString(), anyString());
        assertThat(charges).isEmpty();

        premiumDown = false;
        next.put("AUSDT", t + H);
        round(t + 90_000);
        assertThat(charges).containsExactly(expected("AUSDT", 1));
    }

    private LowFreqFuturesService newService() {
        BinanceProperties props = new BinanceProperties();
        props.setSymbols(List.of("BTCUSDT"));
        return new LowFreqFuturesService(client, props, cache, positionMapper, settlement);
    }

    /** 一分钟一轮：拉行情，再收到点的资金费 */
    private void round(long now) {
        service.nowMs = () -> now;
        service.refresh();
        service.settleDueFunding();
    }

    private void publish(String symbol, int hour) {
        published.computeIfAbsent(symbol, s -> new ArrayList<>()).add(T0 + hour * H);
    }

    private String premiumIndex() {
        return next.entrySet().stream()
                .map(e -> "{\"symbol\":\"%s\",\"markPrice\":\"1\",\"nextFundingTime\":%d}".formatted(e.getKey(), e.getValue()))
                .collect(Collectors.joining(",", "[", "]"));
    }

    private String history(String symbol) {
        return published.getOrDefault(symbol, List.of()).stream()
                .map(t -> "{\"symbol\":\"%s\",\"fundingTime\":%d,\"fundingRate\":\"%s\"}"
                        .formatted(symbol, t + 3, rate((int) ((t - T0) / H)).toPlainString()))
                .collect(Collectors.joining(",", "[", "]"));
    }

    private static BigDecimal rate(int hour) {
        return BigDecimal.valueOf(hour).movePointLeft(5);
    }

    private static String charge(String symbol, long fundingTime, BigDecimal rate) {
        return symbol + "@" + (fundingTime - T0) / H + "h:" + rate.toPlainString();
    }

    private static String expected(String symbol, int hour) {
        return charge(symbol, T0 + hour * H, rate(hour));
    }

    private static FuturesPosition position(long id, String symbol, long createdAtMs) {
        FuturesPosition p = new FuturesPosition();
        p.setId(id);
        p.setSymbol(symbol);
        p.setStatus("OPEN");
        p.setCreatedAt(LocalDateTime.ofInstant(Instant.ofEpochMilli(createdAtMs), ZoneId.systemDefault()));
        return p;
    }
}
