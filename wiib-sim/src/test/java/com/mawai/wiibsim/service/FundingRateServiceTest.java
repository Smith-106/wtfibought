package com.mawai.wiibsim.service;

import com.mawai.wiibcommon.cache.CacheService;
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
 * 资金费按各币自己的结算时刻收：到点了、feed 写的下次结算时刻 T 也往后挪了，就是记下的那一期结算了，
 * 费率用 Binance 结算记录里对上那个时刻的那条，只收结算时刻之前开的仓；每个时刻只收一次（Redis 记着看到的 T），
 * 记录没出来下一分钟再试，sim 停机跨过结算起来后补上；没到点 T 就变了是改了结算间隔，只换记下的时刻。
 * <p>
 * 时间轴从 T0 = 2026-10-20 00:00 UTC 起；结算记录的 fundingTime 比整点多 3ms，费率 = 第几小时 × 0.00001。
 */
class FundingRateServiceTest {

    private static final long H = 3_600_000L;
    private static final long T0 = Instant.parse("2026-10-20T00:00:00Z").toEpochMilli();

    private final BinanceRestClient client = mock(BinanceRestClient.class);
    private final CacheService cache = mock(CacheService.class);
    private final FuturesPositionMapper positionMapper = mock(FuturesPositionMapper.class);
    private final FuturesSettlementService settlement = mock(FuturesSettlementService.class);
    /** 假 Redis（sim 自己记的键），重启前后共用 */
    private final Map<String, String> redis = new HashMap<>();
    /** feed 写的各币下次结算时刻 */
    private final Map<String, Long> next = new HashMap<>();
    /** Binance 已经出了的结算记录：symbol → 结算时刻 */
    private final Map<String, List<Long>> published = new HashMap<>();
    private final List<FuturesPosition> positions = new ArrayList<>();
    /** 收过的：symbol@第几小时:费率 */
    private final List<String> charges = new ArrayList<>();
    private boolean historyDown;
    private FundingRateService service;

    @BeforeEach
    void setUp() {
        doAnswer(inv -> redis.put(inv.getArgument(0), inv.getArgument(1)))
                .when(cache).set(anyString(), anyString(), any(Duration.class));
        when(cache.get(anyString())).thenAnswer(inv -> redis.get(inv.<String>getArgument(0)));
        when(cache.getFundingRate(anyString())).thenAnswer(inv -> {
            Long t = next.get(inv.<String>getArgument(0));
            return t == null ? null : new CacheService.FundingRate(new BigDecimal("0.0001"), t, t - H);
        });
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
        // 每小时一轮：feed 的 T 按各币的间隔往后挪，Binance 出这一期的记录
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
        // 到点了 feed 的 T 还没挪：不收
        round(T0 + H + 30_000);
        assertThat(charges).isEmpty();

        next.put("AUSDT", T0 + 2 * H);
        round(T0 + H + 60_000);
        service = newService();
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
    void 停机跨过结算_起来后补上最早那一期() {
        positions.add(position(1, "AUSDT", T0));
        next.put("AUSDT", T0 + H);
        round(T0 + 30_000);
        // 停机到 02:30：1h、2h 两期都过了，feed 的 T 已经是 3h。补 1h 那期，2h 那期不补
        publish("AUSDT", 1);
        publish("AUSDT", 2);
        next.put("AUSDT", T0 + 3 * H);
        service = newService();
        round(T0 + 2 * H + 30 * 60_000);

        assertThat(charges).containsExactly(expected("AUSDT", 1));
        assertThat(redis).containsEntry("futures:funding:next:AUSDT", String.valueOf(T0 + 3 * H));
    }

    @Test
    void 记录已经滚出最近几条_这一期不收() {
        positions.add(position(1, "AUSDT", T0));
        next.put("AUSDT", T0 + H);
        round(T0 + 30_000);
        // 最近 10 条记录都在 1h 之后：1h 那条取不到了
        for (int k = 2; k <= 11; k++) publish("AUSDT", k);
        next.put("AUSDT", T0 + 12 * H);
        round(T0 + 11 * H + 30_000);

        assertThat(charges).isEmpty();
        assertThat(redis).containsEntry("futures:funding:next:AUSDT", String.valueOf(T0 + 12 * H));
    }

    @Test
    void 结算间隔改了_没到点T就变不算结算_按新时刻收() {
        positions.add(position(1, "AUSDT", T0));
        next.put("AUSDT", T0 + H);
        round(T0 + 30_000);
        // 00:30 改成 4 小时一结：T 从 1h 跳到 4h，1h 那期不存在，不能当它结算了
        next.put("AUSDT", T0 + 4 * H);
        round(T0 + 30 * 60_000);
        assertThat(charges).isEmpty();
        verify(client, never()).getFundingRateHistory(anyString(), anyInt());

        publish("AUSDT", 4);
        next.put("AUSDT", T0 + 8 * H);
        round(T0 + 4 * H + 30_000);
        assertThat(charges).containsExactly(expected("AUSDT", 4));
        verify(client, times(1)).getFundingRateHistory(eq("AUSDT"), anyInt());
    }

    @Test
    void 前端查询读feed写的费率和下次结算时刻() {
        next.put("AUSDT", T0 + 4 * H);

        assertThat(service.query("AUSDT")).isEqualTo(
                new FundingRateService.FundingRateView("AUSDT", new BigDecimal("0.0001"), T0 + 3 * H, T0 + 4 * H));
        assertThat(service.query("NOPE")).isNull();
        assertThat(service.query(null)).isNull();
    }

    private FundingRateService newService() {
        return new FundingRateService(client, cache, positionMapper, settlement);
    }

    /** 一分钟一轮 */
    private void round(long now) {
        service.nowMs = () -> now;
        service.settleDue();
    }

    private void publish(String symbol, int hour) {
        published.computeIfAbsent(symbol, s -> new ArrayList<>()).add(T0 + hour * H);
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
