package com.mawai.wiibsim.service;

import com.mawai.wiibcommon.config.BinanceProperties;
import com.mawai.wiibsim.mapper.FuturesPositionMapper;
import org.junit.jupiter.api.Test;
import org.springframework.data.redis.core.SetOperations;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.ZSetOperations;

import java.util.List;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

/**
 * 交易员集合 = 非 feed 币的 OPEN 仓位所在的币 ∪ 未过期的登记；登记 2 小时后过期；集合没变不写 Redis。
 */
class TraderSymbolServiceTest {

    private static final long T = 1_760_918_400_000L;
    private static final long H = 3_600_000L;

    private final StringRedisTemplate redis = mock(StringRedisTemplate.class);
    @SuppressWarnings("unchecked")
    private final ZSetOperations<String, String> zset = mock(ZSetOperations.class);
    @SuppressWarnings("unchecked")
    private final SetOperations<String, String> set = mock(SetOperations.class);
    private final FuturesPositionMapper positionMapper = mock(FuturesPositionMapper.class);
    private final TraderSymbolService service = service();

    private TraderSymbolService service() {
        when(redis.opsForZSet()).thenReturn(zset);
        when(redis.opsForSet()).thenReturn(set);
        BinanceProperties props = new BinanceProperties();
        props.setSymbols(List.of("BTCUSDT"));
        props.setCryptoFuturesSymbols(List.of("HYPEUSDT"));
        TraderSymbolService s = new TraderSymbolService(redis, positionMapper, props);
        s.nowMs = () -> T;
        return s;
    }

    @Test
    void 集合是非feed币的仓位加上未过期的登记_变了才写() {
        when(positionMapper.openSymbols()).thenReturn(List.of("BTCUSDT", "HYPEUSDT", "PEPEUSDT"));
        when(zset.range(TraderSymbolService.WATCH_KEY, 0, -1)).thenReturn(Set.of("WIFUSDT", "PEPEUSDT"));
        when(set.members(TraderSymbolService.SYMBOLS_KEY)).thenReturn(Set.of("DOGSUSDT"));

        service.refresh();

        // 先清掉过期的登记，再比当前集合：feed 的 BTC、HYPE 不进，多的加、少的删
        verify(zset).removeRangeByScore(TraderSymbolService.WATCH_KEY, Double.NEGATIVE_INFINITY, (double) T);
        verify(set).add(TraderSymbolService.SYMBOLS_KEY, "PEPEUSDT", "WIFUSDT");
        verify(set).remove(TraderSymbolService.SYMBOLS_KEY, "DOGSUSDT");

        // 没变：只读不写
        when(set.members(TraderSymbolService.SYMBOLS_KEY)).thenReturn(Set.of("PEPEUSDT", "WIFUSDT"));
        service.refresh();
        verify(set, times(2)).members(TraderSymbolService.SYMBOLS_KEY);
        verifyNoMoreInteractions(set);
    }

    @Test
    void 登记两小时后过期_登记完立刻重算() {
        when(positionMapper.openSymbols()).thenReturn(List.of());
        when(zset.range(TraderSymbolService.WATCH_KEY, 0, -1)).thenReturn(Set.of("WIFUSDT", "PEPEUSDT"));
        when(set.members(TraderSymbolService.SYMBOLS_KEY)).thenReturn(Set.of());

        service.watch(List.of("WIFUSDT", "PEPEUSDT"));

        verify(zset).add(TraderSymbolService.WATCH_KEY, "WIFUSDT", (double) (T + 2 * H));
        verify(zset).add(TraderSymbolService.WATCH_KEY, "PEPEUSDT", (double) (T + 2 * H));
        verify(set).add(TraderSymbolService.SYMBOLS_KEY, "PEPEUSDT", "WIFUSDT");
    }
}
