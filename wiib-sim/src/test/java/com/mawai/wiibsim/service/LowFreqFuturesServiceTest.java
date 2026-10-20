package com.mawai.wiibsim.service;

import com.mawai.wiibcommon.cache.CacheService;
import com.mawai.wiibcommon.config.BinanceProperties;
import com.mawai.wiibcommon.market.BinanceRestClient;
import com.mawai.wiibsim.mapper.FuturesPositionMapper;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * 低频币价格：全量 premiumIndex 里 feed 没订阅的 USDT 永续，每分钟按标记价写价格，和 feed 一样常驻不过期。
 */
class LowFreqFuturesServiceTest {

    private static final String PREMIUM_ALL = """
            [
              {"symbol":"BTCUSDT","markPrice":"63000.0","lastFundingRate":"0.00010000","nextFundingTime":1760947200000},
              {"symbol":"HYPEUSDT","markPrice":"41.2","lastFundingRate":"0.00010000","nextFundingTime":1760947200000},
              {"symbol":"PEPEUSDT","markPrice":"0.00001234","lastFundingRate":"-0.00050000","nextFundingTime":1760932800000},
              {"symbol":"WIFUSDT","markPrice":"0.81","lastFundingRate":"0.00005000","nextFundingTime":1760936400000},
              {"symbol":"BTCUSDT_261225","markPrice":"64000.0","lastFundingRate":"","nextFundingTime":0},
              {"symbol":"BTCUSDC","markPrice":"63001.0","lastFundingRate":"0.00010000","nextFundingTime":1760947200000}
            ]
            """;

    @Test
    void parseQuotes_只留USDT永续_去掉feed订阅的() {
        Map<String, LowFreqFuturesService.Quote> quotes =
                LowFreqFuturesService.parseQuotes(PREMIUM_ALL, List.of("BTCUSDT", "HYPEUSDT"));

        // BTC、HYPE 是 feed 的；带下划线的是交割合约；USDC 永续不要
        assertThat(quotes).containsOnlyKeys("PEPEUSDT", "WIFUSDT");
        LowFreqFuturesService.Quote pepe = quotes.get("PEPEUSDT");
        assertThat(pepe.markPrice()).isEqualByComparingTo("0.00001234");
        assertThat(pepe.nextFundingTime()).isEqualTo(1760932800000L);
    }

    @Test
    void parseQuotes_空响应与坏响应都给空表() {
        List<String> feed = List.of("BTCUSDT");
        assertThat(LowFreqFuturesService.parseQuotes(null, feed)).isEmpty();
        assertThat(LowFreqFuturesService.parseQuotes("", feed)).isEmpty();
        assertThat(LowFreqFuturesService.parseQuotes("{\"code\":-1003,\"msg\":\"Too many requests\"}", feed)).isEmpty();
    }

    @Test
    void refresh_低频币标记价写两份不过期_feed的币一个不写() {
        BinanceRestClient client = mock(BinanceRestClient.class);
        CacheService cache = mock(CacheService.class);
        when(client.getPremiumIndexAll()).thenReturn(PREMIUM_ALL);
        BinanceProperties props = new BinanceProperties();
        props.setSymbols(List.of("BTCUSDT", "ETHUSDT"));
        props.setCryptoFuturesSymbols(List.of("HYPEUSDT"));
        LowFreqFuturesService service = new LowFreqFuturesService(client, props, cache,
                mock(FuturesPositionMapper.class), mock(FuturesSettlementService.class));

        service.refresh();

        verify(cache).set("market:markprice:PEPEUSDT", "0.00001234");
        verify(cache).set("market:futures-price:PEPEUSDT", "0.00001234");
        verify(cache).set("market:markprice:WIFUSDT", "0.81");
        verify(cache).set("market:futures-price:WIFUSDT", "0.81");
        verify(cache, never()).set(anyString(), anyString(), any(Duration.class));
        // feed 的币（含纯合约的 HYPE）一个不写
        verify(cache, never()).set(eq("market:markprice:BTCUSDT"), anyString());
        verify(cache, never()).set(eq("market:futures-price:BTCUSDT"), anyString());
        verify(cache, never()).set(eq("market:markprice:HYPEUSDT"), anyString());
        verify(cache, never()).set(eq("market:futures-price:HYPEUSDT"), anyString());
    }
}
