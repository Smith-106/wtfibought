package com.mawai.wiibfeed.stream;

import com.mawai.wiibcommon.cache.CacheService;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/** markPrice 每秒一帧：下次结算时刻没变 5 秒内只写一次，变了马上写 */
class FundingRateWriterTest {

    private final CacheService cacheService = mock(CacheService.class);
    private final FundingRateWriter writer = new FundingRateWriter(cacheService);

    @Test
    void 结算时刻没变五秒内只写一次_变了马上写() {
        writer.write("BTCUSDT", "0.00010000", 1700028800000L);
        writer.write("BTCUSDT", "0.00010000", 1700028800000L);
        verify(cacheService, times(1)).putFundingRate(eq("BTCUSDT"), eq(new BigDecimal("0.00010000")),
                eq(1700028800000L), anyLong());

        writer.write("BTCUSDT", "0.00020000", 1700057600000L);
        verify(cacheService).putFundingRate(eq("BTCUSDT"), eq(new BigDecimal("0.00020000")),
                eq(1700057600000L), anyLong());
    }
}
