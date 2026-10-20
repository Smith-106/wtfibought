package com.mawai.wiibfeed.stream;

import com.mawai.wiibcommon.cache.CacheService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 资金费率写 Redis：markPrice 推送每秒一帧带当期费率 r 和下次结算时刻 T。每币记上次写的 T 和时刻，
 * T 变了马上写，没变每 5 秒写一次。feed 配置里的币和交易员的币两边的 markPrice 处理都走这里；
 * sim 按 T 到点收资金费、前端查费率都读它写的。
 */
@Component
@RequiredArgsConstructor
public class FundingRateWriter {

    static final long WRITE_INTERVAL_MS = 5_000L;

    private final CacheService cacheService;

    private record Written(long nextTime, long at) {}

    private final Map<String, Written> written = new ConcurrentHashMap<>();

    void write(String symbol, String rate, long nextTime) {
        long now = System.currentTimeMillis();
        Written prev = written.get(symbol);
        if (prev != null && prev.nextTime() == nextTime && now - prev.at() < WRITE_INTERVAL_MS) return;
        cacheService.putFundingRate(symbol, new BigDecimal(rate), nextTime, now);
        written.put(symbol, new Written(nextTime, now));
    }
}
