package com.mawai.wiibfeed.stream;

import com.mawai.wiibfeed.WsConnection;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

/**
 * 交易员的币的 miniTicker 副连接：和 {@link FuturesMiniTickStreamHandler} 一样，端点拆分后得另开一条物理连接；
 * 地址和解析都用 {@link TraderFuturesStreamHandler} 的。
 */
@Component
@RequiredArgsConstructor
public class TraderMiniTickStreamHandler implements StreamHandler {

    private final TraderFuturesStreamHandler traderHandler;

    @Override public String name() { return "TraderMiniTick"; }
    @Override public long maxIdleSeconds() { return 90; }
    @Override public boolean dynamicSymbols() { return true; }

    @Override
    public void bind(WsConnection conn) {
        traderHandler.setMiniTickConn(conn);
    }

    @Override
    public String buildUrl() {
        return traderHandler.miniTickUrl();
    }

    @Override
    public void onMessage(String raw) {
        traderHandler.handle(raw);
    }
}
