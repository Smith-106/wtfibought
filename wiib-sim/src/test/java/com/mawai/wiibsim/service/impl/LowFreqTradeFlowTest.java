package com.mawai.wiibsim.service.impl;

import com.mawai.wiibcommon.cache.CacheService;
import com.mawai.wiibcommon.config.BinanceProperties;
import com.mawai.wiibcommon.dto.FuturesCloseRequest;
import com.mawai.wiibcommon.dto.FuturesOpenRequest;
import com.mawai.wiibcommon.entity.FuturesOrder;
import com.mawai.wiibcommon.entity.FuturesPosition;
import com.mawai.wiibcommon.entity.User;
import com.mawai.wiibcommon.i18n.MessageCatalog;
import com.mawai.wiibcommon.market.BinanceRestClient;
import com.mawai.wiibsim.config.FuturesLeverageBracketRegistry;
import com.mawai.wiibsim.config.TradeFilterRegistry;
import com.mawai.wiibsim.config.TradingConfig;
import com.mawai.wiibsim.ledger.LedgerTx;
import com.mawai.wiibsim.mapper.FuturesOrderMapper;
import com.mawai.wiibsim.mapper.FuturesPositionMapper;
import com.mawai.wiibsim.mapper.UserMapper;
import com.mawai.wiibsim.service.CrossMarginService;
import com.mawai.wiibsim.service.FuturesPositionIndexService;
import com.mawai.wiibsim.service.FuturesSettlementService;
import com.mawai.wiibsim.service.LowFreqFuturesService;
import com.mawai.wiibsim.service.UserService;
import com.mawai.wiibsim.util.FairLockRegistry;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

/**
 * 低频币走内部下单的整条链：每分钟任务写进去的标记价就是成交价，默认档放行 2 倍全仓，下一轮的价拿来平仓。
 * 例：PEPE 标记价 0.00001 开多 200 万个（名义 20）→ 下一轮 0.0000125 平掉，盈亏 5.00、平仓手续费 0.01。
 */
class LowFreqTradeFlowTest {

    private static final Long UID = 7L;
    private static final String SYM = "PEPEUSDT";

    private final BinanceRestClient client = mock(BinanceRestClient.class);
    /** 假 Redis：任务写什么，下单就读到什么 */
    private final Map<String, String> redis = new HashMap<>();
    private final CacheService cache = mock(CacheService.class);

    @Test
    void 每分钟价格开仓成交_下一轮价格平仓() {
        doAnswer(inv -> redis.put(inv.getArgument(0), inv.getArgument(1)))
                .when(cache).set(anyString(), anyString());
        when(cache.getFuturesPrice(anyString())).thenAnswer(inv -> price("market:futures-price:" + inv.getArgument(0)));
        when(cache.getMarkPrice(anyString())).thenAnswer(inv -> price("market:markprice:" + inv.getArgument(0)));

        UserService userService = mock(UserService.class);
        User user = new User();
        user.setId(UID);
        user.setIsBankrupt(false);
        when(userService.getById(UID)).thenReturn(user);
        UserMapper userMapper = mock(UserMapper.class);
        FuturesPositionMapper positionMapper = mock(FuturesPositionMapper.class);
        FuturesOrderMapper orderMapper = mock(FuturesOrderMapper.class);

        BinanceProperties props = new BinanceProperties();
        props.setSymbols(List.of("BTCUSDT"));
        LowFreqFuturesService lowFreq = new LowFreqFuturesService(client, props, cache,
                positionMapper, mock(FuturesSettlementService.class));
        CrossMarginService crossMargin = mock(CrossMarginService.class);
        FuturesTradingServiceImpl trading = new FuturesTradingServiceImpl(
                userService, userMapper, positionMapper, orderMapper,
                new TradingConfig(), mock(FairLockRegistry.class), cache,
                mock(FuturesPositionIndexService.class), new FuturesLeverageBracketRegistry(), crossMargin,
                new TradeFilterRegistry(mock(BinanceRestClient.class)),
                new MessageCatalog(), new LedgerTx(new TransactionTemplate(mock(PlatformTransactionManager.class))));

        premiumIndex("0.00001000");
        lowFreq.refresh();

        FuturesOpenRequest open = new FuturesOpenRequest();
        open.setSymbol(SYM);
        open.setSide("LONG");
        open.setMarginMode(FuturesPosition.CROSS);
        open.setLeverage(2);
        open.setQuantity(new BigDecimal("2000000"));
        open.setOrderType("MARKET");
        trading.doOpenPosition(UID, open);

        ArgumentCaptor<FuturesPosition> opened = ArgumentCaptor.forClass(FuturesPosition.class);
        verify(positionMapper).insert(opened.capture());
        FuturesPosition pos = opened.getValue();
        assertThat(pos.getEntryPrice()).isEqualByComparingTo("0.00001");
        assertThat(pos.getMargin()).isEqualByComparingTo("10");

        pos.setId(1L);
        when(positionMapper.selectById(1L)).thenReturn(pos);
        when(positionMapper.casClosePosition(anyLong(), anyString(), any(), any())).thenReturn(1);

        premiumIndex("0.00001250");
        lowFreq.refresh();

        FuturesCloseRequest close = new FuturesCloseRequest();
        close.setPositionId(1L);
        close.setOrderType("MARKET");
        trading.doClosePosition(UID, close);

        verify(positionMapper).casClosePosition(eq(1L), eq("CLOSED"),
                argThat(p -> p.compareTo(new BigDecimal("0.0000125")) == 0),
                argThat(pnl -> pnl.compareTo(new BigDecimal("5.00")) == 0));
        verify(crossMargin).settle(eq(UID), argThat(d -> d.compareTo(new BigDecimal("4.99")) == 0));
        ArgumentCaptor<FuturesOrder> orders = ArgumentCaptor.forClass(FuturesOrder.class);
        verify(orderMapper, times(2)).insert(orders.capture());
        assertThat(orders.getAllValues()).extracting(FuturesOrder::getOrderSide, FuturesOrder::getStatus)
                .containsExactly(tuple("OPEN_LONG", "FILLED"), tuple("CLOSE_LONG", "FILLED"));
    }

    private void premiumIndex(String pepeMark) {
        when(client.getPremiumIndexAll()).thenReturn("""
                [{"symbol":"BTCUSDT","markPrice":"63000","lastFundingRate":"0.0001","nextFundingTime":1760947200000},
                 {"symbol":"PEPEUSDT","markPrice":"%s","lastFundingRate":"0.0001","nextFundingTime":1760947200000}]
                """.formatted(pepeMark));
    }

    private BigDecimal price(String key) {
        String v = redis.get(key);
        return v == null ? null : new BigDecimal(v);
    }
}
