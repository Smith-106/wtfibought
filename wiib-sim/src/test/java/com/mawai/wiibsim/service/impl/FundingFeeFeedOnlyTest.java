package com.mawai.wiibsim.service.impl;

import com.mawai.wiibcommon.cache.CacheService;
import com.mawai.wiibcommon.config.BinanceProperties;
import com.mawai.wiibcommon.entity.FuturesPosition;
import com.mawai.wiibsim.config.FuturesLeverageBracketRegistry;
import com.mawai.wiibsim.config.TradingConfig;
import com.mawai.wiibsim.ledger.LedgerTx;
import com.mawai.wiibsim.mapper.FuturesOrderMapper;
import com.mawai.wiibsim.mapper.FuturesPositionMapper;
import com.mawai.wiibsim.mapper.UserMapper;
import com.mawai.wiibsim.service.CrossLiquidationService;
import com.mawai.wiibsim.service.CrossMarginService;
import com.mawai.wiibsim.service.FundingRateService;
import com.mawai.wiibsim.service.FuturesPositionIndexService;
import com.mawai.wiibsim.service.FuturesRiskService;
import com.mawai.wiibsim.service.UserService;
import com.mawai.wiibsim.util.FairLockRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * 0/8/16 点那条资金费只收 feed 币，别的币不碰。
 */
class FundingFeeFeedOnlyTest {

    @Test
    void 每8小时那条只收feed币_别的币不碰() {
        FuturesPositionMapper positionMapper = mock(FuturesPositionMapper.class);
        FairLockRegistry lockRegistry = mock(FairLockRegistry.class);
        FundingRateService fundingRateService = mock(FundingRateService.class);
        when(positionMapper.selectList(any())).thenReturn(List.of(position(1L, "BTCUSDT"), position(2L, "PEPEUSDT")));

        BinanceProperties props = new BinanceProperties();
        props.setSymbols(List.of("BTCUSDT"));
        FuturesSettlementServiceImpl service = new FuturesSettlementServiceImpl(
                mock(UserService.class), mock(UserMapper.class), positionMapper, mock(FuturesOrderMapper.class),
                new TradingConfig(), mock(FuturesLeverageBracketRegistry.class), mock(CacheService.class),
                mock(FuturesPositionIndexService.class), mock(FuturesRiskService.class), mock(CrossMarginService.class),
                mock(CrossLiquidationService.class), lockRegistry, fundingRateService,
                new LedgerTx(new TransactionTemplate(mock(PlatformTransactionManager.class))), props);

        service.chargeFundingFeeAll();

        verify(fundingRateService).rateForSettlement("BTCUSDT");
        verify(lockRegistry).tryLockAsSystem("futures:pos:1");
        verify(fundingRateService, never()).rateForSettlement("PEPEUSDT");
        verify(lockRegistry, never()).tryLockAsSystem("futures:pos:2");
    }

    private static FuturesPosition position(long id, String symbol) {
        FuturesPosition p = new FuturesPosition();
        p.setId(id);
        p.setUserId(7L);
        p.setSymbol(symbol);
        p.setSide("LONG");
        p.setMarginMode(FuturesPosition.CROSS);
        p.setStatus("OPEN");
        return p;
    }
}
