package com.mawai.wiibsim.service;

import com.mawai.wiibcommon.entity.FuturesPosition;
import com.mawai.wiibcommon.market.KlineBar;

import java.math.BigDecimal;
import java.util.List;

public interface FuturesSettlementService {

    void onPriceUpdate(String symbol, BigDecimal price);

    /** 空窗补漏：按空窗期 1m K 线补触发限价单，逐单按创建时间过滤（挂单之后的行情才算） */
    void recoverGap(String symbol, List<KlineBar> bars);

    void executeTriggeredOrders();

    /** 限价单索引对账：DB里的PENDING挂单全部补回ZSet（纯追加、幂等） */
    void reconcileLimitOrderIndex();

    /** feed 币资金费率结算（0/8/16 点）：先拉一次官方费率写缓存，再按缓存给 feed 币持仓扣费 */
    void chargeFundingFeeAll();

    /** 低频币某个结算时刻的资金费：给这些仓位按 rate 收，单仓失败不影响别的 */
    void chargeFundingFee(String symbol, long fundingTime, BigDecimal rate, List<FuturesPosition> positions);
}
