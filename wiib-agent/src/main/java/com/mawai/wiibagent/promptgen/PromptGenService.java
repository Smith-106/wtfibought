package com.mawai.wiibagent.promptgen;

import com.mawai.wiibagent.chat.ChatModelFactory;
import com.mawai.wiibagent.chat.tools.TraderQueryToolkit;
import com.mawai.wiibagent.i18n.LocalizedToolCallbacks;
import com.mawai.wiibagent.i18n.PromptCatalog;
import com.mawai.wiibagent.i18n.UserLangResolver;
import com.mawai.wiibagent.llm.AgentChat;
import com.mawai.wiibagent.llm.ChatEndpoints;
import com.mawai.wiibagent.llm.LlmEndpointService;
import com.mawai.wiibagent.llm.LlmErrorMessages;
import com.mawai.wiibagent.llm.ModelCallLimiter;
import com.mawai.wiibagent.llm.ReactLoop;
import com.mawai.wiibagent.llm.SseChannel;
import com.mawai.wiibagent.trader.TraderChatService;
import com.mawai.wiibagent.trader.TraderService;
import com.mawai.wiibagent.trader.prompt.PlayStatsAssembler;
import com.mawai.wiibcommon.entity.AiTrader;
import com.mawai.wiibcommon.entity.PromptGenSession;
import com.mawai.wiibcommon.entity.UserLlmEndpoint;
import com.mawai.wiibcommon.enums.AgentLang;
import com.mawai.wiibcommon.enums.ErrorCode;
import com.mawai.wiibcommon.exception.BizException;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import static com.mawai.wiibagent.promptgen.PromptGenToolkit.ASK_USER;
import static com.mawai.wiibagent.promptgen.PromptGenToolkit.WRITE_DRAFT;
import static com.mawai.wiibcommon.util.JsonUtils.MAPPER;

/**
 * 交易指令生成器：agent 读 trader 数据、一题一题问用户、信息够了让写稿人出一版，用户提修改再接着问。
 * <pre>
 * /start、/reply 各是一轮：读对话 → 加一条用户消息 → 现建 agent 流式跑 → 对话整份覆盖落库 → done
 *   ask_user   推题卡，这一轮就收工（ReactLoop.stopWhen）
 *   write_draft 写稿人流式写正文，回执进历史时连快照存成一版（只留最近 3 版）
 * </pre>
 * 不开对话压缩：对话本来不长，压了版本快照就对不上。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class PromptGenService {

    /** 单轮模型调用保险丝：正常一轮 2~4 次 */
    static final int RUN_CALL_LIMIT = 12;

    /** 模型思考期间通道可能静默较久，与工作台同值 */
    private static final long HEARTBEAT_SECONDS = 20;

    private final PromptGenStore store;
    private final PromptGenFacts facts;
    private final LlmEndpointService endpointService;
    private final ChatModelFactory chatModelFactory;
    private final TraderService traderService;
    private final TraderChatService traderChatService;
    private final PlayStatsAssembler playStats;
    private final LocalizedToolCallbacks localizedTools;
    private final PromptCatalog prompts;
    private final UserLangResolver userLangResolver;
    /** 心跳专用：只发注释帧，单线程够用；虚拟线程不支持定时调度故用平台线程 */
    private final ScheduledExecutorService heartbeatScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "prompt-gen-heartbeat");
        t.setDaemon(true);
        return t;
    });

    @PreDestroy
    public void stop() {
        heartbeatScheduler.shutdownNow();
    }

    public PromptGenSession session(long userId) {
        return store.session(userId);
    }

    /** 建模放在开流之前：端点不对要回 JSON 错误，不能等进了 SSE 再说 */
    public ChatModel model(long userId, long endpointId) {
        UserLlmEndpoint endpoint = endpointService.get(userId, endpointId);
        if (endpoint == null) {
            throw new BizException(ErrorCode.LLM_CONFIG_MISSING);
        }
        try {
            return chatModelFactory.modelsFor(new ChatEndpoints(userId, endpoint, null)).deep();
        } catch (Exception e) {
            log.warn("[PromptGen] 建模失败 userId={}", userId, e);
            throw new BizException(ErrorCode.LLM_CONFIG_INVALID);
        }
    }

    public PromptGenView view(long userId) {
        List<PromptGenView.Version> versions = store.versions(userId).stream().map(PromptGenView.Version::of).toList();
        PromptGenSession session = store.session(userId);
        if (session == null) {
            return PromptGenView.empty(versions);
        }
        List<PromptGenHistory.Step> steps = PromptGenHistory.steps(PromptGenStore.messages(session.getState()));
        return PromptGenView.of(session, steps, required(userId, steps), versions);
    }

    private Set<String> required(long userId, List<PromptGenHistory.Step> steps) {
        return PromptGenFacts.required(traderService.mine(userId), PromptGenHistory.lastDraft(steps) != null);
    }

    /** 用第 no 版的快照替换当前对话；版本列表不动 */
    public PromptGenView rollback(long userId, int no) {
        if (store.session(userId) == null) {
            throw new BizException(ErrorCode.PROMPT_GEN_NO_SESSION);
        }
        if (!store.restore(userId, no)) {
            throw new BizException(ErrorCode.PROMPT_GEN_VERSION_NOT_FOUND);
        }
        return view(userId);
    }

    /** 开新对话（旧对话整个覆盖，版本保留）并跑第一轮 */
    public void start(SseChannel channel, CompletableFuture<Void> cancel, long userId, ChatModel model,
                      long endpointId, AgentLang outputLang) {
        List<Message> history = List.of(new UserMessage(prompts.get(userLangResolver.of(userId), "promptgen.kickoff")));
        store.open(userId, endpointId, outputLang, history);
        round(channel, cancel, userId, model, outputLang, history);
    }

    /** 这一轮的起跑历史：当前对话 + 用户这句（回答、跳过、或直接说要改哪里）；还没答的必问项不让跳 */
    public List<Message> replyHistory(long userId, PromptGenSession session, String text, boolean skip) {
        List<Message> history = new ArrayList<>(PromptGenStore.messages(session.getState()));
        if (skip) {
            List<PromptGenHistory.Step> steps = PromptGenHistory.steps(history);
            PromptGenHistory.Ask pending = PromptGenHistory.pending(steps);
            if (pending != null && PromptGenHistory.missing(steps, required(userId, steps)).contains(pending.key())) {
                throw new BizException(ErrorCode.PARAM_ERROR);
            }
        }
        history.add(new UserMessage(skip ? prompts.get(userLangResolver.of(userId), "promptgen.skipped") : text));
        return history;
    }

    public void reply(SseChannel channel, CompletableFuture<Void> cancel, long userId, ChatModel model,
                      PromptGenSession session, List<Message> history) {
        round(channel, cancel, userId, model, AgentLang.of(session.getOutputLang()), history);
    }

    private void round(SseChannel channel, CompletableFuture<Void> cancel, long userId, ChatModel model,
                       AgentLang outputLang, List<Message> history) {
        AgentLang lang = userLangResolver.of(userId);
        PromptGenRun run = null;
        ScheduledFuture<?> heartbeat = heartbeatScheduler.scheduleWithFixedDelay(
                channel::heartbeat, HEARTBEAT_SECONDS, HEARTBEAT_SECONDS, TimeUnit.SECONDS);
        try {
            AiTrader trader = traderService.mine(userId);
            boolean drafted = PromptGenHistory.lastDraft(PromptGenHistory.steps(history)) != null;
            PromptGenFacts.Prompts system = facts.prompts(trader, drafted, lang, outputLang);
            run = new PromptGenRun(userId, lang, outputLang, trader, history, channel::send, cancel, store);
            AgentChat writer = AgentChat.builder()
                    .model(model)
                    .systemPrompt(system.writer())
                    .maxAttempts(3).initialDelay(500).maxDelay(4000)
                    .build();
            AgentChat agent = AgentChat.builder()
                    .model(model)
                    .systemPrompt(system.agent())
                    .tools(localizedTools.of(lang,
                            new TraderQueryToolkit(traderChatService, userId, lang),
                            new PromptGenToolkit(run, writer, playStats, prompts)))
                    .maxAttempts(3).initialDelay(500).maxDelay(4000)
                    .build();
            ReactLoop loop = ReactLoop.builder().chat(agent)
                    .streaming(true)
                    .limiter(new ModelCallLimiter(RUN_CALL_LIMIT,
                            prompts.get(lang, "llm.callLimit.notExecuted"),
                            prompts.get(lang, "llm.callLimit.lastCall")))
                    // 同一批里推题卡和写稿别同时跑
                    .serialTools(Set.of(ASK_USER, WRITE_DRAFT))
                    // 真推出了题卡才收工；被拒的提问让 agent 接着按回执办
                    .stopWhen(run::asked)
                    .build();

            run.status("thinking");
            loop.run(history, null, cancel, run);
            store.saveState(userId, PromptGenHistory.trimDangling(run.messages()));
            if (!channel.isClosed()) {
                channel.send("done", MAPPER.createObjectNode());
                channel.complete();
            }
        } catch (Exception e) {
            log.error("[PromptGen] 这一轮失败 userId={}", userId, e);
            // 用户这句话此刻只在内存里，不落库下一轮就不知道问过
            store.saveState(userId, PromptGenHistory.trimDangling(run == null ? history : run.messages()));
            if (!channel.isClosed()) {
                channel.send("error", MAPPER.createObjectNode()
                        .put("message", LlmErrorMessages.classify(e, prompts, lang)));
                channel.complete();
            }
        } finally {
            heartbeat.cancel(false);
        }
    }
}
