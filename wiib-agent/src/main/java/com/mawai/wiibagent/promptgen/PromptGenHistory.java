package com.mawai.wiibagent.promptgen;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.DeserializationFeature;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectReader;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static com.mawai.wiibagent.promptgen.PromptGenToolkit.ASK_USER;
import static com.mawai.wiibagent.promptgen.PromptGenToolkit.WRITE_DRAFT;
import static com.mawai.wiibcommon.util.JsonUtils.MAPPER;

/**
 * 从对话历史里认出生成器关心的几样：题、答、版本、agent 的话、agent 读的数据、用户的话。
 * 页面状态、题数上限、必问项检查、写稿用的问答记录都从这儿推，历史之外不另存状态。
 * <p>
 * 第一条用户消息是开场指令，不算用户说的话。
 */
final class PromptGenHistory {

    /** 读完第一个 JSON 值就停，后面拼的东西不管 */
    private static final ObjectReader FIRST_VALUE =
            MAPPER.reader().without(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);

    private PromptGenHistory() {
    }

    sealed interface Step permits AgentText, Ask, Drafted, Tool, UserText {
    }

    /** agent 自己说的话 */
    record AgentText(String text) implements Step {
    }

    /** agent 读了一次数据：ask_user、write_draft 以外的工具调用 */
    record Tool(String name, JsonNode args) implements Step {
    }

    /** 一道展示出去的题；answer 为 null 就是还在等回答 */
    record Ask(String key, String question, String recommended, String reason, List<String> options,
               String answer) implements Step {
        Ask answered(String text) {
            return new Ask(key, question, recommended, reason, options, text);
        }
    }

    /** 写成的一版 */
    record Drafted(int no, String body, String notes, boolean finalVersion) implements Step {
    }

    /** 没有题挂着时用户直接说的话 */
    record UserText(String text) implements Step {
    }

    static List<Step> steps(List<Message> messages) {
        List<Step> steps = new ArrayList<>();
        for (int i = 0; i < messages.size(); i++) {
            Message message = messages.get(i);
            if (message instanceof AssistantMessage reply) {
                if (reply.getText() != null && !reply.getText().isBlank()) {
                    steps.add(new AgentText(reply.getText().strip()));
                }
                Map<String, String> results = results(messages, i);
                // 同一条里读数据排在题和版本前面：题得是最后一步，用户下一句才配得上它
                List<Step> reads = new ArrayList<>();
                List<Step> cards = new ArrayList<>();
                for (AssistantMessage.ToolCall call : reply.getToolCalls()) {
                    String data = results.get(call.id());
                    // 没配上回执（被叫停）的不算数
                    if (data == null) {
                        continue;
                    }
                    if (!ASK_USER.equals(call.name()) && !WRITE_DRAFT.equals(call.name())) {
                        reads.add(new Tool(call.name(), args(call)));
                        continue;
                    }
                    JsonNode result = json(data);
                    // 被拒的题、没写成的稿也不算数
                    if (ASK_USER.equals(call.name()) && result.path("shown").asBoolean(false)) {
                        cards.add(ask(args(call)));
                    } else if (WRITE_DRAFT.equals(call.name()) && result.has("no")) {
                        cards.add(new Drafted(result.path("no").asInt(), result.path("body").asString(""),
                                result.path("notes").asString(""), result.path("finalVersion").asBoolean(false)));
                    }
                }
                steps.addAll(reads);
                steps.addAll(cards);
            } else if (message instanceof UserMessage user && i > 0) {
                int last = steps.size() - 1;
                if (last >= 0 && steps.get(last) instanceof Ask ask && ask.answer() == null) {
                    steps.set(last, ask.answered(user.getText()));
                } else {
                    steps.add(new UserText(user.getText()));
                }
            }
        }
        return steps;
    }

    /** 还在等回答的那道题 */
    static Ask pending(List<Step> steps) {
        return !steps.isEmpty() && steps.getLast() instanceof Ask ask && ask.answer() == null ? ask : null;
    }

    /** 上一版之后已经问了几题 */
    static int asksSinceDraft(List<Step> steps) {
        int asks = 0;
        for (int i = steps.size() - 1; i >= 0 && !(steps.get(i) instanceof Drafted); i--) {
            if (steps.get(i) instanceof Ask) {
                asks++;
            }
        }
        return asks;
    }

    /** 必问项里还没答的 */
    static List<String> missing(List<Step> steps, Set<String> required) {
        Set<String> answered = steps.stream()
                .filter(s -> s instanceof Ask ask && ask.answer() != null)
                .map(s -> ((Ask) s).key())
                .collect(Collectors.toSet());
        return required.stream().filter(key -> !answered.contains(key)).toList();
    }

    static Drafted lastDraft(List<Step> steps) {
        for (int i = steps.size() - 1; i >= 0; i--) {
            if (steps.get(i) instanceof Drafted drafted) {
                return drafted;
            }
        }
        return null;
    }

    /** agent 交了最终版，之后用户没再开口 */
    static boolean done(List<Step> steps) {
        for (int i = steps.size() - 1; i >= 0; i--) {
            Step step = steps.get(i);
            if (step instanceof Drafted drafted) {
                return drafted.finalVersion();
            }
            if (!(step instanceof AgentText) && !(step instanceof Tool)) {
                return false;
            }
        }
        return false;
    }

    /** 写稿人看的问答记录：答过的题和用户直接说的话，按先后排 */
    static String transcript(List<Step> steps, String userLabel) {
        StringBuilder out = new StringBuilder();
        for (Step step : steps) {
            if (step instanceof Ask ask && ask.answer() != null) {
                out.append("- [").append(ask.key()).append("] ").append(ask.question())
                        .append(" → ").append(ask.answer()).append('\n');
            } else if (step instanceof UserText user) {
                out.append("- ").append(userLabel).append(user.text()).append('\n');
            }
        }
        return out.toString().strip();
    }

    /** 落库前去掉末尾没配上回执的工具调用：留着它，续聊时上游直接 400 */
    static List<Message> trimDangling(List<Message> messages) {
        if (!messages.isEmpty() && messages.getLast() instanceof AssistantMessage reply && reply.hasToolCalls()) {
            return new ArrayList<>(messages.subList(0, messages.size() - 1));
        }
        return messages;
    }

    static String key(String raw) {
        return raw == null ? "" : raw.strip().toLowerCase(Locale.ROOT);
    }

    /** 工具调用的参数；不是 JSON 当空对象 */
    static JsonNode args(AssistantMessage.ToolCall call) {
        return json(call.arguments());
    }

    /**
     * 第 i 条的工具调用对应的回执：ReactLoop 总把回执紧跟在调用后面放，按位置配，不跨整段历史按 id 配——
     * 有的网关不同轮会复用同一个 id
     */
    private static Map<String, String> results(List<Message> messages, int i) {
        if (i + 1 < messages.size() && messages.get(i + 1) instanceof ToolResponseMessage responses) {
            return responses.getResponses().stream().collect(Collectors.toMap(
                    ToolResponseMessage.ToolResponse::id, ToolResponseMessage.ToolResponse::responseData, (a, b) -> b));
        }
        return Map.of();
    }

    private static Ask ask(JsonNode args) {
        List<String> options = new ArrayList<>();
        args.path("options").forEach(option -> options.add(option.asString()));
        return new Ask(key(args.path("key").asString("")), args.path("question").asString(""),
                args.path("recommended").asString(""), args.path("reason").asString(""), options, null);
    }

    /**
     * 工具的 String 返回值被框架再编码成 JSON 字符串，这里剥一层。只读第一个值：保险丝会在最后一条回执后面拼一句预算提示。
     * 不是 JSON（保险丝的占位回执）当空对象
     */
    private static JsonNode json(String raw) {
        try {
            JsonNode node = FIRST_VALUE.readTree(raw);
            return node.isString() ? FIRST_VALUE.readTree(node.asString()) : node;
        } catch (JacksonException e) {
            return MAPPER.createObjectNode();
        }
    }
}
