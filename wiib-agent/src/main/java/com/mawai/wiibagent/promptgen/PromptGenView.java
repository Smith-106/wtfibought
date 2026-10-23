package com.mawai.wiibagent.promptgen;

import com.mawai.wiibcommon.entity.PromptGenSession;
import com.mawai.wiibcommon.entity.PromptGenVersion;

import java.util.List;
import java.util.Set;

/** 页面上的生成器状态：对话部分全部从历史推出来 */
public record PromptGenView(boolean active, Long endpointId, String outputLang, List<Item> items,
                            Question question, Integer currentNo, boolean done, List<Version> versions) {

    /** 对话里的一条。type：agent / user 只带 text；question 带题和回答；version 只带版本号 */
    public record Item(String type, String text, Question question, String answer, Integer no) {
    }

    /** 题卡；required = 还没答的必问项、不能跳，由代码按 key 判，不听模型的。答过的必问项再追问可以跳 */
    public record Question(String key, String question, String recommended, String reason,
                           List<String> options, boolean required) {

        static Question of(PromptGenHistory.Ask ask, Set<String> unanswered) {
            return new Question(ask.key(), ask.question(), ask.recommended(), ask.reason(), ask.options(),
                    unanswered.contains(ask.key()));
        }
    }

    public record Version(int no, String body, String notes, boolean finalVersion) {

        static Version of(PromptGenVersion v) {
            return new Version(v.getVersionNo(), v.getBody(), v.getNotes(), Boolean.TRUE.equals(v.getIsFinal()));
        }

        static Version of(PromptGenHistory.Drafted d) {
            return new Version(d.no(), d.body(), d.notes(), d.finalVersion());
        }
    }

    /** 还没开过对话：只有版本 */
    static PromptGenView empty(List<Version> versions) {
        return new PromptGenView(false, null, null, List.of(), null, null, false, versions);
    }

    static PromptGenView of(PromptGenSession session, List<PromptGenHistory.Step> steps, Set<String> required,
                            List<Version> versions) {
        Set<String> unanswered = Set.copyOf(PromptGenHistory.missing(steps, required));
        List<Item> items = steps.stream().map(step -> switch (step) {
            case PromptGenHistory.AgentText t -> new Item("agent", t.text(), null, null, null);
            case PromptGenHistory.UserText u -> new Item("user", u.text(), null, null, null);
            case PromptGenHistory.Ask a -> new Item("question", null, Question.of(a, unanswered), a.answer(), null);
            case PromptGenHistory.Drafted d -> new Item("version", null, null, null, d.no());
        }).toList();
        PromptGenHistory.Ask pending = PromptGenHistory.pending(steps);
        PromptGenHistory.Drafted last = PromptGenHistory.lastDraft(steps);
        return new PromptGenView(true, session.getEndpointId(), session.getOutputLang(), items,
                pending == null ? null : Question.of(pending, unanswered),
                last == null ? null : last.no(), PromptGenHistory.done(steps), versions);
    }
}
