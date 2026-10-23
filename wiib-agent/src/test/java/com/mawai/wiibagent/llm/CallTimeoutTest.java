package com.mawai.wiibagent.llm;

import com.mawai.wiibcommon.constant.AiProtocols;
import io.micrometer.observation.ObservationRegistry;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.beans.factory.ObjectProvider;

import java.util.Base64;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * openai 协议建模的整体超时：{@code OpenAiChatOptions.timeout} 必须显式给
 * （Spring AI 2.0.1 起逐请求传给 SDK，缺省 60s 会盖掉 client 上设的 10 分钟）。
 */
class CallTimeoutTest {

    /** 不显式设的话 OpenAiChatOptions 自带 60s，2.0.1 起会逐请求盖掉 client 上的 10 分钟，思考模型长回答必超时 */
    @Test
    @SuppressWarnings("unchecked")
    void openai协议建模的默认超时与自研协议同值() {
        ObjectProvider<ObservationRegistry> registry = mock(ObjectProvider.class);
        when(registry.getIfUnique(any())).thenReturn(ObservationRegistry.NOOP);
        ByokModelBuilder builder = new ByokModelBuilder(
                new ApiKeyCrypto(Base64.getEncoder().encodeToString(new byte[32])),
                mock(ToolCallingManager.class), registry);

        ChatModel model = builder.build(AiProtocols.OPENAI, "http://127.0.0.1:1/v1", "sk-test", "m", null, false);

        assertThat(((OpenAiChatOptions) model.getOptions()).getTimeout()).isEqualTo(ByokModelBuilder.CALL_TIMEOUT);
    }
}
