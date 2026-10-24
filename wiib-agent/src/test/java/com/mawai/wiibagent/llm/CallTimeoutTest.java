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
 * openai 协议建模的整体超时：options 不带 timeout，SDK 用 client 上的 {@link ByokModelBuilder#CALL_TIMEOUT}。
 */
class CallTimeoutTest {

    /** options 的 timeout 一旦非空就逐请求盖掉 client 的 10 分钟（Spring AI 2.0.1 默认带 60s），思考模型长回答必超时 */
    @Test
    @SuppressWarnings("unchecked")
    void openai协议options不带超时() {
        ObjectProvider<ObservationRegistry> registry = mock(ObjectProvider.class);
        when(registry.getIfUnique(any())).thenReturn(ObservationRegistry.NOOP);
        ByokModelBuilder builder = new ByokModelBuilder(
                new ApiKeyCrypto(Base64.getEncoder().encodeToString(new byte[32])),
                mock(ToolCallingManager.class), registry);

        ChatModel model = builder.build(AiProtocols.OPENAI, "http://127.0.0.1:1/v1", "sk-test", "m", null, false);

        assertThat(((OpenAiChatOptions) model.getOptions()).getTimeout()).isNull();
    }
}
