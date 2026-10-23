package com.mawai.wiibcommon.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

/** 交易指令生成器写出的一版：每个用户只留最近 3 版，跨对话也算 */
@Data
@TableName("prompt_gen_version")
public class PromptGenVersion {

    @TableId(type = IdType.AUTO)
    private Long id;
    private Long userId;
    /** 版本号：每个用户自己往上数，开新对话不从 1 重来 */
    private Integer versionNo;
    private String body;
    private String notes;
    /** agent 认为没什么可问了，这是最终版 */
    private Boolean isFinal;
    /** 写这版时的正文语言，回退时连同对话一起恢复 */
    private String outputLang;
    /** 交稿那一刻的对话快照，回退到这版就用它替换当前对话 */
    private byte[] state;
    private LocalDateTime createdAt;
}
