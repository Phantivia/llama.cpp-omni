#pragma once

// Qwen3-ForcedAligner: 给定音频与一串对齐单元,回每个单元的起止时刻。
//
// 模型由两个 GGUF 组成:
//   - LM    llama.cpp 的 `qwen3` 架构主干(0.6B)+ 词表
//   - Audio clip 的 `qwen3a` 音频塔 + 多模态投影,另存 `score.weight`(时间桶头)
//
// 单元怎么切由调用方决定,这里原样收下:逐字、逐词或任意粒度都行。

#include <cstddef>
#include <string>
#include <vector>

struct qwen3_aligner;

struct qwen3_aligner_params {
    std::string lm_path;
    std::string audio_path;
    int         n_gpu_layers = -1;    // -1 = 全部卸载到 GPU
    int         n_threads    = 4;
    int         n_ctx        = 4096;  // 约 5 分钟音频
};

// 单元在音频里的位置,单位秒
struct qwen3_aligner_span {
    std::string text;
    double      start = 0.0;
    double      end   = 0.0;
};

// 失败回 nullptr 并填 err
qwen3_aligner * qwen3_aligner_init(const qwen3_aligner_params & params, std::string & err);

// 容忍 nullptr
void qwen3_aligner_free(qwen3_aligner * ctx);

// 模型要求的采样率(16000)
int qwen3_aligner_sample_rate(const qwen3_aligner * ctx);

// 把一段音频文件(wav/mp3/flac 等)解成单声道 16k float PCM
bool qwen3_aligner_decode_audio(qwen3_aligner *      ctx,
                                const void *         data,
                                size_t               size,
                                std::vector<float> & pcm,
                                std::string &        err);

// pcm 必须是 qwen3_aligner_sample_rate() 的单声道数据。
// spans 与 units 一一对应,顺序一致。
bool qwen3_aligner_align(qwen3_aligner *                        ctx,
                         const float *                          pcm,
                         size_t                                 n_samples,
                         const std::vector<std::string> &       units,
                         std::vector<qwen3_aligner_span> &      spans,
                         std::string &                          err);
