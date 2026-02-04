#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta { //元数据结构体
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights { //权重结构体
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    /*
llaisysQwen2ModelCreate - 模型创建

llaisysQwen2ModelDestroy - 模型销毁

llaisysQwen2ModelWeights - 获取权重指针

llaisysQwen2ModelInfer - 推理函数
    */

    struct LlaisysQwen2Model; //前向模型结构体

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);

    /*
llaisysQwen2LoadWeightByName - 权重加载

llaisysQwen2Forward - 完整前向传播（返回logits）

llaisysQwen2ResetCache - 重置KV-Cache

llaisysQwen2GetMeta - 获取模型配置信息
    */
    __export int llaisysQwen2LoadWeightByName(struct LlaisysQwen2Weights *weights, const char *name, llaisysTensor_t tensor);

    __export llaisysTensor_t llaisysQwen2Forward(struct LlaisysQwen2Model *model, const int64_t *token_ids, size_t ntoken, int past_len);
    
    __export void llaisysQwen2ResetCache(struct LlaisysQwen2Model *model);

    __export const LlaisysQwen2Meta* llaisysQwen2GetMeta(struct LlaisysQwen2Model *model);

}
#endif // LLAISYS_MODELS_QWEN2_H
