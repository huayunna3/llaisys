from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
import json
import os
import time
from ctypes import byref, c_int, c_int64, c_size_t

from pathlib import Path
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor

        model_path = Path(model_path)

        #加载权重
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pytorch", device="cpu")
            state_dict_ = {}
            for name_ in data_.keys():
                ## TODO: load the model weights
                state_dict_[name_] = data_.get_tensor(name_)
        #创建模型
        w_naming = Qwen2WeightsNaming()
        if w_naming.match(state_dict_):
            ndev = 1
            dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
            self.meta = Qwen2MetaCStruct(config)
            self.weights = Qwen2WeightsCStruct(self.meta, state_dict_, w_naming, ndev)
            self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
                byref(self.meta), byref(self.weights), device, ndev, dev_ids
            )
            self.weights.release()
            


    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        tokens = list(inputs)
        max_len = len(tokens) + max_new_tokens
        kvcache = LIB_LLAISYS.llaisysQwen2KVCacheCreate(self.model, max_len)

        # 预填充
        print("➡️ 💬 Qwen2: prefilling...", flush=True)
        start_time = time.time()
        ntoken = len(tokens)
        token_ids = (c_int64 * ntoken)(*tokens)
        past_len = c_size_t(0)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model, token_ids, c_size_t(ntoken), kvcache, past_len
        )
        tokens.append(next_token)
        end_time = time.time()
        prefill_time = end_time - start_time
        print(f"LLAISYS Prefill Time: {prefill_time:.4f}s")
        

        # decode
        print(" Qwen2: decoding...\n\n", flush=True)
        start_time = time.time()
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
            ntoken = 1
            token_ids = (c_int64 * 1)(next_token)
            past_len = c_size_t(len(tokens) - 1)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model, token_ids, ntoken, kvcache, past_len
            )
            tokens.append(next_token)
            # print("current tokens: ", tokens, flush=True)
        end_time = time.time()
        decode_time = end_time - start_time
        print(f"LLAISYS Decode Time: {decode_time:.4f}s")

        nlayer = self.meta.nlayer
        LIB_LLAISYS.llaisysQwen2KVCacheDestroy(kvcache, nlayer)
        self.destroy()

        return tokens

    def destroy(self):
        LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
        return []
