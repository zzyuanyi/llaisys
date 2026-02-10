import json
import os
import platform
import time
from ctypes import byref, c_int, c_int64, c_long, c_size_t,c_longlong
from pathlib import Path
from typing import Sequence

import safetensors

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    Qwen2MetaCStruct,
    Qwen2WeightsCStruct,
    Qwen2WeightsNaming,
)


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        print(
            "Qwen2: Initializing Model and Loading Weights\n",
            sep="",
        )

        # load config
        print("Qwen2: loading configs...", flush=True)
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config

        if device == DeviceType.CPU:
            self.config["torch_dtype"] = "float32"

        # load weights
        print("Qwen2: loading weights...\n", flush=True)
        for file in sorted(model_path.glob("*.safetensors")):
            state_dict = {}
            data_ = safetensors.safe_open(file, framework="pytorch", device="cpu")
            for name_ in data_.keys():
                state_dict[name_] = data_.get_tensor(name_)

        # create model
        naming = Qwen2WeightsNaming()
        if naming.match(state_dict):
            ndev = 1
            dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
            self.meta = Qwen2MetaCStruct(config)
            self.weights = Qwen2WeightsCStruct(self.meta, state_dict, naming, ndev)
            self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
                byref(self.meta), byref(self.weights), device, ndev, dev_ids
            )
            self.weights.release()
        else:
            raise ValueError("state_dict fail weights name compare")

    def _get_token_array_type(self):
        """根据平台返回正确的令牌数组类型"""
        # Windows平台使用c_long，其他平台使用c_int64
        if platform.system() == "Windows":
            return c_int64
        else:
            return c_int64

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        tokens = list(inputs)
        max_len = len(tokens) + max_new_tokens
        kvcache = LIB_LLAISYS.llaisysQwen2KVCacheCreate(self.model, max_len)

        # 获取适合当前平台的令牌数组类型
        token_array_type = self._get_token_array_type()
        
        # 记录使用的类型，便于调试
        print(f"Using token array type: {token_array_type.__name__} on {platform.system()}")

        # prefill
        print("Qwen2: prefilling...", flush=True)
        start_time = time.time()
        ntoken = len(tokens)
        
        # 使用平台特定的类型创建令牌数组
        if token_array_type == c_int64:
            # Windows平台使用c_long
            token_ids = (c_int64 * ntoken)(*tokens)
        else:
            # 其他平台使用原逻辑的c_int64
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
        print("Qwen2: decoding...\n", flush=True)
        start_time = time.time()
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
            ntoken = 1
            
            # 使用平台特定的类型创建令牌数组
            if token_array_type == c_int64:
                # Windows平台使用c_long
                token_ids = (c_int64 * 1)(next_token)
            else:
                # 其他平台使用原逻辑的c_int64
                token_ids = (c_int64 * 1)(next_token)
                
            past_len = c_size_t(len(tokens) - 1)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model, token_ids, ntoken, kvcache, past_len
            )
            tokens.append(next_token)
        end_time = time.time()
        decode_time = end_time - start_time
        print(f"LLAISYS Decode Time: {decode_time:.4f}s")

        nlayer = self.meta.nlayer
        LIB_LLAISYS.llaisysQwen2KVCacheDestroy(kvcache, nlayer)
        self.destroy()

        return tokens

    def destroy(self):
        LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
        print(
            "Qwen2 COMPLETE: All resources cleaned up\n",
            sep="",
        )
