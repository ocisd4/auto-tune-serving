"""Engine adapter for vLLM."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .base import EngineAdapter


class VLLMEngineAdapter(EngineAdapter):
    """Adapter for vLLM serving engine."""

    name: str = "vllm"
    default_port: int = 8000
    openai_compatible: bool = True

    def get_default_parameter_space(self) -> Dict[str, Any]:
        """Return safe default search space for vLLM."""
        return {
            "batch_size": {
                "enabled": True,
                "options": [8, 16, 32, 64],
            },
            "gpu_memory_utilization": {
                "enabled": True,
                "min": 0.65,
                "max": 0.85,
                "step": 0.05,
            },
        }

    def extract_baseline_parameters(self, serving_template: Dict[str, Any]) -> Dict[str, Any]:
        """Extract baseline parameters from serving template."""
        baseline: Dict[str, Any] = {}
        if "batchSize" in serving_template:
            try:
                baseline["batch_size"] = int(serving_template["batchSize"])
            except (ValueError, TypeError):
                pass
        if "gpuMemoryUtilization" in serving_template:
            try:
                baseline["gpu_memory_utilization"] = float(serving_template["gpuMemoryUtilization"])
            except (ValueError, TypeError):
                pass
        if "contextLength" in serving_template:
            try:
                baseline["context_length"] = int(serving_template["contextLength"])
            except (ValueError, TypeError):
                pass
        if "prefillSettings" in serving_template and isinstance(serving_template["prefillSettings"], dict):
            if "maxBatchTokens" in serving_template["prefillSettings"]:
                try:
                    baseline["max_num_batched_tokens"] = int(serving_template["prefillSettings"]["maxBatchTokens"])
                except (ValueError, TypeError):
                    pass
        return baseline

    def map_to_serving_patch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map trial parameters to AFSBox ModelServing patch for vLLM."""
        spec_patch: Dict[str, Any] = {}
        parallelism: Dict[str, Any] = {}
        extra_args: List[str] = []

        for k, v in params.items():
            k_lower = k.lower().replace("-", "_")

            # Top-level AFSBox CR fields natively understood by controller
            if k_lower in ("batchsize", "batch_size", "max_num_seqs"):
                spec_patch["batchSize"] = str(int(round(float(v))))
            elif k_lower in ("contextlength", "context_length", "max_model_len"):
                spec_patch["contextLength"] = str(int(round(float(v))))
            elif k_lower in ("replicas", "replica_count"):
                spec_patch["replicas"] = int(round(float(v)))

            # Parallelism
            elif k_lower in ("tp", "tensor_parallel_size"):
                parallelism["tp"] = int(round(float(v)))
            elif k_lower in ("pp", "pipeline_parallel_size"):
                parallelism["pp"] = int(round(float(v)))
            elif k_lower in ("dp", "data_parallel_size"):
                parallelism["dp"] = int(round(float(v)))

            # GPU & Memory
            elif k_lower in ("gpu_memory_utilization", "gpu_mem_util"):
                spec_patch["gpuMemoryUtilization"] = str(round(float(v), 4))
            elif k_lower in ("max_num_batched_tokens", "max_batch_tokens"):
                if "prefillSettings" not in spec_patch:
                    spec_patch["prefillSettings"] = {}
                spec_patch["prefillSettings"]["maxBatchTokens"] = str(int(round(float(v))))
            elif k_lower in ("kv_cache_dtype",):
                spec_patch["kvCacheDtype"] = str(v)
            elif k_lower in ("enable_cuda_graphs", "cuda_graph"):
                if isinstance(v, bool):
                    spec_patch["cudaGraph"] = {"enabled": v}

            # vLLM CLI extra args
            elif isinstance(v, bool):
                flag = k.replace("_", "-")
                extra_args.append(f"--{flag}" if v else f"--no-{flag}")
            elif k.startswith("values.") or k.startswith("params."):
                extra_args.append(f"--{k}={v}")
            else:
                if isinstance(v, float) and v.is_integer():
                    v_str = str(int(v))
                else:
                    v_str = str(v)
                extra_args.append(f"--{k.replace('_', '-')}={v_str}")

        if parallelism:
            spec_patch["parallelism"] = parallelism
        if extra_args:
            spec_patch["extraCommand"] = extra_args

        return spec_patch

    def build_server_command(
        self,
        model: str,
        port: int,
        host: str = "0.0.0.0",
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Build vLLM OpenAI API server launch command."""
        cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--port",
            str(port),
            "--host",
            host,
        ]
        if extra_args:
            cmd.extend(extra_args)
        return cmd
