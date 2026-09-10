"""Engine adapter for SGLang."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .base import EngineAdapter


class SGLangEngineAdapter(EngineAdapter):
    """Adapter for SGLang serving engine."""

    name: str = "sglang"
    default_port: int = 30000
    openai_compatible: bool = True

    def get_default_parameter_space(self) -> Dict[str, Any]:
        """Return default search space for SGLang."""
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
            "chunked_prefill_size": {
                "enabled": False,
                "options": [512, 1024, 2048],
            },
            "schedule_policy": {
                "enabled": False,
                "options": ["lpm", "fcfs"],
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

        # Check prefillSettings
        prefill = serving_template.get("prefillSettings", {})
        if "prefillChunkTokens" in prefill:
            try:
                baseline["chunked_prefill_size"] = int(prefill["prefillChunkTokens"])
            except (ValueError, TypeError):
                pass

        # Check extraCommand for SGLang-specific flags
        for cmd in serving_template.get("extraCommand", []):
            if "--schedule-policy=" in cmd:
                baseline["schedule_policy"] = cmd.split("=")[-1].strip()
            elif "--chunked-prefill-size=" in cmd and "chunked_prefill_size" not in baseline:
                try:
                    baseline["chunked_prefill_size"] = int(cmd.split("=")[-1].strip())
                except ValueError:
                    pass

        return baseline

    def map_to_serving_patch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map trial parameters to AFSBox ModelServing patch for SGLang."""
        spec_patch: Dict[str, Any] = {}
        parallelism: Dict[str, Any] = {}
        extra_args: List[str] = []

        for k, v in params.items():
            k_lower = k.lower().replace("-", "_")

            # Top-level AFSBox CR fields: controller translates batchSize to --max-running-requests
            if k_lower in ("batchsize", "batch_size", "max_running_requests", "max_num_seqs"):
                spec_patch["batchSize"] = str(int(round(float(v))))
            # Controller translates contextLength to --context-length
            elif k_lower in ("contextlength", "context_length", "max_model_len"):
                spec_patch["contextLength"] = str(int(round(float(v))))
            elif k_lower in ("replicas", "replica_count"):
                spec_patch["replicas"] = int(round(float(v)))

            # Parallelism: controller translates tp to --tp-size
            elif k_lower in ("tp", "tp_size", "tensor_parallel_size"):
                parallelism["tp"] = int(round(float(v)))
            elif k_lower in ("pp", "pp_size", "pipeline_parallel_size"):
                parallelism["pp"] = int(round(float(v)))
            elif k_lower in ("dp", "dp_size", "data_parallel_size"):
                parallelism["dp"] = int(round(float(v)))

            # GPU & Memory: controller translates gpuMemoryUtilization to --mem-fraction-static
            elif k_lower in ("gpu_memory_utilization", "gpu_mem_util", "mem_fraction_static"):
                spec_patch["gpuMemoryUtilization"] = str(round(float(v), 4))
            elif k_lower in ("max_prefill_tokens", "max_num_batched_tokens"):
                if "prefillSettings" not in spec_patch:
                    spec_patch["prefillSettings"] = {}
                spec_patch["prefillSettings"]["maxBatchTokens"] = str(int(round(float(v))))
            elif k_lower in ("chunked_prefill_size", "prefill_chunk_tokens"):
                if "prefillSettings" not in spec_patch:
                    spec_patch["prefillSettings"] = {}
                chunk_int = int(round(float(v)))
                spec_patch["prefillSettings"]["prefillChunkTokens"] = str(chunk_int)
                extra_args.append(f"--chunked-prefill-size={chunk_int}")
            elif k_lower in ("kv_cache_dtype",):
                spec_patch["kvCacheDtype"] = str(v)

            # SGLang-specific CLI arguments
            elif k_lower in ("schedule_policy", "policy"):
                extra_args.append(f"--schedule-policy={v}")
            elif k_lower in ("attention_backend",):
                extra_args.append(f"--attention-backend={v}")
            elif k_lower in ("enable_mixed_chunk",):
                if v:
                    extra_args.append("--enable-mixed-chunk")
            elif k_lower in ("disable_radix_cache",):
                if v:
                    extra_args.append("--disable-radix-cache")
            elif isinstance(v, bool):
                flag = k.replace("_", "-")
                if v:
                    extra_args.append(f"--{flag}")
                else:
                    extra_args.append(f"--disable-{flag}")
            elif k.startswith("values.") or k.startswith("params."):
                extra_args.append(f"--{k}={v}")
            else:
                extra_args.append(f"--{k.replace('_', '-')}={v}")

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
        """Build SGLang server launch command."""
        cmd = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--port",
            str(port),
            "--host",
            host,
        ]
        if extra_args:
            cmd.extend(extra_args)
        return cmd
