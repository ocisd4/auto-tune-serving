"""AFSBox Kubernetes execution backend.

Orchestrates trials by updating AFSBox ModelServing experimental instances
and triggering AFSBox Benchmark (AIPerf) jobs via Kubernetes Custom Resources.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..core.trial import ExecutionInfo, TrialConfig, TrialResult
from ..engines.base import EngineAdapter
from ..engines.factory import get_engine_adapter
from .backends import ExecutionBackend, JobHandle

logger = logging.getLogger(__name__)


def _deep_update(target: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dictionary."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_update(target[k], v)
        else:
            target[k] = v
    return target


def _select_best_pareto_candidate(
    pareto_front: List[Dict[str, Any]],
    objectives: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Pick a single representative trial out of a Pareto front by equal-weighted
    score, instead of just taking whichever entry happens to be first in the list.

    Pareto-optimal solutions are non-dominated by definition — there is no single
    objectively "best" one, they're pure trade-offs. `pareto_front`'s own order
    (Optuna's ``study.best_trials``) is not a ranking; previously this module took
    ``pareto_front[0]`` as "bestCandidate", which just surfaced whichever trial
    happened to sort first out of the non-dominated set (in practice this tended
    to be an early/chronologically-first trial, not a meaningfully better one —
    e.g. it could be shadowed by a trial with 2x the throughput that ran later).

    To pick a single defensible representative, score each trial by the mean of
    its per-objective improvement over baseline (``baseline_improvements``,
    already direction-corrected upstream in study_controller.get_optimization_results
    so "positive == better" regardless of maximize/minimize) and take the max.
    All objectives are weighted equally — the study config has no per-objective
    weight field (see core/config.py ObjectiveConfig) — so this is the least
    presumptuous "real" multi-objective ranking without inventing a weighting
    scheme the user never configured.

    Falls back to min-max normalized raw ``values`` (direction-corrected via
    ``objectives``) when no trial has usable baseline_improvements — e.g. the
    baseline comparison was unavailable or baseline trials were disabled — so
    this still produces a meaningful pick rather than silently reverting to
    "first in list".
    """
    if not pareto_front:
        return None

    def trial_name(p: Dict[str, Any]) -> Optional[str]:
        trial_num = p.get("trial")
        return f"trial_{trial_num}" if trial_num is not None else None

    def score_from_improvements(p: Dict[str, Any]) -> Optional[float]:
        improvements = [v for v in (p.get("baseline_improvements") or []) if v is not None]
        if not improvements:
            return None
        return sum(improvements) / len(improvements)

    scored = [(trial_name(p), score_from_improvements(p)) for p in pareto_front]
    if any(score is not None for _, score in scored):
        best_name, _ = max(
            (item for item in scored if item[1] is not None),
            key=lambda item: item[1],
        )
        return best_name

    # Fallback: no baseline_improvements anywhere — normalize raw `values` per
    # objective across the front (min-max, direction-corrected) and average.
    directions = [obj.get("direction", "maximize") for obj in (objectives or [])]
    values_matrix = [p.get("values") or [] for p in pareto_front]
    n_objectives = max((len(v) for v in values_matrix), default=0)
    if n_objectives == 0:
        # No values to rank by at all — nothing meaningful to pick beyond order.
        return trial_name(pareto_front[0])

    normalized_scores = [0.0] * len(pareto_front)
    for obj_idx in range(n_objectives):
        column = [v[obj_idx] for v in values_matrix if len(v) > obj_idx]
        if not column:
            continue
        lo, hi = min(column), max(column)
        direction = directions[obj_idx] if obj_idx < len(directions) else "maximize"
        for i, v in enumerate(values_matrix):
            if len(v) <= obj_idx:
                continue
            if hi == lo:
                normalized = 1.0  # every trial tied on this objective — no signal, don't penalize
            else:
                normalized = (v[obj_idx] - lo) / (hi - lo)
                if direction == "minimize":
                    normalized = 1.0 - normalized
            normalized_scores[i] += normalized / n_objectives

    best_idx = max(range(len(pareto_front)), key=lambda i: normalized_scores[i])
    return trial_name(pareto_front[best_idx])

AFSBOX_GROUP = "afsbox.asus.com"
AFSBOX_VERSION = "v1beta1"
PLURAL_SERVINGS = "modelservings"
PLURAL_BENCHMARKS = "benchmarks"
PLURAL_TUNINGS = "modeltunings"


LABEL_SERVING = "afsbox.asus.com/serving"
LABEL_TUNING = "afsbox.asus.com/model-tuning"
LABEL_CANDIDATE = "afsbox.asus.com/candidate"


class AFSBoxK8sBackend(ExecutionBackend):
    """Execution backend interfacing with AFSBox Kubernetes CRDs (ModelServing and Benchmark)."""

    def __init__(
        self,
        tuning_name: Optional[str] = None,
        namespace: str = "default",
        serving_name: Optional[str] = None,
        serving_template: Optional[Dict[str, Any]] = None,
        cleanup_serving: bool = False,
        deploy_timeout_seconds: int = 1800,
        poll_interval_seconds: int = 10,
        engine: Optional[str] = None,
    ):
        """Initialize AFSBox Kubernetes execution backend.

        Args:
            tuning_name: Name of parent ModelTuning CR (if run as part of a ModelTuning session).
            namespace: Kubernetes namespace where ModelServing and Benchmark CRs reside.
            serving_name: Explicit name of experiment ModelServing CR.
            serving_template: Custom ModelServing spec dictionary template to generate serving if missing.
            cleanup_serving: Whether to delete the ModelServing CR upon study completion.
            deploy_timeout_seconds: Max seconds to wait for ModelServing to become Ready after patch.
            poll_interval_seconds: Interval between status polls.
            engine: Inference serving engine type ('vllm', 'sglang', 'llamacpp').
        """
        self.tuning_name = tuning_name
        self.namespace = namespace
        self.serving_name = (
            serving_name
            or (f"{tuning_name}-exp" if tuning_name else "optuna-tune-exp")
        )
        self.serving_template = serving_template
        self.cleanup_serving = cleanup_serving
        self.deploy_timeout_seconds = deploy_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.active_trials: Dict[str, Dict[str, Any]] = {}
        self._cached_tuning_cr: Optional[Dict[str, Any]] = None
        self._cached_tuning_test_suite: Optional[List[Dict[str, Any]]] = None
        self._serving_created_by_us: bool = False
        self._serving_just_created: bool = False
        self._engine_adapter: Optional[EngineAdapter] = get_engine_adapter(engine) if engine else None

        # Lazy load kubernetes client to avoid import errors when running in environments without it
        try:
            from kubernetes import client, config

            try:
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes configuration")
            except Exception:
                config.load_kube_config()
                logger.info("Loaded local kube-config")

            self.custom_api = client.CustomObjectsApi()
        except ImportError:
            raise RuntimeError(
                "The 'kubernetes' package is required for the AFSBox backend. "
                "Please install it via 'pip install kubernetes'."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kubernetes client for AFSBox backend: {e}")

    @property
    def engine_adapter(self) -> EngineAdapter:
        """Get or lazily initialize the engine adapter for this session."""
        if self._engine_adapter is not None:
            return self._engine_adapter

        engine_type = None
        if self.serving_template and isinstance(self.serving_template, dict):
            engine_type = self.serving_template.get("engine", {}).get("type")

        if not engine_type:
            spec = self._get_parent_tuning_spec()
            if spec:
                engine_type = spec.get("servingTemplate", {}).get("engine", {}).get("type")

        self._engine_adapter = get_engine_adapter(engine_type)
        logger.info(
            "Configured AFSBox backend with engine adapter: %s (default port: %d)",
            self._engine_adapter.name,
            self._engine_adapter.default_port,
        )
        return self._engine_adapter

    def _get_parent_tuning_cr(self) -> Optional[Dict[str, Any]]:
        """Fetch parent ModelTuning CR if available."""
        if not self.tuning_name:
            return None
        if self._cached_tuning_cr is not None:
            return self._cached_tuning_cr
        try:
            tuning_obj = self.custom_api.get_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_TUNINGS,
                name=self.tuning_name,
            )
            self._cached_tuning_cr = tuning_obj
            return tuning_obj
        except Exception as e:
            logger.warning("Could not fetch ModelTuning %s: %s", self.tuning_name, e)
            return None

    def _get_parent_tuning_suite(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch testSuite from parent ModelTuning CR if available."""
        if self._cached_tuning_test_suite is not None:
            return self._cached_tuning_test_suite
        cr = self._get_parent_tuning_cr()
        if cr:
            suite = cr.get("spec", {}).get("testSuite")
            if suite:
                self._cached_tuning_test_suite = suite
                logger.info("Cached parent ModelTuning %s testSuite (%d items)", self.tuning_name, len(suite))
                return suite
        return None

    def _get_parent_tuning_spec(self) -> Optional[Dict[str, Any]]:
        """Fetch spec from parent ModelTuning CR if available."""
        cr = self._get_parent_tuning_cr()
        return cr.get("spec", {}) if cr else None

    def _build_benchmark_suite(self, trial_config: TrialConfig) -> List[Dict[str, Any]]:
        """Construct Benchmark CR suite from ModelTuning CR or trial_config.benchmark_config."""
        parent_suite = self._get_parent_tuning_suite()
        if parent_suite:
            logger.info("Using testSuite inherited from ModelTuning CR %s", self.tuning_name)
            return parent_suite

        bc = trial_config.benchmark_config
        if not bc:
            return [
                {
                    "name": "perf-eval",
                    "type": "concurrency",
                    "timeoutSeconds": 3600,
                    "params": {
                        "concurrency": 8,
                        "requestCount": 100,
                        "streaming": True,
                        "ignoreEOS": True,
                    },
                }
            ]

        # Determine benchmark item parameters
        concurrency = bc.rate or bc.concurrency or 8
        request_count = bc.samples or 100
        timeout_seconds = max(bc.max_seconds or 3600, 3600)

        params: Dict[str, Any] = {
            "streaming": True,
            "ignoreEOS": True,
        }

        # Input & output token distribution
        if bc.prompt_tokens:
            params["isl"] = {
                "mean": int(bc.prompt_tokens),
                "stddev": int(bc.prompt_tokens_stdev or 0),
            }
        if bc.output_tokens:
            params["osl"] = {
                "mean": int(bc.output_tokens),
                "stddev": int(bc.output_tokens_stdev or 0),
            }

        # Dataset replay vs request-rate vs fixed concurrency
        if bc.dataset and bc.dataset.lower() in ("sharegpt", "sonnet"):
            suite_type = "dataset-replay"
            params["dataset"] = bc.dataset.lower()
            params["concurrency"] = int(concurrency)
            params["requestCount"] = int(request_count)
        elif bc.request_rate and str(bc.request_rate).lower() != "inf":
            suite_type = "request-rate"
            params["requestRate"] = str(bc.request_rate)
            params["concurrency"] = int(concurrency)
            params["requestCount"] = int(request_count)
        elif concurrency == 1:
            suite_type = "latency"
            params["requestCount"] = int(request_count)
        else:
            suite_type = "concurrency"
            params["concurrency"] = int(concurrency)
            params["requestCount"] = int(request_count)

        return [
            {
                "name": "perf-eval",
                "type": suite_type,
                "timeoutSeconds": int(timeout_seconds),
                "params": params,
            }
        ]

    def _get_experiment_serving_name(self) -> str:
        """Get the deterministic experiment serving name."""
        return self.serving_name

    def ensure_serving_exists(self, trial_config: Optional[TrialConfig] = None) -> None:
        """Ensure experiment ModelServing CR exists in Kubernetes; create it if not present."""
        serving_name = self.serving_name
        try:
            existing = self.custom_api.get_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_SERVINGS,
                name=serving_name,
            )
            labels = existing.get("metadata", {}).get("labels", {})
            if (
                labels.get("app.kubernetes.io/managed-by") in ("auto-tuning-vllm", "auto-tune-serving")
                or (self.tuning_name and labels.get(LABEL_TUNING) == self.tuning_name)
            ):
                self._serving_created_by_us = True
            logger.info("Target ModelServing %s already exists in %s (managed_by_us=%s)", serving_name, self.namespace, self._serving_created_by_us)
            return
        except Exception as e:
            from kubernetes.client.exceptions import ApiException

            is_not_found = (
                (isinstance(e, ApiException) and e.status == 404)
                or "NotFound" in str(e)
                or "404" in str(e)
            )
            if not is_not_found:
                logger.warning("Error checking ModelServing %s: %s", serving_name, e)
            logger.info(
                "ModelServing %s not found in namespace %s. Generating and creating it...",
                serving_name,
                self.namespace,
            )

        # Build initial ModelServing spec
        serving_spec: Dict[str, Any] = {}

        # 1. From provided serving_template (dict)
        if self.serving_template:
            if "spec" in self.serving_template:
                serving_spec = copy.deepcopy(self.serving_template["spec"])
            else:
                serving_spec = copy.deepcopy(self.serving_template)
            logger.info("Using provided serving_template for %s", serving_name)
        # 2. Inherit from parent ModelTuning CR spec.servingTemplate
        elif self.tuning_name:
            parent_spec = self._get_parent_tuning_spec()
            if parent_spec and "servingTemplate" in parent_spec:
                serving_spec = copy.deepcopy(parent_spec["servingTemplate"])
                logger.info("Inherited servingTemplate from parent ModelTuning %s", self.tuning_name)

        # 3. Default fallback spec
        if not serving_spec:
            logger.info("Building default baseline ModelServing spec for %s", serving_name)
            model_name = "facebook-opt-125m"
            served_name = "opt-125m"
            if trial_config and trial_config.benchmark_config and trial_config.benchmark_config.model:
                raw_model = trial_config.benchmark_config.model
                served_name = raw_model.split("/")[-1]
                model_name = raw_model.replace("/", "-")

            serving_spec = {
                "image": "docker.io/vllm/vllm-openai:v0.27.1",
                "engine": {
                    "servicePort": 8000,
                    "type": "vllm",
                },
                "modelType": "llm",
                "servedModelName": served_name,
                "command": [
                    "vllm",
                    "serve",
                    "${MODEL_PATH}",
                    "--served-model-name=${SERVED_MODEL_NAME}",
                    "--port=${SERVICE_PORT}",
                    "--enforce-eager",
                    "--chat-template=/vllm-workspace/examples/template_chatml.jinja",
                ],
                "model": {
                    "valueFrom": {
                        "kind": "ClusterModelRepository",
                        "name": model_name,
                    }
                },
                "replicas": 1,
                "externalAccess": False,
                "cacheAwareRouting": False,
                "gpuClaim": {
                    "className": "nvidia-gb10",
                    "requests": {
                        "memoryMB": 16384,
                    },
                },
                "parallelism": {
                    "tp": 1,
                },
                "contextLength": "2048",
                "batchSize": "32",
                "gpuMemoryUtilization": "0.8",
            }

        if "image" not in serving_spec:
            serving_spec["image"] = "docker.io/vllm/vllm-openai:v0.27.1"

        if "command" not in serving_spec and "extraCommand" not in serving_spec:
            serving_spec["command"] = [
                "vllm",
                "serve",
                "${MODEL_PATH}",
                "--served-model-name=${SERVED_MODEL_NAME}",
                "--port=${SERVICE_PORT}",
                "--enforce-eager",
                "--chat-template=/vllm-workspace/examples/template_chatml.jinja",
            ]

        # Apply candidate parameters from trial 0 if present
        if trial_config and trial_config.parameters:
            initial_patch = self._map_parameters_to_serving_patch(trial_config.parameters)
            _deep_update(serving_spec, initial_patch)

        # Ensure extraCommand has chat template if extraCommand is used and command doesn't have it
        if "command" not in serving_spec:
            if "extraCommand" not in serving_spec:
                serving_spec["extraCommand"] = ["--chat-template=/vllm-workspace/examples/template_chatml.jinja"]
            elif not any("--chat-template" in cmd for cmd in serving_spec["extraCommand"]):
                serving_spec["extraCommand"].append("--chat-template=/vllm-workspace/examples/template_chatml.jinja")

        body = {
            "apiVersion": f"{AFSBOX_GROUP}/{AFSBOX_VERSION}",
            "kind": "ModelServing",
            "metadata": {
                "name": serving_name,
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/managed-by": "auto-tune-serving",
                },
            },
            "spec": serving_spec,
        }
        if self.tuning_name:
            body["metadata"]["labels"][LABEL_TUNING] = self.tuning_name

        try:
            self.custom_api.create_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_SERVINGS,
                body=body,
            )
            self._serving_created_by_us = True
            self._serving_just_created = True
            logger.info("Successfully created ModelServing %s in %s", serving_name, self.namespace)
        except Exception as e:
            if "AlreadyExists" in str(e):
                logger.info("ModelServing %s already exists (concurrent creation)", serving_name)
            else:
                logger.error("Failed to create ModelServing %s: %s", serving_name, e)
                raise RuntimeError(f"AFSBox ModelServing creation failed: {e}")

    def _map_parameters_to_serving_patch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map Optuna parameter dictionary to AFSBox ModelServing spec patch using engine adapter."""
        return self.engine_adapter.map_to_serving_patch(params)

    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit a trial to AFSBox by updating ModelServing and triggering a Benchmark CR."""
        serving_name = self._get_experiment_serving_name()
        trial_id = trial_config.trial_id
        trial_num = trial_config.trial_number if trial_config.trial_number is not None else 0
        bench_name = f"{self.tuning_name or 'optuna'}-c{trial_num}"

        logger.info(
            "Submitting trial %s (trial #%d) to AFSBox ModelServing %s",
            trial_id,
            trial_num,
            serving_name,
        )

        exec_info = ExecutionInfo()
        exec_info.mark_vllm_started()

        # 0. Ensure experiment ModelServing exists (auto-create if missing)
        self.ensure_serving_exists(trial_config)

        # 1. Update ModelServing with candidate parameters
        if self._serving_just_created:
            target_gen = 1
            self._serving_just_created = False
            logger.info(
                "ModelServing %s was created for trial %s (target generation: %s)",
                serving_name,
                trial_id,
                target_gen,
            )
        else:
            spec_patch = self._map_parameters_to_serving_patch(trial_config.parameters)
            try:
                # Patch experiment ModelServing spec
                body = {"spec": spec_patch}
                patched_obj = self.custom_api.patch_namespaced_custom_object(
                    group=AFSBOX_GROUP,
                    version=AFSBOX_VERSION,
                    namespace=self.namespace,
                    plural=PLURAL_SERVINGS,
                    name=serving_name,
                    body=body,
                )
                target_gen = patched_obj.get("metadata", {}).get("generation", 0)
                logger.info(
                    "Patched ModelServing %s with spec: %s (target generation: %s)",
                    serving_name,
                    spec_patch,
                    target_gen,
                )
            except Exception as e:
                logger.error("Failed to patch ModelServing %s: %s", serving_name, e)
                raise RuntimeError(f"AFSBox ModelServing patch failed: {e}")

        # 2. Wait for ModelServing to become Ready
        time.sleep(2)
        start_wait = time.time()
        is_ready = False
        while time.time() - start_wait < self.deploy_timeout_seconds:
            try:
                serving_obj = self.custom_api.get_namespaced_custom_object(
                    group=AFSBOX_GROUP,
                    version=AFSBOX_VERSION,
                    namespace=self.namespace,
                    plural=PLURAL_SERVINGS,
                    name=serving_name,
                )
                status = serving_obj.get("status", {})
                phase = status.get("phase", "")
                observed_gen = status.get("observedGeneration", 0)
                if phase == "Ready" and observed_gen >= target_gen:
                    is_ready = True
                    break
                elif phase == "Failed":
                    conditions = status.get("conditions", [])
                    err_msg = status.get("message") or (conditions[0].get("message") if conditions else "Unknown error")
                    logger.error("ModelServing %s entered Failed phase: %s", serving_name, err_msg)
                    raise RuntimeError(f"ModelServing {serving_name} failed: {err_msg}")
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning("Checking ModelServing %s status warning: %s", serving_name, e)

            time.sleep(self.poll_interval_seconds)

        if not is_ready:
            raise TimeoutError(
                f"Timed out waiting {self.deploy_timeout_seconds}s for ModelServing {serving_name} to become Ready"
            )

        exec_info.mark_vllm_ready()
        logger.info("ModelServing %s is Ready. Launching Benchmark CR %s...", serving_name, bench_name)

        # 3. Create Benchmark CR to initiate AIPerf load test
        exec_info.mark_benchmark_started()
        suite = self._build_benchmark_suite(trial_config)
        endpoint_model = None

        # Extract the actual servedModelName from ready ModelServing status if available
        actual_served_model = None
        if serving_obj:
            output_dict = serving_obj.get("status", {}).get("output", {})
            if isinstance(output_dict, dict):
                smn = output_dict.get("servedModelName")
                if isinstance(smn, dict) and smn.get("value"):
                    actual_served_model = smn.get("value")
                elif isinstance(smn, str):
                    actual_served_model = smn

        if actual_served_model:
            endpoint_model = actual_served_model
            logger.info("Using actual servedModelName from ModelServing output: %s", endpoint_model)
        elif self.serving_template and self.serving_template.get("servedModelName"):
            endpoint_model = self.serving_template.get("servedModelName")
        elif trial_config.benchmark_config and trial_config.benchmark_config.model:
            endpoint_model = trial_config.benchmark_config.model
        else:
            endpoint_model = "opt-125m"

        if trial_config.benchmark_config and endpoint_model:
            trial_config.benchmark_config.model = endpoint_model

        service_port = self.engine_adapter.default_port
        parent_spec = self._get_parent_tuning_spec()
        if self.serving_template and isinstance(self.serving_template, dict):
            service_port = self.serving_template.get("engine", {}).get("servicePort", service_port)
        elif parent_spec and isinstance(parent_spec, dict):
            service_port = parent_spec.get("servingTemplate", {}).get("engine", {}).get("servicePort", service_port)

        endpoint_dict = {
            "url": f"http://{serving_name}.{self.namespace}.svc.cluster.local:{service_port}",
        }
        if endpoint_model:
            endpoint_dict["modelName"] = endpoint_model

        if parent_spec and isinstance(parent_spec, dict):
            parent_endpoint = parent_spec.get("endpoint") or {}
            if "apiKeySecretRef" in parent_endpoint:
                endpoint_dict["apiKeySecretRef"] = parent_endpoint["apiKeySecretRef"]

        bench_body = {
            "apiVersion": f"{AFSBOX_GROUP}/{AFSBOX_VERSION}",
            "kind": "Benchmark",
            "metadata": {
                "name": bench_name,
                "namespace": self.namespace,
                "labels": {
                    LABEL_SERVING: serving_name,
                    LABEL_CANDIDATE: trial_id,
                },
            },
            "spec": {
                "displayName": f"Optuna / {trial_id}",
                "target": {
                    "modelServingRef": {"name": serving_name},
                    "endpoint": endpoint_dict,
                },
                "suite": suite,
            },
        }

        if self.tuning_name:
            bench_body["metadata"]["labels"][LABEL_TUNING] = self.tuning_name

        try:
            self.custom_api.create_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_BENCHMARKS,
                body=bench_body,
            )
            logger.info("Created Benchmark CR %s", bench_name)
        except Exception as e:
            # If already exists from previous attempt, ignore conflict
            if "AlreadyExists" not in str(e):
                logger.error("Failed to create Benchmark CR %s: %s", bench_name, e)
                raise RuntimeError(f"AFSBox Benchmark CR creation failed: {e}")

        handle = JobHandle(trial_id=trial_id, backend_job_id=bench_name, status="running")
        self.active_trials[trial_id] = {
            "handle": handle,
            "bench_name": bench_name,
            "exec_info": exec_info,
            "trial_config": trial_config,
        }

        # Update candidate status to Testing with parameters
        candidate_vars = dict(trial_config.parameters) if trial_config.parameters else {}
        if not candidate_vars and trial_config.trial_type == "baseline" and self.serving_template:
            if "batchSize" in self.serving_template:
                try:
                    candidate_vars["batch_size"] = int(self.serving_template["batchSize"])
                except (ValueError, TypeError):
                    pass
            if "gpuMemoryUtilization" in self.serving_template:
                try:
                    candidate_vars["gpu_memory_utilization"] = float(self.serving_template["gpuMemoryUtilization"])
                except (ValueError, TypeError):
                    pass
        self._sync_candidate_status_to_tuning(
            candidate_name=trial_id,
            bench_name=bench_name,
            phase="Testing",
            variables=candidate_vars,
        )

        return handle

    def poll_trials(
        self, job_handles: List[JobHandle]
    ) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll active Benchmark CRs and collect results."""
        completed_results: List[TrialResult] = []
        remaining_handles: List[JobHandle] = []

        for handle in job_handles:
            trial_id = handle.trial_id
            trial_info = self.active_trials.get(trial_id)
            if not trial_info:
                continue

            bench_name = trial_info["bench_name"]
            exec_info: ExecutionInfo = trial_info["exec_info"]
            trial_config: TrialConfig = trial_info["trial_config"]

            try:
                bench_obj = self.custom_api.get_namespaced_custom_object(
                    group=AFSBOX_GROUP,
                    version=AFSBOX_VERSION,
                    namespace=self.namespace,
                    plural=PLURAL_BENCHMARKS,
                    name=bench_name,
                )
                status = bench_obj.get("status", {})
                phase = status.get("phase", "")

                if phase == "Completed":
                    exec_info.mark_benchmark_completed()
                    exec_info.mark_completed("success")

                    # Extract metrics from results or BenchmarkReport
                    results_list = status.get("results", [])
                    detailed_metrics = {}
                    if results_list and "metrics" in results_list[0]:
                        detailed_metrics = results_list[0]["metrics"]

                    report_ref = status.get("reportRef", {}).get("name")
                    if not detailed_metrics and report_ref:
                        try:
                            report_obj = self.custom_api.get_namespaced_custom_object(
                                group=AFSBOX_GROUP,
                                version=AFSBOX_VERSION,
                                namespace=self.namespace,
                                plural="benchmarkreports",
                                name=report_ref,
                            )
                            items = report_obj.get("spec", {}).get("items", [])
                            if items and "metrics" in items[0]:
                                detailed_metrics = items[0]["metrics"]
                        except Exception as err:
                            logger.warning("Failed to fetch BenchmarkReport %s: %s", report_ref, err)

                    # Extract objective values
                    objective_values = self._extract_objectives(
                        detailed_metrics, trial_config.optimization_config
                    )

                    # Format metrics dictionary for CR status (all values formatted as strings)
                    metrics_str_map: Dict[str, str] = {}
                    for k, v in detailed_metrics.items():
                        if isinstance(v, float):
                            metrics_str_map[k] = f"{v:.4f}"
                        elif isinstance(v, (int, str)):
                            metrics_str_map[k] = str(v)

                    # Extract candidate parameters/variables
                    candidate_vars = dict(trial_config.parameters) if trial_config.parameters else {}
                    if not candidate_vars and trial_config.trial_type == "baseline" and self.serving_template:
                        if "batchSize" in self.serving_template:
                            try:
                                candidate_vars["batch_size"] = int(self.serving_template["batchSize"])
                            except (ValueError, TypeError):
                                pass
                        if "gpuMemoryUtilization" in self.serving_template:
                            try:
                                candidate_vars["gpu_memory_utilization"] = float(self.serving_template["gpuMemoryUtilization"])
                            except (ValueError, TypeError):
                                pass

                    result = TrialResult(
                        trial_id=trial_id,
                        trial_number=trial_config.trial_number,
                        trial_type=trial_config.trial_type,
                        objective_values=objective_values,
                        detailed_metrics=detailed_metrics,
                        execution_info=exec_info,
                        success=True,
                    )
                    completed_results.append(result)
                    self._sync_candidate_status_to_tuning(
                        candidate_name=trial_id,
                        bench_name=bench_name,
                        phase="Completed",
                        variables=candidate_vars,
                        metrics=metrics_str_map,
                    )
                    logger.info("Trial %s Benchmark %s completed: %s", trial_id, bench_name, objective_values)

                elif phase == "Failed":
                    exec_info.mark_completed("failed")
                    err_msg = status.get("message", "Benchmark failed")
                    candidate_vars = dict(trial_config.parameters) if trial_config.parameters else {}
                    result = TrialResult(
                        trial_id=trial_id,
                        trial_number=trial_config.trial_number,
                        trial_type=trial_config.trial_type,
                        objective_values=[],
                        execution_info=exec_info,
                        success=False,
                        error_message=err_msg,
                    )
                    completed_results.append(result)
                    self._sync_candidate_status_to_tuning(
                        candidate_name=trial_id,
                        bench_name=bench_name,
                        phase="Failed",
                        message=err_msg,
                        variables=candidate_vars,
                    )
                    logger.warning("Trial %s Benchmark %s failed: %s", trial_id, bench_name, err_msg)

                else:
                    remaining_handles.append(handle)

            except Exception as e:
                logger.error("Error polling Benchmark %s: %s", bench_name, e)
                remaining_handles.append(handle)

        return completed_results, remaining_handles

    def _sync_candidate_status_to_tuning(
        self,
        candidate_name: str,
        bench_name: str,
        phase: str,
        message: str = "",
        variables: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, str]] = None,
    ):
        """Sync trial candidate progress back to parent ModelTuning.status.candidates."""
        if not self.tuning_name:
            return
        try:
            tuning_obj = self.custom_api.get_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_TUNINGS,
                name=self.tuning_name,
            )
            status = tuning_obj.get("status", {})
            candidates = status.get("candidates", [])

            found = False
            for c in candidates:
                if c.get("name") == candidate_name:
                    c["phase"] = phase
                    c["benchmarkRef"] = bench_name
                    if message:
                        c["message"] = message
                    if variables:
                        c["variables"] = variables
                    if metrics:
                        c["metrics"] = metrics
                    found = True
                    break
            if not found:
                item = {
                    "name": candidate_name,
                    "benchmarkRef": bench_name,
                    "phase": phase,
                    "message": message,
                }
                if variables:
                    item["variables"] = variables
                if metrics:
                    item["metrics"] = metrics
                candidates.append(item)

            status["candidates"] = candidates
            status["currentCandidate"] = candidate_name if phase not in ("Completed", "Failed") else ""
            self.custom_api.patch_namespaced_custom_object_status(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_TUNINGS,
                name=self.tuning_name,
                body={"status": status},
            )
        except Exception as e:
            logger.debug("Failed to sync candidate status to ModelTuning %s: %s", self.tuning_name, e)

    def _extract_objectives(
        self, metrics: Dict[str, Any], optimization_config: Any
    ) -> List[float]:
        """Extract objective values based on configured optimization targets."""
        def _parse_val(v: Any) -> float:
            if v is None:
                return 0.0
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            if not s:
                return 0.0
            if s.endswith("ms"):
                try:
                    return float(s[:-2])
                except ValueError:
                    pass
            elif s.endswith("m"):  # milli (e.g. 1234900m -> 1234.9)
                try:
                    return float(s[:-1]) / 1000.0
                except ValueError:
                    pass
            elif s.endswith("k") or s.endswith("K"):
                try:
                    return float(s[:-1]) * 1000.0
                except ValueError:
                    pass
            elif s.endswith("M"):
                try:
                    return float(s[:-1]) * 1000000.0
                except ValueError:
                    pass
            elif s.endswith("u"):
                try:
                    return float(s[:-1]) / 1000000.0
                except ValueError:
                    pass
            elif s.endswith("n"):
                try:
                    return float(s[:-1]) / 1000000000.0
                except ValueError:
                    pass
            try:
                return float(s)
            except ValueError:
                logger.warning("Could not parse metric value to float: %r", v)
                return 0.0

        values: List[float] = []

        if not optimization_config or not hasattr(optimization_config, "objectives"):
            # Default fallback: maximize throughput
            tps = metrics.get("output_tokens_per_sec_per_user") or metrics.get("output_token_throughput") or 0.0
            return [_parse_val(tps)]

        for obj in optimization_config.objectives:
            m_name = obj.metric.lower()
            if "token" in m_name and "second" in m_name:
                v = (
                    metrics.get("outputTokensPerSec")
                    or metrics.get("outputTokensPerSecPerUser")
                    or metrics.get("output_tokens_per_sec_per_user")
                    or metrics.get("output_token_throughput")
                    or 0.0
                )
                values.append(_parse_val(v))
            elif "first_token" in m_name or "ttft" in m_name:
                percentile = getattr(obj, "percentile", "p50").lower()
                ttft_data = metrics.get("ttft")
                if isinstance(ttft_data, dict):
                    v = ttft_data.get(percentile) or ttft_data.get("p50") or ttft_data.get("avg") or 0.0
                else:
                    v = metrics.get(f"ttft_{percentile}") or metrics.get("ttft_p50") or 0.0
                values.append(_parse_val(v))
            elif "inter_token" in m_name or "itl" in m_name:
                percentile = getattr(obj, "percentile", "p50").lower()
                itl_data = metrics.get("itl")
                if isinstance(itl_data, dict):
                    v = itl_data.get(percentile) or itl_data.get("p50") or itl_data.get("avg") or 0.0
                else:
                    v = metrics.get(f"itl_{percentile}") or metrics.get("itl_p50") or 0.0
                values.append(_parse_val(v))
            elif "latency" in m_name or "e2e" in m_name:
                percentile = getattr(obj, "percentile", "p50").lower()
                e2e_data = metrics.get("e2e")
                if isinstance(e2e_data, dict):
                    v = e2e_data.get(percentile) or e2e_data.get("p50") or e2e_data.get("avg") or 0.0
                else:
                    v = metrics.get(f"e2e_{percentile}") or metrics.get("e2e_p50") or 0.0
                values.append(_parse_val(v))
            else:
                v = metrics.get(obj.metric, 0.0)
                values.append(_parse_val(v))

        return values

    def sync_final_results_to_tuning(self, results: Dict[str, Any]):
        """Sync final best candidate and pareto frontier to ModelTuning.status."""
        if not self.tuning_name:
            return
        try:
            tuning_obj = self.custom_api.get_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_TUNINGS,
                name=self.tuning_name,
            )
            status = tuning_obj.get("status", {})
            if results.get("type") == "multi_objective":
                pareto_front = results.get("pareto_front", [])
                pareto_candidates = []
                for p in pareto_front:
                    trial_num = p.get("trial")
                    if trial_num is not None:
                        pareto_candidates.append(f"trial_{trial_num}")
                status["paretoFrontier"] = pareto_candidates
                # 用加權分數挑代表解，不是「前沿清單第一筆」——後者只是
                # Optuna study.best_trials 回傳的原始順序（實測就是接近
                # trial 執行的時間先後），不是任何排名，容易把「剛好先跑
                # 完又落在前沿上」的解錯認成「最好」，見
                # _select_best_pareto_candidate 檔頭完整說明。
                best = _select_best_pareto_candidate(pareto_front, results.get("objectives"))
                if best:
                    status["bestCandidate"] = best
            else:
                best_num = results.get("best_trial_number")
                if best_num is not None:
                    status["bestCandidate"] = f"trial_{best_num}"

            status["currentCandidate"] = ""
            self.custom_api.patch_namespaced_custom_object_status(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_TUNINGS,
                name=self.tuning_name,
                body={"status": status},
            )
            logger.info(
                "Synced final results to ModelTuning %s status: best=%s, pareto=%s",
                self.tuning_name,
                status.get("bestCandidate"),
                status.get("paretoFrontier"),
            )
        except Exception as e:
            logger.warning("Failed to sync final results to ModelTuning %s: %s", self.tuning_name, e)

    def shutdown(self):
        """Clean shutdown of backend resources."""
        if self.cleanup_serving and (self._serving_created_by_us or self.tuning_name):
            self._delete_serving()
        logger.info("AFSBoxK8sBackend shutdown completed.")

    def cleanup_all_trials(self):
        """Clean up active benchmarks and experiment serving if configured."""
        logger.info("Cleaning up active trials for AFSBox backend.")
        self.active_trials.clear()
        if self.cleanup_serving and (self._serving_created_by_us or self.tuning_name):
            self._delete_serving()

    def _delete_serving(self):
        """Delete experiment ModelServing CR if created by this backend."""
        serving_name = self._get_experiment_serving_name()
        try:
            logger.info("Deleting ModelServing %s in namespace %s...", serving_name, self.namespace)
            self.custom_api.delete_namespaced_custom_object(
                group=AFSBOX_GROUP,
                version=AFSBOX_VERSION,
                namespace=self.namespace,
                plural=PLURAL_SERVINGS,
                name=serving_name,
            )
            self._serving_created_by_us = False
            logger.info("Deleted ModelServing %s", serving_name)
        except Exception as e:
            if "NotFound" not in str(e):
                logger.warning("Failed to delete ModelServing %s: %s", serving_name, e)


def synthesize_study_config_from_cr(tuning_name: str, namespace: str = "default") -> str:
    """Synthesize a study configuration YAML file from parent ModelTuning CR."""
    import tempfile
    import yaml
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

    custom_api = client.CustomObjectsApi()
    tuning_obj = custom_api.get_namespaced_custom_object(
        group=AFSBOX_GROUP,
        version=AFSBOX_VERSION,
        namespace=namespace,
        plural=PLURAL_TUNINGS,
        name=tuning_name,
    )
    spec = tuning_obj.get("spec", {})
    opt_spec = spec.get("optimization", {}) or {}
    test_suite = spec.get("testSuite", [])
    serving_template = spec.get("servingTemplate", {})

    objectives = []
    for obj in opt_spec.get("objectives", []):
        objectives.append({
            "metric": obj.get("metric", "output_tokens_per_second"),
            "direction": obj.get("direction", "maximize"),
            "percentile": obj.get("percentile", "p50"),
        })
    if not objectives:
        objectives = [
            {"metric": "output_tokens_per_second", "direction": "maximize"},
            {"metric": "time_to_first_token_ms", "direction": "minimize"},
        ]

    # Extract servedModelName from existing ModelServing output or servingTemplate
    served_model_name = None
    try:
        existing_serving = custom_api.get_namespaced_custom_object(
            group=AFSBOX_GROUP,
            version=AFSBOX_VERSION,
            namespace=namespace,
            plural=PLURAL_SERVINGS,
            name=f"{tuning_name}-serving",
        )
        smn = existing_serving.get("status", {}).get("output", {}).get("servedModelName", {})
        if isinstance(smn, dict) and smn.get("value"):
            served_model_name = smn.get("value")
        elif isinstance(smn, str):
            served_model_name = smn
    except Exception:
        pass

    if not served_model_name:
        served_model_name = (
            serving_template.get("servedModelName")
            or serving_template.get("model", {}).get("valueFrom", {}).get("name")
            or "default"
        )

    suite_params = test_suite[0].get("params", {}) if test_suite else {}
    isl = suite_params.get("isl", {})
    osl = suite_params.get("osl", {})
    bench_dict = {
        "benchmark_type": "aiperf",
        "model": served_model_name,
        "samples": suite_params.get("requestCount", 10),
        "rate": suite_params.get("concurrency", 4),
        "prompt_tokens": isl.get("mean", 64) if isinstance(isl, dict) else 64,
        "output_tokens": osl.get("mean", 32) if isinstance(osl, dict) else 32,
        "dataset": suite_params.get("dataset"),
        "max_seconds": max(test_suite[0].get("timeoutSeconds", 3600), 3600) if test_suite else 3600,
    }

    # Detect engine type and get adapter
    engine_type = serving_template.get("engine", {}).get("type") if isinstance(serving_template, dict) else None
    engine_adapter = get_engine_adapter(engine_type)
    logger.info("Synthesizing study config using engine adapter: %s", engine_adapter.name)

    # Default parameters tailored to the engine
    params_dict = engine_adapter.get_default_parameter_space()

    # Dynamically extract parameters from BenchmarkTemplate sweeps if referenced
    template_name = tuning_obj.get("metadata", {}).get("labels", {}).get("platform.afsbox.asus.com/template")
    if template_name:
        for tpl_ns in [namespace, "afsbox-system", "default"]:
            try:
                tpl_obj = custom_api.get_namespaced_custom_object(
                    group="platform.afsbox.asus.com",
                    version="v1beta1",
                    namespace=tpl_ns,
                    plural="benchmarktemplates",
                    name=template_name,
                )
                sweeps = tpl_obj.get("spec", {}).get("sweeps", [])
                if sweeps:
                    INT_PARAMS = {
                        "batch_size",
                        "max_num_batched_tokens",
                        "context_length",
                        "tensor_parallel_size",
                        "pipeline_parallel_size",
                        "data_parallel_size",
                        "replicas",
                        "max_num_seqs",
                        "max_model_len",
                        "max_batch_tokens",
                        "chunked_prefill_size",
                    }
                    extracted_params = {}
                    for sw in sweeps:
                        var_name = sw.get("variable", "")
                        var_lower = var_name.lower().replace("-", "_")
                        if var_lower in ("batchsize", "batch_size", "max_num_seqs"):
                            param_name = "batch_size"
                        elif var_lower in ("gpumemoryutilization", "gpu_memory_utilization", "gpu_mem_util"):
                            param_name = "gpu_memory_utilization"
                        elif var_lower in ("prefillsettings.maxbatchtokens", "max_num_batched_tokens", "max_batch_tokens", "maxbatchtokens"):
                            param_name = "max_num_batched_tokens"
                        elif var_lower in ("contextlength", "context_length", "max_model_len"):
                            param_name = "context_length"
                        elif var_lower in ("parallelism.tp", "tp", "tensor_parallel_size"):
                            param_name = "tensor_parallel_size"
                        elif var_lower in ("kvcachedtype", "kv_cache_dtype"):
                            param_name = "kv_cache_dtype"
                        else:
                            param_name = var_name.replace(".", "_")

                        kind = sw.get("kind", "values")
                        if kind == "values":
                            raw_vals = sw.get("values", [])
                            conv_vals = []
                            for v in raw_vals:
                                try:
                                    if param_name in INT_PARAMS:
                                        conv_vals.append(int(float(v)))
                                    elif "." in str(v):
                                        conv_vals.append(float(v))
                                    else:
                                        conv_vals.append(int(v))
                                except ValueError:
                                    conv_vals.append(str(v))
                            extracted_params[param_name] = {
                                "enabled": True,
                                "options": conv_vals,
                            }
                        elif kind == "range":
                            min_val = sw.get("min", 0)
                            max_val = sw.get("max", 1)
                            step_val = sw.get("step")
                            is_int_param = (
                                param_name in INT_PARAMS
                                or (
                                    isinstance(min_val, int)
                                    and isinstance(max_val, int)
                                    and (step_val is None or isinstance(step_val, int))
                                )
                            )
                            if is_int_param:
                                r_conf = {
                                    "enabled": True,
                                    "min": int(float(min_val)),
                                    "max": int(float(max_val)),
                                }
                                if step_val is not None:
                                    r_conf["step"] = int(float(step_val))
                            else:
                                r_conf = {
                                    "enabled": True,
                                    "min": float(min_val),
                                    "max": float(max_val),
                                }
                                if step_val is not None:
                                    r_conf["step"] = float(step_val)
                            extracted_params[param_name] = r_conf
                    if extracted_params:
                        logger.info(
                            "Extracted %d parameters from BenchmarkTemplate %s sweeps: %s",
                            len(extracted_params),
                            template_name,
                            list(extracted_params.keys()),
                        )
                        params_dict = extracted_params
                break
            except Exception as e:
                logger.debug("Attempt to fetch BenchmarkTemplate %s in ns %s: %s", template_name, tpl_ns, e)

    n_trials = opt_spec.get("nTrials", 20)
    n_startup = max(1, min(5, n_trials - 1)) if n_trials > 1 else 1

    baseline_params = engine_adapter.extract_baseline_parameters(serving_template)

    probe = serving_template.get("probe", {}) if isinstance(serving_template, dict) else {}
    startup_mins = probe.get("startupTimeoutMinutes") if isinstance(probe, dict) else None
    deploy_timeout = int(startup_mins * 60) if startup_mins else 1800
    import os
    env_timeout = os.getenv("AFSBOX_DEPLOY_TIMEOUT_SECONDS") or os.getenv("DEPLOY_TIMEOUT_SECONDS")
    if env_timeout:
        try:
            deploy_timeout = int(env_timeout)
        except ValueError:
            pass

    study_dict = {
        "study": {
            "name": tuning_name,
        },
        "engine": engine_adapter.name,
        "backend": "afsbox",
        "afsbox": {
            "namespace": namespace,
            "tuning_name": tuning_name,
            "serving_name": f"{tuning_name}-exp",
            "serving_template": serving_template,
            "cleanup_serving": True,
            "deploy_timeout_seconds": deploy_timeout,
            "poll_interval_seconds": 5,
        },
        "baseline": {
            "enabled": True,
            "concurrency_levels": [bench_dict["rate"]],
            "parameters": baseline_params,
        },
        "optimization": {
            "approach": "multi_objective" if len(objectives) > 1 else "single_objective",
            "objectives": objectives,
            "sampler": opt_spec.get("sampler", "tpe"),
            "n_trials": n_trials,
            "n_startup_trials": n_startup,
            "max_concurrent_trials": 1,
        },
        "benchmark": bench_dict,
        "parameters": params_dict,
    }

    tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=f"_{tuning_name}.yaml", delete=False)
    yaml.safe_dump(study_dict, tmp_file, sort_keys=False)
    tmp_file.close()
    logger.info("Synthesized study configuration from ModelTuning CR %s: %s", tuning_name, tmp_file.name)
    return tmp_file.name

