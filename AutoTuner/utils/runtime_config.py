from typing import Any, Mapping

DEFAULT_DP_ALLREDUCE_BANDWIDTH_GBPS = 50.0
DEFAULT_DP_ALLREDUCE_LATENCY_US = 30.0


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got {value!r}") from exc


def parse_ddp_simulate_config(
    ddp_simulate_config: Mapping[str, Any] | None,
) -> dict[str, float]:
    raw_config = ddp_simulate_config or {}
    if not isinstance(raw_config, Mapping):
        raise ValueError("ddp_simulate_config must be a JSON object")
    bandwidth_gbps = _coerce_float(
        raw_config.get(
            "dp_allreduce_bandwidth_gbps", DEFAULT_DP_ALLREDUCE_BANDWIDTH_GBPS
        ),
        "ddp_simulate_config.dp_allreduce_bandwidth_gbps",
    )
    latency_us = _coerce_float(
        raw_config.get("dp_allreduce_latency_us", DEFAULT_DP_ALLREDUCE_LATENCY_US),
        "ddp_simulate_config.dp_allreduce_latency_us",
    )
    return {
        "bandwidth_gbps": bandwidth_gbps,
        "latency_s": latency_us / 1e6,
        "latency_us": latency_us,
    }
