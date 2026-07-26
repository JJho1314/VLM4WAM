# OLA Eight-GPU Deep Stress Test Design

Date: 2026-07-26

Status: Approved approach, pending written-spec review

## Objective

Run a bounded one-hour health and stability test across all eight NVIDIA H100
80GB GPUs on the OLA node. The test must exercise device memory, BF16 Tensor
Cores, and NCCL collectives while producing enough telemetry to distinguish a
compute, memory, communication, thermal, or driver failure.

This is a diagnostic run, not an indefinite resource holder. It must restore
the node to its prior state when it exits.

## Execution Model

Launch one PyTorch process per GPU with `torchrun --nproc_per_node=8`. Each
process binds to exactly one local CUDA device. Rank zero coordinates phase
transitions, safety decisions, and the final report.

The one-hour run has three phases:

1. **Memory validation — 10 minutes**
   - Allocate and physically initialize up to 80% of each device's total
     memory while preserving a fixed workspace reserve.
   - Repeatedly write deterministic byte patterns to fixed-size chunks and
     verify sampled regions after synchronization.
   - Fail on an allocation, write, or verification mismatch.
2. **Compute and communication — 40 minutes**
   - Reuse preallocated BF16 matrices for repeated Tensor Core GEMMs.
   - Periodically check results for finite values and stable checksums.
   - Run NCCL all-reduce operations at a fixed interval using a preallocated
     communication buffer.
   - Measure per-rank GEMM throughput and collective bandwidth.
3. **Mixed reliability — 10 minutes**
   - Alternate memory pattern checks, GEMMs, and NCCL collectives.
   - Confirm that every rank remains responsive and that final checksums agree.

Phase deadlines use monotonic time. The total configured duration may not
exceed 3,600 seconds in the OLA launcher.

## Resource Limits

- Use all eight visible GPUs and fail if the world size is not exactly eight.
- Default memory target: 80% of total memory per GPU.
- Hard maximum memory target: 85%.
- Preserve at least 8 GiB per GPU for CUDA context, GEMM, NCCL, and monitoring
  workspaces.
- Do not modify application clocks, persistence mode, power caps, ECC settings,
  or driver configuration.
- Reuse allocated tensors rather than generating unbounded allocations.

Before launch, stop only the dedicated `qwen35_planx_gpu_hold` screen. Do not
touch unrelated screens or processes. If preflight finds any other compute
process on the eight GPUs, abort instead of competing for the devices.

## Safety and Failure Handling

Rank zero samples `nvidia-smi` telemetry throughout the run. The test stops all
ranks if any of these conditions occurs:

- GPU temperature reaches 87 degrees Celsius;
- an unrecoverable CUDA or NCCL exception occurs;
- a NaN or infinity is found in a checked GEMM result;
- a memory pattern check fails;
- an uncorrectable volatile ECC counter increases;
- an NVIDIA XID error appears in kernel logs when log access is available;
- any rank misses a distributed synchronization deadline.

SIGINT and SIGTERM trigger coordinated cleanup. Normal and abnormal exits
destroy the process group and release all GPU allocations.

## Telemetry and Artifacts

Write under a timestamped directory in
`/data/users/junjie/runs/ola_8gpu_stress/`:

- `stress.log`: combined rank output;
- `telemetry.csv`: timestamped per-GPU temperature, power, utilization,
  memory, clocks, and ECC counters;
- `result.json`: configuration, phase durations, per-rank iterations,
  GEMM TFLOPS, NCCL bandwidth, peak temperature, peak power, memory high-water
  mark, error list, and final pass/fail status;
- `nvidia_smi_before.txt` and `nvidia_smi_after.txt`;
- `dmesg_xid_before.txt` and `dmesg_xid_after.txt` when permitted.

The run passes only when all eight ranks complete all phases, no safety
threshold is crossed, no new ECC/XID fault is observed, and every required
artifact is present.

## Testing Strategy

Unit tests cover:

- duration, memory-fraction, temperature, and world-size validation;
- safe ballast allocation calculations and required workspace headroom;
- phase boundary selection from monotonic elapsed time;
- parsing `nvidia-smi` CSV rows;
- safety-stop decisions for temperature, ECC, and non-finite results;
- aggregation of eight rank reports into the final JSON schema.

A short 60-second eight-GPU smoke run precedes the one-hour run. The smoke uses
the same code path with shorter phase durations and a lower memory target. The
deep run starts only if the smoke exits successfully and produces valid
artifacts.
