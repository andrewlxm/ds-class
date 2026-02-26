# Measurement of Real FLOPs Using Hardware Performance Counters

To obtain realistic floating-point operation counts rather than theoretical estimates, we implemented a runtime measurement approach based on Linux hardware performance counters. The implementation relies on the perf tool, which interfaces directly with the Performance Monitoring Unit (PMU) of modern CPUs. This method enables counting floating-point instructions actually executed by the processor during model inference.

The measurement code wraps the target computation between start and stop calls of a custom counter class. Internally, the counter launches a background perf stat process attached to the current process ID, recording floating-point arithmetic events while the workload runs.

## Supported Floating-Point Events

Modern CPUs execute vectorized instructions that process multiple floating-point values simultaneously. Therefore, the implementation probes several hardware events corresponding to different SIMD widths:

- Scalar double-precision operations
- 128-bit packed double operations
- 256-bit packed double operations
- 512-bit packed double operations

Each event is multiplied by the number of double-precision values processed per instruction (1, 2, 4, or 8 respectively). This produces a more accurate estimate of the actual number of floating-point operations executed rather than simply counting instructions.

If floating-point events are not available (for example in virtualized environments), the system falls back to counting retired instructions as an approximate proxy.

## Runtime Measurement Procedure

The measurement process follows these steps:
    1. Detect supported floating-point hardware events
    2. Launch perf stat attached to the running process
    3. Execute the target workload (e.g., model prediction)
    4. Stop the counter using a SIGINT signal
    5. Parse the recorded statistics to compute total FLOPs

## Requirements

To ensure correct operation, the following conditions must be satisfied:

- Operating System: Linux
- CPU: Intel processor based on Skylake architecture or newer
- PMU Access: The OS must allow access to Performance Monitoring Unit (PMU) counters

    - Appropriate perf_event_paranoid configuration or root privileges may be required

- Execution: Run the provided deployment script:

```
./deploy.sh
```

Under these conditions, the reported FLOPs reflect actual hardware-level execution rather than theoretical complexity estimates.