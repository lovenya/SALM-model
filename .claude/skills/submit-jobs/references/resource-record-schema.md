# Resource Record Schema

Use this shape for generated monitoring/accounting JSON. Store files under ignored `results/`, `logs/`, or scratch output directories; do not commit them.

```json
{
  "schema_version": 1,
  "cluster": "tamia",
  "job_id": "312372",
  "job_name": "salm-full-s1-light",
  "account": "aip-csubakan",
  "partition": "gpubase_b",
  "submitted_at": "2026-06-03T07:35:33Z",
  "state": "COMPLETED|FAILED|CANCELLED|TIMEOUT",
  "exit_code": "0:0",
  "script": "submission_scripts/run_job.sbatch",
  "config": "configs/run.json",
  "input_manifest": "data/splits/ttas_val.jsonl",
  "output_manifest": ".../manifest.jsonl",
  "log_paths": [".../slurm.out", ".../lane.out"],
  "environment": {
    "modules": ["python/3.10", "cuda/12.6", "arrow/17.0.0"],
    "venv": "/path/to/venv",
    "python": "3.10.x",
    "cuda_visible_devices": "0"
  },
  "requested": {
    "nodes": 1,
    "gpus": 4,
    "cpus": 8,
    "memory_gb": 128,
    "walltime_seconds": 21600
  },
  "observed": {
    "elapsed_seconds": 4644,
    "max_rss_mb": 10101,
    "peak_gpu_memory_mb": 16610,
    "avg_gpu_utilization_pct": null,
    "model_load_seconds": 23.735,
    "avg_generation_seconds": 2.032,
    "samples_total": 2277,
    "samples_success": 2258,
    "samples_error": 19
  },
  "recommended_next": {
    "gpus": 4,
    "cpus": 4,
    "memory_gb": 32,
    "walltime_seconds": 7200,
    "basis": "1.3x observed peak plus stage packing constraints"
  }
}
```
