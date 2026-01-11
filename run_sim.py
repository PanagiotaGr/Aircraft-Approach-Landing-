from sim.metrics import compute_metrics

metrics = compute_metrics(log, config)
print("\n=== Performance Metrics ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.3f}")
    else:
        print(f"{k}: {v}")
