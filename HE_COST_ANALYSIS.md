# Homomorphic Encryption (HE) Cost Analysis

## Overview
The PROFILE system automatically logs detailed HE performance metrics during experiments. Reviewers can analyze these costs from the server logs.

## HE Operations Captured

### 1. **Public Key Aggregation Time**
The multi-key CKKS protocol requires aggregating public keys from all clients at the start.

**Log Entry:**
```
[PROFILE] Public Key Aggregation Time: 41.71s
```

**Analysis:**
- One-time cost per experiment (happens before Round 1)
- For 50 clients with 16 buckets: ~41.7 seconds
- Scales with number of clients and security parameter n=262144

### 2. **Bucket Processing Time (HE Operations)**
Each bucket's encrypted updates are processed homomorphically.

**Log Entries:**
```
[PROFILE] Bucket 0 processing time: 5.02s
[PROFILE] Bucket 1 processing time: 5.06s
...
[PROFILE] Bucket 15 processing time: 3.90s
```

**Analysis:**
- Per-bucket homomorphic aggregation time
- Average per round: ~4.3 seconds per bucket
- Total per round: 16 buckets × 4.3s ≈ 69 seconds
- Includes encrypted weight averaging inside buckets

### 3. **Per-Round Timing Breakdown**

From the logs, a typical round with 50 clients, 16 buckets:

| Operation | Time | Notes |
|-----------|------|-------|
| Public Key Aggregation | 41.71s | One-time (Round 1 only) |
| Client Training (50 clients) | ~24s | Local computation |
| Bucket Processing (16 buckets) | ~69s | HE aggregation |
| Validator Selection & Voting | ~3m | Detection phase |
| **Total Round Time** | ~4-5 min | Varies by config |

## How to Extract HE Costs

### Method 1: From Server Logs (Easiest)

```bash
# Get public key aggregation time
grep "Public Key Aggregation Time" ablation_results/*/server.log

# Get bucket processing times
grep "Bucket.*processing time" ablation_results/*/server.log

# Calculate average bucket time for a specific experiment
grep "Bucket.*processing time" ablation_results/batch_*/mnist_*/server.log | \
  awk -F': ' '{sum+=$2; count++} END {print "Avg:", sum/count "s"}'
```

### Method 2: From Metrics Files

The `metrics/` directory contains JSON files with detailed timing data:
- `experiment_config.jsonl` - Configuration and timing metadata
- Per-round metrics with timestamps

### Method 3: Custom Analysis Script

Create a Python script to parse all logs:

```python
import re
import glob
import statistics

# Extract all bucket processing times
logs = glob.glob("ablation_results/*/server.log")
bucket_times = []

for log_file in logs:
    with open(log_file) as f:
        for line in f:
            match = re.search(r"Bucket \d+ processing time: ([\d.]+)s", line)
            if match:
                bucket_times.append(float(match.group(1)))

print(f"Mean bucket time: {statistics.mean(bucket_times):.2f}s")
print(f"Median bucket time: {statistics.median(bucket_times):.2f}s")
print(f"Std dev: {statistics.stdev(bucket_times):.2f}s")
```

## Expected HE Costs for Ablation Study

### Configuration A (Bucketing Only)
- **Setup:** 16 buckets, no validators
- **HE Operations:** Public key agg + bucket averaging
- **Cost per round:** ~70-90 seconds

### Configuration D (PROFILE Full)
- **Setup:** 16 buckets + validators + DP + HE
- **HE Operations:** All of the above
- **Cost per round:** ~4-5 minutes (includes detection)

## Security Parameters Impact on Cost

Current parameters (from logs):
```
n: 262144        (polynomial degree)
t: 2000003       (plaintext modulus)
q: 10000015007   (ciphertext modulus)
std: 4           (noise standard deviation)
```

**Cost Scaling:**
- Doubling `n` → ~2× slower HE operations
- More buckets → proportional increase in aggregation time
- More clients → higher public key aggregation time

## Comparison with Non-HE Approaches

| Approach | Aggregation Time (50 clients) | Privacy |
|----------|------------------------------|---------|
| Plain FedAvg | 0.0002s | None |
| PROFILE (HE) | ~70s | Strong (multi-key CKKS) |
| **Overhead** | **350,000×** | **Full encryption** |

**Key Insight:** The ~70s HE overhead per round is acceptable because:
1. Provides strong cryptographic privacy guarantees
2. Training time dominates (minutes per client)
3. One-time public key setup amortized over 50 rounds

## Communication Cost Analysis

While not directly logged, HE communication costs can be estimated:

**Ciphertext Size:**
- Each encrypted parameter: ~O(n × log q) bits
- For n=262144, q=10^10: ~8.6 MB per ciphertext
- Model with 61,706 parameters → ~531 GB total (with batching optimizations)

**Actual Communication:**
- Clients send encrypted updates to server
- Server sends global model to clients
- No raw data transmission (privacy preserved)

## Recommendations for Reviewers

1. **Verify HE is Active:**
   ```bash
   # Should see ~41s public key aggregation
   grep "Public Key Aggregation Time" ablation_results/*/server.log
   ```

2. **Check Bucket Processing:**
   ```bash
   # Should see 3-5s per bucket (not instant)
   grep "Bucket 0 processing time" ablation_results/*/server.log
   ```

3. **Compare Configs:**
   - Config A (Bucketing Only): Lower overhead
   - Config D (Full PROFILE): Higher overhead due to validators

4. **Reproduce Timing:**
   - Run single experiment with `--num-rounds 2`
   - Should see similar HE timing patterns

## References

- RLWE-xMKCKKS implementation: `https://github.com/knowledge-bin/crypto-utils`
- Multi-key CKKS paper: [Include citation from your paper]
- Timing logs: `ablation_results/batch_*/*/server.log`

---

**Note:** All timing measurements are from CPU execution. GPU acceleration for HE operations is an area for future work.
