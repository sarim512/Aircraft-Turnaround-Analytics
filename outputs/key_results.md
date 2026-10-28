| Metric | Value |
|---|---|
| Flights analysed | 7,079,061 |
| Turnarounds reconstructed | 5,184,074 (6,097 aircraft, 15 airlines, 344 airports) |
| Median actual / planned ground time | 63 min / 56 min |
| Delay-generating turnarounds | 8.9% (naive 'actual > planned' label: 65.9%) |
| Multi-day gaps a naive clock-modulo method mistakes for turnarounds | 30,404 |
| Best-fitting distribution | Log-normal mixture (k=2) (KS = 0.019) |
| KL divergence, hubs vs regional | 0.84 bits (JS distance 0.45) |
| Median design effect (aircraft clustering) | 9.6× for mean ground time, 2.3× for delay rate |
| ROC-AUC: calendar → schedule → real-time | 0.653 → 0.690 → 0.710 |
| PR-AUC: calendar → schedule → real-time | 0.159 → 0.244 → 0.276 |
| Delays caught only with live inbound data | 6.5% of delay-generating turnarounds |
| Riskiest / safest departure window | Sun 22:00 (13.7%, n=26,328) / Mon 05:00 (4.3%, n=12,746) |
| Highest / lowest risk month | Jul (12.6%) / Oct (5.9%) |
| Highest-risk large airports | BWI, FLL, MCO, MIA, MDW |
