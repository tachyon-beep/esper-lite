"""Corrected S1 (fossilize timing via FOSSILIZE last_action rows, same epoch
semantics) and F5 (all-off regex without mandatory space). Read-only."""

import time

import duckdb

GLOB = "telemetry/causal_r1_n5/control_s4[1-5]/telemetry_*/events.jsonl"

con = duckdb.connect()
con.execute("SET threads TO 8")
t0 = time.time()
con.execute(
    f"""
CREATE TABLE ev AS
SELECT
  regexp_extract(filename, 'control_(s\\d+)', 1) AS run,
  event_type,
  epoch,
  json_extract_string(data, '$.kind') AS kind,
  json_extract(data, '$.env_id')::INTEGER AS env_id,
  json_extract(data, '$.episode_idx')::INTEGER AS episode_idx,
  json_extract_string(data, '$.action_name') AS action_name,
  json_extract(data, '$.action_success')::BOOLEAN AS action_success,
  json_array_length(data, '$.slot_ids') AS n_slots,
  CASE WHEN event_type = 'COUNTERFACTUAL_MATRIX_COMPUTED'
       THEN json_array_length(data, '$.configs') END AS n_configs,
  CASE WHEN event_type = 'COUNTERFACTUAL_MATRIX_COMPUTED'
       THEN regexp_matches(json_extract_string(data, '$.configs'),
                           '\\[(false, ?)*false\\]')
       END AS has_alloff
FROM read_json(
  '{GLOB}',
  format='newline_delimited',
  columns={{'event_type':'VARCHAR','epoch':'BIGINT','data':'JSON'}},
  filename=true, maximum_object_size=33554432)
WHERE event_type = 'COUNTERFACTUAL_MATRIX_COMPUTED'
   OR (event_type = 'ANALYTICS_SNAPSHOT'
       AND json_extract_string(data, '$.kind') = 'last_action')
"""
)
print(f"scan {time.time() - t0:.0f}s")

print("== sanity: per-episode max last_action epoch (quantiles across env-episodes) ==")
print(con.sql("""
WITH mx AS (SELECT run, env_id, episode_idx, MAX(epoch) mx, MIN(epoch) mn, COUNT(*) n
            FROM ev WHERE kind='last_action' GROUP BY 1,2,3)
SELECT MIN(mx) min_max_epoch, quantile_cont(mx,0.5) med_max_epoch, MAX(mx) max_max_epoch,
       MIN(mn) min_min_epoch, ROUND(AVG(n),1) mean_rows_per_ep FROM mx"""))

print("== S1 (fixed): FOSSILIZE-step position within episode (epoch/episode-max) ==")
print(con.sql("""
WITH mx AS (SELECT run, env_id, episode_idx, MAX(epoch) mx
            FROM ev WHERE kind='last_action' GROUP BY 1,2,3),
f AS (
  SELECT e.epoch::DOUBLE / NULLIF(m.mx,0) frac
  FROM ev e JOIN mx m USING (run, env_id, episode_idx)
  WHERE e.kind='last_action' AND e.action_name='FOSSILIZE' AND e.action_success)
SELECT COUNT(*) n_foss,
       ROUND(quantile_cont(frac,0.10),3) p10, ROUND(quantile_cont(frac,0.25),3) p25,
       ROUND(quantile_cont(frac,0.50),3) p50, ROUND(quantile_cont(frac,0.75),3) p75,
       ROUND(quantile_cont(frac,0.90),3) p90,
       COUNT(*) FILTER (frac > 0.9) n_final_decile,
       COUNT(*) FILTER (frac <= 0.5) n_first_half
FROM f"""))

print("== S1 decile histogram ==")
print(con.sql("""
WITH mx AS (SELECT run, env_id, episode_idx, MAX(epoch) mx
            FROM ev WHERE kind='last_action' GROUP BY 1,2,3),
f AS (
  SELECT LEAST(FLOOR(e.epoch::DOUBLE / NULLIF(m.mx,0) * 10), 9)::INTEGER decile
  FROM ev e JOIN mx m USING (run, env_id, episode_idx)
  WHERE e.kind='last_action' AND e.action_name='FOSSILIZE' AND e.action_success)
SELECT decile, COUNT(*) n FROM f GROUP BY 1 ORDER BY 1"""))

print("== F5 (fixed): all-off config presence by matrix arity ==")
print(con.sql("""
SELECT n_slots, n_configs, COUNT(*) n_matrices,
       COUNT(*) FILTER (has_alloff) n_with_alloff,
       ROUND(100.0*COUNT(*) FILTER (has_alloff)/COUNT(*),2) pct_with_alloff
FROM ev WHERE event_type='COUNTERFACTUAL_MATRIX_COMPUTED'
GROUP BY 1,2 ORDER BY 1,2"""))

print("== F5 (fixed): env-episodes with multi-slot matrices lacking any all-off ==")
print(con.sql("""
WITH m AS (
  SELECT run, env_id, episode_idx,
         MAX(CASE WHEN n_slots >= 2 THEN 1 ELSE 0 END) has_multi,
         MAX(CASE WHEN n_slots >= 2 AND has_alloff THEN 1 ELSE 0 END) multi_alloff
  FROM ev WHERE event_type='COUNTERFACTUAL_MATRIX_COMPUTED' GROUP BY 1,2,3)
SELECT SUM(has_multi) n_multi_eps,
       SUM(has_multi) - SUM(multi_alloff) n_no_alloff,
       ROUND(100.0*(SUM(has_multi)-SUM(multi_alloff))/NULLIF(SUM(has_multi),0),2) pct_no_alloff
FROM m"""))

print(f"total {time.time() - t0:.0f}s")
