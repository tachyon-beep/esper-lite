"""Zero-GPU occurrence probes (pre-registered: gate comment #99, 2026-07-03).

Probes F1, F2a, F5, S1, S2 + coverage/bootstrap proxy over the n=5 control runs
in telemetry/causal_r1_n5/. Read-only. Single scan -> compact table -> queries.
"""

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
  seed_id  AS top_seed_id,
  slot_id  AS top_slot_id,
  epoch,
  json_extract_string(data, '$.kind') AS kind,
  COALESCE(json_extract(data, '$.env_id')::INTEGER,
           json_extract(data, '$.seed_residency.env_id')::INTEGER) AS env_id,
  COALESCE(json_extract(data, '$.episode_idx')::INTEGER,
           json_extract(data, '$.seed_residency.episode_idx')::INTEGER) AS episode_idx,
  -- last_action fields
  json_extract_string(data, '$.action_name') AS action_name,
  json_extract(data, '$.action_success')::BOOLEAN AS action_success,
  json_extract_string(data, '$.slot_id') AS tgt_slot,
  json_extract(data, '$.total_reward')::DOUBLE AS total_reward,
  json_extract(data, '$.reward_components.bounded_attribution')::DOUBLE AS ba,
  json_extract(data, '$.reward_components.seed_contribution')::DOUBLE AS sc,
  json_extract(data, '$.reward_components.num_fossilized_seeds')::INTEGER AS num_foss,
  -- lifecycle fields
  json_extract_string(data, '$.to_stage') AS to_stage,
  json_extract_string(data, '$.blueprint_id') AS blueprint_id,
  json_extract(data, '$.counterfactual')::DOUBLE AS lc_cf,
  json_extract(data, '$.epochs_total')::INTEGER AS epochs_total,
  -- residency fields
  json_extract_string(data, '$.seed_residency.seed_id') AS r_seed_id,
  json_extract(data, '$.seed_residency.params')::INTEGER AS r_params,
  json_extract(data, '$.seed_residency.cf_weighted_integral')::DOUBLE AS cfw,
  json_extract(data, '$.seed_residency.cf_weighted_integral_committed')::DOUBLE AS cfw_c,
  json_extract(data, '$.seed_residency.cf_weighted_integral_uncommitted')::DOUBLE AS cfw_u,
  json_extract(data, '$.seed_residency.j_per_param')::DOUBLE AS jpp,
  -- cf-matrix fields
  json_array_length(data, '$.slot_ids') AS n_slots,
  CASE WHEN event_type = 'COUNTERFACTUAL_MATRIX_COMPUTED'
       THEN regexp_matches(json_extract_string(data, '$.configs'), '\\[(false, )*false\\]')
       END AS has_alloff
FROM read_json(
  '{GLOB}',
  format='newline_delimited',
  columns={{'event_type':'VARCHAR','seed_id':'VARCHAR','slot_id':'VARCHAR',
            'epoch':'BIGINT','data':'JSON'}},
  filename=true, maximum_object_size=33554432)
WHERE event_type IN ('SEED_GERMINATED','SEED_FOSSILIZED','SEED_PRUNED',
                     'SEED_STAGE_CHANGED','COUNTERFACTUAL_MATRIX_COMPUTED')
   OR (event_type = 'ANALYTICS_SNAPSHOT'
       AND json_extract_string(data, '$.kind') IN ('last_action','seed_residency'))
"""
)
print(f"scan done in {time.time() - t0:.0f}s")
print(con.sql("SELECT event_type, kind, COUNT(*) n FROM ev GROUP BY 1,2 ORDER BY n DESC"))

# sanity: epoch semantics + seed_id join rate ------------------------------
print("== sanity: within-episode epoch range (expect ~consistent max) ==")
print(con.sql("""
SELECT run, MAX(epoch) mx_epoch, COUNT(DISTINCT (env_id, episode_idx)) n_env_episodes
FROM ev WHERE kind='last_action' GROUP BY run ORDER BY run"""))
print("== sanity: fossilized seed_id -> residency join rate (expect ~1.0) ==")
print(con.sql("""
WITH f AS (SELECT DISTINCT run, top_seed_id FROM ev WHERE event_type='SEED_FOSSILIZED'),
     r AS (SELECT DISTINCT run, r_seed_id FROM ev WHERE kind='seed_residency')
SELECT COUNT(*) n_foss, SUM(CASE WHEN r.r_seed_id IS NOT NULL THEN 1 ELSE 0 END) n_matched
FROM f LEFT JOIN r ON f.run=r.run AND f.top_seed_id=r.r_seed_id"""))

# ============ F1: PRUNE positive-ba mass ==================================
print("\n== F1: per-action positive/negative ba mass (successful actions) ==")
print(con.sql("""
SELECT action_name,
       COUNT(*) n,
       COUNT(*) FILTER (ba > 0) n_pos_ba,
       ROUND(SUM(ba) FILTER (ba > 0), 1) pos_ba_mass,
       ROUND(SUM(ba) FILTER (ba < 0), 1) neg_ba_mass,
       ROUND(100.0 * SUM(ba) FILTER (ba > 0)
             / SUM(SUM(ba) FILTER (ba > 0)) OVER (), 2) pct_of_total_pos_mass
FROM ev WHERE kind='last_action' AND action_success
GROUP BY 1 ORDER BY pos_ba_mass DESC NULLS LAST"""))

print("== F1: PRUNE-step seed_contribution distribution (successful PRUNE) ==")
print(con.sql("""
SELECT COUNT(*) n_prune,
       COUNT(sc) n_sc_measured,
       ROUND(MIN(sc),2) min_sc, ROUND(quantile_cont(sc,0.10),2) p10,
       ROUND(quantile_cont(sc,0.50),2) p50, ROUND(quantile_cont(sc,0.90),2) p90,
       ROUND(MAX(sc),2) max_sc,
       COUNT(*) FILTER (sc < 0) n_neg_sc,
       ROUND(SUM(ba) FILTER (sc < 0 AND ba > 0),1) pos_ba_from_neg_sc
FROM ev WHERE kind='last_action' AND action_success AND action_name='PRUNE'"""))

print("== F1/S6: PRUNE positive-ba mass by targeted slot ==")
print(con.sql("""
SELECT tgt_slot, COUNT(*) n_prune,
       ROUND(SUM(ba) FILTER (ba > 0),1) pos_ba_mass,
       ROUND(AVG(ba) FILTER (ba > 0),3) mean_pos_ba
FROM ev WHERE kind='last_action' AND action_success AND action_name='PRUNE'
GROUP BY 1 ORDER BY 1"""))

print("== F1: per-episode corr(prune_count, prune-step pos-ba sum), per run ==")
print(con.sql("""
WITH pe AS (
  SELECT run, env_id, episode_idx,
         COUNT(*) FILTER (action_name='PRUNE' AND action_success) prune_n,
         COALESCE(SUM(ba) FILTER (action_name='PRUNE' AND action_success AND ba>0),0) prune_pos_ba
  FROM ev WHERE kind='last_action' GROUP BY 1,2,3)
SELECT run, ROUND(corr(prune_n, prune_pos_ba),3) corr_pruneN_posBA,
       ROUND(AVG(prune_n),2) mean_prune_per_ep
FROM pe GROUP BY run ORDER BY run"""))

# ============ F2a: per-slot ba mass vs per-slot value mass =================
print("\n== F2a: targeted-slot share of |ba| mass (WAIT canonicalization check) ==")
print(con.sql("""
SELECT tgt_slot,
       COUNT(*) n_steps,
       ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (),1) pct_steps,
       ROUND(SUM(ABS(ba)),1) abs_ba_mass,
       ROUND(100.0*SUM(ABS(ba))/SUM(SUM(ABS(ba))) OVER (),1) pct_ba_mass
FROM ev WHERE kind='last_action' GROUP BY 1 ORDER BY 1"""))

print("== F2a: WAIT-step share and WAIT-slot concentration ==")
print(con.sql("""
SELECT action_name IN ('WAIT') is_wait, tgt_slot, COUNT(*) n,
       ROUND(SUM(ABS(ba)),1) abs_ba_mass
FROM ev WHERE kind='last_action' GROUP BY 1,2 ORDER BY 1 DESC,2"""))

print("== F2a: per-slot residency value mass (seed slot via germination join) ==")
print(con.sql("""
WITH germ AS (
  SELECT DISTINCT run, top_seed_id seed_id, top_slot_id slot_id
  FROM ev WHERE event_type='SEED_GERMINATED'),
r AS (SELECT run, r_seed_id seed_id, cfw, jpp FROM ev WHERE kind='seed_residency')
SELECT g.slot_id,
       COUNT(*) n_seed_rows,
       ROUND(SUM(GREATEST(r.cfw,0)),1) pos_cfw_mass,
       ROUND(100.0*SUM(GREATEST(r.cfw,0))/SUM(SUM(GREATEST(r.cfw,0))) OVER (),1) pct_pos_cfw,
       ROUND(SUM(GREATEST(r.jpp,0)),4) pos_jpp_mass,
       ROUND(100.0*SUM(GREATEST(r.jpp,0))/SUM(SUM(GREATEST(r.jpp,0))) OVER (),1) pct_pos_jpp
FROM r JOIN germ g ON r.run=g.run AND r.seed_id=g.seed_id
GROUP BY 1 ORDER BY 1"""))

print("== F2c/S2: lifecycle counts by slot ==")
print(con.sql("""
SELECT top_slot_id slot, event_type, COUNT(*) n
FROM ev WHERE event_type IN ('SEED_GERMINATED','SEED_FOSSILIZED','SEED_PRUNED')
GROUP BY 1,2 ORDER BY 2,1"""))

print("== corr(J, episode-summed ba) and corr(J, episode-summed reward), per run ==")
print(con.sql("""
WITH j AS (
  SELECT run, env_id, episode_idx, SUM(jpp) J
  FROM ev WHERE kind='seed_residency' GROUP BY 1,2,3),
rw AS (
  SELECT run, env_id, episode_idx, SUM(ba) sum_ba, SUM(total_reward) sum_rew
  FROM ev WHERE kind='last_action' GROUP BY 1,2,3)
SELECT j.run, COUNT(*) n_ep,
       ROUND(corr(J, sum_ba),3) corr_J_ba,
       ROUND(corr(J, sum_rew),3) corr_J_reward
FROM j JOIN rw ON j.run=rw.run AND j.env_id=rw.env_id AND j.episode_idx=rw.episode_idx
GROUP BY 1 ORDER BY 1"""))

# ============ S1: fossilize timing within episode ==========================
print("\n== S1: fossilize epoch position within episode (deciles of epoch/max_epoch) ==")
print(con.sql("""
WITH mx AS (
  SELECT run, env_id, episode_idx, MAX(epoch) mx
  FROM ev WHERE kind='last_action' GROUP BY 1,2,3),
f AS (
  SELECT e.run, e.env_id, e.episode_idx, e.epoch,
         e.epoch::DOUBLE / NULLIF(m.mx,0) frac
  FROM ev e JOIN mx m ON e.run=m.run AND e.env_id=m.env_id AND e.episode_idx=m.episode_idx
  WHERE e.event_type='SEED_FOSSILIZED')
SELECT COUNT(*) n_foss,
       ROUND(quantile_cont(frac,0.10),2) p10, ROUND(quantile_cont(frac,0.50),2) p50,
       ROUND(quantile_cont(frac,0.90),2) p90,
       COUNT(*) FILTER (frac > 0.9) n_final_decile,
       COUNT(*) FILTER (frac < 0.5) n_first_half
FROM f"""))

# ============ Coverage / bootstrap-opportunity =============================
print("\n== COVERAGE: bootstrap opportunity — episodes with >=1 fossilize ==")
print(con.sql("""
WITH eps AS (
  SELECT DISTINCT run, env_id, episode_idx FROM ev WHERE kind='last_action'),
f AS (
  SELECT run, env_id, episode_idx, COUNT(*) k
  FROM ev WHERE event_type='SEED_FOSSILIZED' GROUP BY 1,2,3)
SELECT eps.run,
       COUNT(*) n_episodes,
       COUNT(f.k) n_with_foss,
       ROUND(100.0*COUNT(f.k)/COUNT(*),1) pct_with_foss,
       MAX(f.k) max_k,
       ROUND(AVG(COALESCE(f.k,0)),3) foss_per_ep
FROM eps LEFT JOIN f ON eps.run=f.run AND eps.env_id=f.env_id AND eps.episode_idx=f.episode_idx
GROUP BY 1 ORDER BY 1"""))

print("== COVERAGE: k distribution (fossilized slots per env-episode, episodes with k>=1) ==")
print(con.sql("""
WITH f AS (
  SELECT run, env_id, episode_idx, COUNT(*) k
  FROM ev WHERE event_type='SEED_FOSSILIZED' GROUP BY 1,2,3)
SELECT k, COUNT(*) n_episodes FROM f GROUP BY k ORDER BY k"""))

print("== COVERAGE: positive value-mass share reachable by the fossilize-only gate ==")
print(con.sql("""
WITH fseeds AS (SELECT DISTINCT run, top_seed_id seed_id FROM ev WHERE event_type='SEED_FOSSILIZED'),
r AS (SELECT run, r_seed_id seed_id, cfw, cfw_u, jpp FROM ev WHERE kind='seed_residency')
SELECT COUNT(*) n_seed_rows,
       COUNT(*) FILTER (f.seed_id IS NOT NULL) n_fossilized,
       ROUND(SUM(GREATEST(r.cfw,0)),1) total_pos_cfw,
       ROUND(SUM(GREATEST(r.cfw,0)) FILTER (f.seed_id IS NOT NULL),1) foss_pos_cfw,
       ROUND(100.0*SUM(GREATEST(r.cfw,0)) FILTER (f.seed_id IS NOT NULL)
             / NULLIF(SUM(GREATEST(r.cfw,0)),0),2) pct_mass_fossilized,
       ROUND(100.0*SUM(GREATEST(r.jpp,0)) FILTER (f.seed_id IS NOT NULL)
             / NULLIF(SUM(GREATEST(r.jpp,0)),0),2) pct_jpp_mass_fossilized
FROM r LEFT JOIN fseeds f ON r.run=f.run AND r.seed_id=f.seed_id"""))

print("== COVERAGE: HOLDING-reached seeds — fossilized vs not (count + pos-cfw mass) ==")
print(con.sql("""
WITH hold AS (
  SELECT DISTINCT run, top_seed_id seed_id
  FROM ev WHERE event_type='SEED_STAGE_CHANGED' AND to_stage='HOLDING'),
fseeds AS (SELECT DISTINCT run, top_seed_id seed_id FROM ev WHERE event_type='SEED_FOSSILIZED'),
r AS (SELECT run, r_seed_id seed_id, SUM(GREATEST(cfw,0)) pos_cfw
      FROM ev WHERE kind='seed_residency' GROUP BY 1,2)
SELECT COUNT(*) n_holding_seeds,
       COUNT(*) FILTER (f.seed_id IS NOT NULL) n_fossilized,
       ROUND(SUM(r.pos_cfw),1) holding_pos_cfw,
       ROUND(SUM(r.pos_cfw) FILTER (f.seed_id IS NOT NULL),1) fossilized_pos_cfw,
       ROUND(100.0*SUM(r.pos_cfw) FILTER (f.seed_id IS NOT NULL)
             / NULLIF(SUM(r.pos_cfw),0),2) pct_holding_mass_fossilized
FROM hold h
LEFT JOIN fseeds f ON h.run=f.run AND h.seed_id=f.seed_id
LEFT JOIN r ON h.run=r.run AND h.seed_id=r.seed_id"""))

# ============ F5: all-off config presence in CF matrices ====================
print("\n== F5: CF-matrix all-off config presence ==")
print(con.sql("""
SELECT n_slots,
       COUNT(*) n_matrices,
       COUNT(*) FILTER (has_alloff) n_with_alloff,
       ROUND(100.0*COUNT(*) FILTER (has_alloff)/COUNT(*),2) pct_with_alloff
FROM ev WHERE event_type='COUNTERFACTUAL_MATRIX_COMPUTED'
GROUP BY 1 ORDER BY 1"""))

print("== F5: env-episodes with multi-slot matrices lacking any all-off matrix ==")
print(con.sql("""
WITH m AS (
  SELECT run, env_id, episode_idx,
         MAX(CASE WHEN n_slots >= 2 THEN 1 ELSE 0 END) has_multi,
         MAX(CASE WHEN n_slots >= 2 AND has_alloff THEN 1 ELSE 0 END) multi_alloff
  FROM ev WHERE event_type='COUNTERFACTUAL_MATRIX_COMPUTED' GROUP BY 1,2,3)
SELECT SUM(has_multi) n_multi_eps,
       SUM(has_multi) - SUM(multi_alloff) n_multi_eps_no_alloff,
       ROUND(100.0*(SUM(has_multi)-SUM(multi_alloff))/NULLIF(SUM(has_multi),0),2) pct_no_alloff
FROM m"""))

print(f"\ntotal wall time {time.time() - t0:.0f}s")
