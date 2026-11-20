# ELK Lab: monitoring an ML model on 2025 earthquake data

This is my version of `ELK_Labs/ELK_Docker`. The original ships ML training logs and synthetic drift logs (Iris, `make_classification`) through Logstash into Elasticsearch and Kibana. I kept that architecture, but I replaced the data, the model, the logs and the Kibana side with my own, so it is not a copy.

```
data/usgs_earthquakes_2025.csv ──┐
logs/training.log (plain text) ──┼─> Logstash ─> Elasticsearch ─> Kibana dashboards
logs/drift.log    (plain text) ──┘   csv, grok, kv, date,        earthquakes, ml-training, ml-drift
                                     geo_point, fingerprint
        ▲
   ml_pipeline.py trains a classifier and replays the second half of 2025 week by week
```

## What I changed from the original lab

| Original `ELK_Docker` | This version |
|---|---|
| Iris model trained in a loop, plus random synthetic "drift" | Real data: all 29,068 earthquakes of magnitude 2.5+ in 2025 from USGS. A classifier flags events of M4.5 or higher |
| Drift is injected by the script (5% of samples are altered on purpose) | Drift is measured with a two-sample KS test of each weekly batch against the training data. Nothing is injected |
| Logs only | Logstash also ingests the earthquake CSV itself (csv filter, date filter, `geo_point` for a map) |
| Index names and a grok layout for F1, FPR and FNR lines | My own log format parsed with grok plus the `kv` filter, a `date` filter that uses the week the data comes from, and an index template that makes `location` a `geo_point` |
| Re-reading a file on restart creates duplicates | Each document id is a hash of its source line, so a restart overwrites instead of duplicating |
| Metricbeat container, `restart: always`, 256 MB heaps | No Metricbeat, healthchecks with `depends_on: service_healthy`, 512 MB heaps, one Logstash worker (see problems below) |
| No dashboards | Two Kibana dashboards, built from files in `kibana/` so they can be re-imported |

## Data

`fetch_data.py` downloads the [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/) for 2025, one month per request (the API caps a request at 20,000 events), and keeps `type == earthquake`. Result: 29,068 events in `data/usgs_earthquakes_2025.csv` (3.5 MB), so the lab does not need the API to run. `nst`, `gap` and `dmin` are missing for about 1,300 events each (`evidence/fetch_data.txt`).

## The ML part

`ml_pipeline.py` predicts whether an event is M4.5 or higher from its location, depth and the quality of the readings (`nst`, `gap`, `dmin`, `rms`). It is a toy task. The lab is about the logging, ELK pipeline and monitoring, not seismology.

- Train: January to May. Validation: June. Replay: July to December in weekly batches, 27 batches (the first and last are partial weeks).
- Six configurations (2 logistic regression, 4 gradient boosting). The best validation F1 wins.
- Drift rule, fixed before I looked at any results: a feature drifts in a batch when its KS statistic against the training data is at least **0.20**. Features checked: latitude, longitude, depth, gap, dmin, rms, nst.

## Files

- `fetch_data.py`, `ml_pipeline.py`, `requirements.txt`
- `docker-compose.yml`: Elasticsearch, Logstash and Kibana 7.16.2
- `logstash/pipeline/logstash.conf`, `logstash/templates/*.json`
- `kibana/build_dashboard.py` writes `kibana/dashboards.ndjson` and imports it into Kibana
- `logs/`: the two logs `ml_pipeline.py` wrote, which Logstash reads
- `screenshots/`, `evidence/`: output of the runs below

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ml_pipeline.py              # writes logs/training.log and logs/drift.log
                                   # (fetch_data.py is optional, the CSV is included)

docker compose up -d               # Elasticsearch :9200, Logstash, Kibana :5601
python kibana/build_dashboard.py   # imports the dashboards (also sets Kibana to show times in UTC)
```

Then open `http://localhost:5601/app/dashboards#/view/d-quakes` (the map needs internet for its tiles) and `.../view/d-training` (its time range is the day I ran it, 19-22 Nov 2026, so change the range if you run it later). Stop with `docker compose down`, add `-v` to delete the Elasticsearch data.

## What happened when I ran it

Run on macOS with Docker Desktop (8 GB assigned to Docker). The stack used about 2.2 GB.

- Everything ingested: 29,068 earthquakes, 8 training log documents, 71 drift log documents, zero parse failures, correct field types. Cluster health green (`evidence/stack_status.txt`). After wiping the indices and restarting Logstash three more times, the counts stayed identical.
- The best model was run 5 (gradient boosting, 200 trees, learning rate 0.05) with a validation F1 of 0.791. All six runs are in `logs/training.log`.
- The drift detector raised **42 warnings in 17 of the 27 weeks**. It ranks the week of **28 July** first (KS 0.691). That week has 1,874 events, against a median of 521 for July to December, and contains the **M8.8 Kamchatka earthquake of 29 July** plus an M7.0. I checked the other top weeks against the data:

| Week starting | Max KS | Events | What the data shows |
|---|---|---|---|
| 2025-07-28 | 0.691 | 1,874 | M8.8 and M7.0. Busiest places: Petropavlovsk-Kamchatsky and Severo-Kuril'sk |
| 2025-07-14 | 0.542 | 1,215 | M7.3 and M7.4. 634 events near Sand Point, Alaska |
| 2025-08-04 | 0.477 | 827 | No M7+. Mostly events near Severo-Kuril'sk and Vilyuchinsk, Russia (the Kamchatka region) |
| 2025-12-01 | 0.471 | 842 | An M7.0. 404 events near Yakutat, Alaska |
| 2025-09-15 | 0.438 | 599 | An M7.8 near Petropavlovsk-Kamchatsky |

- **Drift did not wreck the model.** The mean weekly F1 was 0.739 in the 17 flagged weeks and 0.745 in the 10 unflagged weeks, with batch F1 between 0.685 and 0.793. A drift flag tells you the inputs changed, not that predictions got worse.

## Problems I hit and what I did about them

- **Duplicates on restart.** `sincedb_path => /dev/null` makes Logstash re-read every file when it restarts, which doubled my indices. Fixed with the hash-based document id.
- **Logstash crashed during one restart** (`uninitialized constant ...Fingerprint::OpenSSL`). With many pipeline workers, the `fingerprint` filter can race while loading OpenSSL. The crash stopped ingestion partway, and `ml-drift` had 57 of 71 documents. I only noticed because the count was off. Fix: `PIPELINE_WORKERS=1`. After that I wiped the indices and reloaded, then restarted three times with correct counts and no pipeline errors each time. It is a race, so this lowers the risk and is not a proof.
- **Yellow cluster.** The two `ml-*` indices wanted a replica a single node cannot hold. Fixed with an index template (`ml-template.json`) that sets zero replicas.
- **Kibana dashboard bugs.** `week` is mapped as a `date` by Elasticsearch's date detection, so `week.keyword` does not exist, which broke a table. A horizontal bar chart did not draw its labels, so I used a table. Times showed in my browser's time zone, so the build script sets Kibana's time zone to UTC.

## Notes and limits

- Ran on Docker Desktop on a Mac, not on Ubuntu in a VM.
- Elasticsearch, Logstash and Kibana 7.16.2 are old (from the original lab) and security is switched off. This is for local use only.
- Latitude and longitude drift in almost any week with a regional swarm, because the training data is spread worldwide. The 0.20 threshold is simple and I did not tune it.
- The training dashboard's time range only covers the day I ran the pipeline.
- Dashboard screenshots were taken with a headless browser against the final stack.
- I did not use Metricbeat or Filebeat.
