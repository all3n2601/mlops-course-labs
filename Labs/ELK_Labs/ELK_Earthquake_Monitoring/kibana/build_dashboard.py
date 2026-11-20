"""Build the Kibana index patterns, charts and two dashboards, and import them.

Writes kibana/dashboards.ndjson (so it can also be imported by hand through
Stack Management > Saved Objects) and posts it to Kibana on localhost:5601.
Standard library only.
"""
import json
import subprocess
import sys

KIBANA = "http://localhost:5601"
OUT = "kibana/dashboards.ndjson"
MIG = {"visualization": "7.14.0", "index-pattern": "7.11.0", "dashboard": "7.9.3"}


def index_pattern(pid, title):
    return {"id": pid, "type": "index-pattern", "migrationVersion": {"index-pattern": MIG["index-pattern"]},
            "attributes": {"title": title, "timeFieldName": "@timestamp"}, "references": []}


def search_source(query="", filters=None):
    return json.dumps({"query": {"query": query, "language": "kuery"}, "filter": filters or [],
                       "indexRefName": "kibanaSavedObjectMeta.searchSourceJSON.index"})


def vis(vid, title, pattern_id, state, query=""):
    return {"id": vid, "type": "visualization", "migrationVersion": {"visualization": MIG["visualization"]},
            "attributes": {"title": title, "description": "", "uiStateJSON": "{}", "version": 1,
                           "visState": json.dumps({"title": title, **state}),
                           "kibanaSavedObjectMeta": {"searchSourceJSON": search_source(query)}},
            "references": [{"name": "kibanaSavedObjectMeta.searchSourceJSON.index",
                            "type": "index-pattern", "id": pattern_id}]}


def count_agg():
    return {"id": "1", "enabled": True, "type": "count", "params": {}, "schema": "metric"}


def terms(agg_id, field, schema, size=5, order_by="1"):
    return {"id": agg_id, "enabled": True, "type": "terms", "schema": schema,
            "params": {"field": field, "orderBy": order_by, "order": "desc", "size": size,
                       "otherBucket": False, "missingBucket": False}}


def per_week(agg_id="2"):
    return {"id": agg_id, "enabled": True, "type": "date_histogram", "schema": "segment",
            "params": {"field": "@timestamp", "useNormalizedEsInterval": True, "scaleMetricValues": False,
                       "interval": "w", "drop_partials": False, "min_doc_count": 1, "extended_bounds": {}}}


def xy(kind, aggs, mode="stacked", horizontal=False, y_title="Count"):
    cat_pos, val_pos = ("left", "bottom") if horizontal else ("bottom", "left")
    return {"type": "horizontal_bar" if horizontal else kind, "aggs": aggs, "params": {
        "type": kind, "grid": {"categoryLines": False},
        "categoryAxes": [{"id": "CategoryAxis-1", "type": "category", "position": cat_pos, "show": True,
                          "style": {}, "scale": {"type": "linear"},
                          "labels": {"show": True, "filter": True, "truncate": 100}, "title": {}}],
        "valueAxes": [{"id": "ValueAxis-1", "name": "LeftAxis-1", "type": "value", "position": val_pos,
                       "show": True, "style": {}, "scale": {"type": "linear", "mode": "normal"},
                       "labels": {"show": True, "rotate": 0, "filter": False, "truncate": 100},
                       "title": {"text": y_title}}],
        "seriesParams": [{"show": True, "type": kind, "mode": mode,
                          "data": {"label": y_title, "id": "1"}, "valueAxis": "ValueAxis-1",
                          "drawLinesBetweenPoints": True, "lineWidth": 2, "showCircles": True}],
        "addTooltip": True, "addLegend": True, "legendPosition": "right", "times": [],
        "addTimeMarker": False, "labels": {"show": False},
        "thresholdLine": {"show": False, "value": 10, "width": 1, "style": "full", "color": "#E7664C"}}}


def metric_vis(aggs, font=48):
    return {"type": "metric", "aggs": aggs, "params": {
        "addTooltip": True, "addLegend": False, "type": "metric",
        "metric": {"percentageMode": False, "useRanges": False, "colorSchema": "Green to Red",
                   "metricColorMode": "None", "colorsRange": [{"from": 0, "to": 10000}],
                   "labels": {"show": True}, "invertColors": False,
                   "style": {"bgFill": "#000", "bgColor": False, "labelColor": False, "subText": "",
                             "fontSize": font}}}}


def table_vis(aggs):
    return {"type": "table", "aggs": aggs, "params": {
        "perPage": 10, "showPartialRows": False, "showMetricsAtAllLevels": False,
        "showTotal": False, "totalFunc": "sum", "percentageCol": ""}}


def dashboard(did, title, panels, time_from, time_to):
    panels_json, refs = [], []
    for i, (vid, x, y, w, h) in enumerate(panels):
        panels_json.append({"version": "7.16.2", "type": "visualization",
                            "gridData": {"x": x, "y": y, "w": w, "h": h, "i": str(i)},
                            "panelIndex": str(i), "embeddableConfig": {}, "panelRefName": f"panel_{i}"})
        refs.append({"name": f"panel_{i}", "type": "visualization", "id": vid})
    return {"id": did, "type": "dashboard", "migrationVersion": {"dashboard": MIG["dashboard"]},
            "attributes": {"title": title, "description": "", "hits": 0, "version": 1,
                           "timeRestore": True, "timeFrom": time_from, "timeTo": time_to,
                           "refreshInterval": {"pause": True, "value": 0},
                           "optionsJSON": json.dumps({"useMargins": True, "hidePanelTitles": False}),
                           "panelsJSON": json.dumps(panels_json),
                           "kibanaSavedObjectMeta": {"searchSourceJSON": json.dumps(
                               {"query": {"query": "", "language": "kuery"}, "filter": []})}},
            "references": refs}


def main():
    objects = [
        index_pattern("eq-pattern", "earthquakes"),
        index_pattern("drift-pattern", "ml-drift"),
        index_pattern("train-pattern", "ml-training"),

        vis("v-total", "Earthquakes in 2025 (M2.5+)", "eq-pattern", metric_vis([count_agg()], 40)),
        vis("v-big", "Earthquakes M7.0 and above", "eq-pattern", metric_vis([count_agg()], 40), query="mag >= 7"),
        vis("v-weekly", "Earthquakes per week by size class", "eq-pattern",
            xy("histogram", [count_agg(), per_week(), terms("3", "mag_class", "group")])),
        vis("v-places", "Top 10 places in the selected period", "eq-pattern",
            table_vis([count_agg(), terms("2", "place.keyword", "bucket", 10)])),
        vis("v-map", "Where the earthquakes happened", "eq-pattern", {
            "type": "tile_map", "aggs": [count_agg(), {
                "id": "2", "enabled": True, "type": "geohash_grid", "schema": "segment",
                "params": {"field": "location", "autoPrecision": True, "isFilteredByCollar": True,
                           "useGeocentroid": True, "mapZoom": 2, "mapCenter": [0, 0], "precision": 2}}],
            "params": {"colorSchema": "Yellow to Red", "mapType": "Scaled Circle Markers",
                       "isDesaturated": True, "addTooltip": True, "heatClusterSize": 1.5,
                       "legendPosition": "bottomright", "mapZoom": 2, "mapCenter": [0, 0],
                       "wms": {"enabled": False, "options": {"format": "image/png", "transparent": True}}}}),

        vis("v-drift-week", "Drift warnings per week, by feature", "drift-pattern",
            xy("histogram", [count_agg(), per_week(), terms("3", "feature.keyword", "group", 7)],
               y_title="Drift warnings"), query="drift : true"),
        vis("v-f1-week", "Model F1 on each weekly batch", "drift-pattern",
            xy("line", [{"id": "1", "enabled": True, "type": "avg", "schema": "metric",
                         "params": {"field": "f1"}}, per_week()], mode="normal", y_title="Average F1"),
            query="f1 : *"),
        vis("v-drift-total", "Drift warnings", "drift-pattern", metric_vis([count_agg()], 40), query="drift : true"),
        vis("v-ks-table", "Weeks with the strongest drift (KS statistic)", "drift-pattern", table_vis([
            {"id": "1", "enabled": True, "type": "max", "schema": "metric", "params": {"field": "ks_stat"}},
            {"id": "4", "enabled": True, "type": "count", "schema": "metric", "params": {}},
            terms("2", "week", "bucket", 8, order_by="1")]), query="drift : true"),

        vis("v-train-f1", "Validation F1 by training run", "train-pattern",
            xy("histogram", [{"id": "1", "enabled": True, "type": "max", "schema": "metric",
                              "params": {"field": "val_f1"}},
                             {"id": "2", "enabled": True, "type": "terms", "schema": "segment",
                              "params": {"field": "run", "orderBy": "_key", "order": "asc", "size": 10,
                                         "otherBucket": False, "missingBucket": False}}],
               mode="normal", y_title="val_f1"), query="run : *"),
        vis("v-train-table", "Training runs", "train-pattern", table_vis([
            {"id": "1", "enabled": True, "type": "max", "schema": "metric", "params": {"field": "val_f1"}},
            {"id": "4", "enabled": True, "type": "max", "schema": "metric", "params": {"field": "val_recall"}},
            {"id": "5", "enabled": True, "type": "max", "schema": "metric", "params": {"field": "val_fpr"}},
            {"id": "2", "enabled": True, "type": "terms", "schema": "bucket",
             "params": {"field": "run", "orderBy": "_key", "order": "asc", "size": 10,
                        "otherBucket": False, "missingBucket": False}}]), query="run : *"),

        dashboard("d-quakes", "Earthquakes 2025: events and model drift", [
            ("v-total", 0, 0, 8, 7), ("v-big", 8, 0, 8, 7), ("v-drift-total", 16, 0, 8, 7),
            ("v-weekly", 0, 7, 24, 14), ("v-map", 24, 0, 24, 21),
            ("v-drift-week", 0, 21, 24, 14), ("v-f1-week", 24, 21, 24, 14),
            ("v-places", 0, 35, 24, 14), ("v-ks-table", 24, 35, 24, 14)],
            "2025-01-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z"),
        dashboard("d-training", "Model training runs", [
            ("v-train-f1", 0, 0, 24, 14), ("v-train-table", 24, 0, 24, 14)],
            "2026-11-19T00:00:00.000Z", "2026-11-22T00:00:00.000Z"),
    ]
    with open(OUT, "w") as f:
        for obj in objects:
            f.write(json.dumps(obj) + "\n")
    print(f"wrote {OUT}: {len(objects)} saved objects")

    subprocess.run(["curl", "-s", "-X", "POST", f"{KIBANA}/api/kibana/settings",
                    "-H", "kbn-xsrf: true", "-H", "Content-Type: application/json",
                    "-d", json.dumps({"changes": {"dateFormat:tz": "UTC"}})], capture_output=True)
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{KIBANA}/api/saved_objects/_import?overwrite=true",
         "-H", "kbn-xsrf: true", "--form", f"file=@{OUT}"], capture_output=True, text=True)
    response = json.loads(result.stdout)
    print("import success:", response.get("success"), "| objects:", response.get("successCount"))
    for err in response.get("errors", []):
        print("ERROR:", err.get("id"), err.get("error"))
    sys.exit(0 if response.get("success") else 1)


if __name__ == "__main__":
    main()
