import json

schema_path = "datasets/duee/schema.json"

labels = []
with open("datasets/duee/schema.json") as f:
    for l in f:
        l = json.loads(l)
        t = l["event_type"]
        for r in ["触发词"] + [s["role"] for s in l["role_list"]]:
            labels.append(f"{t}+{r}")
