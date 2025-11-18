import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("eval_out_frcnn_test/per_class_metrics.csv")

class_names = [
    "Ants","Bees","Beetles","Caterpillars","Earthworms","Earwigs",
    "Grasshoppers","Moths","Slugs","Snails","Wasps","Weevils","Worms"
][:len(df)]

df["class_name"] = class_names

plt.figure(figsize=(10,5))
plt.bar(df["class_name"], df["AP"], color="skyblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average Precision (AP)")
plt.title("Per-Class Detection Accuracy (Test Set)")
plt.tight_layout()
plt.savefig("eval_out_frcnn_test/per_class_mAP_bar.png", dpi=300)
plt.show()
import matplotlib.pyplot as plt

map_values = {
    'mAP': 0.2147,
    'mAP@50': 0.4613,
    'mAP@75': 0.1764,
    'mAR@100': 0.3986
}

plt.figure(figsize=(6,4))
plt.bar(map_values.keys(), map_values.values(), color='orange')
plt.ylabel("Score")
plt.title("Overall Detection Metrics (Test Set)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("eval_out_frcnn_test/overall_mAP_summary.png", dpi=300)
plt.show()
