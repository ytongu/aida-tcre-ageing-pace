#!/usr/bin/env python3
import synapseclient
import os
import sys

OUTDIR = "${DATA_DIR}/lnc_project_synapse_dataset/syn49637038/01_download_fastq"

BATCHES = {
    "AS051": "syn50209112",
    "AS052": "syn50215079",
    "AS053": "syn50229489",
    "AS054": "syn50233314",
    "AS055": "syn50271167",
    "AS057": "syn50235032",
    "AS058": "syn50236541",
    "AS059": "syn50270729",
    "AS060": "syn50270840",
    "AS061": "syn50270945",
    "AS056": "syn50234663",
    "AS062": "syn50271047",
}

syn = synapseclient.Synapse()
syn.login(silent=True)

for batch_name, folder_id in sorted(BATCHES.items()):
    print(f"\n=== {batch_name} ({folder_id}) ===")
    batch_dir = os.path.join(OUTDIR, batch_name)
    os.makedirs(batch_dir, exist_ok=True)

    children = list(syn.getChildren(folder_id, includeTypes=["file"]))
    gex_files = [f for f in children if "5GEX" in f["name"] or "5pGEX" in f["name"] or "5-GEX" in f["name"]]

    print(f"  Total files: {len(children)}, 5GEX files: {len(gex_files)}")

    for f in gex_files:
        dest = os.path.join(batch_dir, f["name"])
        if os.path.exists(dest):
            print(f"  [SKIP] {f['name']}")
            continue
        print(f"  [DL] {f['name']} ({f['id']})")
        syn.get(f["id"], downloadLocation=batch_dir)

    print(f"  {batch_name} done!")

print("\n=== All 5GEX downloads complete ===")
