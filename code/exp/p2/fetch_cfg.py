"""Pull the small config/tokenizer/scheduler files of a diffusers repo from
ModelScope (HF is unreachable here).  Large weights are skipped: UNet / VAE /
text-encoder weights come from the single-file safetensors we already have.
"""
import json
import os
import sys
import urllib.request

REPO = "AI-ModelScope/stable-diffusion-v1-5"
ROOT = r"E:\研究\models\sd15_cfg"
BASE = f"https://www.modelscope.cn/models/{REPO}/resolve/master/"
API = f"https://www.modelscope.cn/api/v1/models/{REPO}/repo/files"
MAX_MB = 60.0


def ls(root=""):
    u = f"{API}?Revision=master&Root={root}"
    d = json.loads(urllib.request.urlopen(u, timeout=30).read())["Data"]["Files"]
    return d


def walk(root=""):
    for f in ls(root):
        path, size, typ = f["Path"], f.get("Size", 0), f.get("Type", "blob")
        if typ == "tree":
            yield from walk(path)
        else:
            yield path, size


def main():
    todo, skipped = [], []
    for path, size in walk():
        if path.endswith((".ckpt", ".safetensors", ".bin", ".pt", ".pth")):
            skipped.append((path, size))
            continue
        todo.append((path, size))
    print(f"to fetch {len(todo)} files, skip {len(skipped)} weight files")
    for p, s in skipped:
        print(f"  skip {s/1e6:8.1f}MB  {p}")

    for path, size in todo:
        dst = os.path.join(ROOT, path.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            continue
        url = BASE + urllib.parse.quote(path)
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
                f.write(r.read())
            print(f"  ok  {size/1e6:8.2f}MB  {path}")
        except Exception as e:
            print(f"  ERR {path}: {type(e).__name__} {e}")
    print("done ->", ROOT)


if __name__ == "__main__":
    import urllib.parse
    main()
