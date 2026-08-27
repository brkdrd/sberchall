#!/usr/bin/env bash
# Push a notebook to Kaggle without hand-editing metadata.
#
# `kaggle kernels push` reads exactly one kaggle/kernel-metadata.json, and its "id" must be
# "<your-handle>/<slug>". Leaving the USERNAME placeholder in makes Kaggle try to resolve a
# user literally called USERNAME, which fails with:  Permission 'users.get' was denied.
# This script fills the handle in from your credentials instead.
#
#   ./kaggle/push.sh                                  # the runner notebook (05)
#   ./kaggle/push.sh notebooks/03_massive_multistart.ipynb qaoa-multistart
set -euo pipefail
cd "$(dirname "$0")/.."

NB="${1:-notebooks/05_kaggle_runner.ipynb}"
SLUG="${2:-qaoa-pipeline-runner}"
[ -f "$NB" ] || { echo "no such notebook: $NB" >&2; exit 1; }

USER="${KAGGLE_USERNAME:-}"
for f in "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" "$HOME/.config/kaggle/kaggle.json"; do
    [ -n "$USER" ] && break
    [ -f "$f" ] && USER=$(python3 -c "import json;print(json.load(open('$f'))['username'])")
done
[ -n "$USER" ] || { echo "No Kaggle handle. Set KAGGLE_USERNAME, or put kaggle.json in ~/.kaggle/ (Kaggle -> Settings -> API -> Create New Token)." >&2; exit 1; }

# internet must stay on: the notebook clones this repo from GitHub
python3 - "$USER" "$SLUG" "$NB" <<'PY'
import json, sys
user, slug, nb = sys.argv[1:4]
json.dump({
    "id": f"{user}/{slug}",
    "title": slug.replace("-", " ").title(),
    "code_file": f"../{nb}",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
}, open("kaggle/kernel-metadata.json", "w"), indent=2)
print(f"kaggle/kernel-metadata.json -> {user}/{slug}  ({nb})")
PY

kaggle kernels push -p kaggle
echo
echo "watch it with:  kaggle kernels status $USER/$SLUG"
echo "fetch output:   kaggle kernels output $USER/$SLUG -p out/"
