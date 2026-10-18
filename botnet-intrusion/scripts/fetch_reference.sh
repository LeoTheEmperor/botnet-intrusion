#!/usr/bin/env bash
# Downloads the authors' Kitsune-py AfterImage.py (github.com/ymirsky/Kitsune-py) into reference/.
# It is used ONLY by src/validate_extractor.py to check our own implementation numerically;
# it is not part of this repository. `np.Inf` is patched to `np.inf` for NumPy 2.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reference
COMMIT=28a654b5813936380d264c0934136efda672174a
curl -sSL "https://raw.githubusercontent.com/ymirsky/Kitsune-py/${COMMIT}/AfterImage.py" \
  | tr -d '\r' | sed 's/np\.Inf/np.inf/g' > reference/AfterImage_ref.py
echo "saved reference/AfterImage_ref.py ($(wc -l < reference/AfterImage_ref.py) lines)"
