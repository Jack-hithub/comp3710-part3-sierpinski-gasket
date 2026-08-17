#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
bundle_dir="$project_dir/dist"
bundle_path="$bundle_dir/COMP3710_Part3_Colab_Bundle.zip"

mkdir -p "$bundle_dir"
cd "$project_dir"

rm -f "$bundle_path"
zip -q -r "$bundle_path" \
  src \
  tests \
  README.md \
  PART3_NOTES.md \
  BENCHMARK_RESULTS.md \
  REFERENCES.md \
  environment.yml \
  -x '*/__pycache__/*' '*.pyc'

echo "Created: $bundle_path"

