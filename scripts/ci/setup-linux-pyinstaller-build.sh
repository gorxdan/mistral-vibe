#!/usr/bin/env bash
set -euo pipefail

# Build in manylinux for glibc compatibility, then clear executable-stack
# flags rejected by hardened Linux kernels.
python_version="${PYTHON_VERSION:-3.12}"
patchelf_version="${PATCHELF_VERSION:-0.18.0}"

uv python install "${python_version}"

# Pin the patchelf release tarball by sha256. NixOS/patchelf publishes no signed
# SHA256SUMS asset, so these are community-computed (trust-on-first-use): they
# guard against future asset replacement, not a compromise predating this pin.
# Update the table when bumping PATCHELF_VERSION or adding a build arch.
arch="$(uname -m)"
declare -A patchelf_sha256=(
  ["0.18.0:x86_64"]="ce84f2447fb7a8679e58bc54a20dc2b01b37b5802e12c57eece772a6f14bf3f0"
  ["0.18.0:aarch64"]="ae13e2effe077e829be759182396b931d8f85cfb9cfe9d49385516ea367ef7b2"
)
expected_sha="${patchelf_sha256[${patchelf_version}:${arch}]:-}"
if [ -z "${expected_sha}" ]; then
  echo "::error::No pinned patchelf sha256 for ${patchelf_version}/${arch}; update scripts/ci/setup-linux-pyinstaller-build.sh" >&2
  exit 1
fi

tarball="patchelf-${patchelf_version}-${arch}.tar.gz"
curl -fsSL "https://github.com/NixOS/patchelf/releases/download/${patchelf_version}/${tarball}" -o "${tarball}"
echo "${expected_sha}  ${tarball}" | sha256sum -c -
tar xz -C /usr/local -f "${tarball}"
rm -f "${tarball}"

find "$(uv python dir)" -name 'libpython*.so*' -type f -exec patchelf --clear-execstack {} \;
