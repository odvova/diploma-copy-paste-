#!/bin/bash
# Run with: bash install_deps.sh
# Installs XeLaTeX + all packages needed for the thesis
sudo dnf install -y \
  texlive-xetex \
  texlive-collection-xetex \
  texlive-collection-fontsrecommended \
  texlive-collection-langcyrillic \
  texlive-collection-latexrecommended \
  texlive-collection-latexextra \
  texlive-polyglossia \
  texlive-fontspec \
  texlive-titlesec \
  texlive-caption \
  texlive-fancyhdr \
  texlive-booktabs \
  texlive-float \
  texlive-multirow \
  texlive-chngcntr \
  texlive-tocloft \
  texlive-listings \
  texlive-geometry \
  texlive-setspace \
  texlive-amsmath \
  texlive-extsizes \
  texlive-babel-ukrainian \
  texlive-hyphen-ukrainian

echo "Done. Now run: bash compile.sh"
