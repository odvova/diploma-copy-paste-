#!/bin/bash
# Compile the thesis (run twice for correct TOC and references)
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode main.tex && \
xelatex -interaction=nonstopmode main.tex && \
echo "=== Compiled successfully: main.pdf ==="
