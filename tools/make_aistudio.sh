#!/bin/bash

TO_DIR="./run/aistudio"
AISTUDIO_IPYNB="./templates/aistudio/main.ipynb"
PADDLEX_CLS_IPYNB="./templates/paddlex_cls/aismain.ipynb"
PADDLEX_DET_IPYNB="./templates/paddlex_det/aismain.ipynb"
PADDLEX_SEG_IPYNB="./templates/paddlex_seg/aismain.ipynb"

if [ ! -d $TO_DIR ]; then
    mkdir $TO_DIR
fi

cp "$AISTUDIO_IPYNB" "$TO_DIR"
cp "$PADDLEX_CLS_IPYNB" "$TO_DIR/paddlex_cls.ipynb"
cp "$PADDLEX_DET_IPYNB" "$TO_DIR/paddlex_det.ipynb"
cp "$PADDLEX_SEG_IPYNB" "$TO_DIR/paddlex_seg.ipynb"
