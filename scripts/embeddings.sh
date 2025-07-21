#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 --dataset_name"
  echo "$#" "$1"
  exit 1
fi

# Use a case statement to handle different datasets
case $1 in
  --openreview)
    python pre_comp_emb.py --dataset openreview --model_name_or_path 'stsb-roberta-base-v2'
    ;;
  --pubmed)
    python pre_comp_emb.py --dataset pubmed --model_name_or_path 'sentence-t5-base'
    ;;
  --cas)
    python pre_comp_emb.py --dataset cas --model_name_or_path 'sentence-t5-base'
    ;;
  --psytar)
    python pre_comp_emb.py --dataset cls/psytar --model_name_or_path 'sentence-t5-base'
    ;;
  --hallmarks_of_cancer)
    python pre_comp_emb.py --dataset cls/hallmarks_of_cancer --model_name_or_path 'kamalkraj/BioSimCSE-BioLinkBERT-BASE'
    ;;
  --yelp)
    python pre_comp_emb.py --dataset yelp --model_name_or_path 'stsb-roberta-base-v2'
    ;;
  --mimic)
    python pre_comp_emb.py --dataset cls/mimic --model_name_or_path 'kamalkraj/BioSimCSE-BioLinkBERT-BASE'
    ;;
  --luckycat37)
    python pre_comp_emb.py --dataset  'cls/luckycat37' --model_name_or_path 'sentence-t5-base'
    ;;
  --asylex)
    python pre_comp_emb.py --dataset asylex --model_name_or_path 'sentence-t5-base'
    ;;
  --aaai)
    CUDA_VISIBLE_DEVICES=7 python pre_comp_emb.py --dataset aaai --model_name_or_path 'sentence-t5-base'
    ;;
  --n2c2_2008)
    python pre_comp_emb.py --dataset cls/n2c2_2008 --model_name_or_path 'kamalkraj/BioSimCSE-BioLinkBERT-BASE'
    ;;
  *)
    echo "Invalid dataset. Available datasets are: --openreview, --pubmed, --yelp, --psytar, --hallmarks_of_cancer"
    exit 1
    ;;
esac
