#!/bin/bash

DUTS_TRAIN_URL="http://saliencydetection.net/duts/download/DUTS-TR.zip"
DUTS_TEST_URL="http://saliencydetection.net/duts/download/DUTS-TE.zip"
MSRA10K_URL="http://mftp.mmcheng.net/Data/MSRA10K_Imgs_GT.zip"
PASCALS_URL="https://cbs.ic.gatech.edu/salobj/download/salObj.zip"

dataset_dir="data_temp"
mkdir -p $dataset_dir

download_and_extract() {
  url=$1
  filename=$(basename $url)

  echo "Downloading $filename..."
  curl -L -o "$dataset_dir/$filename" $url

  echo "Extracting $filename..."
  unzip -q "$dataset_dir/$filename" -d $dataset_dir
  rm "$dataset_dir/$filename"
}

download_duts() {
  download_and_extract $DUTS_TRAIN_URL
  mv "$dataset_dir/DUTS-TR/DUTS-TR-Image" "$dataset_dir/DUTS-TR/images"
  mv "$dataset_dir/DUTS-TR/DUTS-TR-Mask" "$dataset_dir/DUTS-TR/masks"
  mv "$dataset_dir/DUTS-TR" "$dataset_dir/duts_train_data"
  
  download_and_extract $DUTS_TEST_URL
  mv "$dataset_dir/DUTS-TE/DUTS-TE-Image" "$dataset_dir/DUTS-TE/images"
  mv "$dataset_dir/DUTS-TE/DUTS-TE-Mask" "$dataset_dir/DUTS-TE/masks"
  mv "$dataset_dir/DUTS-TE" "$dataset_dir/duts_test_data"
}

download_msra() {
  download_and_extract $MSRA10K_URL
  rm -f "$dataset_dir/Readme.txt"
  mkdir -p "$dataset_dir/MSRA10K_Imgs_GT/masks"
  mv "$dataset_dir/MSRA10K_Imgs_GT/Imgs/"*.png "$dataset_dir/MSRA10K_Imgs_GT/masks"
  mv "$dataset_dir/MSRA10K_Imgs_GT/Imgs" "$dataset_dir/MSRA10K_Imgs_GT/images"
  mv "$dataset_dir/MSRA10K_Imgs_GT" "$dataset_dir/msra_data"
}

download_pascals() {
  download_and_extract $PASCALS_URL
  rm -rf "$dataset_dir/algmaps" "$dataset_dir/benchmark" "$dataset_dir/code" "$dataset_dir/results" \
        "$dataset_dir/readme.pdf" "$dataset_dir/tips_for_matlab.txt" "$dataset_dir/datasets/fixations" \
        "$dataset_dir/datasets/segments" "$dataset_dir/datasets/imgs/bruce" "$dataset_dir/datasets/imgs/cerf" \
        "$dataset_dir/datasets/imgs/ft" "$dataset_dir/datasets/imgs/judd" "$dataset_dir/datasets/imgs/pascal" \
        "$dataset_dir/datasets/masks/bruce" "$dataset_dir/datasets/masks/ft" "$dataset_dir/datasets/masks/pascal"
  mv "$dataset_dir/datasets/imgs/imgsal"/* "$dataset_dir/datasets/imgs"
  mv "$dataset_dir/datasets/masks/imgsal"/* "$dataset_dir/datasets/masks"
  rm -rf "$dataset_dir/datasets/imgs/imgsal" "$dataset_dir/datasets/imgs/Thumbs.db" "$dataset_dir/datasets/masks/imgsal"
  mv "$dataset_dir/datasets/imgs" "$dataset_dir/datasets/images"
  mv "$dataset_dir/datasets" "$dataset_dir/pascals_data"
}

usage() {
  echo "Usage: $0 [-d] [-m] [-p]"
  echo "  -d    Download DUTS dataset (train and test)"
  echo "  -m    Download MSRA10K dataset"
  echo "  -p    Download Pascal-S dataset"
  echo "If no options are provided, all datasets will be downloaded."
  exit 1
}

all=false
while getopts "dmp" opt; do
  case $opt in
    d)
      download_duts
      ;;
    m)
      download_msra
      ;;
    p)
      download_pascals
      ;;
    *)
      usage
      ;;
  esac
done

# Check if no options were provided
if [ $OPTIND -eq 1 ]; then
  echo "No options provided; downloading all datasets."
  download_duts
  download_msra
  download_pascals
fi