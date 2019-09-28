#!/bin/bash

function download_gd() {
  fileid=$1
  filename=$2
  curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
  CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
  curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${fileid}" -o ${filename}
}

fileid="1DsBAUcxYCOqnSggwKuyF-86qei7vGgu0"
filename="weights.28-3.73.hdf5"
download_gd $fileid $filename


