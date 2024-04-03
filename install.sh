#!/bin/bash

# Made by Pelochus
# Check for more info: https://github.com/Pelochus/ezrknn-llm/

echo
echo "#########################################"
echo "Checking root permission..."
echo "#########################################"
echo

if [ "$EUID" -ne 0 ]; then 
  echo "Please run this script as root!"
  exit
fi

echo
echo "#########################################"
echo "Cloning repository..."
echo "#########################################"
echo

git clone https://github.com/Pelochus/ezrknn-llm.git
cd ezrknn-llm/

echo
echo "#########################################"
echo "Installing RKNN LLM..."
echo "#########################################"
echo

cp ./rkllm-runtime/runtime/Linux/librkllm_api/aarch64/* /usr/lib
cp ./rkllm-runtime/runtime/Linux/librkllm_api/include/* /usr/local/include

echo
echo "#########################################"
echo "Done!"
echo "#########################################"
echo
