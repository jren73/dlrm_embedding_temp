#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "Syntax: run.sh [-h|c|p|e]"
   echo "options:"
   echo "c     Train cache model from scratch."
   echo "p     Train prefetch model from scratch."
   echo "e     Train model from existing checkpoint."
   echo "h     Print this Help."
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################
chk=0
while getopts ":hcpe:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      c) # Enter a name
         model_type=0;;
      p) model_type=1;;
      e) chk=1;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done


if [ "$model_type" -eq 0 ]
then
      if [ "$chk" -eq 0 ] 
      then
         python train_caching.py --config=example_caching.json --traceFile=dataset/sample_0 --model_type=0 --infFile=dataset/dataset_0_sampled_80_4.txt
      else
         python train_caching.py --config=example_caching.json --traceFile=dataset/sample_0 --model_type=1 --infFile=dataset/dataset_0_sampled_80_4.txt
      fi

else
      if [ "$chk" -eq 0 ] 
      then
         python train_prefetch.py --config=example_prefetching.json --traceFile=dataset/sample_0 --model_type=0 --infFile=dataset/dataset_0_sampled_80_4.txt
      else
         python train_prefetch.py --config=example_prefetching.json --traceFile=dataset/sample_0 --model_type=1 --infFile=dataset/dataset_0_sampled_80_4.txt
      fi
fi




