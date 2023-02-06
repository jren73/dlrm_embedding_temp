#!/bin/bash
############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "Syntax: run.sh [-h|c|p]"
   echo "options:"
   echo "c     Train cache model from scratch."
   echo "p     Train cache model from scratch."
   echo "h     Print this Help."
   echo
}

############################################################
############################################################
# Main program                                             #
############################################################
############################################################

while getopts ":hncp:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      n) # Enter a name
         Name=$OPTARG;;
      c) # Enter a name
         model_type=0;;
      p) model_type=1;;
     \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done


if [ "$model_type" -eq 0 ]
then
        python3 train.py --config=example_caching.json --traceFile=dataset/sample_0 --model_type=0 --infFile=dataset/dataset_0_sampled_80_4.txt

else
        python3 train.py --config=example_caching.json --traceFile=dataset/sample_1 --model_type=0 --infFile=dataset/dataset_0_sampled_80_4.txt

fi




