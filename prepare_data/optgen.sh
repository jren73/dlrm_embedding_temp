#!/bin/bash
#generate training data in parallel
cache_ratio=$1
INPUT=$2

#echo "$cache_ratio"
#echo "$2"
SUBSTRING=$(echo $INPUT| cut -d'_' -f 6)
m=${#INPUT}
n=${#SUBSTRING}
pos=$(($m-$n))
file_path="${INPUT:0:$((pos))}"
echo "$file_path"
for k in $( seq 0 70 )
do
  echo "$k"
  file=${k}".txt"
  file=$file_path$file
  echo "$file"
  screen_name="my_screen"
  screen -dmS $screen_name${k}

  cmd=$"python3 optgen.py "$cache_ratio" "$file;

  echo "$cmd"

  screen -x -S $screen_name${k} -p 0 -X stuff "$cmd"
  screen -x -S $screen_name${k} -p 0 -X stuff $'\n'

done
#to kill all screen, use killall screen
