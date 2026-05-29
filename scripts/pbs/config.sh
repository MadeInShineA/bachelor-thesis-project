#!/bin/bash
# Please note that this script was provided by Ayumu-san
## This script need two inputs, the first input is job_dir, the second input is the number of max job to throw
username="olivier.amacker"

case "$(uname -n)" in
  "pine00")
    NODENAME="pine"
    NODE=(02 03 04 06 07 09 10 12 13 14 15 16)
    # NODE=(02 03 06 07 09 10 11 12 13 14 15 16)
    declare -A NODE_CORES=( ["02"]=8 ["03"]=8 ["04"]=8 ["05"]=8 ["06"]=8 ["07"]=8 ["08"]=8 ["09"]=8 ["10"]=8 ["11"]=8 ["12"]=8 ["13"]=8 ["14"]=8 ["15"]=8 ["16"]=8 )
    MAXJOB=96
    ;;
  "beers")
    NODENAME="cns-node"
    # NODE=(01 02 04 05 06 07 08 09 10 11 12 13 14 15 16)
    # NODE=(01 02 04 05 06 08 09 10 11 12 13 14 15 16)
    NODE=(01 02 04 05 06 08 10 11 12 13 14 15 16)
    declare -A NODE_CORES=( ["01"]=24 ["02"]=24 ["03"]=24 ["04"]=24 ["05"]=24 ["06"]=24 ["07"]=24 ["08"]=24 ["09"]=24 ["10"]=24 ["11"]=24 ["12"]=24 ["13"]=24 ["14"]=24 ["15"]=24 ["16"]=24 )
    MAXJOB=128
    ;;
  "pnp01")
    NODENAME="pnp-node"
    NODE=(01 02 03 04 05 06 07 08 09 10)
    # NODE=(01 02 03)
    declare -A NODE_CORES=( ["01"]=24 ["02"]=24 ["03"]=24 ["04"]=48 ["05"]=64 ["06"]=64 ["07"]=64 ["08"]=64 ["09"]=64 ["10"]=64 )
    MAXJOB=136
    # MAXJOB=68
    ;;
  *)
    echo "Unknown host"
    exit 1
    ;;
esac

FIRSATNODE=${NODE[0]}
NUMNODE=${#NODE[@]}
