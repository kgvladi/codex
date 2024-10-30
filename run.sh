#!/bin/bash
file="list"
while read -r line
 do
#   python3.10 lymph_node_seg_pat.py $line &
   python3.10 ln_seg_cell.py $line &
 done < "$file"
