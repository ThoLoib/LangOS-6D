#!/bin/bash

echo "Running descriptions_hcat_attributes.py..."
python3 descriptions_ycbv_attributes.py

echo "Running t2t_attributes_hcat.py..."
(cd object_retrieval && python3 t2t_attributes_ycbv.py)

#echo "Running t2t_attributes_ycbv_gso.py..."
#(cd object_retrieval && python3 t2t_attributes_ycbv_gso.py)

#echo "Running descriptions_hcat_attributes.py..."
#python3 descriptions_ycbv_gso_comma.py

echo "All tasks completed successfully."

