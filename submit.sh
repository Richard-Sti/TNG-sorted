env="/mnt/zfsusers/rstiskalek/TNG-sorted/venv_tngsorted/bin/python3"

minpart=999
mem=16


cm="addqueue -q berg -n 1 -m $mem $env main.py --minpart $minpart"

echo $cm
$cm
