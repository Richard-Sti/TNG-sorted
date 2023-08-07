env="/mnt/zfsusers/rstiskalek/TNG-sorted/venv_tngsorted/bin/python3"

minpart=99999
mem=150


cm="addqueue -q berg -n 1 -m $mem $env main.py --minpart $minpart"

echo $cm
$cm
