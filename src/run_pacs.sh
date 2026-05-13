python -m fedlerate.cli.main partition -d PACS-strict

strategies=(
    "fedavg" 
    "fedprox" 
    "fedper" 
    "ditto"
    "ifca" 
    "per_ifca"
    "fcfl" 
    "per_fcfl" 
    "fedmix"
)

for strategy in ${strategies[@]}; do
    fedlerate-train --config-dir ../tests/configs/pacs --config-name $strategy
done;