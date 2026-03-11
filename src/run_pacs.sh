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
    python test_app.py --config-path ../tests/configs/pacs --config-name $strategy
done;