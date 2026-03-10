python -m fedlerate.cli.main partition -d MNIST

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
    python test_app.py --config-path ../tests/configs/mnist --config-name $strategy
done;