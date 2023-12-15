# Can Wikipedia Teach Us About the World?
## Impact of Language-Pretraining on Model-Based RL

Currently WIP.


Install packages from `requirements.txt`

Execute the following command to run an experiment:  
```
python -O twm/main.py --game Breakout --seed 0 --device cuda:0 --cpu_p 1.0 --wandb disabled 
```

Use `--wandb online` to log the metrics in weights and biases.  
To use other hyperparameters, edit the file `twm/config.py`.