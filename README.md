# Can Wikipedia Teach Us About the World?
## Impact of Language-Pretraining on Model-Based RL

Currently WIP.


Install packages from `requirements.txt`

Execute the following command to run an experiment:  
```
python -O twm/main.py --game CartPole --seed 0 --device cuda:0 --cpu_p 1.0 --config cartpole --wandb disabled 
```

Use `--wandb online` to log the metrics in weights and biases.  
To use other hyperparameters, edit the file `twm/config.py`.

See `analysis` folder for notebooks used to generate analysis for the report.