Run experiment in background:

`nohup python -u sheeprl.py exp=dreamer_v3_XL_crafter fabric.accelerator=gpu algo.total_steps=200000 > sheeprl_output.log 2>&1 &`

Monitor stdout in realtime:

`tail -f sheeprl_output.log`

Show gpu utilization:

`nvidia-smi`

SSH with port-forwarding tunnel:

`ssh -L 6006:localhost:6006 ***@***.***.***.***`

Run tensorboard:

`tensorboard --logdir path/to/logdir --port 6006`