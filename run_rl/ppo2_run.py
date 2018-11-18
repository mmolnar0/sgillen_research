import baselines.run

arg_dict = {}
arg_dict["alg"] = "ppo2"
arg_dict["env"] = "Acrobot-v1"
arg_dict["num_timsteps"] = "0"
arg_dict["load_path"] = "~/work_dir/"

baselines.run.main()