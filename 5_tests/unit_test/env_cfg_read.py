import json

with open("3_configurations/environment_configuration/EnvCfg.json", 'r') as file:
            
    epoch_data = json.loads(file.read())


print(epoch_data["Experiment"][0]["Path"][0])