import yaml


class ConfigLoader:
    def __init__(self, environment):
        self.environment = environment
        self.config = self.load_config()

    def load_config(self):
        config_file = f'config/settings.{self.environment}.yaml'
        print(f'Loading settings from {config_file}')
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def get_config(self):
        return self.config
