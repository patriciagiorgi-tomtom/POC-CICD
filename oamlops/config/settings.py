import yaml

class ConfigLoader:
    def __init__(self, environment):
        self.environment = environment
        self.config = self.load_config()

    def load_config(self):
        if self.environment == 'dev':
            config_file = 'config/settings.dev.yaml'
        elif self.environment == 'qa':
            config_file = 'config/settings.qa.yaml'
        elif self.environment == 'prod':
            config_file = 'config/settings.prod.yaml'
        else:
            raise ValueError("the environment should be dev, qa or prod")

        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def get_config(self):
        return self.config
