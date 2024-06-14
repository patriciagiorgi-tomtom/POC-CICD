import mlflow.pyfunc
from mmdet.apis import DetInferencer


class MMDetectionModel(mlflow.pyfunc.PythonModel):
    def __init__(self, inferencer, registry_parameters):
        self.inferencer = inferencer
        self.registry_parameters = registry_parameters

    def predict(self, context, model_input, params=None):
        return self.inferencer(model_input)
