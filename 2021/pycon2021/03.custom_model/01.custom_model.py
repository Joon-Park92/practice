import mlflow.pyfunc


# Define the model class
class AddN(mlflow.pyfunc.PythonModel):

    def __init__(self, n):
        self.n = n

    def predict(self, context, model_input):

        # some process using necessary_file
        with open(context.artifacts["necessary_file"], 'r') as f:
            # do sth with necessary file 
            # e.g. processing inputs like vocab mapping
            pass

        return model_input.apply(lambda column: column + self.n)


if __name__ == "__main__":

    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.get_experiment_by_name("Pycon2021")

    with mlflow.start_run(experiment_id=experiment.experiment_id):

        artifacts = {
            "necessary_file" : "./necessary_file",
        }

        # Construct and save the model
        model_path = "add_n_model"
        add5_model = AddN(n=5)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model = add5_model,
            artifacts=artifacts)
