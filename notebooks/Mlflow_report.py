from mlflow.tracking import MlflowClient
import mlflow
import sys
from io import StringIO
import modelling_utils as mutils
from sklearn.metrics import classification_report

class Mlflow_report():
     
    def __init__(self):
        self.client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
        # mlflow.set_tracking_uri(r"C:\Users\felix\Documents\jul24_bds_int_heartbeat\notebooks\mlflow")


    def start_server(self):
        #!mlflow server --host 0.0.0.0 --port 8080
        print("starting server not possible from Class.")
        print("Use: !mlflow server --host 0.0.0.0 --port 8080 in notebook und interrupt it!")
        print("Adress: http://127.0.0.1:8080")
        
    def set_tracking_uri(self):
        mlflow.set_tracking_uri("http://localhost:8080")
    
    def report_autolog(self, disable=False):
        """
        Starts autologing of training. 
            disable: True: disables autologin, False: Enabeles autologin
        """
        # mlflow.autolog(disable=disable)
        mlflow.autolog(disable=disable, exclusive=True, log_datasets=False)

    def rename(self, name):
        """ Changes the name of the last Run to 'name' (str) """
        last_runID = mlflow.search_runs().loc[0, "run_id"]
        self.client.set_tag(last_runID, "mlflow.runName", name)
    
    def performancesummary(self, y_train_labels, y_pred_train, y_test_labels, y_pred_test):
        """ 
        Adds the performance_summary as describtion 
        """

        # Model summary is useless, as it will be saved in the artifacts
        # Save Model Summary as model describtion
        # s = StringIO()
        # model.summary(print_fn=lambda x: s.write(x + '\n'))
        # model_summary = s.getvalue()
        # s.close()
        # self.client.set_tag(last_runID, "mlflow.note.content", model_summary)

        # Save performance_summary string als description
        # Create Classification Report and Crosstab
        # formating is strange in mlflow - some spaces are needed after a new line (1 to 4 or more)
        performance_summary = ''
        performance_summary += "Classification Report TRAIN:\n \n" + classification_report(y_train_labels, y_pred_train, digits=3).replace("\n", "\n ") + " \n "
        performance_summary += ' ' + "- "*53 + ' \n '
        performance_summary += "Classification Report TEST: \n \n " + classification_report(y_train_labels, y_pred_train, digits=3).replace("\n", "\n ")
        performance_summary += "- "*53 + ' \n '

        # get crosstab as string from print-function
        old_stdout = sys.stdout  
        result = StringIO()
        sys.stdout = result
        result_string = print(mutils.print_crosstab(y_train_labels, y_pred_train, y_test_labels, y_pred_test, normalize=False))
        string = result.getvalue()
        sys.stdout = old_stdout

        # It seems that mlflow needs a few spaces after a new line for the write formatting
        performance_summary += "Crosstab: \n \n      " + string.replace("\n", "\n      ")
        performance_summary = performance_summary.replace("\n weighted avg   ", "\n    weighted avg")

        last_runID = mlflow.search_runs().loc[0, "run_id"]
        self.client.set_tag(last_runID, "mlflow.note.content", performance_summary)

        # try: 
        #     mlflow.end_run()
        # except:
        #     print("nothing")
        # # Save the name of the "Sequential"-layer of the DNN
        # with mlflow.start_run(run_id=last_runID) as run:
        #     mlflow.log_params({"Modelname": model.name})

        return performance_summary


    def add_param(self, param_dict):
        # Saves parameter to mlflow - (Dictionary!)
        last_runID = mlflow.search_runs().loc[0, "run_id"]

        try: 
            mlflow.end_run()
        except:
            print("nothing")
        # Save the name of the "Sequential"-layer of the DNN
        with mlflow.start_run(run_id=last_runID) as run:
            mlflow.log_params(param_dict)



    def report_manually(self, history, run_name, artifact_path, epochs, batch_size, model, X_test):
        """ Not working 100% """
        # Define Parameter-Dict to be saved
        params = {"optimizer": history.model.optimizer.name, 
                  "learning_rate": history.model.optimizer.learning_rate.value.numpy(),
                  "epochs": epochs,
                  "batch_size": batch_size}
        
        # Define Metrics-Dict to be saved
        metrics = history.model.get_metrics_result()
        # Metric must be a scaler - compress f1_scores to be a scaler (min of all)
        if "f1_score" in history.model.get_metrics_result().keys():
            metrics["f1_score_min"] = np.min(metrics["f1_score"])
            del metrics["f1_score"]

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(sk_model=model, input_example=X_test, artifact_path=artifact_path)
    

    def log_history_figure(self, fig, name="Train History"):
        if name[-5:] != ".html":
            name = name + ".html"
        last_runID = mlflow.search_runs().loc[0, "run_id"]
        try: 
            mlflow.end_run()
        except:
            print("nothing")

        with mlflow.start_run(run_id=last_runID) as run:
            mlflow.log_figure(fig, name)

    def end_run(self):
        try: 
            mlflow.end_run()
        except:
            print("not running")

    def save_as_dataframe(self, filepath=None):
        """ Returns the Database as a Dataframe and saves it (if filepath != None) """
        # filepath = "..\\reports\\Streamlit\\mlflow_database"
        data = mlflow.search_runs()
        if filepath is not None:
            data.to_pickle(filepath)
        return data