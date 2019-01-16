import os
import json
import pandas as pd

class RayResultsParser():
    def __init__(self):
        pass

    def _load_run(self, path):

        result_file = os.path.join(path, "result.json")

        if not os.path.exists(result_file):
            return None

        with open(result_file,'r') as f:
            lines = f.readlines()

        if len(lines) > 0:
            return json.loads(lines[-1])
        else:
            return None

    def _load_all_runs(self, path):
        runs = os.listdir(path)

        result_list = list()
        for run in runs:
            runpath = os.path.join(path, run)

            run = self._load_run(runpath)
            if run is None:
                continue
            else:
                result = run

            config = result.pop("config")

            # merge result dict and config dict
            for key, value in config.items():
                result[key] = value

            result_list.append(result)

        return result_list

    def _get_n_best_runs(self,
                         experimentpath,
                         n=5,
                         group_by=["hidden_dims", "learning_rate", "num_rnn_layers"]):

        resultlist = self._load_all_runs(experimentpath)

        if len(resultlist) == 0:
            print("Warning! Experiment {} returned no runs".format(experimentpath))
            return None

        result = pd.DataFrame(resultlist)
        # average accuracy over the same columns (particularily over the fold variable...)
        grouped = result.groupby(group_by)["accuracy"]

        nfolds = grouped.count().rename("nfolds")
        mean_accuracy = grouped.mean().rename("mean_accuracy")
        std_accuracy = grouped.std().rename("std_accuracy")

        score = pd.concat([mean_accuracy, std_accuracy, nfolds], axis=1)

        top = score.nlargest(n, "mean_accuracy")
        top["runs"] = len(score)

        dataset = os.path.basename(experimentpath)
        top.reset_index(inplace=True)
        top["dataset"] = dataset

        return top

    def get_sota_experiment(self, path, outpath=None):
        data = self._load_all_runs(path)
        data = pd.DataFrame(data).set_index(["dataset", "earliness_factor"])
        data[["accuracy", "earliness"]].to_csv(outpath)



    def get_best_hyperparameters(self, path, outpath=None, group_by=["hidden_dims", "learning_rate", "num_rnn_layers"]):

        experiments = os.listdir(path)

        best_hyperparams = list()
        for experiment in experiments:

            experimentpath = os.path.join(path,experiment)

            if os.path.isdir(experimentpath):
                print("parsing experiment "+experiment)
                result = self._get_n_best_runs(experimentpath=experimentpath, n=1, group_by=group_by)
                if result is not None:
                    best_hyperparams.append(result)

        summary = pd.concat(best_hyperparams)

        if outpath is not None:
            csvfile = os.path.join(outpath, "hyperparams_conv1d.csv")

            if not os.path.exists(outpath):
                os.makedirs(outpath,exist_ok=True)

            print("writing "+csvfile)
            summary.to_csv(csvfile)

        return summary

def parse_hyperparameters():
    parser = RayResultsParser()
    summary = parser.get_best_hyperparameters("/data/remote/hyperparams_conv1d_v2_secondrun/conv1d",
                                              outpath="/data/remote/hyperparams_conv1d_v2_secondrun/hyperparams.csv",
                                              group_by=["hidden_dims", "learning_rate", "num_layers",
                                                        "shapelet_width_increment"])

    print(summary.set_index("dataset")[["mean_accuracy", "std_accuracy", "runs"]])

if __name__=="__main__":

    parser = RayResultsParser()

    os.makedirs("../viz/data/sota_comparison", exist_ok=True)
    print("writing to ../viz/data/sota_comparison/runs.csv")
    parser.get_sota_experiment("/data/remote/sota_comparison/sota_comparison",
                               outpath="../viz/data/sota_comparison/runs.csv")