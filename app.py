import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split


class HyperparameterTuningApp:
    def __init__(self):
        """
        Initializes the HyperparameterTuningApp class.

        Attributes:
        - dataset_name (str): Name of the selected dataset.
        - classifier (str): Name of the selected classifier.
        - cv_count (int): Number of cross-validation folds.
        - X (numpy.ndarray or pandas.DataFrame): Input features of the dataset.
        - y (numpy.ndarray or pandas.Series): Target variable of the dataset.
        - model (type): Chosen classifier model.
        - parameters (dict): Hyperparameter grid to search over.
        - clf (sklearn.model_selection.GridSearchCV): GridSearchCV instance for hyperparameter tuning.
        """
        self.dataset_name = None
        self.classifier = None
        self.cv_count = None
        self.X = None
        self.y = None
        self.model = None
        self.parameters = None
        self.clf = None

    def run(self):
        """
        Runs the HyperparameterTuningApp.

        This method executes the main logic of the application by calling other methods
        in a specific sequence to set up the app and perform hyperparameter tuning.
        """
        self.set_page_config()
        self.show_title()
        self.select_dataset()
        self.select_classifier()
        self.select_cv_count()
        self.load_dataset()
        self.split_dataset()
        self.display_dataset_info()
        self.get_classifier()
        self.run_grid_search()
        self.display_tuning_results()

    def set_page_config(self):
        st.set_page_config(page_title="Hyperparameter Tuning", layout="wide")

    def show_title(self):
        st.title("OptiFine: Fine-Tune Your Models for Peak Performance!")
        st.markdown(
            """

    Welcome to the Hyperparameter Tuning web application! This app demonstrates the power of hyperparameter tuning 
    using `**GridSearchCV**` from `Scikit-learn`. By optimizing the model's hyperparameters, you can achieve better 
    performance on your datasets. \n
    Please note that this app provides a simplified demonstration, and there might be other combinations of 
    parameters and algorithms that can yield even better accuracy scores for a given dataset. Feel free to 
    explore and experiment with different options to enhance your understanding of hyperparameter tuning.
            """
        )

    def select_dataset(self):
        st.sidebar.header("Select Dataset")
        self.dataset_name = st.sidebar.selectbox(
            "", ("Iris Plants", "Wine Recognition")
        )
        st.title("")
        st.write(f"## **{self.dataset_name} Dataset**")

    def select_classifier(self):
        """
        Allows the user to select a classifier.

        This method adds a sidebar section to the web app where the user can select a classifier
        from the given options ("Random Forest", "SVM", "Logistic Regression"). The selected
        classifier name is stored in the `classifier` attribute.
        """
        self.classifier = st.sidebar.selectbox(
            "Select Classifier", ("Random Forest", "SVM", "Logistic Regression")
        )

    def select_cv_count(self):
        self.cv_count = st.sidebar.slider("Cross-validation count", 2, 5, 3)
        st.sidebar.write("---")
        st.sidebar.subheader("Parameters")
        st.sidebar.write("")

    def load_dataset(self):
        if self.dataset_name == "Iris Plants":
            self.X, self.y = datasets.load_iris(return_X_y=True)
        elif self.dataset_name == "Wine Recognition":
            self.X, self.y = datasets.load_wine(return_X_y=True)

    def split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3
        )

    def display_dataset_info(self):
        st.write("Shape of dataset:", self.X.shape)
        st.write("number of classes:", len(np.unique(self.y)))

    def get_classifier(self):
        if self.classifier == "SVM":
            self.get_svm_classifier()
        elif self.classifier == "Random Forest":
            self.get_random_forest_classifier()
        else:
            self.get_logistic_regression_classifier()

    def get_svm_classifier(self):
        st.sidebar.write("**Kernel Type**")
        st.sidebar.write("Specifies the kernel type to be used in the algorithm.")
        kernel_type = st.sidebar.multiselect(
            "", options=["linear", "rbf", "poly"], default=["linear", "rbf", "poly"]
        )
        st.sidebar.subheader("")

        st.sidebar.write("**Regularization Parameter**")
        st.sidebar.write(
            "The strength of the regularization is inversely proportional to C."
        )
        c1 = st.sidebar.slider("C1", 1, 7, 1)
        c2 = st.sidebar.slider("C2", 8, 14, 10)
        c3 = st.sidebar.slider("C3", 15, 20, 20)

        self.parameters = {"C": [c1, c2, c3], "kernel": kernel_type}
        self.model = svm.SVC()

    def get_random_forest_classifier(self):
        st.sidebar.write("**Number of Estimators**")
        st.sidebar.write("The number of trees in the forest.")
        n1 = st.sidebar.slider("n_estimators1", 1, 40, 5)
        n2 = st.sidebar.slider("n_estimators2", 41, 80, 50)
        n3 = st.sidebar.slider("n_estimators3", 81, 120, 100)
        st.sidebar.header("")

        st.sidebar.write("**Max depth**")
        st.sidebar.write(
            "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure."
        )
        md1 = st.sidebar.slider("max_depth1", 1, 7, 1)
        md2 = st.sidebar.slider("max_depth2", 8, 14, 10)
        md3 = st.sidebar.slider("max_depth3", 15, 20, 20)

        self.parameters = {"n_estimators": [n1, n2, n3], "max_depth": [md1, md2, md3]}
        self.model = RandomForestClassifier()

    def get_logistic_regression_classifier(self):
        st.sidebar.write("**Penalty**")
        st.sidebar.write("Used to specify the norm used in the penalization.")
        penalty = st.sidebar.multiselect("", options=["l1", "l2"], default=["l1", "l2"])
        st.sidebar.subheader("")

        st.sidebar.write("**Regularization Parameter**")
        st.sidebar.write(
            "Inverse of regularization strength; must be a positive float."
        )
        c1 = st.sidebar.slider("C1", 0.01, 1.00, 0.05)
        c2 = st.sidebar.slider("C2", 2, 19, 10)
        c3 = st.sidebar.slider("C3", 20, 100, 80, 10)

        self.parameters = {"penalty": penalty, "C": [c1, c2, c3]}
        self.model = LogisticRegression(solver="liblinear", max_iter=200)

    def run_grid_search(self):
        self.clf = GridSearchCV(
            estimator=self.model,
            param_grid=self.parameters,
            cv=self.cv_count,
            return_train_score=False,
        )
        self.clf.fit(self.X, self.y)

    def display_tuning_results(self):
        df = pd.DataFrame(self.clf.cv_results_)

        st.header("Tuning Results")
        results_df = st.multiselect(
            "",
            options=[
                "mean_fit_time",
                "std_fit_time",
                "mean_score_time",
                "std_score_time",
                "std_test_score",
                "rank_test_score",
                "mean_train_score",
                "std_train_score",
            ],
            default=[
                "mean_score_time",
                "std_score_time",
            ],
        )
        test_score_columns = [
            col for col in df.columns if "split" in col and "test_score" in col
        ]
        results_df += test_score_columns
        df_results = df[results_df]
        st.write(df_results)

        st.subheader("**Parameters and Mean test score**")
        params_df = df[["params", "mean_test_score"]].copy()
        params_df["mean_test_score"] = params_df["mean_test_score"].map("{:.3f}".format)
        params_df = params_df.join(pd.json_normalize(params_df["params"])).drop(
            columns=["params"]
        )

        # Move "Mean Test Score" column to the last position
        mean_test_score_col = params_df.pop("mean_test_score")
        params_df["Mean Test Score"] = mean_test_score_col
        st.dataframe(params_df)

        # Additional evaluation metrics
        st.subheader("Additional Evaluation Metrics")
        if len(test_score_columns) > 0:
            y_pred = self.clf.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1 = f1_score(self.y_test, y_pred, average="weighted")

            metrics_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Score": [accuracy, precision, recall, f1],
            }

            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)

        st.subheader("Best Parameters")
        best_params = self.clf.best_params_
        best_params_str = "\n".join(
            [f"{key}: {value}" for key, value in best_params.items()]
        )
        st.info(best_params_str)

        st.subheader("Best Score")
        best_score = "{:.3f}".format(self.clf.best_score_)
        st.info(best_score)


if __name__ == "__main__":
    app = HyperparameterTuningApp()
    app.run()
