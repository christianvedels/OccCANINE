import numpy as np
from scipy.stats import mode

from .formatter import (
    hisco_blocky5,
    construct_general_purpose_formatter
)

class EvalEngine:
    '''
    Evaluate the performance of a model
    model: model object (instance of OccCANINE)
    ground_truth: pd.DataFrame
    predicitons: pd.DataFrame
    pred_col: str (prefix of the columns contain the predictions)
    '''

    def __init__(self, model, ground_truth, predicitons, pred_col, digits = None):
        """
        Initialize the evaluation metrics class.
        Args:
            model (object): The model object containing formatter, block_size, and system attributes.
            ground_truth (pd.DataFrame): The ground truth data.
            predicitons (pd.DataFrame): The predicted data.
            pred_col (str): The prefix of the columns to be used for predictions and ground truth.
            digits (int, optional): The number of digits to use for each occupational code. Defaults to None (which then includes all).
        Attributes:
            pred_col (str): The prefix of the columns to be used for predictions and ground truth.
            y_pred (pd.DataFrame): The filtered predictions based on pred_col.
            y_true (pd.DataFrame): The filtered ground truth based on pred_col.
            formatter (object): The formatter object from the model.
            block_size (int): The block size from the model's formatter.
            system (object): The system object from the model.
            use_within_block_sep (bool): A flag indicating whether to use within block separation.
        """
        self.pred_col = pred_col

        # Extract cols starting with pred_col
        self.y_pred = predicitons.filter(regex=pred_col)
        self.y_true = ground_truth.filter(regex=pred_col)

        # No label
        self.no_label = ground_truth.filter(regex=pred_col).shape[1] == 0

        # Get things from model
        self.formatter = model.formatter
        self.block_size = model.formatter.block_size
        self.system = model.system
        self.use_within_block_sep = False

        # Format
        self.y_pred = self.format(self.y_pred)
        self.y_true = self.format(self.y_true)

        self.digits = digits # All if none

    def update_data(self, ground_truth, predicitons):
        """
        Update the ground truth and predictions data.
        Args:
            ground_truth (pd.DataFrame): The new ground truth data.
            predicitons (pd.DataFrame): The new predicted data.
        """
        self.y_pred = predicitons.filter(regex=self.pred_col)
        self.y_true = ground_truth.filter(regex=self.pred_col)
        self.y_pred = self.format(self.y_pred)
        self.y_true = self.format(self.y_true)

    def format(self, y):
        """
        Formats the columns of the given DataFrame by ensuring that each element in the columns
        has a length equal to `self.block_size`. If an element's length is one less than `self.block_size`,
        a '0' is prepended to the element.
        Parameters:
        y (pd.DataFrame): The DataFrame to be formatted. Each element in the DataFrame is expected to be
                          convertible to a string.
        Returns:
        pd.DataFrame: The formatted DataFrame with updated column values.
        """
        y = y.astype(str)
        for j in y.columns:
            col_j = y[j].tolist()
            for i in range(len(col_j)):
                if len(col_j[i]) == (self.block_size-1):
                    col_j[i] = '0' + col_j[i]
                if len(col_j[i]) == 2:
                    if col_j[i][0] == '-':
                        # Should handle e.g. -1, -2, -3, etc. to become -0001, -0002, -0003
                        col_j[i] = col_j[i][0] + '0' * (self.block_size-2) + col_j[i][1]

            # Store updated column
            y.loc[:, j] = col_j

        return y

    def _acc(self, y_true, y_pred, digits = None):
        '''
        Calculate the accuracy of **one observation**.
        The accuracy is determined by averaging the proportion of predictions
        that are in the true values and the proportion of true values that are
        in the predictions, divided by the maximum number of predictions or true values.
        Parameters:
            y_true (list): A list of true values.
            y_pred (list): A list of predicted values.
            digits (int, optional): The number of digits to use of each occupational code. Defaults to None.
        Returns:
            float: The accuracy score of the observation.
        '''
        if self.no_label:
            return float('NaN')

        # Remove digits
        if digits is not None:
            y_true = [x[:digits] for x in y_true]
            y_pred = [x[:digits] for x in y_pred]

        # Discard duplicates in labels (and pred, though this does not tend to occur)
        y_true = list(set(y_true))
        y_pred = list(set(y_pred))

        # Count number of y_pred in y_true
        pred_in_true = sum([x in y_true for x in y_pred])

        # Count number of y_true in y_pred
        true_in_pred = sum([x in y_pred for x in y_true])

        # Max number of preds
        max_preds = max(len(y_true), len(y_pred))

        # average of the two above divided by max
        res = (pred_in_true + true_in_pred) / (2 * max_preds)

        return res

    def accuracy(self, return_per_obs=False):
        """
        Calculate the accuracy of predictions.
        This method iterates over the true and predicted values, cleans the data by removing NaNs and empty strings,
        and then checks if any of the true codes are in the predicted codes. The accuracy is computed as the ratio
        of correct predictions to the total number of predictions.
        Args:
            return_per_obs (bool, optional): If True, returns the accuracy for each observation. Defaults to False.
        Returns:
            float: The accuracy of the predictions.
        """
        if self.no_label:
            if return_per_obs:
                return float('NaN')*len(self.y_pred)
            else:
                return float('NaN')

        correct_predictions = 0
        per_obs_accuracy = []
        for i in range(len(self.y_true)):
            y_true_i = self.y_true.iloc[i]
            y_pred_i = self.y_pred.iloc[i]

            # To list
            y_true_i = y_true_i.tolist()
            y_pred_i = y_pred_i.tolist()

            # Remove NaN
            y_true_i = [x for x in y_true_i if str(x) != 'nan']
            y_pred_i = [x for x in y_pred_i if str(x) != 'nan']

            # Remove empty strings
            y_true_i = [x for x in y_true_i if x != " "]
            y_pred_i = [x for x in y_pred_i if x != " "]

            # Check if any of the true codes are in the predicted codes
            acc = self._acc(y_true_i, y_pred_i, self.digits)
            correct_predictions += acc
            per_obs_accuracy.append(acc)

        if return_per_obs:
            return per_obs_accuracy
        else:
            return correct_predictions / len(self.y_true)

    def _prec(self, y_true, y_pred, digits = None):
        """
        Calculate the precision of **one obsertion**.

        Precision is the ratio of correctly predicted positive observations to the total predicted positives.

        Args:
            y_true (list): The list of true values.
            y_pred (list): The list of predicted values.

        Returns:
            float: The precision score. If `y_pred` is empty, returns 0.
        """
        if self.no_label:
            return float('NaN')

        # Remove digits
        if digits is not None:
            y_true = [x[:digits] for x in y_true]
            y_pred = [x[:digits] for x in y_pred]

        # Discard duplicates in labels (and pred, though this does not tend to occur)
        y_true = list(set(y_true))
        y_pred = list(set(y_pred))

        if len(y_pred) == 0:
            return 0

        pred_in_true = sum([1 for pred in y_pred if pred in y_true])
        return pred_in_true / len(y_pred)

    def precision(self, return_per_obs=False):
        """
        Calculate the precision metric for the given true and predicted values.
        This method iterates over each observation in the true and predicted values,
        removes NaN values and empty strings, and then calculates the precision for
        each observation. The final precision is the average precision over all observations.
        Args:
            return_per_obs (bool, optional): If True, returns the precision for each observation. Defaults to False.
        Returns:
            float or list: The average precision over all observations or a list of precision values per observation.
        """
        if self.no_label:
            if return_per_obs:
                return float('NaN')*len(self.y_pred)
            else:
                return float('NaN')

        total_precision = 0
        per_obs_precision = []
        for i in range(len(self.y_true)):
            y_true_i = self.y_true.iloc[i]
            y_pred_i = self.y_pred.iloc[i]

            # To list
            y_true_i = y_true_i.tolist()
            y_pred_i = y_pred_i.tolist()

            # Remove NaN
            y_true_i = [x for x in y_true_i if str(x) != 'nan']
            y_pred_i = [x for x in y_pred_i if str(x) != 'nan']

            # Remove empty strings
            y_true_i = [x for x in y_true_i if x != " "]
            y_pred_i = [x for x in y_pred_i if x != " "]

            # Calculate precision for this observation
            prec = self._prec(y_true_i, y_pred_i, self.digits)
            total_precision += prec
            per_obs_precision.append(prec)

        if return_per_obs:
            return per_obs_precision
        else:
            return total_precision / len(self.y_true)

    def recall(self, return_per_obs=False):
        """
        Calculate the average recall for all observations.
        This method iterates over each pair of true and predicted values, processes them to remove NaNs and empty strings,
        and then calculates the recall for each observation using the `_recall` method. The average recall is then computed
        by dividing the total recall by the number of observations.
        Args:
            return_per_obs (bool, optional): If True, returns the recall for each observation. Defaults to False.
        Returns:
            float or list: The average recall across all observations or a list of recall values per observation.
        """
        if self.no_label:
            if return_per_obs:
                return float('NaN')*len(self.y_pred)
            else:
                return float('NaN')

        total_recall = 0
        per_obs_recall = []
        for i in range(len(self.y_true)):
            y_true_i = self.y_true.iloc[i]
            y_pred_i = self.y_pred.iloc[i]

            # To list
            y_true_i = y_true_i.tolist()
            y_pred_i = y_pred_i.tolist()

            # Remove NaN
            y_true_i = [x for x in y_true_i if str(x) != 'nan']
            y_pred_i = [x for x in y_pred_i if str(x) != 'nan']

            # Remove empty strings
            y_true_i = [x for x in y_true_i if x != " "]
            y_pred_i = [x for x in y_pred_i if x != " "]

            # Calculate recall for this observation
            rec = self._recall(y_true_i, y_pred_i, self.digits)
            total_recall += rec
            per_obs_recall.append(rec)

        if return_per_obs:
            return per_obs_recall
        else:
            return total_recall / len(self.y_true)

    def _recall(self, y_true, y_pred, digits = None):
        """
        Calculate the recall metric of *one observation*.

        Recall is the ratio of the number of relevant instances that have been retrieved
        over the total number of relevant instances. It is also known as sensitivity or
        true positive rate.

        Args:
            y_true (list): A list of true labels.
            y_pred (list): A list of predicted labels.

        Returns:
            float: The recall value. If y_true is empty, returns 0.
        """
        if self.no_label:
            return float('NaN')

        # Remove digits
        if digits is not None:
            y_true = [x[:digits] for x in y_true]
            y_pred = [x[:digits] for x in y_pred]

        # Discard duplicates in labels (and pred, though this does not tend to occur)
        y_true = list(set(y_true))
        y_pred = list(set(y_pred))

        if len(y_true) == 0:
            return 0
        true_in_pred = sum([1 for true in y_true if true in y_pred])
        return true_in_pred / len(y_true)

    def f1(self, return_per_obs=False):
        """
        Calculate the average F1 score for all observations in the dataset.
        The method iterates over each observation in `y_true` and `y_pred`, processes the lists to remove NaN values and empty strings,
        and then calculates the F1 score for each observation using the `_f1` method. The average F1 score is then computed and returned.
        Args:
            return_per_obs (bool, optional): If True, returns the F1 score for each observation. Defaults to False.
        Returns:
            float or list: The average F1 score for all observations or a list of F1 scores per observation.
        """
        if self.no_label:
            if return_per_obs:
                return float('NaN')*len(self.y_pred)
            else:
                return float('NaN')

        total_f1 = 0
        per_obs_f1 = []
        for i in range(len(self.y_true)):
            y_true_i = self.y_true.iloc[i]
            y_pred_i = self.y_pred.iloc[i]

            # To list
            y_true_i = y_true_i.tolist()
            y_pred_i = y_pred_i.tolist()

            # Remove NaN
            y_true_i = [x for x in y_true_i if str(x) != 'nan']
            y_pred_i = [x for x in y_pred_i if str(x) != 'nan']

            # Remove empty strings
            y_true_i = [x for x in y_true_i if x != " "]
            y_pred_i = [x for x in y_pred_i if x != " "]

            # Calculate F1 score for this observation
            f1_score = self._f1(y_true_i, y_pred_i, self.digits)
            total_f1 += f1_score
            per_obs_f1.append(f1_score)

        if return_per_obs:
            return per_obs_f1
        else:
            return total_f1 / len(self.y_true)

    def _f1(self, y_true, y_pred, digits = None):
        """
        Calculate the F1 score for a single observation.
        The F1 score is the harmonic mean of precision and recall, providing a balance between the two metrics.
        It is particularly useful for imbalanced datasets where one class is more frequent than the other.
        Args:
            y_true (int): The ground truth label.
            y_pred (int): The predicted label.
        Returns:
            float: The F1 score. If both precision and recall are zero, returns 0.
        """
        if self.no_label:
            return float('NaN')

        # Remove digits
        if digits is not None:
            y_true = [x[:digits] for x in y_true]
            y_pred = [x[:digits] for x in y_pred]

        # Discard duplicates in labels (and pred, though this does not tend to occur)
        y_true = list(set(y_true))
        y_pred = list(set(y_pred))

        # Calculate precision and recall for this observation
        precision = self._prec(y_true, y_pred, self.digits)
        recall = self._recall(y_true, y_pred, self.digits)

        # Calculate F1 score for this observation
        if precision + recall == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)
