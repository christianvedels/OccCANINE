import numpy as np
import pandas as pd


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


class TopKEvalEngine(EvalEngine):
    '''
    Evaluate the performance of a model for top-k predictions at each rank level (atomistically).

    For top-k predictions, each observation has multiple rows (one per k position).
    This engine evaluates performance at each rank level independently (atomistically),
    meaning rank i only considers the predictions at position i, not cumulative top-i.

    model: model object (instance of OccCANINE)
    ground_truth: pd.DataFrame (original ground truth, one row per observation)
    predictions: pd.DataFrame (top-k predictions with 'top-k-pos' column and rowid/identifier)
    pred_col: str (prefix of the columns containing the predictions)
    group_col: str (column name to group observations by, e.g., 'rowid')
    '''

    def __init__(self, model, ground_truth, predicitons, pred_col, group_col='rowid', digits=None):
        """
        Initialize the top-k evaluation metrics class.

        Args:
            model (object): The model object containing formatter, block_size, and system attributes.
            ground_truth (pd.DataFrame): The ground truth data (one row per observation).
            predicitons (pd.DataFrame): The predicted data with top-k predictions (multiple rows per observation).
            pred_col (str): The prefix of the columns to be used for predictions and ground truth.
            group_col (str): Column name to group observations by (default: 'rowid').
            digits (int, optional): The number of digits to use for each occupational code. Defaults to None.
        """
        # Store group column
        self.group_col = group_col

        # Check if predictions have the group column
        if group_col not in predicitons.columns:
            raise ValueError(f"Predictions must have a '{group_col}' column to group observations")

        # Check if predictions have top-k-pos column
        if 'top-k-pos' not in predicitons.columns:
            raise ValueError("Predictions must have a 'top-k-pos' column for top-k evaluation")

        # Store original predictions for grouping
        self.predictions_topk = predicitons

        # Determine k from the data
        self.k = int(predicitons['top-k-pos'].max() + 1)

        # Store references for reuse
        self.pred_col = pred_col
        self.digits = digits
        self.ground_truth = ground_truth

        # Get things from model for formatting
        self.formatter = model.formatter
        self.block_size = model.formatter.block_size
        self.system = model.system
        self.use_within_block_sep = False

        # Format ground truth and predictions
        y_true_formatted = self.format(ground_truth.filter(regex=pred_col))
        y_pred_formatted = self.format(predicitons.filter(regex=pred_col))

        # Store ground truth with group column for easy lookup
        if group_col in ground_truth.columns:
            self.ground_truth_with_id = ground_truth[[group_col]].copy()
            self.ground_truth_with_id = pd.concat([self.ground_truth_with_id, y_true_formatted], axis=1)
        else:
            raise ValueError(f"Ground truth must have a '{group_col}' column")

        # Store predictions with group column and rank for easy lookup
        self.predictions_with_id = predicitons[[group_col, 'top-k-pos']].copy()
        self.predictions_with_id = pd.concat([self.predictions_with_id, y_pred_formatted], axis=1)
        # Store original index to preserve input order
        self.predictions_with_id['_original_index'] = range(len(self.predictions_with_id))

        # Validate no duplicates in _original_index
        if self.predictions_with_id['_original_index'].duplicated().any():
            duplicates = self.predictions_with_id[self.predictions_with_id['_original_index'].duplicated(keep=False)]
            raise ValueError(f"Duplicate values found in _original_index. This should never happen.\n{duplicates}")

        # Check if we have labels
        self.no_label = ground_truth.filter(regex=pred_col).shape[1] == 0

        # Create a reusable EvalEngine instance (will update y_pred and y_true as needed)
        # Initialize with empty DataFrames but correct structure
        empty_pred = pd.DataFrame(columns=y_pred_formatted.columns)
        empty_true = pd.DataFrame(columns=y_true_formatted.columns)

        self.temp_engine = EvalEngine(
            model,
            ground_truth=empty_true,
            predicitons=empty_pred,
            pred_col=pred_col,
            digits=digits
        )
        # Set attributes directly to avoid reinitializing
        self.temp_engine.formatter = self.formatter
        self.temp_engine.block_size = self.block_size
        self.temp_engine.system = self.system
        self.temp_engine.use_within_block_sep = self.use_within_block_sep
        self.temp_engine.no_label = self.no_label

    def _get_topk_predictions_for_obs(self, obs_id, rank=None):
        """
        Get predictions for a single observation at a specific rank.

        Args:
            obs_id: The identifier for the observation
            rank (int, optional): Specific rank to get predictions for (0-indexed). If None, returns all.

        Returns:
            list: Predictions at the specified rank (or all ranks if rank is None)
        """
        obs_preds = self.predictions_topk[self.predictions_topk[self.group_col] == obs_id]

        # Filter by specific rank if specified
        if rank is not None:
            obs_preds = obs_preds[obs_preds['top-k-pos'] == rank]

        # Sort by top-k-pos to ensure correct order
        obs_preds = obs_preds.sort_values('top-k-pos')

        # Extract prediction columns
        pred_cols = [col for col in obs_preds.columns if col.startswith(self.pred_col)]

        # Collect all predictions
        all_preds = []
        for _, row in obs_preds.iterrows():
            for col in pred_cols:
                pred = str(row[col])
                if pred != 'nan' and pred != ' ':
                    # Format the prediction
                    if len(pred) == (self.block_size - 1):
                        pred = '0' + pred
                    if len(pred) == 2 and pred[0] == '-':
                        pred = pred[0] + '0' * (self.block_size - 2) + pred[1]
                    all_preds.append(pred)

        return all_preds

    def _prepare_rank_data(self, rank):
        """
        Prepare prediction and ground truth data for a specific rank.

        This hidden method:
        1. Finds indices for the specified rank
        2. Extracts columns and indices (and resets)
        3. Tests that RowIDs match between predictions and ground truth
        4. Tests that lengths match between predictions and ground truth

        Args:
            rank (int): The rank position (0-indexed) to prepare data for.

        Side effects:
            Updates self.temp_engine.y_pred and self.temp_engine.y_true

        Raises:
            ValueError: If RowIDs don't match between predictions and ground truth
        """
        # 1. Find indices for this rank
        rank_indices = self.predictions_with_id['top-k-pos'] == rank

        # 2. Extract columns and indices (and reset)
        rank_preds = self.predictions_with_id[rank_indices].reset_index(drop=True)
        pred_rowids = rank_preds[self.group_col]
        self.temp_engine.y_pred = rank_preds.drop(columns=[self.group_col, 'top-k-pos', '_original_index']).reset_index(drop=True)

        # Get ground truth by index position (same order as predictions)
        gt = self.ground_truth_with_id
        gt_rowids = gt[self.group_col]
        self.temp_engine.y_true = gt.drop(columns=[self.group_col]).reset_index(drop=True)

        # 3. Test that RowIDs match
        if not pred_rowids.reset_index(drop=True).equals(gt_rowids.reset_index(drop=True)):
            raise ValueError(f"RowID mismatch at rank {rank}. Predictions and ground truth are not aligned.")

        # 4. Test that lengths match
        if len(self.temp_engine.y_pred) != len(self.temp_engine.y_true):
            raise ValueError(f"Length mismatch at rank {rank}. Predictions and ground truth lengths differ.")

    def accuracy(self, return_per_obs=True):
        """
        Calculate the accuracy for top-k predictions at each rank level (atomically).
        Each rank is evaluated independently -8 only predictions at that specific rank position.

        Args:
            return_per_obs (bool): If True, returns a flat list matching the order and length of the input predictions.

        Returns:
            dict or list: Dictionary mapping rank to accuracy, or flat list with one value per row in predictions.
        """

        # Collect results for all ranks
        all_accs = []
        ids = []
        topk_pos_list = []
        for rank in range(self.k):
            self._prepare_rank_data(rank)
            accs = self.temp_engine.accuracy(return_per_obs=True)
            all_accs.extend(accs)
            ids = ids + self.ground_truth_with_id["RowID"].tolist()
            topk_pos_list = topk_pos_list + [rank] * len(accs)

        # To dataframe
        results_df = pd.DataFrame({
            self.group_col: ids,
            "accuracy": all_accs,
            "top-k-pos": topk_pos_list
        })

        # Join to copy of self.predictions_with_id
        results_df = results_df.merge(
            self.predictions_with_id[[self.group_col, 'top-k-pos', '_original_index']],
            on=[self.group_col, 'top-k-pos'],
            how='left'
        )
        # Sort
        results_df = results_df.sort_values(by=['_original_index'])

        # Return results
        if return_per_obs:
            return results_df['accuracy'].tolist()
        else:
            return np.mean(all_accs)


    def precision(self, return_per_obs=True):
        """
        Calculate the precision for top-k predictions at each rank level (atomically).
        Each rank is evaluated independently - only predictions at that specific rank position.

        Args:
            return_per_obs (bool): If True, returns a flat list matching the order and length of the input predictions.

        Returns:
            dict or list: Dictionary mapping rank to precision, or flat list with one value per row in predictions.
        """

        # Collect results for all ranks
        all_precs = []
        ids = []
        topk_pos_list = []
        for rank in range(self.k):
            self._prepare_rank_data(rank)
            precs = self.temp_engine.precision(return_per_obs=True)
            all_precs.extend(precs)
            ids = ids + self.ground_truth_with_id[self.group_col].tolist()
            topk_pos_list = topk_pos_list + [rank] * len(precs)

        # To dataframe
        results_df = pd.DataFrame({
            self.group_col: ids,
            "precision": all_precs,
            "top-k-pos": topk_pos_list
        })

        # Join to copy of self.predictions_with_id
        results_df = results_df.merge(
            self.predictions_with_id[[self.group_col, 'top-k-pos', '_original_index']],
            on=[self.group_col, 'top-k-pos'],
            how='left'
        )
        # Sort
        results_df = results_df.sort_values(by=['_original_index'])

        # Return results
        if return_per_obs:
            return results_df['precision'].tolist()
        else:
            return np.mean(all_precs)

    def recall(self, return_per_obs=True):
        """
        Calculate the recall for top-k predictions at each rank level (atomically).
        Each rank is evaluated independently - only predictions at that specific rank position.

        Args:
            return_per_obs (bool): If True, returns a flat list matching the order and length of the input predictions.

        Returns:
            dict or list: Dictionary mapping rank to recall, or flat list with one value per row in predictions.
        """

        # Collect results for all ranks
        all_recs = []
        ids = []
        topk_pos_list = []
        for rank in range(self.k):
            self._prepare_rank_data(rank)
            recs = self.temp_engine.recall(return_per_obs=True)
            all_recs.extend(recs)
            ids = ids + self.ground_truth_with_id[self.group_col].tolist()
            topk_pos_list = topk_pos_list + [rank] * len(recs)

        # To dataframe
        results_df = pd.DataFrame({
            self.group_col: ids,
            "recall": all_recs,
            "top-k-pos": topk_pos_list
        })

        # Join to copy of self.predictions_with_id
        results_df = results_df.merge(
            self.predictions_with_id[[self.group_col, 'top-k-pos', '_original_index']],
            on=[self.group_col, 'top-k-pos'],
            how='left'
        )
        # Sort
        results_df = results_df.sort_values(by=['_original_index'])

        # Return results
        if return_per_obs:
            return results_df['recall'].tolist()
        else:
            return np.mean(all_recs)

    def f1(self, return_per_obs=True):
        """
        Calculate the F1 score for top-k predictions at each rank level (atomically).
        Each rank is evaluated independently - only predictions at that specific rank position.

        Args:
            return_per_obs (bool): If True, returns a flat list matching the order and length of the input predictions.

        Returns:
            dict or list: Dictionary mapping rank to F1, or flat list with one value per row in predictions.
        """

        # Collect results for all ranks
        all_f1s = []
        ids = []
        topk_pos_list = []
        for rank in range(self.k):
            self._prepare_rank_data(rank)
            f1s = self.temp_engine.f1(return_per_obs=True)
            all_f1s.extend(f1s)
            ids = ids + self.ground_truth_with_id[self.group_col].tolist()
            topk_pos_list = topk_pos_list + [rank] * len(f1s)

        # To dataframe
        results_df = pd.DataFrame({
            self.group_col: ids,
            "f1": all_f1s,
            "top-k-pos": topk_pos_list
        })

        # Join to copy of self.predictions_with_id
        results_df = results_df.merge(
            self.predictions_with_id[[self.group_col, 'top-k-pos', '_original_index']],
            on=[self.group_col, 'top-k-pos'],
            how='left'
        )
        # Sort
        results_df = results_df.sort_values(by=['_original_index'])

        # Return results
        if return_per_obs:
            return results_df['f1'].tolist()
        else:
            return np.mean(all_f1s)

