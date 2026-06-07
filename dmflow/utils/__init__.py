import warnings

import torch
import numpy as np
from scipy.stats import entropy


def load_state_dict_from_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    return state_dict


def masked_mean(data, mask):
    """
    Compute mean of values where mask indicates non-selected positions

    Args:
        data: Tensor of shape [B, N, d] containing the data values
        mask: Tensor of shape [B, N, 1] where 1 indicates positions to include in mean calculation
    Returns:
        Mean value of data at positions where mask is 1
    """

    # Expand mask to match data dimensions for element-wise operations
    expanded_mask = mask.expand_as(data)

    # Use logical NOT operation to identify positions where mask is 0 (included positions)

    # Extract values from positions to be included in mean calculation
    selected_values = data[expanded_mask]

    # Calculate mean if there are any selected values, otherwise return 0
    if selected_values.numel() > 0:
        return selected_values.mean()
    else:
        return torch.tensor(0.0, device=data.device, dtype=data.dtype)


def auto_numpy_conversion(*tensor_arg_names: str, copy: bool = False):
    from functools import wraps
    import inspect
    from typing import Any, Dict, Optional

    """
    Decorator: Automatically converts specified arguments from torch.Tensor to numpy,
    runs the function, and converts the return value back to the original type.

    Args:
        *tensor_arg_names (str): Names of arguments that may be torch.Tensor or np.ndarray.
        copy (bool): If True, makes a copy of the data during conversion. Default: False.

    Returns:
        Decorated function that transparently handles both torch.Tensor and np.ndarray.

    Example:
        @auto_numpy_conversion('input_data', 'labels')
        def process(input_data: np.ndarray, labels: np.ndarray, config: dict) -> np.ndarray:
            # Inside: input_data and labels are np.ndarray
            # Return: automatically converted back if inputs were tensors
            ...
    """

    def decorator(func: callable) -> callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Bind arguments to parameter names
            sig: inspect.Signature = inspect.signature(func)
            try:
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
            except TypeError as e:
                raise TypeError(f"Error binding arguments for {func.__name__}: {e}")

            # Track original tensor metadata for conversion back
            conversion_info: Dict[str, Optional[Dict[str, Any]]] = {}

            # Convert specified arguments to numpy if they are tensors
            for name in tensor_arg_names:
                if name not in bound_args.arguments:
                    continue

                value = bound_args.arguments[name]

                if isinstance(value, torch.Tensor):
                    device = value.device
                    dtype = value.dtype
                    # Move to CPU and convert to numpy
                    np_value = value.detach().cpu().numpy()
                    if copy:
                        np_value = np_value.copy()
                    bound_args.arguments[name] = np_value
                    conversion_info[name] = {
                        "type": "tensor",
                        "device": device,
                        "dtype": dtype,
                    }
                elif isinstance(value, np.ndarray):
                    conversion_info[name] = {"type": "numpy"}
                else:
                    conversion_info[name] = {"type": "other"}

            # Execute the original function with converted inputs
            result = func(*bound_args.args, **bound_args.kwargs)

            # If result is a numpy array, convert it back to tensor if needed
            if isinstance(result, np.ndarray):
                # Find the first argument that was a tensor to recover device and dtype
                for name in tensor_arg_names:
                    info = conversion_info.get(name)
                    if info and info["type"] == "tensor":
                        result = torch.tensor(
                            result, dtype=info["dtype"], device=info["device"]
                        )
                        break  # Use the first tensor's device

            return result

        return wrapper

    return decorator


class MultiLabelSelector:
    """
    A class that encapsulates various strategies for multi-label selection
    from a probability matrix.

    All methods ensure that the number of selected labels for each sample
    does not exceed 'max_labels'.
    """

    def __init__(
        self,
        k=5,
        threshold=0.5,
        percentile=90,
        adaptive_ratio=0.5,
        entropy_threshold=0.8,
        min_votes=3,
        max_labels=5,
        single_label_threshold_ratio=3.0,
    ):
        """
        Initializes the selector with parameters for various methods.

        Args:
            k (int): The number of categories to select in the 'top_k' method.
            threshold (float): The probability threshold used in the fixed 'threshold' method.
            percentile (int): The percentile (0-100) to use as a dynamic threshold in the
                              'adaptive_threshold' method.
            adaptive_ratio (float): The ratio (0.0-1.0) of the max probability to use as a
                                    threshold in the 'adaptive_top_k' method.
            entropy_threshold (float): The normalized entropy threshold used in the 'entropy_based'
                                       method to determine uncertainty.
            min_votes (int): The minimum number of votes required for a label to be selected
                             in the 'combined' method. Default is 3 for the 5 base methods.
            max_labels (int): The maximum number of labels that can be selected for any sample,
                              enforced across all methods.
            single_label_threshold_ratio (float): If max_prob / second_max_prob > this ratio,
                                                  treat as single-label case. Set to None to disable.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between [0, 1]")
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between [0, 100]")
        if not 0.0 <= adaptive_ratio <= 1.0:
            raise ValueError("adaptive_ratio must be between [0, 1]")
        if (
            single_label_threshold_ratio is not None
            and single_label_threshold_ratio <= 1.0
        ):
            raise ValueError(
                "single_label_threshold_ratio must be > 1.0 or None to disable."
            )

        self.k = k
        self.threshold = threshold
        self.percentile = percentile
        self.adaptive_ratio = adaptive_ratio
        self.entropy_threshold = entropy_threshold
        self.min_votes = min_votes
        self.max_labels = max_labels
        self.single_label_threshold_ratio = single_label_threshold_ratio

    def _enforce_max_labels(
        self, prob_matrix: np.ndarray, binary_matrix: np.ndarray
    ) -> np.ndarray:
        """Helper function: ensures the number of labels per row does not exceed max_labels."""
        for i in range(binary_matrix.shape[0]):
            if np.sum(binary_matrix[i]) > self.max_labels:
                selected_indices = np.where(binary_matrix[i] == 1)[0]
                selected_probs = prob_matrix[i, selected_indices]
                top_indices_within_selection = np.argsort(selected_probs)[
                    -self.max_labels :
                ]
                new_row = np.zeros_like(binary_matrix[i])
                new_row[selected_indices[top_indices_within_selection]] = 1
                binary_matrix[i] = new_row
            elif np.sum(binary_matrix[i]) == 0:
                # Ensure at least one label is selected
                max_index = np.argmax(prob_matrix[i])
                binary_matrix[i, max_index] = 1
        return binary_matrix

    # --- Raw Selection Logic (before enforcing max_labels) ---

    def _raw_top_k(self, prob_matrix: np.ndarray) -> np.ndarray:
        binary_matrix = np.zeros_like(prob_matrix, dtype=int)
        k_val = min(self.k, prob_matrix.shape[1])
        for i in range(prob_matrix.shape[0]):
            top_k_indices = np.argsort(prob_matrix[i, :])[-k_val:]
            binary_matrix[i, top_k_indices] = 1
        return binary_matrix

    def _raw_threshold(self, prob_matrix: np.ndarray) -> np.ndarray:
        return (prob_matrix > self.threshold).astype(int)

    def _raw_adaptive_threshold(self, prob_matrix: np.ndarray) -> np.ndarray:
        binary_matrix = np.zeros_like(prob_matrix, dtype=int)
        for i in range(prob_matrix.shape[0]):
            prob_vector = prob_matrix[i, :]
            if np.all(prob_vector == 0):
                continue
            threshold = np.percentile(prob_vector[prob_vector > 0], self.percentile)
            binary_matrix[i, prob_vector >= threshold] = 1
        return binary_matrix

    def _raw_adaptive_top_k(self, prob_matrix: np.ndarray) -> np.ndarray:
        binary_matrix = np.zeros_like(prob_matrix, dtype=int)
        for i in range(prob_matrix.shape[0]):
            prob_vector = prob_matrix[i, :]
            if np.all(prob_vector == 0):
                continue
            max_prob = np.max(prob_vector)
            threshold = max_prob * self.adaptive_ratio
            binary_matrix[i, prob_vector >= threshold] = 1
        return binary_matrix

    def _raw_entropy_based(self, prob_matrix: np.ndarray) -> np.ndarray:
        binary_matrix = np.zeros_like(prob_matrix, dtype=int)
        n_dimensions = prob_matrix.shape[1]
        max_entropy = np.log2(n_dimensions)
        for i in range(prob_matrix.shape[0]):
            prob_vector = prob_matrix[i, :]
            current_entropy = entropy(prob_vector, base=2)
            normalized_entropy = current_entropy / max_entropy if max_entropy > 0 else 0
            if normalized_entropy > self.entropy_threshold:
                binary_matrix[i, np.argmax(prob_vector)] = 1
            else:
                max_prob = np.max(prob_vector)
                threshold = max_prob * self.adaptive_ratio
                binary_matrix[i, prob_vector >= threshold] = 1
        return binary_matrix

    # --- Public-Facing Methods ---

    def _top_k(self, prob_matrix: np.ndarray) -> np.ndarray:
        """Selects the top k categories. This method is inherently constrained by k."""
        k_val = min(self.k, self.max_labels)  # Ensure k doesn't exceed the global max
        binary_matrix = np.zeros_like(prob_matrix, dtype=int)
        for i in range(prob_matrix.shape[0]):
            top_k_indices = np.argsort(prob_matrix[i, :])[-k_val:]
            binary_matrix[i, top_k_indices] = 1
        return binary_matrix

    def _threshold(self, prob_matrix: np.ndarray) -> np.ndarray:
        """Selects all categories exceeding a fixed threshold."""
        raw_selection = self._raw_threshold(prob_matrix)
        return self._enforce_max_labels(prob_matrix, raw_selection)

    def _adaptive_threshold(self, prob_matrix: np.ndarray) -> np.ndarray:
        """Selects categories based on a dynamic, percentile-based threshold."""
        raw_selection = self._raw_adaptive_threshold(prob_matrix)
        return self._enforce_max_labels(prob_matrix, raw_selection)

    def _adaptive_top_k(self, prob_matrix: np.ndarray) -> np.ndarray:
        """Selects categories with probabilities exceeding (max_probability * adaptive_ratio)."""
        raw_selection = self._raw_adaptive_top_k(prob_matrix)
        return self._enforce_max_labels(prob_matrix, raw_selection)

    def _entropy_based(self, prob_matrix: np.ndarray) -> np.ndarray:
        """Adjusts the selection strategy based on entropy."""
        raw_selection = self._raw_entropy_based(prob_matrix)
        return self._enforce_max_labels(prob_matrix, raw_selection)

    def _combined(self, prob_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """Fuses the results of multiple algorithms by voting."""
        methods_to_combine = kwargs.get(
            "methods_to_combine",
            [
                "top_k",
                "threshold",
                "adaptive_threshold",
                "adaptive_top_k",
                "entropy_based",
            ],
        )

        votes = np.zeros_like(prob_matrix, dtype=int)

        method_map = {
            "top_k": self._raw_top_k,
            "threshold": self._raw_threshold,
            "adaptive_threshold": self._raw_adaptive_threshold,
            "adaptive_top_k": self._raw_adaptive_top_k,
            "entropy_based": self._raw_entropy_based,
        }

        for method_name in methods_to_combine:
            if method_name in method_map:
                votes += method_map[method_name](prob_matrix)
            else:
                warnings.warn(
                    f"Selection method {method_name!r} is not recognized and will be skipped.",
                    RuntimeWarning,
                )

        binary_matrix = (votes >= self.min_votes).astype(int)
        return self._enforce_max_labels(prob_matrix, binary_matrix)

    def _is_single_label_decision(self, prob_vector: np.ndarray) -> int:
        """
        Determines if the highest probability is significantly larger than the second.
        Returns the index of the single label if yes, else -1.
        """
        if self.single_label_threshold_ratio is None:
            return -1

        # Remove zero or near-zero values? Optional: depends on use case
        # We'll work with all values for now
        sorted_indices = np.argsort(prob_vector)[::-1]
        max_prob = prob_vector[sorted_indices[0]]
        second_max_prob = prob_vector[sorted_indices[1]]

        if second_max_prob <= 1e-8:  # Avoid division by near-zero
            return sorted_indices[0] if max_prob > 1e-8 else -1

        ratio = max_prob / second_max_prob
        if ratio > self.single_label_threshold_ratio:
            return sorted_indices[0]
        return -1

    @auto_numpy_conversion("prob_matrix")
    def select(
        self, prob_matrix: np.ndarray, method: str, set_norm: bool, **kwargs
    ) -> np.ndarray:
        """
        Selects labels according to the specified method.
        First checks for single-label dominance (if enabled), otherwise uses multi-label logic.

        Args:
            prob_matrix (np.ndarray): An n*d probability matrix.
            method (str): The name of the method to use.
                          Options: 'top_k', 'threshold', 'adaptive_threshold',
                                   'adaptive_top_k', 'entropy_based', 'combined'
            **kwargs: Additional arguments for specific methods.
                      For 'combined': methods_to_combine (list of strings)

        Returns:
            np.ndarray: An n*d binary (0/1) matrix.
        """
        prob_matrix = np.asarray(prob_matrix)
        n_samples, n_classes = prob_matrix.shape
        result = np.zeros((n_samples, n_classes), dtype=int)

        # Get the multi-label selection function
        methods = {
            "top_k": self._top_k,
            "threshold": self._threshold,
            "adaptive_threshold": self._adaptive_threshold,
            "adaptive_top_k": self._adaptive_top_k,
            "entropy_based": self._entropy_based,
            "combined": self._combined,
        }

        if method not in methods:
            raise ValueError(
                f"Unknown method: {method}. Available methods: {list(methods.keys())}"
            )

        multi_label_func = methods[method]
        resulted_prob_matrix = []

        # Process each sample
        for i in range(n_samples):
            prob_vector = prob_matrix[i]

            if not set_norm:
                prob_sum = np.sum(prob_vector)
                if np.abs(prob_sum - 1.0) > 0.05:
                    idx = np.argmax(prob_vector)
                    result[i, idx] = 1
                    resulted_prob_matrix.append(prob_vector)
                    continue
                else:
                    prob_vector = prob_vector / (prob_sum + 1e-8)
            else:
                prob_vector = prob_vector / (np.sum(prob_vector) + 1e-8)

            resulted_prob_matrix.append(prob_vector)

            # Step 1: Check for single-label dominance
            single_label_idx = self._is_single_label_decision(prob_vector)
            if single_label_idx != -1:
                result[i, single_label_idx] = 1
            else:
                # Step 2: Fall back to multi-label method
                full_result = multi_label_func(
                    prob_matrix[[i]], **kwargs
                )  # Call on single row
                result[i] = full_result[0]

        # result = multi_label_func(prob_matrix, **kwargs)

        return torch.LongTensor(result), torch.FloatTensor(
            np.asarray(resulted_prob_matrix)
        )


def make_default_disorder_selector() -> MultiLabelSelector:
    return MultiLabelSelector(
        k=2,
        threshold=0.2,
        percentile=95,
        adaptive_ratio=0.2,
        entropy_threshold=0.9,
        min_votes=4,
        max_labels=5,
    )
