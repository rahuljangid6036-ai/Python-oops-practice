import numpy as np

class NumpyFeatureProcessor:
    def __init__(self, data):
        """
        Initialize the processor with raw data inputs and placeholders for outputs.
        """
        self.data = data
        self.array = None
        self.min_max_data = None
        self.standardized_data = None

    def validate_input(self):
        """
        Validates that input data is a non-empty list containing only numerical values.
        """
        if not isinstance(self.data, list):
            raise TypeError("Input must be a Python list.")
        if len(self.data) == 0:
            raise ValueError("Input list cannot be empty.")
        for item in self.data:
            # explicit check for bool since isinstance(True, int) evaluates to True
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError("Dataset contains non-numeric values.")

    def convert_to_array(self):
        """
        Converts the validated list into a NumPy array.
        """
        self.array = np.array(self.data)

    def get_array_info(self):
        """
        Returns basic properties of the underlying NumPy array.
        """
        return {
            "array": self.array,
            "dtype": self.array.dtype,
            "ndim": self.array.ndim,
            "shape": self.array.shape,
            "size": self.array.size
        }

    def calculate_minimum(self):
        """Calculates minimum using NumPy vectorization."""
        return np.min(self.array)

    def calculate_maximum(self):
        """Calculates maximum using NumPy vectorization."""
        return np.max(self.array)

    def calculate_mean(self):
        """Calculates mean using NumPy vectorization."""
        return np.mean(self.array)

    def calculate_standard_deviation(self):
        """Calculates standard deviation using NumPy vectorization."""
        return np.std(self.array)

    def min_max_scale(self):
        """Performs Min-Max Scaling mapping data to [0, 1]."""
        minimum = self.calculate_minimum()
        maximum = self.calculate_maximum()
        
        if maximum == minimum:
            # Handle edge case where all items are identical
            self.min_max_data = np.zeros_like(self.array, dtype=float)
        else:
            self.min_max_data = (self.array - minimum) / (maximum - minimum)
        return self.min_max_data

    def standardize(self):
        """Performs Z-Score Standardization mapping data to mean=0, std=1."""
        mean = self.calculate_mean()
        std_dev = self.calculate_standard_deviation()
        
        if std_dev == 0:
            raise ZeroDivisionError("Standard deviation is zero. Cannot standardize a constant dataset.")
        
        self.standardized_data = (self.array - mean) / std_dev
        return self.standardized_data

    def display_report(self):
        """Displays the processing metadata, statistical metrics, and transformed arrays."""
        info = self.get_array_info()
        minimum = self.calculate_minimum()
        maximum = self.calculate_maximum()
        mean = self.calculate_mean()
        std_dev = self.calculate_standard_deviation()
        
        self.min_max_scale()
        try:
            self.standardize()
            z_score_str = "[" + ", ".join(f"{x:.4f}" for x in self.standardized_data) + "]"
        except ZeroDivisionError:
            z_score_str = "Error: Standard deviation is zero (Constant Dataset)"

        report = [
            "==================================================",
            "          NUMPY FEATURE PROCESSING REPORT",
            "==================================================",
            f"\nOriginal Data:\n{self.data}",
            f"\nNumPy Array:\n{self.array}",
            f"\nData Type: {info['dtype']}",
            f"Dimensions: {info['ndim']}",
            f"Shape: {info['shape']}",
            f"Size: {info['size']}",
            f"\nMinimum: {minimum}",
            f"Maximum: {maximum}",
            f"Mean: {mean:.4f}" if mean % 1 != 0 else f"Mean: {mean:.1f}",
            f"Standard Deviation: {std_dev:.4f}",
            f"\nMin-Max Scaled:\n[" + ", ".join(f"{x:.2f}" for x in self.min_max_data) + "]",
            f"\nZ-Score Standardized:\n{z_score_str}",
            "=================================================="
        ]
        print("\n".join(report))

    def compare_scaling_methods(self):
        """Bonus Challenge (Part F): Displays comparison metrics and insights."""
        self.min_max_scale()
        try:
            self.standardize()
            has_z = True
        except ZeroDivisionError:
            has_z = False
            
        print("\n" + "="*50)
        print("          BONUS CHALLENGE: SCALING COMPARISON")
        print("="*50)
        print(f"{'Original':<12} | {'Min-Max (0 to 1)':<18} | {'Z-Score (Mean=0)':<18}")
        print("-" * 54)
        for i in range(len(self.data)):
            orig = self.data[i]
            m_max = f"{self.min_max_data[i]:.4f}"
            z_val = f"{self.standardized_data[i]:.4f}" if has_z else "N/A (Std=0)"
            print(f"{orig:<12} | {m_max:<18} | {z_val:<18}")
        print("="*50)
        
        print("\n--- Analytical Insights ---")
        print("1. Which transformation produces values between 0 and 1?")
        print("   -> Min-Max Scaling strictly bounds values between 0 and 1.")
        print("2. Which transformation produces values centered around zero?")
        print("   -> Z-Score Standardization centers values directly around 0.")
        print("3. What happens to the mean after standardization?")
        print("   -> The mean becomes exactly 0.0 (or infinitesimally close to 0 due to float64 representation).")
        print("4. Why might an ML algorithm benefit from either transformation?")
        print("   -> Distance-based systems (KNN, K-Means) or gradient-descent algorithms (Neural Networks, Logistic Regression)")
        print("      prevent features with vast absolute ranges from dominating weights, ensuring stable and faster convergence.")

def run_pipeline(data, test_name):
    print(f"\n⚡ {test_name}")
    try:
        processor = NumpyFeatureProcessor(data)
        processor.validate_input()
        processor.convert_to_array()
        processor.display_report()
        processor.compare_scaling_methods()
    except Exception as e:
        print(f"❌ Execution Halted via Exception Handler: {e}")
    print("\n" + "-"*60)

def main():
    run_pipeline([10, 20, 30, 40, 50], "Test Case 1 – Normal Dataset")
    run_pipeline([-10, -5, 0, 5, 10], "Test Case 2 – Negative Values")
    run_pipeline([1.5, 2.5, 3.5, 4.5], "Test Case 3 – Decimal Values")
    run_pipeline([100, 100, 100], "Test Case 4 – Constant Dataset")
    run_pipeline([10, 20, "30", 40], "Test Case 5 – Invalid Dataset")
    run_pipeline([], "Test Case 6 – Empty Dataset")

if __name__ == "__main__":
    main()
