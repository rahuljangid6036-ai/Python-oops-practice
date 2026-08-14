class MissingValueHandler:
    def __init__(self, data):
        self.data = data
        self.cleaned_data = None

    def validate_input(self):
        if not isinstance(self.data, list):
            raise TypeError("Error: Input must be a list.")
        if len(self.data) == 0:
            raise ValueError("Error: Input list cannot be empty.")
        
        for item in self.data:
            if item is not None and not isinstance(item, (int, float)):
                raise ValueError("Error: Dataset contains invalid values.")

    def find_missing_indexes(self):
        indexes = []
        for i in range(len(self.data)):
            if self.data[i] is None:
                indexes.append(i)
        return indexes

    def count_missing_values(self):
        return len(self.find_missing_indexes())

    def calculate_mean(self):
        total_sum = 0
        count = 0
        for item in self.data:
            if item is not None:
                total_sum += item
                count += 1
        if count == 0:
            raise ValueError("Error because no valid values exist to calculate the mean.")
        return total_sum / count

    def calculate_median(self):
        valid_values = []
        for item in self.data:
            if item is not None:
                valid_values.append(item)
        if len(valid_values) == 0:
            raise ValueError("Error because no valid values exist to calculate the median.")
        
        # Manual sort
        valid_values.sort()
        n = len(valid_values)
        if n % 2 == 1:
            return valid_values[n // 2]
        else:
            mid1 = valid_values[(n // 2) - 1]
            mid2 = valid_values[n // 2]
            return (mid1 + mid2) / 2

    def fill_with_mean(self):
        mean_val = self.calculate_mean()
        self.cleaned_data = list(self.data)
        for i in range(len(self.cleaned_data)):
            if self.cleaned_data[i] is None:
                self.cleaned_data[i] = mean_val

    def fill_with_median(self):
        median_val = self.calculate_median()
        self.cleaned_data = list(self.data)
        for i in range(len(self.cleaned_data)):
            if self.cleaned_data[i] is None:
                self.cleaned_data[i] = median_val

    def fill_with_zero(self):
        self.cleaned_data = list(self.data)
        for i in range(len(self.cleaned_data)):
            if self.cleaned_data[i] is None:
                self.cleaned_data[i] = 0

    def fill_missing_values(self, strategy="mean"):
        self.validate_input()
        if strategy == "mean":
            self.fill_with_mean()
        elif strategy == "median":
            self.fill_with_median()
        elif strategy == "zero":
            self.fill_with_zero()
        else:
            raise ValueError(f"Error: Unknown imputation strategy '{strategy}'.")

    def display_report(self):
        try:
            self.validate_input()
            missing_count = self.count_missing_values()
            missing_idx = self.find_missing_indexes()
            available_count = len(self.data) - missing_count
            
            mean_val = "N/A"
            if available_count > 0:
                mean_val = self.calculate_mean()
            
            if self.cleaned_data is None:
                self.fill_missing_values("mean")

            
            print("MISSING VALUE REPORT")
            print(f"Original Data : {self.data}")
            print(f"Total Values  : {len(self.data)}")
            print(f"Missing Values: {missing_count}")
            print(f"Missing Indexes: {missing_idx}")
            print(f"Available Values: {available_count}")
            print(f"Mean          : {mean_val}")
            print(f"Cleaned Data  : {self.cleaned_data}")
           
        except Exception as error:
            print("Error:", error)


def main():
    test_cases = [
        [25, 30, None, 40, None, 35, 28],

        [None, None, None],
        [],
        [10, 20, "30", None],
        [-10, -20, None, -30]
    ]

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n--- Test Case {idx} ---")
        try:
            handler = MissingValueHandler(tc)
            handler.display_report()
        except Exception as error:
            print("Error:", error)


if __name__ == "__main__":
    main()
