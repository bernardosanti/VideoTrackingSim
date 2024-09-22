import numpy as np

class LowPassFilter:
    def __init__(self, cutoff_freq_hz, sample_time_s):
        self.cutoff_freq_hz = cutoff_freq_hz
        self.sample_time_s = sample_time_s
        self.alpha = self._calculate_alpha()
        self.prev_output = 0.0

    def _calculate_alpha(self):
        # Calculate the filter coefficient (alpha)
        tau = 1.0 / (2 * np.pi * self.cutoff_freq_hz)
        return self.sample_time_s / (tau + self.sample_time_s)

    def filter(self, input_value):
        # Apply the low-pass filter
        output = self.alpha * input_value + (1 - self.alpha) * self.prev_output
        self.prev_output = output
        return output

    def initialize(self, initial_value):
        """
        Initialize the filter's output value to avoid startup transients.
        """
        self.prev_output = initial_value
        
if __name__ == "__main__":
    # Example usage:
    cutoff_frequency_hz = 5  # cutoff frequency in Hz
    sample_time_s = 0.01  # sample time in seconds (e.g., 10 ms)

    # Create filter instance
    lpf = LowPassFilter(cutoff_frequency_hz, sample_time_s)

    # Filter a signal (for example, using a loop or in real-time)
    input_signal = [1, 0.5, 0.2, 0.1, 0.05]  # Example signal

    filtered_signal = [lpf.filter(value) for value in input_signal]
    print(filtered_signal)
