class PID:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, output_limits=(None, None), integral_limits=(None, None), derivative_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_min, self.output_max = output_limits
        self.integral_min, self.integral_max = integral_limits
        self.derivative_min, self.derivative_max = derivative_limits
        
        self.integral = 0.0
        self.previous_error = 0.0

    def set_gains(self, Kp, Ki, Kd):
        """Set the proportional, integral, and derivative gains."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def set_limits(self, output_min, output_max):
        """Set the output limits."""
        self.output_min = output_min
        self.output_max = output_max
    
    def set_integral_limits(self, integral_min, integral_max):
        """Set the integral component limits."""
        self.integral_min = integral_min
        self.integral_max = integral_max

    def set_derivative_limits(self, derivative_min, derivative_max):
        """Set the derivative component limits."""
        self.derivative_min = derivative_min
        self.derivative_max = derivative_max

    def calculate(self, error, dt):
        """
        Calculate the PID control output based on the error input and time difference.
        
        :param error: The current error value
        :param dt: Time difference between current and previous calculation
        :return: Control output
        """
        # Proportional term
        P = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I = self.Ki * self.integral
        
        # Apply integral limits
        if self.integral_min is not None:
            I = max(self.integral_min, I)
        if self.integral_max is not None:
            I = min(self.integral_max, I)
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        D = self.Kd * derivative
        
        # Apply derivative limits
        if self.derivative_min is not None:
            D = max(self.derivative_min, D)
        if self.derivative_max is not None:
            D = min(self.derivative_max, D)
        
        # PID output
        output = P + I + D
        
        # Save the current error for the next derivative calculation
        self.previous_error = error
        
        # Apply output limits
        if self.output_min is not None:
            output = max(self.output_min, output)
        if self.output_max is not None:
            output = min(self.output_max, output)

        # Print the status
        self.print_status(P, I, D, output)
        
        return output
    
    def print_status(self, P, I, D, output):
        """Print the current PID status."""
        print(f"Proportional (P): {P:.4f}")
        print(f"Integral (I): {I:.4f}")
        print(f"Derivative (D): {D:.4f}")
        print(f"PID Output: {output:.4f}")
        print("-" * 30)

if __name__ == "__main__":
    # Example usage:
    pid = PID(Kp=1.0, Ki=0.1, Kd=0.01, output_limits=(-10, 10), integral_limits=(-1, 1), derivative_limits=(-0.1, 0.1))
    pid.set_gains(Kp=2.0, Ki=0.5, Kd=0.05)
    pid.set_limits(-5, 5)
    pid.set_integral_limits(-0.5, 0.5)
    pid.set_derivative_limits(-0.05, 0.05)

    error = 1.0
    dt = 0.1
    output = pid.calculate(error, dt)
    print(f"PID Output: {output}")
