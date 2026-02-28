import random
import matplotlib.pyplot as plt 
import numpy as np

def plot_function(func, range_start=0, range_end = 100, step=1000):
    # Define the range of x values you want to evaluate the function over
    x_values = np.linspace(range_start, range_end, step)  # You can adjust the range and number of points as needed
    
    # Evaluate the function for each x value
    y_values = func(x_values)
    
    # Create a plot
    plt.figure(figsize=(10, 6))  # You can adjust the figure size as needed
    plt.plot(x_values, y_values, label='Function Plot')

    # Add titles and labels
    plt.title('Plot of the Function')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    
    # Add a legend
    plt.legend()
    
    # Show grid
    plt.grid(True)



def generate_random_bias(total_length = 10000, std = 10):
    # Define the range for the uniform distribution of the total array length
    min_length, max_length = 500, 2000
    
    # Randomly draw the total length of the array from the uniform distribution
    
    
    sampled_length = 0
    segments = []
    while sampled_length < total_length:
        segment_length = random.randint(min_length, max_length)
        segments.append(
            np.repeat(np.random.normal(0, std, 1),segment_length)
        )
        sampled_length += segment_length

    
    # Concatenate the segments into a single numpy array
    concatenated_array = np.concatenate(segments)
    
    return concatenated_array[:total_length]