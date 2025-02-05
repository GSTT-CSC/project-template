"""Based on the paper: Charan J, Biswas T. How to calculate sample size for different study designs in medical research? Indian J Psychol Med. 2013 Apr;35(2):121-6. doi: 10.4103/0253-7176.116232. PMID: 24049221; PMCID: PMC3775042."""

import math
import scipy.stats as stats

def calculate_sample_size(odds_ratio, baseline_probability, significance_level, power, allocation_ratio):
    """Calculates minimum sample size for AI fairness evaluation using logistic regression."""
    
    # Check that input values are within valid ranges
    if not (0 < baseline_probability < 1):
        raise ValueError("Baseline probability must be between 0 and 1.")
    
    if odds_ratio <= 0:
        raise ValueError("Odds ratio must be greater than 0.")
    
    if not (0 < significance_level < 1):
        raise ValueError("Significance level must be between 0 and 1.")
    
    if not (0 < power < 1):
        raise ValueError("Power must be between 0 and 1.")
    
    # Two-tailed Z-score for significance level
    z_alpha = stats.norm.ppf(1 - significance_level / 2)  
    
    # Z-score for power (1 - β)
    z_beta = stats.norm.ppf(power)  

    # Compute probabilities based on the given odds ratio
    p1 = baseline_probability  # Probability of outcome in reference group
    p2 = (odds_ratio * p1) / (1 + (odds_ratio * p1) - p1)  # Adjusted probability for second group

    # Compute variance and effect size
    var1 = p1 * (1 - p1)
    var2 = p2 * (1 - p2)
    pooled_variance = (var1 + var2) / 2
    effect_size = abs(p1 - p2)

    # Sample size formula for logistic regression
    n_per_group = ((z_alpha * math.sqrt(2 * pooled_variance) + z_beta * math.sqrt(var1 + var2)) ** 2) / effect_size**2

    # Adjust for allocation ratio (e.g., if male/female ratio isn't 1:1)
    total_sample_size = n_per_group * (1 + allocation_ratio)
    
    return math.ceil(total_sample_size)

def get_valid_input(prompt, validation_function, error_message):
    while True:
        try:
            user_input = float(input(prompt))
            # Validate input using the provided validation function
            validation_function(user_input)
            return user_input
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except Exception as e:
            print(error_message)

def main():
    print("AI Fairness Sample Size Calculator (Logistic Regression)")
    
    # Explain the terms to the user
    print("\nExplanation of Terms:")
    print("1. **Odds Ratio**: This measures the strength of the association between two groups. It represents how much more likely the outcome is in one group compared to the other.")

    
    print("\n2. **Allocation Ratio**: This refers to the ratio of sample sizes between two groups in your study. For example:")
    
    # Get user inputs with immediate validation and feedback
    odds_ratio = get_valid_input("Enter odds ratio (must be > 0): ", 
                                lambda x: x > 0, 
                                "Odds ratio must be greater than 0.")
    
    allocation_ratio = get_valid_input("Enter allocation ratio (e.g., 1 for 50/50, 2 for 2:1): ", 
                                       lambda x: x > 0, 
                                       "Allocation ratio must be greater than 0.")
    
    baseline_probability = get_valid_input("Enter baseline probability (between 0 and 1): ", 
                                           lambda x: 0 < x < 1, 
                                           "Baseline probability must be between 0 and 1.")
    
    significance_level = get_valid_input("Enter significance level (α) (between 0 and 1): ", 
                                         lambda x: 0 < x < 1, 
                                         "Significance level must be between 0 and 1.")
    
    power = get_valid_input("Enter power (1 - β) (between 0 and 1): ", 
                            lambda x: 0 < x < 1, 
                            "Power must be between 0 and 1.")
    
    # Calculate the required sample size
    sample_size = calculate_sample_size(odds_ratio, baseline_probability, significance_level, power, allocation_ratio)
    print(f"Required Sample Size: {sample_size}")

if __name__ == "__main__":
    main()
