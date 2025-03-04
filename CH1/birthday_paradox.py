import numpy as np
# import matplotlib.pyplot as plt

# # Vectorized version of baby_prob
# def baby_prob(n_babies, year_days=365):
#     # Create an array of terms for the product
#     terms = (year_days - np.arange(n_babies)) / year_days
#     # Calculate the product of terms using np.prod
#     negprob = np.prod(terms)  # Removed axis=0
#     # Calculate the probability
#     probability = 1 - negprob
#     return probability

# # Generate an array of number of babies
# numbabies = np.arange(1, 366)

# # Apply the baby_prob function to the entire array
# probabilities = np.vectorize(baby_prob)(numbabies)  # Vectorized function call

# # Plot the results
# plt.plot(numbabies, probabilities)
# plt.xlabel('Number of Babies')
# plt.ylabel('Probability of Shared Birthday')
# plt.title('Birthday Paradox')
# plt.grid(True)
# plt.show()


# dp = np.gradient(probabilities)
# plt.plot(numbabies, dp)
# plt.show()

N = 941
n = 0

if N <= 1:
    print('Not Prime')
else:
    for i in range(2, int(np.sqrt(N)) + 1):  # Fixed range
        if N % i == 0:
            n += 1
            break 
    if n > 0:
        print('Not Prime')
    else:
        print('Prime')