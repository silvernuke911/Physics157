import time

def progress_bar(progress, total, start_time, scale=0.50):
    # Creates a progress bar on the command line, input is progress, total, and a present start time
    # progress and total can be any number, and this can be placed in a for or with loop

    percent = 100 * (float(progress) / float(total))                        # Calculate the percentage of progress
    bar = '█' * round(percent*scale) + '-' * round((100-percent)*scale)     # Create the progress bar string
    elapsed_time = time.time() - start_time                                 # Calculate elapsed time
    if progress > 0:                                                        # Estimate total time and remaining time
        estimated_total_time = elapsed_time * total / progress
        remaining_time = estimated_total_time - elapsed_time
        remaining_seconds = int(remaining_time)
        remaining_milliseconds = int((remaining_time - remaining_seconds) * 1_000)
        elapsed_milliseconds = int((elapsed_time -int(elapsed_time)) * 1_000)
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))
        remaining_str = f"{remaining_str}.{remaining_milliseconds:03d}"
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        elapsed_str = f"{elapsed_str}.{elapsed_milliseconds:03d}"
    else:
        elapsed_str   = '...'
        remaining_str = '...'
    print(f'|{bar}| {percent:.2f}% Time remaining: {remaining_str}  Time Elapsed = {elapsed_str}', end='\r')    # Print the progress bar with the remaining time

# n = 10000
# x = [0] * (n + 1)
# # x=[]
# start_time = time.time()            # Record the start time
# for i in range(n + 1):              # Process function
#     x[i] = i
#     # x.append(i)                        
#     progress_bar(i, n, start_time,0.5)
# print()