import os
def check_and_prompt_overwrite(filename):
    extension = os.path.splitext(filename)[1]
    def get_user_input(prompt, valid_responses, invalid_input_limit=3):
        attempts = 0
        while attempts < invalid_input_limit:
            response = input(prompt).lower().strip()
            if response in valid_responses:
                return response
            print("Invalid input. Valid inputs are [ Y , N , YES , NO ]")
            attempts += 1
        print("Exceeded maximum invalid input limit. Operation aborted.")
        return 'ABORT'

    def handle_file_exists(filename):
        while True:
            response = get_user_input(f"{filename} already exists, do you want to overwrite it? (Y/N): ", ['yes', 'y', 'no', 'n'], 5)
            if response in ['yes', 'y']:
                print                       ('\nx---------------------WARNING---------------------x')
                sure_response = get_user_input("Are you really sure you want to OVERWRITE it? (Y/N): ", ['yes', 'y', 'no', 'n'], 5)
                if sure_response in ['yes', 'y']:
                    print("Proceeding with overwrite...")
                    return True, filename
                elif sure_response in ['no', 'n']:
                    print('Operation aborted.')
                    return False, filename
                elif sure_response == 'ABORT':
                    return False, filename
            elif response in ['no', 'n']:
                return handle_rename(filename)
            elif response == 'ABORT':
                return False, filename

    def handle_rename(filename):
        while True:
            rename_response = get_user_input('Would you like to rename it? (Y/N): ', ['yes', 'y', 'no', 'n'],3)
            if rename_response in ['yes', 'y', '1']:
                return get_new_filename()
            elif rename_response in ['no', 'n', '0']:
                print('Operation aborted.')
                return False, filename
            elif rename_response == 'ABORT':
                return False, filename

    def get_new_filename():
        while True:
            new_filename = input('Input the new name of the file: ').strip()
            # If the user doesn't specify an extension, add the original extension
            if not new_filename.endswith(extension):
                new_filename += extension
            if new_filename == ('ABORT' + extension):
                print('Operation aborted.')
                return False, new_filename
            if not os.path.isfile(new_filename):
                print(f'Proceeding with creation of {new_filename}')
                return True, new_filename
            print(f'{new_filename} already exists. Please put another file name.')
    if os.path.isfile(filename):
        return handle_file_exists(filename)
    return True, filename