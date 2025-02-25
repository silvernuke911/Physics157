def encrypt_to_hex(filename):
    try:
        # Read the content of the file
        with open(filename, 'r') as file:
            content = file.read()

        # Convert the content to ASCII hexadecimal (uppercase and spaced)
        hex_content = ' '.join(f"{ord(char):02X}" for char in content)

        # Print the hexadecimal content
        print("Hexadecimal representation:")
        print(hex_content)

        # Save the hexadecimal content to a new file
        new_filename = filename.replace('.txt', '_hex.dat')
        with open(new_filename, 'w') as hex_file:
            hex_file.write(hex_content)

        print(f"Hex content saved to {new_filename}")

    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def decrypt_from_hex(filename):
    try:
        # Read the hexadecimal content from the file
        with open(filename, 'r') as hex_file:
            hex_content = hex_file.read()

        # Convert the hexadecimal content back to plain text
        plain_text = bytes.fromhex(hex_content).decode('utf-8')

        # Print the decrypted plain text
        print("Decrypted plain text:")
        print('---------------------------------------------')
        print(plain_text)
        print('---------------------------------------------')
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
    except ValueError:
        print(f"Error: The file {filename} does not contain valid hexadecimal data.")
    except Exception as e:
        print(f"An error occurred: {e}")

encrypt_to_hex('long.txt')
decrypt_from_hex('long_hex.dat')