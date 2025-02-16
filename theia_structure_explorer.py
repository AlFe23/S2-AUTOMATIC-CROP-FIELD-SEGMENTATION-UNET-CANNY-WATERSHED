import os

def print_hierarchy(path, level=0):
    try:
        # Check if the path exists
        if not os.path.exists(path):
            print(f"Error: The path '{path}' does not exist.")
            return

        # Check if the path is indeed a directory
        if not os.path.isdir(path):
            print(f"Error: The path '{path}' is not a directory.")
            return

        # Get the list of files and directories in the given directory
        items = os.listdir(path)

        if not items:  # Skip empty directories
            print(f"{'  ' * level}No items found in '{path}'")
            return

        # Loop through each item in the directory
        for item in items:
            # Get the full path of the item
            item_path = os.path.join(path, item)

            # Check if the item is a symlink
            if os.path.islink(item_path):
                print(f"{'  ' * level}{item}/ (symlink)")
                continue  # Skip symlinks or handle them as needed

            # Print indentation based on the current level
            print('  ' * level + f'{item}/' if os.path.isdir(item_path) else '  ' * level + f'{item}')

            # Recursively call the function if the item is a directory
            if os.path.isdir(item_path):
                print_hierarchy(item_path, level + 1)

    except PermissionError:
        print(f"Error: Permission denied to access '{path}'")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")

# Example usage
path = "/mnt/h/S2_THEIA_test/SENTINEL2A_20210405-105854-998_L2A_T31TCJ_C_V3-0"
print_hierarchy(path)

