import h5py

def rename_groups_in_hdf5(file_path):
    """
    Rename all variables within each group in an HDF5 file by stripping the first two characters.
    """
    with h5py.File(file_path, 'r+') as f:
        # Iterate through all groups
        for group_name in f.keys():
            if isinstance(f[group_name], h5py.Group):
                group = f[group_name]
                
                # Get all variable names in this group
                var_names = list(group.keys())
                
                group.attrs['data_origin'] = 'Themis_B'
                # Create a mapping of old names to new names
                rename_map = {}
                for old_name in var_names:
                    if old_name.startswith('B') or old_name.startswith('C'):
                        new_name = old_name[2:]  # Strip first two characters
                        if new_name and new_name != old_name:
                            rename_map[old_name] = new_name
                
                # Rename variables within the group
                for old_name, new_name in rename_map.items():
                    group.move(old_name, new_name)
                    print(f"Renamed '{old_name}' to '{new_name}' in group '{group_name}'")
                

# Usage
if __name__ == "__main__":
    training_data_name = "Data/training_data/test.h5"
    test_data_name = "Data/testing_data/test1.h5"
    rename_groups_in_hdf5(training_data_name)
    rename_groups_in_hdf5(test_data_name)