from pathlib import Path

def setup_validation_analyses(path_to_recording:Path):
    validation_folder_name = 'validation'

    path_to_validation = path_to_recording/validation_folder_name
    path_to_validation.mkdir(exist_ok=True)
    

if __name__ == '__main__':
    path_to_recording = Path(r'D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking')

    setup_validation_analyses(path_to_recording)