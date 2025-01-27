# 스크립트 코드

import os
import json

# def count_rotate_left(actions):
#     """Counts the number of RotateLeft actions with success == true."""
#     return sum(1 for action in actions if action["action"] == "RotateLeft" and action["success"])

def count_rotate_left(actions):
    """Counts the number of consecutive RotateLeft actions with success == true, excluding sequences ending with STOP."""
    max_consecutive_count = 0
    current_consecutive_count = 0

    for action in actions:
        if action["action"] == "RotateLeft" and action["success"]:
            current_consecutive_count += 1
        else:
            # RotateLeft가 아닌 경우 Reset
            current_consecutive_count = 0

    return current_consecutive_count

# def count_mode(custom_logs):
#     """Counts the number of logs with the mode key."""
#     return sum(1 for log in custom_logs if "mode" in log)

def count_mode(custom_logs, success):
    """Counts the number of logs with the mode key."""
    count = 0
    if not success:
        for log in custom_logs:
            if log.get("mode") == 3:
                count += 1
    return count

def check_conditions(filepath):
    """Checks if the conditions are met for the given file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            episode_metrics = data.get("episode_metrics", {})
            actions_taken = episode_metrics.get("actions_taken", [])
            # print(count_rotate_left(actions_taken))
            custom_logs = episode_metrics.get("custom_logs", [])

            success = episode_metrics.get("success", True)
            mode_3_count = count_mode(custom_logs, success)

            if mode_3_count > 0:
                # print(mode_3_count)
                return True, mode_3_count

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return False, 0

def main(path):
    """Main function to iterate over files and print those that meet the conditions."""
    files_with_conditions_met = []
    num_files_with_conditions_met = 0
    total_mode_3_count = 0

    exoloit = 0

    for filename in os.listdir(path):
        if filename.endswith(".json"):
            filepath = os.path.join(path, filename)
            condition_met, mode_3_count = check_conditions(filepath)
            if condition_met and mode_3_count > 10:
                files_with_conditions_met.append((filename, mode_3_count))
                num_files_with_conditions_met += 1
                total_mode_3_count += mode_3_count

    files_with_conditions_met.sort()
    pattern = "Val"
    filtered_file_names = [filename for filename, count in files_with_conditions_met if pattern in filename]
    print(f'file name list : {filtered_file_names}')

    # 조건에 맞는 파일 이름 및 mode 3 개수 출력
    # print(files_with_conditions_met)
    for filename, mode_3_count in files_with_conditions_met:
        exoloit += 1
        # print(f'{filename}')
        # print(mode_3_count)
    print(exoloit)
    print(f'총 파일 수: {num_files_with_conditions_met}')

if __name__ == "__main__":
    # 경로 설정: JSON 파일이 있는 디렉토리 경로로 변경
    path = "./results/dup_spatial_fbe_gleeglip-b32-openai-center/"
    main(path)
