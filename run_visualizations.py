import os

# 기본 명령어
base_command = "python path_visualization.py --out-dir {out_dir} --thor-floor {thor_floor} --result-json {result_json} --thor-build /home/ailab/cow_ours/pasture_builds/thor_build_dup/dup.x86_64"
# a = 'python path_visualization.py --out-dir viz_FloorPlan_Val1_5_BaseballBat_0/ --thor-floor FloorPlan_Val1_5 --result-json /home/ailab/cow/ex_7_viz/CoW/normal_spatial/FloorPlan_Val1_5_BaseballBat_0.json --thor-build /home/ailab/cow/pasture_builds/thor_build_dup/dup.x86_64'
# b = 'python path_visualization.py --out-dir viz_FloorPlan_Val1_3_AlarmClock_0/ --thor-floor FloorPlan_Val1_3 --result-json /home/ailab/cow/ex_7_viz/CoW/normal_spatial/FloorPlan_Val1_3_AlarmClock_0.json --thor-build /home/ailab/cow/pasture_builds/thor_build_dup/dup.x86_64'

# 파일 목록
files = ['FloorPlan_Val1_2_Vase_0.json']

# json_dir = "/home/ailab/cow/ex_7_viz/CoW/normal_spatial"
json_dir = "/home/ailab/cow_ours/results/dup_spatial_fbe_gleeglip-b32-openai-center"
output_base_dir = "viz_"

# 명령어 생성 및 실행
for file_name in files:
    # 파일 이름에서 floor와 object 추출
    floor = file_name.split('_')[1]
    object_name = file_name.split('_')[2]
    file_name_no_ext = file_name.replace('.json', '')

    # out-dir과 thor-floor 설정
    out_dir = f"{output_base_dir}{file_name_no_ext}/"
    thor_floor = f"FloorPlan_{floor}_{object_name}"

    # result-json 설정
    result_json = os.path.join(json_dir, file_name)

    # 최종 명령어 생성
    command = base_command.format(out_dir=out_dir, thor_floor=thor_floor, result_json=result_json)
    # 명령어 출력 (또는 os.system(command)로 실행 가능)
    # print(command)
    os.system(command)  # 실제 실행 시에는 이 줄의 주석을 제거