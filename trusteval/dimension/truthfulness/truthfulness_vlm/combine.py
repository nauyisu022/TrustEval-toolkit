import json
import os

enhance_output_lst = []

# enhanced data
enhance_scan_dir = "/home/rayguan/Desktop/trustAGI/TrustAGI-code/data"

enhance_file_list = [os.path.join(enhance_scan_dir, ff + "/" + ff + "_annotation.json") for ff in os.listdir(enhance_scan_dir) 
                if not os.path.isfile(ff) and os.path.exists(os.path.join(enhance_scan_dir, ff + "/" + ff + "_annotation.json"))
                ]

print(enhance_file_list)

for js_file in enhance_file_list:
    with open(js_file, 'r') as file:
        data = json.load(file)
    for d in data:
        if type(d["Quality assessment"]) == bool and d["Quality assessment"]:
            if "enhanced_prompt" in d:
                d["ori_prompt"] = d["prompt"]
                d["prompt"] = d["enhanced_prompt"]

            if "enhanced_ground_truth" in d:
                d["ori_ground_truth"] = d["ground_truth"]
                d["ground_truth"] = d["enhanced_ground_truth"]

            enhance_output_lst.append(d)

print(len(enhance_output_lst))

#####################

ori_output_lst = []

# original data
ori_scan_dir = "/home/rayguan/Desktop/trustAGI/TrustAGI-anno-platform/data"

ori_file_list = [os.path.join(ori_scan_dir, ff + "/" + ff + "_annotation.json") for ff in os.listdir(ori_scan_dir) 
                if not os.path.isfile(ff) and os.path.exists(os.path.join(ori_scan_dir, ff + "/" + ff + "_annotation.json"))
                ]

ori_file_list.append("/home/rayguan/Desktop/trustAGI/TrustAGI-code/data/hallusionBench_data/hallusionBench_data.json")

print(ori_file_list)

for js_file in ori_file_list:
    with open(js_file, 'r') as file:
        data = json.load(file)
    for d in data:
        d["Semantic shift"] = False
        d["Quality assessment"] = True
        d["feedback"] = ""
        d["enhancement_method"] = "keep_original"
        ori_output_lst.append(d)

print(len(ori_output_lst))

# # Open and read the JSON file
# with open('data.json', 'r') as file:
#     data = json.load(file)

enhance_output_lst.extend(ori_output_lst)


with open("./hallusion_data.json", 'w') as f:
    json.dump(enhance_output_lst, f, indent=4)