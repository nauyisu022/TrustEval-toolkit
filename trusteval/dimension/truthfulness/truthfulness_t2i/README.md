# Genereate Any Scene
install the following packages:
```
pip install inflect
pip install openai
pip install t2v_metrics
```
# Generate new batch of prompts:
run the following command to generate the prompts:
```
python generation.py --max_complexity 8 --max_attributes 5 --total_prompts 300
```

run the following command to rephrase the prompts:
```
python llm_rephase.py --input_file prompts_batch_0_1000.json
```

# Generated file structure:
- prompt: the original prompt, constructed directly by the generation.py, consist of multiple keywords that produced by template.
- global_attributes: the global attributes of the prompt, which is not important for generation and do not involve in the evaluation.
- scene_graph: metadata of the prompt, which can be used for the following VQA evaluation.
- llm_rephrased_prompt: the rephrased prompt that will be used for image generation.

example:
```
"0": {
        "prompt": "A painter (skilled worker); a landscaping; a pointed-leaf maple.",
        "global_attributes": {},
        "scene_graph": {
            "nodes": [
                [
                    "object_1",
                    {
                        "type": "object_node",
                        "value": "painter (skilled_worker)"
                    }
                ],
                [
                    "object_2",
                    {
                        "type": "object_node",
                        "value": "landscaping"
                    }
                ],
                [
                    "object_3",
                    {
                        "type": "object_node",
                        "value": "pointed-leaf_maple"
                    }
                ]
            ],
            "edges": []
        },
        "llm_rephrased_prompt": "A skilled painter meticulously working on a beautiful landscaping scene, featuring a pointed-leaf maple tree."
    },
```

一个跑eval的example是
```apt-get update && apt-get install ffmpeg libsm6 libxext6 -y python image_metric.py --metric tifa_score
--image_folder /image_fluxdev/flux1-dev \
--gen_data_file <generated_file_name>```

这个意思就是用vqascore去eval flux1-dev生成出来的图片