import os
import json
import torch

def compute_metrics(gold_index, predictions):

    hit_1 = 1 if gold_index == predictions[0] else 0
    rank_pos = (predictions == gold_index).nonzero(as_tuple=True)[0] + 1
    mrr = 1 / rank_pos

    return hit_1, mrr[0].item()

os.makedirs('./submissions_outs', exist_ok=True)

with open("./ds_test_anon_with_labels.json", "r", encoding="utf8") as f_gold:

    data = json.load(f_gold)
    total = len(data)

    for x in os.listdir('./submissions'):

        hit_1_total = 0
        mrr_total = 0
        total_sub = 0

        with open(os.path.join('./submissions_outs', x.replace("csv", "jsonl")), "w", encoding="utf8") as f_out:

            with open(os.path.join('./submissions', x), "r", encoding="utf8") as f:

                for idx, (l, inst) in enumerate(zip(f, data)):
                    
                    line_data = l.strip().replace("\"", "").split(", ")

                    mapping = {img: idx for idx, img in enumerate(inst["candidates"])}

                    line_data = [mapping[y] for y in line_data]
                    
                    label_idx = inst["candidates"].index(inst["label"] + ".jpg")
                    
                    hit1, mrr = compute_metrics(label_idx, torch.tensor(line_data).int())

                    new_dict = {}
                    new_dict["idx"] = idx
                    new_dict["hit@1"] = hit1
                    new_dict["mrr"] = mrr
                    
                    json.dump(new_dict, f_out)
                    f_out.write('\n')

                    hit_1_total += hit1
                    mrr_total += mrr

                    total_sub += 1
        
        assert total_sub == len(data)

        print(x)       
        print("HIT@1: " + str(hit_1_total / total))
        print("MRR: " + str(mrr_total / total))
        print("*" * 8)

