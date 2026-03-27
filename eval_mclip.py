from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
import requests
import torch
import json

def compute_metrics(gold_index, predictions):

    hit_1 = 1 if gold_index == predictions[0] else 0
    rank_pos = (predictions == gold_index).nonzero(as_tuple=True)[0] + 1
    mrr = 1 / rank_pos

    return hit_1, mrr[0].item()

# We use the original clip-ViT-B-32 for encoding images
img_model = SentenceTransformer('clip-ViT-B-32')

# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

# Now we load and encode the images
def load_image(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)

hit_1_total = 0
mmr_total = 0

with open("ds_test_anon_with_labels.json", "r", encoding="utf8") as f:

    data = json.load(f)
    score = 0
    total = len(data)

    for x in data:

        # We load 3 images. You can either pass URLs or
        # a path on your disc
        img_paths = x["candidates"]
        label_idx = img_paths.index(x["label"] + ".jpg")

        images = [load_image('./ds/images/' + img) for img in img_paths]

        # Map images to the vector space
        img_embeddings = img_model.encode(images)

        # Now we encode our text:
        texts = [
            x['query']
        ]

        text_embeddings = text_model.encode(texts)

        # Compute cosine similarities:
        cos_sim = util.cos_sim(text_embeddings, img_embeddings)

        for text, scores in zip(texts, cos_sim):
            sorted, indices = torch.sort(scores, descending=True)
            print(sorted)
            
            hit1, mrr = compute_metrics(label_idx, indices)

            hit_1_total += hit1
            mmr_total += mrr

print(hit_1_total / total)
print(mmr_total / total)
