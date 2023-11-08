import json
import torch
from transformers import AutoModel, AutoTokenizer
import argparse
from tqdm import tqdm

def load_dataset(path):
    with open(path, 'r') as file:
        data = json.load(file)
    print(f"Dataset loaded with structure: {type(data)}")
    if isinstance(data, dict):
        # print some example keys to help debug
        print(f"Example keys: {list(data.keys())[:5]}")
    return data

def load_model(device):
    model_name = "princeton-nlp/sup-simcse-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    print("Loaded model!")
    return model, tokenizer

def get_embeddings(model, tokenizer, texts, device, batch_size):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    embeddings = []

    for i in tqdm(range(0, len(inputs['input_ids']), batch_size)):
        input_ids = inputs['input_ids'][i:i+batch_size].to(device)
        attention_mask = inputs['attention_mask'][i:i+batch_size].to(device)
        batch_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        with torch.no_grad():
            batch_embeddings = model(**batch_inputs, output_hidden_states=True, return_dict=True).pooler_output
            embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings)
    return embeddings

def compute_similarity_matrix(embeddings):
    normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    return normalized_embeddings @ normalized_embeddings.T

def process_data_and_compute_similarity(dataset, model, tokenizer, device, batch_size):
    # Assuming that the dataset structure is a dictionary of dictionaries
    # and each sub-dictionary contains 'sentences' key with a list of sentences
    sentences = [sentence for video_id, video_data in dataset.items() for sentence in video_data['sentences']]
    query_embeddings = get_embeddings(model, tokenizer, sentences, device, batch_size)
    similarity_matrix = compute_similarity_matrix(query_embeddings)
    return similarity_matrix

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model(device)
    dataset = load_dataset(args.dataset_path)
    similarity_matrix = process_data_and_compute_similarity(dataset, model, tokenizer, device, args.batch_size)
    output_filename = f"similarity_matrix_{args.dataset_name}_{args.dataset_type}.pt"
    torch.save(similarity_matrix, output_filename)
    print(f"Saved similarity matrix to {output_filename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute text embeddings and similarity matrices.")
    
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset json file, including the type as part of the filename.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset, used for naming the output file.")
    parser.add_argument("--dataset_type", type=str, default="test",
                        help="Type of the dataset, train of test")
    parser.add_argument("--batch_size", type=int, default=48,
                        help="Batch size for processing embeddings.")

    args = parser.parse_args()
    main(args)