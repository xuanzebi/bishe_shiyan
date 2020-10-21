import os
from utils.bmes_decode import bmes_decode
import json

def convert_file(input_file, output_file, tag2query_file):
    """
    Convert NER raw data to MRC format
    """
    origin_count = 0
    new_count = 0
    tag2query = json.load(open(tag2query_file,'r',encoding='utf-8'))
    mrc_samples = []
    old_datas = json.load(open(input_file,'r',encoding='utf-8'))

    for line in old_datas:
        src = line[0]
        labels = line[1]
        origin_count += 1

        src_list = src.split()
        labels_list = labels.split()

        tags = bmes_decode(char_label_list=[(char, label) for char, label in zip(src.split(), labels.split())])
        for tag_idx,(label, query) in enumerate(tag2query.items()):
            mrc_samples.append(
                {
                    "context": src,
                    "start_position": [tag.begin for tag in tags if tag.tag == label],
                    "end_position": [tag.end-1 for tag in tags if tag.tag == label],
                    "query": query,
                    "qas_id": f"{origin_count-1}.{tag_idx+1}",
                    "entity_label":label
                }
            )
            new_count += 1

    json.dump(mrc_samples, open(output_file, "w",encoding='utf-8'), ensure_ascii=False, sort_keys=True, indent=2)
    print(f"Convert {origin_count} samples to {new_count} samples and save to {output_file}")

def main():
    msra_raw_dir = "D:/bishe_shiyan/NER_CH_DATA/CyssecurityNER"
    msra_mrc_dir = "D:/bishe_shiyan/mrc-for-flat-nested-ner/datasets/cysecurity"
    tag2query_file = "queries/cysecurity.json"

    for phase in ["train_data", "dev_data", "test_data"]:
        old_file = os.path.join(msra_raw_dir, f"{phase}.json")
        new_file = os.path.join(msra_mrc_dir, f"cys_mrc.{phase.split('_')[0]}")
        convert_file(old_file, new_file, tag2query_file)


if __name__ == '__main__':
    main()
