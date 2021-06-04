# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""GRU preprocess script."""
import os
import argparse
from src.dataset import create_gru_dataset
from src.config import config

parser = argparse.ArgumentParser(description='GRU preprocess')
parser.add_argument("--dataset_path", type=str, default="",
                    help="Dataset path, default: f`sns.")
parser.add_argument('--device_num', type=int, default=1, help='Use device nums, default is 1')
parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')
args = parser.parse_args()

if __name__ == "__main__":
    mindrecord_file = args.dataset_path
    if not os.path.exists(mindrecord_file):
        print("dataset file {} not exists, please check!".format(mindrecord_file))
        raise ValueError(mindrecord_file)
    dataset = create_gru_dataset(epoch_count=config.num_epochs, batch_size=config.eval_batch_size, \
        dataset_path=mindrecord_file, rank_size=args.device_num, rank_id=0, do_shuffle=False, is_training=False)

    source_ids_path = os.path.join(args.result_path, "00_data")
    target_ids_path = os.path.join(args.result_path, "01_data")
    os.makedirs(source_ids_path)
    os.makedirs(target_ids_path)

    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        file_name = "gru_bs" + str(config.eval_batch_size) + "_" + str(i) + ".bin"
        data["source_ids"].tofile(os.path.join(source_ids_path, file_name))
        data["target_ids"].tofile(os.path.join(target_ids_path, file_name))

    print("="*20, "export bin files finished", "="*20)
