# Speaker Recognition - ECAPA-TDNN Model
This repo contains the training, inference, and evaluation of speaker recognition models.
There are two types of model, based on ECAPA-TDNN architecture: Vanilla ECAPA-TDNN (ECAPAModel.py) and Adversarial-trained ECAPA-TDNN (AdversarialModel.py), the Adversarial performs generally better, specially for multilingual speakers. For benchmarking, please use the Vanilla ECAPA-TDNN.

The are two types of loss function implemented: the self-implemented Large Margin Cosine Loss (for multilingual speakers, based on [CosFace paper](https://arxiv.org/abs/1801.09414)), and AAM-Softmax loss as the implementation of the [original paper](https://arxiv.org/abs/2005.07143). 


## Training and Evaluation
- Prepare a text file `train_list` for the training corpus, containing the lists of path to audios of speakers. The format is as follows: `speaker_id \t path`, where `path` is the ABOSULTE path of the audio.
- Prepare the `eval_list`, which is the development corpus. Each line contains `label \t enrol_path \t test_path`.
- We also have `eval_path` containing the absolute path to the directory containing the test audios. The path in `eval_list` can be relative w.r.t the the `eval_path` directory, otherwise you can leave them as absolute path and no need to care about `eval_path`.


To train the model (you can refer to `trainECAPAModel.py` for the meaning of each argument). Remember to change `n_class` to the corresponding number of unique speakers in the training corpus. The model checkpoints and loggings are stored in `./exp`, you may want to change this for every different model:
```
python trainECAPAModel.py --save_path ./exp \
--lr 0.0005 \
--max_epoch 50 \
--test_step 10 \
--train_list \
--train_path \
--eval_list \
--eval_path \
--musan_path /home3/thanhpv/speaker_verification/slt/ECAPA-TDNN/voxceleb_trainer/data_augment/musan_split \
--rir_path /home3/thanhpv/speaker_verification/slt/ECAPA-TDNN/voxceleb_trainer/data_augment/RIRS_NOISES/simulated_rirs \
--batch_size 100 \
--initial_model \
--n_class 4  
```

To evaluate the model (just use an additional `--eval` flag, change `eval_list` to the test corpus, and `initial_model` to the model to be evaluated):
```
python trainECAPAModel.py --save_path ./exp_emo \
--lr 0.0005 \
--max_epoch 50 \
--test_step 10 \
--train_list \
--train_path \
--eval_list \
--eval_path \
--musan_path /home3/thanhpv/speaker_verification/slt/ECAPA-TDNN/voxceleb_trainer/data_augment/musan_split \
--rir_path /home3/thanhpv/speaker_verification/slt/ECAPA-TDNN/voxceleb_trainer/data_augment/RIRS_NOISES/simulated_rirs \
--batch_size 100 \
--initial_model \
--n_class 4 \
--eval
```

## Speaker Embedding Extraction
Run the `infer_embedding.py` file, preparing a .csv file where each line contains the audio path to infer speaker embedding, with one header column: `path`.


## References
Original unofficial implementation of the Vanilla ECAPA-TDNN model: [GitHub](https://github.com/TaoRuijie/ECAPA-TDNN?tab=readme-ov-file)

## Contacts
Vu Hoang - [Mail](longvu200502@gmail.com), [Facebook](https://www.facebook.com/long.vu.02/).
