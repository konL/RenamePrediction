from bert4keras.snippets import sequence_padding, DataGenerator

from prediction.keras4bert_dataset import load_data
config_path='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_config.json'
checkpoint_path='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\bert_model.ckpt'

dict_path='C:\\Users\\delll\\Desktop\\liangjh\\iden_project\\uncased_L-12_H-768_A-12\\vocab.txt'
from bert4keras.tokenizers import Tokenizer
tokenizer = Tokenizer(dict_path)
class data_generator(DataGenerator):

    """
    数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        token_len=0
        index=1
        # for is_end, (text,text1, label) in self.sample(random):
        for is_end, (oldname,text,text1,body1,body2, edge,label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text,text1, maxlen=312)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
proj="dubbo"
test_data, test = load_data(
        "C:\\project\\IdentifierStyle\\data\\VersionDB\\prepocessed_data_extend\\test_data\\dubbo_test.csv",
      )
print(test.columns)
print(test['oldStmt_body'])