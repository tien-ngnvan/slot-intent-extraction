from transformers import AutoConfig, TFAutoModel

def convert_tfmodel(dir_pytorch_model, dir_config):
    config = AutoConfig.from_pretrained(dir_config)
    tf_model = TFAutoModel.from_pretrained(dir_pytorch_model, config=config, from_pt=True)
    tf_model.layers[0].trainable = False

    return tf_model

if __name__ == '__main__':
    bert = convert_tfmodel('vinai/phobert-base', 'vinai/phobert-base')